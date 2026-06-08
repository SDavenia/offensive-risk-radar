import os
import json
import gc
import re

from dotenv import load_dotenv

import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from utils.prompts import ANNOTATION_PROMPT_COMMENTS as ANNOTATION_PROMPT
from utils.utils import newspapers, make_dev_test_split, add_context_columns, parse_label
from utils.inference_utils import ModelSpec, REGISTRY, load_model, build_messages
from utils.inference_utils import run_inference_comments as run_inference

BATCH_SIZE = 8       
MAX_LENGTH = 4096    

def create_prompt(row):
    return ANNOTATION_PROMPT.format(
        video_title=row["video_title"] if pd.notna(row["video_title"]) else "N/A",
        video_description=row["video_description"] if pd.notna(row["video_description"]) else "N/A",
        head_comment=row["head_comment_text"] if pd.notna(row["head_comment_text"]) else "N/A",
        previous_comment=row["previous_comment_text"] if pd.notna(row["previous_comment_text"]) else "N/A",
        target_comment=row["text"],
    )


def load_gold_data(use_fake_data: bool = True) -> pd.DataFrame:
    """Load all *_gold.csv files + their metadata, attach context, build prompts."""
    base = "VideosComments_fake" if use_fake_data else "VideosComments"
    input_gold_dir = f"{base}/youtube/annotated_comments"
    input_gold_metadata_dir = f"{base}/youtube/annotated_metadata"

    all_dfs = []
    for newspaper in newspapers:
        gold_file_path = os.path.join(input_gold_dir, newspaper)
        gold_metadata_path = os.path.join(input_gold_metadata_dir, newspaper)

        for filename in os.listdir(gold_file_path):
            if not filename.endswith("_gold.csv"):
                continue

            gold_file_path_full = os.path.join(gold_file_path, filename)
            metadata_file_path_full = os.path.join(
                gold_metadata_path, filename.replace(".csv", ".json")
            )
            if not os.path.exists(metadata_file_path_full):
                raise ValueError(f"Metadata file not found for {filename}: {metadata_file_path_full}")

            with open(metadata_file_path_full, "r") as f:
                metadata = json.load(f)
            video_title = metadata.get("title", "")
            video_description = metadata.get("description", "")

            # Truncate long descriptions at the last sentence boundary before 1000 chars.
            if len(video_description) > 1000:
                trunc_point = video_description.rfind(".", 0, 1000)
                video_description = (
                    video_description[:trunc_point + 1] if trunc_point != -1
                    else video_description[:1000]
                )

            df_gold = pd.read_csv(gold_file_path_full)
            df_gold["newspaper"] = newspaper
            df_gold["video_title"] = video_title
            df_gold["video_description"] = video_description

            df_gold = add_context_columns(df_gold)
            df_gold["annotation_prompt"] = df_gold.apply(create_prompt, axis=1)
            all_dfs.append(df_gold)

    return pd.concat(all_dfs, ignore_index=True)

def score(y_true: list, y_pred: list, label: str) -> dict:
    """Drop unparseable predictions, report, return F1 on the offensive class."""
    n_none = sum(p is None for p in y_pred)
    if n_none:
        print(f"  [{label}] {n_none}/{len(y_pred)} unparseable — excluded from metrics")

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        print(f"  [{label}] no parseable predictions — F1 = 0")
        return {"f1_offensive": 0.0, "report": "no parseable predictions", "n_unparseable": n_none}

    yt, yp = zip(*pairs)
    report = classification_report(yt, yp, target_names=["non-offensive", "offensive"], digits=4)
    f1 = f1_score(yt, yp, pos_label=1, zero_division=0)
    print(f"\n--- {label} ---\n{report}\nF1 (offensive): {f1:.4f}")
    return {"f1_offensive": f1, "report": report, "n_unparseable": n_none}


def run_one_model(key: str, prompts: list, labels: list, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Load a model, run it, free it, score. Self-contained so GPU memory is
    released before the next model loads (finally => freed even on error)."""
    spec = REGISTRY[key]
    model, proc = load_model(spec)
    print(f"Running with model: {key} ({spec.name})")
    try:
        preds = run_inference(model, proc, spec, prompts, batch_size=batch_size, test=test)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
    if test:
        # Save outputs to try_modelname.csv 
        df_out = pd.DataFrame({
            "prompt": prompts,
            "label": labels,
            "pred": preds,
        })
        df_out.to_csv(f"try_{key}.csv", index=False)

    metrics = score(labels, preds, label=f"{key} | dev")
    metrics.update(key=key, preds=preds)
    return metrics


def select_and_evaluate(dev_df, test_df, model_keys, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Run every model on dev, pick best F1(offensive), evaluate that one on test."""
    dev_prompts, dev_labels = dev_df["annotation_prompt"].tolist(), dev_df["label"].tolist()

    dev_results = [run_one_model(k, dev_prompts, dev_labels, batch_size, test=test) for k in model_keys]

    best = max(dev_results, key=lambda m: m["f1_offensive"])
    print(f"\n{'='*60}\nBest on dev: {best['key']}  (F1={best['f1_offensive']:.4f})\n{'='*60}")

    # Final, single evaluation on the untouched test split.
    test_prompts, test_labels = test_df["annotation_prompt"].tolist(), test_df["label"].tolist()
    spec = REGISTRY[best["key"]]
    model, proc = load_model(spec)
    try:
        test_preds = run_inference(model, proc, spec, test_prompts, batch_size=batch_size, test=test)
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()
    test_metrics = score(test_labels, test_preds, label=f"{best['key']} | test")
    
    summary = {
        "best_model": best["key"],
        "dev_f1_offensive": best["f1_offensive"],
        "test_f1_offensive": test_metrics["f1_offensive"],
        "dev_report": best["report"],
        "test_report": test_metrics["report"],
    }
    with open("results_summary_comments.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved -> results_summary_comments.json")
    return summary


def main():
    load_dotenv()  
    gold_df = load_gold_data(use_fake_data=False)   # flip to False for the real run

    # TODO: For small testing runs
    test = False
    if test:
        gold_df = gold_df.sample(50, random_state=42).reset_index(drop=True)

    dev_df, test_df = make_dev_test_split(gold_df)
    print(f"Correctly split into dev and test set")
    select_and_evaluate(dev_df, test_df, model_keys=list(REGISTRY), batch_size=BATCH_SIZE, test=test)


if __name__ == "__main__":
    main()