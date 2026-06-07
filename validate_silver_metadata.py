import os
import json
import gc
import re

from dotenv import load_dotenv

import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from utils.taxonomy import TAXONOMY, _format_taxonomy
from utils.prompts import ANNOTATION_PROMPT_METADATA as ANNOTATION_PROMPT
from utils.inference_utils import ModelSpec, REGISTRY
from utils.utils import newspapers

from utils.taxonomy import canonical_topic, parse_topic
from utils.inference_utils import build_messages, load_model
from utils.inference_utils import run_inference_topic as run_inference

TAXONOMY_BLOCK = _format_taxonomy(TAXONOMY)
BATCH_SIZE = 8         
MAX_LENGTH = 4096      

# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def create_prompt(row) -> str:
    return ANNOTATION_PROMPT.format(
        taxonomy=TAXONOMY_BLOCK,
        article_title=row["video_title"] if row["video_title"] else "N/A",
        article_description=row["video_description"] if row["video_description"] else "N/A",
    )


def load_gold_data(use_fake_data: bool = True) -> pd.DataFrame:
    """Load every gold-annotated article from annotated_metadata.

    Each metadata JSON describes one article (title, description) and carries the
    gold label in its "topic" field. One row == one article."""
    base = "VideosComments_fake" if use_fake_data else "VideosComments"
    input_gold_metadata_dir = f"{base}/youtube/annotated_metadata"

    rows = []
    for newspaper in newspapers:
        metadata_dir = os.path.join(input_gold_metadata_dir, newspaper)
        if not os.path.isdir(metadata_dir):
            continue
        # Only load gold ones
        for filename in sorted(os.listdir(metadata_dir)):
            if not filename.endswith("_gold.json"):
                continue

            with open(os.path.join(metadata_dir, filename), "r", encoding="utf-8") as f:
                metadata = json.load(f)

            topic = metadata.get("topic")
            if topic is None or (isinstance(topic, str) and not topic.strip()):
                # No gold topic on this article -> not usable for selection/eval.
                print(f"WARNING: no 'topic' field in {newspaper}/{filename} — skipped")
                continue

            video_title = metadata.get("title", "") or ""
            video_description = metadata.get("description", "") or ""

            # Truncate long descriptions at the last sentence boundary before 1000 chars.
            if len(video_description) > 1000:
                trunc_point = video_description.rfind(".", 0, 1000)
                video_description = (
                    video_description[:trunc_point + 1] if trunc_point != -1
                    else video_description[:1000]
                )

            rows.append({
                "newspaper": newspaper,
                "filename": filename,
                "video_title": video_title,
                "video_description": video_description,
                "topic_raw": topic,                  # original gold value, kept for audit
                "topic": canonical_topic(topic),     # mapped to a top-level category
            })

    if not rows:
        raise ValueError(f"No annotated metadata with a 'topic' field found under {input_gold_metadata_dir}")

    df = pd.DataFrame(rows)
    df["annotation_prompt"] = df.apply(create_prompt, axis=1)
    return df


def make_dev_test_split(gold_df, test_size: float = 0.4, seed: int = 42):
    """
    60/40 dev/test, stratified on topic so rare categories are not all dumped
    into one side. With many categories a stratum may be too small to split
    (train_test_split needs >= 2 members per stratum); fall back to a plain
    random split in that case.
    """
    strat = gold_df["topic"]
    if strat.value_counts().min() < 2:
        print("WARNING: some topics have < 2 examples — splitting without stratification")
        strat = None
    return train_test_split(gold_df, test_size=test_size, random_state=seed, stratify=strat)



# --------------------------------------------------------------------------- #
# Scoring, selection, evaluation
# --------------------------------------------------------------------------- #

def score(y_true: list, y_pred: list, label: str) -> dict:
    """Drop unparseable predictions, report, return macro-F1 across categories."""
    n_none = sum(p is None for p in y_pred)
    if n_none:
        print(f"  [{label}] {n_none}/{len(y_pred)} unparseable — excluded from metrics")

    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    if not pairs:
        print(f"  [{label}] no parseable predictions — macro-F1 = 0")
        return {"macro_f1": 0.0, "report": "no parseable predictions", "n_unparseable": n_none}

    yt, yp = zip(*pairs)
    report = classification_report(yt, yp, digits=4, zero_division=0)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"\n--- {label} ---\n{report}\nMacro-F1: {macro_f1:.4f}")
    return {"macro_f1": macro_f1, "report": report, "n_unparseable": n_none}


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
            "topic": labels,
            "pred": preds,
        })
        df_out.to_csv(f"try_{key}.csv", index=False)

    metrics = score(labels, preds, label=f"{key} | dev")
    metrics.update(key=key, preds=preds)
    return metrics


def select_and_evaluate(dev_df, test_df, model_keys, batch_size: int = BATCH_SIZE, test: bool = False) -> dict:
    """Run every model on dev, pick best macro-F1, evaluate that one on test."""
    dev_prompts, dev_labels = dev_df["annotation_prompt"].tolist(), dev_df["topic"].tolist()

    dev_results = [run_one_model(k, dev_prompts, dev_labels, batch_size, test=test) for k in model_keys]

    best = max(dev_results, key=lambda m: m["macro_f1"])
    print(f"\n{'='*60}\nBest on dev: {best['key']}  (macro-F1={best['macro_f1']:.4f})\n{'='*60}")

    # Final, single evaluation on the untouched test split.
    test_prompts, test_labels = test_df["annotation_prompt"].tolist(), test_df["topic"].tolist()
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
        "dev_macro_f1": best["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "dev_report": best["report"],
        "test_report": test_metrics["report"],
    }
    with open("results_summary_topics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved -> results_summary_topics.json")
    return summary


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    load_dotenv()  # for Hugging Face API keys, if needed

    gold_df = load_gold_data(use_fake_data=False)   # flip to False for the real run
    # gold_df.to_csv("try.csv", index=False)         # keep for inspection

    # TODO: For small testing runs
    test = False
    if test:
        n = min(50, len(gold_df))
        gold_df = gold_df.sample(n, random_state=42).reset_index(drop=True)

    dev_df, test_df = make_dev_test_split(gold_df)
    print(f"Correctly split into dev and test set")
    select_and_evaluate(dev_df, test_df, model_keys=list(REGISTRY), batch_size=BATCH_SIZE, test=test)


if __name__ == "__main__":
    main()