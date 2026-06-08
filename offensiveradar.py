import os
import gc
import argparse

from dotenv import load_dotenv

import torch
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from utils.inference_utils import ModelSpec, REGISTRY, GLOBAL_LOAD_KWARGS, load_model, build_messages
from utils.utils import parse_label
from utils.inference_utils import run_inference_comments as run_inference
from utils.prompts import OFFENSIVE_RADAR_PROMPTS as PROMPTS

BATCH_SIZE = 8          
MAX_LENGTH = 4096

REQUIRED_COLUMNS = ["video_text", "video_description", "offensive_trigger"]


def load_dataset(input_path: str, type_filter: str = "all"):
    """Read the per-video CSV"""
    df = pd.read_csv(input_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    if type_filter != "all":
        before = len(df)
        df = df[df["type"].astype(str) == type_filter].copy()
        print(f"Type filter '{type_filter}': {len(df)}/{before} videos kept")

    label = pd.to_numeric(df["offensive_trigger"], errors="coerce")
    n_dropped = int(label.isna().sum())
    if n_dropped:
        print(f"Dropping {n_dropped} videos with no offensive_trigger label — undefined ground truth")
    df = df[label.notna()].copy()
    df["true_label"] = pd.to_numeric(df["offensive_trigger"], errors="coerce").astype(int)

    df["__title"] = (
        df["video_text"].fillna("N/A").astype(str).str.strip().replace("", "N/A")
    )
    df["__description"] = (
        df["video_description"].fillna("N/A").astype(str).str.strip().replace("", "N/A")
    )

    n_pos = int(df["true_label"].sum())
    print(f"Loaded {len(df)} videos | triggering (label 1): {n_pos} | "
          f"non-triggering (label 0): {len(df) - n_pos}")
    return df.reset_index(drop=True)


TARGET_NAMES = ["non-triggering (No)", "triggering (Si)"]

def write_outputs(out_dir: str, model_key: str, prompt_id: int,
                  df: pd.DataFrame, preds: list):
    """Write results/<model>/prompt_<id>.csv (every input column + prediction)
    and .txt (metrics)."""
    model_dir = os.path.join(out_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    csv_path = os.path.join(model_dir, f"prompt_{prompt_id}.csv")
    txt_path = os.path.join(model_dir, f"prompt_{prompt_id}.txt")

    out_df = df.drop(columns=["__title", "__description"], errors="ignore").copy()
    out_df["pred_label"] = preds
    out_df.to_csv(csv_path, index=False)

    pairs = [(t, p) for t, p in zip(df["true_label"].tolist(), preds) if p is not None]
    n_total = len(preds)
    n_unparseable = n_total - len(pairs)
    n_pos = int(df["true_label"].sum())

    lines = [
        f"Model:   {model_key}",
        f"Prompt:  {prompt_id}",
        f"Label:   offensive_trigger (taken directly from the dataset)",
        f"Videos evaluated:        {n_total}",
        f"Unparseable (excluded):  {n_unparseable}",
        f"Label distribution:      triggering (1): {n_pos} | non-triggering (0): {n_total - n_pos}",
        "",
    ]

    if pairs:
        y_true, y_pred = zip(*pairs)
        report = classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=TARGET_NAMES, digits=4, zero_division=0,
        )
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average="macro", zero_division=0,
        )
        lines.append(report)
        lines.append("")
        lines.append(f"Macro-averaged -> precision: {p:.4f} | recall: {r:.4f} | f1: {f1:.4f}")
    else:
        lines.append("No parseable predictions — metrics unavailable.")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  -> {csv_path}")
    print(f"  -> {txt_path}")


def parse_args():
    ap = argparse.ArgumentParser(description="Predict offensiveness-triggering videos with multiple LLMs and prompts.")
    ap.add_argument("--input", default="video_dataset.csv",
                    help="Per-video dataset CSV (must contain video_text, video_description, offensive_trigger).")
    ap.add_argument("--out-dir", default="results",
                    help="Root output directory (default: results).")
    ap.add_argument("--models", nargs="+", choices=list(REGISTRY), default=list(REGISTRY),
                    help="Which models to run (default: all in the registry).")
    ap.add_argument("--type", choices=["all", "gold", "silver"], default="all",
                    help="Restrict ground truth to gold/silver videos (default: all).")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return ap.parse_args()


def main():
    load_dotenv()  
    args = parse_args()

    df = load_dataset(args.input, type_filter=args.type)
    if df.empty:
        print("No videos to evaluate — aborting.")
        return

    tds = list(zip(df["__title"].tolist(), df["__description"].tolist()))
    prompt_sets = [
        [tmpl.format(video_title=t, video_description=d) for (t, d) in tds]
        for tmpl in PROMPTS
    ]

    for model_key in args.models:
        spec = REGISTRY[model_key]
        print(f"\n{'#'*60}\nModel: {model_key} ({spec.name})\n{'#'*60}")
        model, proc = load_model(spec)
        try:
            for prompt_id, prompts in enumerate(prompt_sets):
                print(f"\n[{model_key}] prompt {prompt_id}")
                preds = run_inference(model, proc, spec, prompts, batch_size=args.batch_size)
                write_outputs(args.out_dir, model_key, prompt_id, df, preds)
        finally:
            del model, proc
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nDone. Results under '{args.out_dir}/'")


if __name__ == "__main__":
    main()