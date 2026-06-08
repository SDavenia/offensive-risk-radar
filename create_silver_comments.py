import os
import json
import gc
import re
import argparse

from dotenv import load_dotenv

import torch
import pandas as pd


from utils.prompts import ANNOTATION_PROMPT_COMMENTS as ANNOTATION_PROMPT
from utils.utils import newspapers, add_context_columns, parse_label
from utils.inference_utils import ModelSpec, REGISTRY, GLOBAL_LOAD_KWARGS, load_model, build_messages
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


def load_metadata(metadata_file_path_full):
    """Return (title, description) for a video, with the description truncated at
    the last sentence boundary before 1000 chars. Missing metadata is non-fatal
    for silver labelling: we just fall back to empty strings."""
    if not os.path.exists(metadata_file_path_full):
        print(f"  WARNING: metadata not found ({metadata_file_path_full}) — using empty title/description")
        return "", ""

    with open(metadata_file_path_full, "r") as f:
        metadata = json.load(f)
    video_title = metadata.get("title", "")
    video_description = metadata.get("description", "")

    if len(video_description) > 1000:
        trunc_point = video_description.rfind(".", 0, 1000)
        video_description = (
            video_description[:trunc_point + 1] if trunc_point != -1
            else video_description[:1000]
        )
    return video_title, video_description


def prepare_df(df, video_title, video_description, newspaper):
    """Attach metadata columns + per-comment context + the rendered prompt."""
    df["newspaper"] = newspaper
    df["video_title"] = video_title
    df["video_description"] = video_description
    df = add_context_columns(df)
    df["annotation_prompt"] = df.apply(create_prompt, axis=1)
    return df



def annotate_silver(model, proc, spec: ModelSpec,
                    comments_dir: str, metadata_dir: str, annotated_dir: str,
                    batch_size: int = BATCH_SIZE, overwrite: bool = False):
    """
    Pool every comment across every file into ONE batched inference run, so that
    a 1-comment file no longer wastes a whole batch. Predictions are scattered
    back to each file afterwards via the [start:end] slice it occupies.
    """
    n_skipped_gold = n_skipped_silver = n_skipped_empty = n_written = n_errors = 0

    jobs = []          
    all_prompts = [] 

    for newspaper in newspapers:
        comments_np = os.path.join(comments_dir, newspaper)
        metadata_np = os.path.join(metadata_dir, newspaper)
        annotated_np = os.path.join(annotated_dir, newspaper)

        if not os.path.isdir(comments_np):
            print(f"WARNING: no comments dir for '{newspaper}' ({comments_np}) — skipping")
            continue
        os.makedirs(annotated_np, exist_ok=True)

        for filename in sorted(os.listdir(comments_np)):
            if not filename.endswith(".csv"):
                continue
            if filename.endswith("_gold.csv") or filename.endswith("_silver.csv"):
                continue

            file_id = filename[:-4]
            gold_path = os.path.join(annotated_np, f"{file_id}_gold.csv")
            silver_path = os.path.join(annotated_np, f"{file_id}_silver.csv")

            if os.path.exists(gold_path):
                n_skipped_gold += 1
                continue
            if os.path.exists(silver_path) and not overwrite:
                n_skipped_silver += 1
                continue

            comment_path = os.path.join(comments_np, filename)
            try:
                df = pd.read_csv(comment_path)
            except pd.errors.EmptyDataError:
                print(f"[{newspaper}/{file_id}] empty file (no header/rows) — writing empty silver")
                pd.DataFrame(columns=["label"]).to_csv(silver_path, index=False)
                n_skipped_empty += 1
                continue

            if df.empty:
                print(f"[{newspaper}/{file_id}] no comments — writing empty silver")
                out_cols = [c for c in df.columns.tolist() if c != "label"] + ["label"]
                df["label"] = pd.Series(dtype=object)
                df[out_cols].to_csv(silver_path, index=False)
                n_skipped_empty += 1
                continue
            try:
                out_cols = [c for c in df.columns.tolist() if c != "label"] + ["label"]
                video_title, video_description = load_metadata(
                    os.path.join(metadata_np, f"{file_id}.json")
                )
                df = prepare_df(df, video_title, video_description, newspaper)
                prompts = df["annotation_prompt"].tolist()
            except Exception as e:
                n_errors += 1
                print(f"  ERROR preparing {newspaper}/{file_id}: {e}")
                continue

            start = len(all_prompts)
            all_prompts.extend(prompts)
            jobs.append({
                "newspaper": newspaper,
                "file_id": file_id,
                "df": df,
                "out_cols": out_cols,
                "silver_path": silver_path,
                "start": start,
                "end": start + len(prompts),
            })

    if not all_prompts:
        print("Nothing to annotate.")
        print(f"skipped (gold): {n_skipped_gold} | "
              f"skipped (existing silver): {n_skipped_silver} | "
              f"skipped (empty): {n_skipped_empty} | errors: {n_errors}")
        return

    # Batched inference over all comments
    print(f"\nAnnotating {len(all_prompts)} comments across {len(jobs)} files "
          f"in batches of {batch_size}...")
    all_preds = run_inference(model, proc, spec, all_prompts, batch_size=batch_size)

    # Rebuild silver files with predictions in the right place
    for job in jobs:
        df = job["df"]
        preds = all_preds[job["start"]:job["end"]]
        df["label"] = preds

        n_none = sum(p is None for p in preds)
        if n_none:
            print(f"  [{job['newspaper']}/{job['file_id']}] "
                  f"{n_none}/{len(preds)} unparseable — written as empty label")

        try:
            df[job["out_cols"]].to_csv(job["silver_path"], index=False)
            print(f"  -> {job['silver_path']}  ({len(df)} comments)")
            n_written += 1
        except Exception as e:
            n_errors += 1
            print(f"  ERROR writing {job['newspaper']}/{job['file_id']}: {e}")

    print(f"\n{'='*60}")
    print(f"Done. silver written: {n_written} | "
          f"skipped (gold): {n_skipped_gold} | "
          f"skipped (existing silver): {n_skipped_silver} | "
          f"skipped (empty): {n_skipped_empty} | "
          f"errors: {n_errors}")
    print(f"{'='*60}")

def parse_args():
    ap = argparse.ArgumentParser(description="Silver-annotate offensive comments with the best model.")
    ap.add_argument("--model", required=True, choices=list(REGISTRY),
                    help="Which (best) model to use for silver annotation.")
    ap.add_argument("--use-fake-data", action="store_true",
                    help="Use the VideosComments_fake tree instead of VideosComments.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate even if a *_silver.csv already exists.")
    return ap.parse_args()


def main():
    load_dotenv()  # for Hugging Face API keys, if needed
    args = parse_args()

    base = "VideosComments_fake" if args.use_fake_data else "VideosComments"
    comments_dir = f"{base}/youtube/anonymized_comments"
    metadata_dir = f"{base}/youtube/metadata"
    annotated_dir = f"{base}/youtube/annotated_comments"

    spec = REGISTRY[args.model]
    print(f"Loading best model: {args.model} ({spec.name})")
    model, proc = load_model(spec)
    try:
        annotate_silver(
            model, proc, spec,
            comments_dir=comments_dir,
            metadata_dir=metadata_dir,
            annotated_dir=annotated_dir,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    finally:
        del model, proc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()