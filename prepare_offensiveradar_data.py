import os
import json
import argparse

import numpy as np
import pandas as pd

from utils.utils import newspapers

OUTPUT_COLUMNS = [
    "video_id", "newspaper", "video_text", "video_description", "video_topic",
    "num_comments", "num_offensive_comments", "offensive_score", "offensive_trigger",
    "type",
]

OFFENSIVE_THRESHOLD = 0.7
LAMBDA = 1.0

def offensive_score(n, n_off, lambda1=1.0):
    if n == 0:
        return 0.0
    if n_off == 0:
        return 0.0
    return np.arctan((n_off / n) + lambda1 * np.log(n_off)) / (np.pi / 2)

def parse_annotated_filename(filename: str):
    for ext in (".json", ".csv"):
        for suffix, kind in (("_gold", "gold"), ("_silver", "silver")):
            tag = suffix + ext
            if filename.endswith(tag):
                return filename[:-len(tag)], kind
    return None


def index_annotated(dir_path: str) -> dict:
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for fn in sorted(os.listdir(dir_path)):
        parsed = parse_annotated_filename(fn)
        if parsed is None:
            continue
        file_id, kind = parsed
        if file_id in out and out[file_id][1] == "gold":
            continue                                   
        if file_id not in out or kind == "gold":
            out[file_id] = (os.path.join(dir_path, fn), kind)
    return out

def read_meta(meta_path: str):
    """Return (title, description, topic) from a metadata JSON."""
    with open(meta_path, "r", encoding="utf-8") as f:
        md = json.load(f)
    title = (md.get("title", "") or "").strip()
    description = (md.get("description", "") or "").strip()
    topic = md.get("topic", "")
    return title, description, topic


def read_comment_stats(comm_path: str):
    """Return (num_comments, n_offensive). Files with no rows (including
    completely empty files) yield (0, 0). """
    try:
        df = pd.read_csv(comm_path)
    except pd.errors.EmptyDataError:
        return 0, 0                                    
    num_comments = len(df)
    if num_comments == 0:
        return 0, 0
    if "label" not in df.columns:
        print(f"  WARNING: no 'label' column in {comm_path} — treated as 0 offensive")
        return num_comments, 0
    labels = pd.to_numeric(df["label"], errors="coerce")    
    n_offensive = int((labels == 1).sum())
    return num_comments, n_offensive


def resolve_type(meta_type, comm_type, newspaper, file_id):
    """Single gold/silver tag from the two source suffixes; flag disagreements."""
    present = [t for t in (meta_type, comm_type) if t is not None]
    if not present:
        return ""
    if len(set(present)) == 1:
        return present[0]
    print(f"  WARNING: {newspaper}/{file_id} has mismatched sources "
          f"(metadata={meta_type}, comments={comm_type}) — recorded as combined")
    return f"{meta_type}/{comm_type}"


def build_dataset(annotated_metadata_dir: str, annotated_comments_dir: str) -> pd.DataFrame:
    rows = []
    n_no_meta = n_no_comments = 0

    for newspaper in newspapers:
        meta_idx = index_annotated(os.path.join(annotated_metadata_dir, newspaper))
        comm_idx = index_annotated(os.path.join(annotated_comments_dir, newspaper))

        all_ids = sorted(set(meta_idx) | set(comm_idx))
        if not all_ids:
            print(f"WARNING: nothing found for newspaper '{newspaper}'")
            continue

        for file_id in all_ids:
            title, description, topic = "", "", ""
            meta_type = comm_type = None
            if file_id in meta_idx:
                meta_path, meta_type = meta_idx[file_id]
                try:
                    title, description, topic = read_meta(meta_path)
                except Exception as e:
                    print(f"  ERROR reading metadata {newspaper}/{file_id}: {e}")
            else:
                n_no_meta += 1
                print(f"  WARNING: {newspaper}/{file_id} has comments but no metadata "
                      f"— text/description/topic left empty")

            num_comments, n_offensive = 0, 0
            if file_id in comm_idx:
                comm_path, comm_type = comm_idx[file_id]
                try:
                    num_comments, n_offensive = read_comment_stats(comm_path)
                except Exception as e:
                    print(f"  ERROR reading comments {newspaper}/{file_id}: {e}")
            else:
                n_no_comments += 1
                print(f"  WARNING: {newspaper}/{file_id} has metadata but no comments "
                      f"— offensive_score = 0.0")

            score = offensive_score(num_comments, n_offensive, lambda1=LAMBDA)

            rows.append({
                "video_id": file_id,
                "newspaper": newspaper,
                "video_text": title,
                "video_description": description,
                "video_topic": topic,
                "num_comments": num_comments,
                "num_offensive_comments": n_offensive,
                "offensive_score": round(score, 4),
                "offensive_trigger": 1 if score > OFFENSIVE_THRESHOLD else 0,
                "type": resolve_type(meta_type, comm_type, newspaper, file_id),
            })

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if n_no_meta or n_no_comments:
        print(f"\nMissing counterparts -> videos without metadata: {n_no_meta}, "
              f"without comments: {n_no_comments}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Build a per-video dataset from gold + silver annotations.")
    parser.add_argument("--out", default="video_dataset.csv",
                        help="Output CSV path (default: video_dataset.csv).")
    return parser.parse_args()


def main():
    args = parse_args()
    base =  "VideosComments"
    annotated_metadata_dir = f"{base}/youtube/annotated_metadata"
    annotated_comments_dir = f"{base}/youtube/annotated_comments"

    df = build_dataset(annotated_metadata_dir, annotated_comments_dir)
    df.to_csv(args.out, index=False)

    print(f"\n{'='*60}")
    print(f"Wrote {len(df)} videos -> {args.out}")
    if len(df):
        print(f"  offensive triggers (score > {OFFENSIVE_THRESHOLD}): "
              f"{int(df['offensive_trigger'].sum())}")
        print(f"  total comments: {int(df['num_comments'].sum())}")
        print(f"  total offensive comments: {int(df['num_offensive_comments'].sum())}")
        print(f"  mean offensive_score: {df['offensive_score'].mean():.4f}")
        print(f"  by type: {df['type'].value_counts().to_dict()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()