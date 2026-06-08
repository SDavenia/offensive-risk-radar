import os
import json
import gc
import argparse

from dotenv import load_dotenv

import torch

from utils.taxonomy import TAXONOMY, _format_taxonomy
from utils.prompts import ANNOTATION_PROMPT_METADATA as ANNOTATION_PROMPT
from utils.inference_utils import ModelSpec, REGISTRY
from utils.utils import newspapers

from utils.inference_utils import load_model
from utils.inference_utils import run_inference_topic as run_inference

TAXONOMY_BLOCK = _format_taxonomy(TAXONOMY)
BATCH_SIZE = 8          
MAX_LENGTH = 4096     

def read_metadata(path: str) -> dict:
    """Load a raw article metadata JSON as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prompt_fields(metadata: dict):
    """
    Extract (title, description) for the prompt
    """
    title = metadata.get("title", "") or ""
    description = metadata.get("description", "") or ""
    if len(description) > 1000:
        trunc_point = description.rfind(".", 0, 1000)
        description = description[:trunc_point + 1] if trunc_point != -1 else description[:1000]
    return title, description


def create_prompt(title: str, description: str) -> str:
    return ANNOTATION_PROMPT.format(
        taxonomy=TAXONOMY_BLOCK,
        article_title=title if title else "N/A",
        article_description=description if description else "N/A",
    )

def collect_pending(metadata_dir: str, annotated_dir: str, overwrite: bool):
    """
    Find missing silver metadata files and associate them with their raw metadata and annotation prompts. Skip articles
    """
    pending = []
    n_skipped_gold = n_skipped_silver = n_errors = 0

    for newspaper in newspapers:
        metadata_np = os.path.join(metadata_dir, newspaper)
        annotated_np = os.path.join(annotated_dir, newspaper)

        if not os.path.isdir(metadata_np):
            print(f"WARNING: no metadata dir for '{newspaper}' ({metadata_np}) — skipping")
            continue
        os.makedirs(annotated_np, exist_ok=True)

        for filename in sorted(os.listdir(metadata_np)):
            if not filename.endswith(".json"):
                continue
            if filename.endswith("_gold.json") or filename.endswith("_silver.json"):
                continue

            file_id = filename[:-5]                      
            gold_path = os.path.join(annotated_np, f"{file_id}_gold.json")
            silver_path = os.path.join(annotated_np, f"{file_id}_silver.json")

            if os.path.exists(gold_path):
                n_skipped_gold += 1
                continue                                
            # If already done, skp it
            if os.path.exists(silver_path) and not overwrite:
                n_skipped_silver += 1
                continue                                

            try:
                metadata = read_metadata(os.path.join(metadata_np, filename))
            except Exception as e:
                n_errors += 1
                print(f"  ERROR reading {newspaper}/{file_id}: {e}")
                continue

            title, description = prompt_fields(metadata)
            pending.append({
                "newspaper": newspaper,
                "file_id": file_id,
                "metadata": metadata,
                "silver_path": silver_path,
                "prompt": create_prompt(title, description),
            })

    return pending, n_skipped_gold, n_skipped_silver, n_errors


def annotate_silver(model, proc, spec: ModelSpec,
                    metadata_dir: str, annotated_dir: str,
                    batch_size: int = BATCH_SIZE, overwrite: bool = False):
    """
    Find missing silver metadata files, run inference to predict their topic, and write them out. Skip articles that
    """
    pending, n_skipped_gold, n_skipped_silver, n_errors = collect_pending(
        metadata_dir, annotated_dir, overwrite
    )

    print(f"\n{len(pending)} articles to annotate "
          f"(skipped gold: {n_skipped_gold}, existing silver: {n_skipped_silver}, "
          f"read errors: {n_errors})")

    n_written = n_none = 0
    if pending:
        preds = run_inference(
            model, proc, spec, [p["prompt"] for p in pending], batch_size=batch_size
        )

        for item, pred in zip(pending, preds):
            topic = pred if pred is not None else ""    
            if pred is None:
                n_none += 1
            out = dict(item["metadata"])                
            out["topic"] = topic                        
            try:
                with open(item["silver_path"], "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                n_written += 1
            except Exception as e:
                n_errors += 1
                print(f"  ERROR writing {item['silver_path']}: {e}")

        if n_none:
            print(f"{n_none}/{len(preds)} unparseable model outputs — written with empty topic")

    print(f"\n{'='*60}")
    print(f"Done. silver written: {n_written} | "
          f"skipped (gold): {n_skipped_gold} | "
          f"skipped (existing silver): {n_skipped_silver} | "
          f"errors: {n_errors}")
    print(f"{'='*60}")

def parse_args():
    ap = argparse.ArgumentParser(description="Silver-annotate article topics with the best model.")
    ap.add_argument("--model", required=True, choices=list(REGISTRY),
                    help="Which (best) model to use for silver annotation.")
    ap.add_argument("--use-fake-data", action="store_true",
                    help="Use the VideosComments_fake tree instead of VideosComments.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-annotate even if a *_silver.json already exists.")
    return ap.parse_args()


def main():
    load_dotenv()  # for Hugging Face API keys, if needed
    args = parse_args()

    base = "VideosComments_fake" if args.use_fake_data else "VideosComments"
    metadata_dir = f"{base}/youtube/metadata"
    annotated_dir = f"{base}/youtube/annotated_metadata"

    spec = REGISTRY[args.model]
    print(f"Loading best model: {args.model} ({spec.name})")
    model, proc = load_model(spec)
    try:
        annotate_silver(
            model, proc, spec,
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