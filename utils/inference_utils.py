
import torch
import os
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    Gemma3ForConditionalGeneration,
)
from utils.taxonomy import parse_topic

@dataclass
class ModelSpec:
    name: str
    model_class: type
    proc_class: type
    is_vlm: bool
    supports_system_role: bool      
                                    
    max_new_tokens: int = 24        
                                    
    load_kwargs: dict = field(default_factory=dict)

REGISTRY = {
    "mistral": ModelSpec(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=True,
    ),
    "gemma2": ModelSpec(
        "google/gemma-2-27b-it",
        AutoModelForCausalLM, AutoTokenizer,
        is_vlm=False, supports_system_role=False,   
    ),
    "gemma3": ModelSpec(
        "google/gemma-3-27b-it",
        Gemma3ForConditionalGeneration, AutoProcessor,
        is_vlm=True, supports_system_role=True,
    ),
}

GLOBAL_LOAD_KWARGS = dict(
    torch_dtype=torch.bfloat16,     
    device_map="auto",              
)


def build_messages(prompt: str, spec: ModelSpec) -> list:
    """
    Wrap the annotation prompt in the chat structure the model expects.
    No custom system prompt: the whole instruction lives in the user turn, and each
    model's own default system prompt (if any) is left in place by its chat template.

    Content shape is the only thing that varies:
      - text-only causal LM (tokenizer) -> plain string
      - VLM (processor)                 -> typed list of parts
    """
    if spec.is_vlm:
        content = [{"type": "text", "text": prompt}]
    else:
        content = prompt
    return [{"role": "user", "content": content}]


def load_model(spec: ModelSpec):
    """
    Instantiate model + tokenizer/processor from a ModelSpec.
    Uses the explicit class in the spec; applies bf16 + auto sharding globally
    (spec.load_kwargs can override per model); sets left padding for batched gen.
    """
    proc = spec.proc_class.from_pretrained(spec.name)

    load_kwargs = {**GLOBAL_LOAD_KWARGS, **spec.load_kwargs}   # spec wins on conflict
    HF_TOKEN = os.getenv("HF_TOKEN")
    model = spec.model_class.from_pretrained(spec.name, token=HF_TOKEN, **load_kwargs)
    model.eval()

    # For a VLM the real tokenizer is proc.tokenizer; for a text model proc IS it.
    tok = proc.tokenizer if spec.is_vlm else proc
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, proc



def run_inference_topic(model, proc, spec: ModelSpec, prompts: list,
                  batch_size: int, max_length: int) -> list:
    """Render -> batch-tokenize -> generate -> decode new tokens -> parse topic."""
    tok = proc.tokenizer if spec.is_vlm else proc

    # Render to formatted STRINGS first, then batch-tokenize with uniform left padding.
    rendered = [
        proc.apply_chat_template(
            build_messages(p, spec),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    preds = []
    n_batches = (len(rendered) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(rendered), batch_size), 1):
        batch = rendered[start:start + batch_size]
        print(f"    batch {bi}/{n_batches}", end="\r")

        # add_special_tokens=False: the chat template already inserted BOS/specials;
        # tokenizing with the default True would add a SECOND BOS (Gemma is sensitive).
        inputs = proc(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=spec.max_new_tokens,
                do_sample=False,                       
                pad_token_id=tok.pad_token_id,
            )

        # Left padding makes prompt length uniform, so one slice recovers the
        # generated tokens for every row in the batch.
        input_len = inputs["input_ids"].shape[1]
        new_tokens = out[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        preds.extend(parse_topic(d) for d in decoded)

    print()
    return preds
