"""
Edge-LLM Algorithmic Accuracy & Ablation Suite for SNAX FP4 Subsystem (Tier A).

Evaluates small/edge LLMs (e.g., Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B/3B, SmolLM2-1.7B):
  1. Perplexity (PPL) on WikiText-2 and C4
  2. Zero-shot Accuracy on PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, MMLU (via lm-eval)
  3. Ablation: FP16 vs. MXFP4 vs. NVFP4 vs. Log8-FP4 vs. Log8-FP4 + Exp-Free Softmax Requantizer
  4. Attention Output (O) L1 Error & Cosine Similarity: Standard FP16 Softmax vs. Exp-Free Softmax
"""

import argparse
import contextlib
import csv
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantizers import (
    exp_free_softmax_requantizer,
    quantize_log8_fp4,
    quantize_mxfp4,
    quantize_nvfp4,
)

QUANT_FNS: Dict[str, Optional[Callable[[torch.Tensor], torch.Tensor]]] = {
    "fp16": None,
    "mxfp4": quantize_mxfp4,
    "nvfp4": quantize_nvfp4,
    "log8_fp4": quantize_log8_fp4,
}


class QuantLinearWrapper(nn.Module):
    """Wraps nn.Linear to simulate W4A4 block-scaled quantization on the fly."""

    def __init__(self, linear: nn.Linear, quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.linear = linear
        self.quant_fn = quant_fn
        # Pre-quantize static weights once for speed
        if quant_fn is not None:
            with torch.no_grad():
                self.q_weight = quant_fn(linear.weight.data)
        else:
            self.q_weight = linear.weight.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.quant_fn is not None:
            x = self.quant_fn(x)
        return F.linear(x, self.q_weight, self.linear.bias)


def apply_linear_quantization(
    model: nn.Module,
    quant_mode: str,
    quant_lm_head: bool = False,
) -> List[tuple]:
    """Replace nn.Linear modules with QuantLinearWrapper. Returns list for restoration."""
    quant_fn = QUANT_FNS[quant_mode]
    if quant_fn is None:
        return []

    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if not quant_lm_head and "lm_head" in name:
                continue
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            wrapped = QuantLinearWrapper(module, quant_fn)
            setattr(parent, child_name, wrapped)
            replaced.append((parent, child_name, module))
    return replaced


def restore_linear_modules(replaced: List[tuple]) -> None:
    for parent, child_name, orig_module in replaced:
        setattr(parent, child_name, orig_module)


@contextlib.contextmanager
def patch_attention_sdpa(
    use_exp_free: bool,
    kv_quant_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    base_scale_mode: str = "log2e",
    metrics_log: Optional[List[Dict[str, float]]] = None,
):
    """
    Context manager that intercepts F.scaled_dot_product_attention (used by
    Llama, Qwen2, SmolLM2, Mistral in eager/sdpa mode) to simulate:
      - Q/K/V Log8-FP4 quantization
      - Exp-free Log8-FP4 Softmax Requantizer
      - Optional L1 / Cosine Similarity measurement of O vs FP16 Softmax
    """
    orig_sdpa = F.scaled_dot_product_attention

    def custom_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        if enable_gqa and query.shape[1] != key.shape[1]:
            repeat_factor = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        q_in = kv_quant_fn(query) if kv_quant_fn is not None else query
        k_in = kv_quant_fn(key) if kv_quant_fn is not None else key
        v_in = kv_quant_fn(value) if kv_quant_fn is not None else value

        d_k = q_in.shape[-1]
        scale_factor = (1.0 / math.sqrt(d_k)) if scale is None else scale
        scores = torch.matmul(q_in.float(), k_in.float().transpose(-2, -1)) * scale_factor

        q_len, k_len = scores.shape[-2], scores.shape[-1]
        mask_bool = None
        if is_causal:
            causal = torch.tril(torch.ones((q_len, k_len), dtype=torch.bool, device=scores.device))
            mask_bool = causal
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                m = attn_mask
            else:
                m = attn_mask > -1e4
            mask_bool = m if mask_bool is None else (mask_bool & m)

        # Reference FP16 Softmax output (when logging attention error)
        if metrics_log is not None:
            ref_scores = scores.masked_fill(~mask_bool, float("-inf")) if mask_bool is not None else scores
            p_ref = torch.softmax(ref_scores, dim=-1)
            o_ref = torch.matmul(p_ref, value.float())

        if use_exp_free:
            p_hat = exp_free_softmax_requantizer(scores, attn_mask=mask_bool, base_scale_mode=base_scale_mode)
            o_out = torch.matmul(p_hat.float(), v_in.float()).to(query.dtype)
        else:
            if mask_bool is not None:
                scores = scores.masked_fill(~mask_bool, float("-inf"))
            p_std = torch.softmax(scores, dim=-1)
            o_out = torch.matmul(p_std, v_in.float()).to(query.dtype)

        if metrics_log is not None:
            o_f = o_out.float()
            l1_err = (o_f - o_ref).abs().mean().item()
            rel_l1 = ((o_f - o_ref).abs().sum() / (o_ref.abs().sum() + 1e-12)).item()
            cos_sim = F.cosine_similarity(o_f.reshape(-1, d_k), o_ref.reshape(-1, d_k), dim=-1).mean().item()
            metrics_log.append({"l1_err": l1_err, "rel_l1": rel_l1, "cos_sim": cos_sim})

        return o_out

    F.scaled_dot_product_attention = custom_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str,
    seq_len: int = 1024,
    max_chunks: int = 64,
    device: str = "cuda",
) -> float:
    """Sliding-window / chunked NLL perplexity on WikiText-2 or C4."""
    if dataset_name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
        enc = tokenizer(text, return_tensors="pt").input_ids
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for i, row in enumerate(ds):
            texts.append(row["text"])
            if i >= 256:
                break
        enc = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    total_tokens = enc.shape[1]
    nlls = []
    num_chunks = min(max_chunks, total_tokens // seq_len)
    for i in tqdm(range(num_chunks), desc=f"PPL ({dataset_name})", leave=False):
        batch = enc[:, i * seq_len : (i + 1) * seq_len].to(device)
        outputs = model(batch, labels=batch)
        nlls.append(outputs.loss.float())

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return ppl


@torch.no_grad()
def evaluate_attention_error(
    model: nn.Module,
    tokenizer,
    seq_len: int = 512,
    num_batches: int = 4,
    base_scale_mode: str = "log2e",
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure L1 error & Cosine Similarity of Attention O vs FP16 Softmax."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(ds["text"][:200]), return_tensors="pt").input_ids
    metrics_log: List[Dict[str, float]] = []

    with patch_attention_sdpa(
        use_exp_free=True,
        kv_quant_fn=quantize_log8_fp4,
        base_scale_mode=base_scale_mode,
        metrics_log=metrics_log,
    ):
        for i in range(min(num_batches, enc.shape[1] // seq_len)):
            batch = enc[:, i * seq_len : (i + 1) * seq_len].to(device)
            model(batch)

    if not metrics_log:
        return {"l1_err": 0.0, "rel_l1": 0.0, "cos_sim": 1.0}

    return {
        "l1_err": sum(m["l1_err"] for m in metrics_log) / len(metrics_log),
        "rel_l1": sum(m["rel_l1"] for m in metrics_log) / len(metrics_log),
        "cos_sim": sum(m["cos_sim"] for m in metrics_log) / len(metrics_log),
    }


def evaluate_zeroshot(
    model: nn.Module,
    tokenizer,
    tasks: List[str],
    batch_size: int = 8,
    limit: Optional[int] = None,
) -> Dict[str, float]:
    """Run lm-evaluation-harness on the active (possibly quantized) model."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        print("[WARN] lm-eval not installed. Run `pip install lm-eval` to enable zero-shot tasks.")
        return {}

    lm_obj = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm_obj,
        tasks=tasks,
        limit=limit,
        log_samples=False,
    )
    scores = {}
    for task_name, task_res in results["results"].items():
        acc = task_res.get("acc_norm,none", task_res.get("acc,none", None))
        if acc is not None:
            scores[task_name] = acc * 100.0
    return scores


def main():
    parser = argparse.ArgumentParser(description="Tier-A Edge LLM Evaluation for SNAX FP4 Subsystem")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B", help="HF model ID or local path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["fp16", "mxfp4", "nvfp4", "log8_fp4", "log8_fp4_expfree"],
        choices=["fp16", "mxfp4", "nvfp4", "log8_fp4", "log8_fp4_expfree"],
        help="Quantization & Softmax configurations to benchmark",
    )
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate WikiText-2 and C4 Perplexity")
    parser.add_argument("--eval_attn_err", action="store_true", help="Measure Attention O L1 & Cosine Sim")
    parser.add_argument("--eval_zeroshot", action="store_true", help="Evaluate Zero-shot tasks via lm-eval")
    parser.add_argument(
        "--zeroshot_tasks",
        nargs="+",
        default=["piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "mmlu"],
    )
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_chunks", type=int, default=64)
    parser.add_argument("--zeroshot_limit", type=int, default=None, help="Optional sample limit per task for quick test")
    parser.add_argument("--base_scale_mode", type=str, default="log2e", choices=["log2e", "rtl_ln2"])
    parser.add_argument("--quant_lm_head", action="store_true", help="Also quantize lm_head")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    if not (args.eval_ppl or args.eval_attn_err or args.eval_zeroshot):
        args.eval_ppl = True
        args.eval_attn_err = True

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.model} on {args.device} (attn_implementation='sdpa')...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()

    all_rows = []

    if args.eval_attn_err:
        attn_stats = evaluate_attention_error(
            model, tokenizer, seq_len=min(512, args.seq_len), base_scale_mode=args.base_scale_mode, device=args.device
        )
        print(
            f"\n[Attention O Error | Exp-Free Log8-FP4 Softmax ({args.base_scale_mode}) vs FP16] "
            f"L1={attn_stats['l1_err']:.6f}, Rel-L1={attn_stats['rel_l1']*100:.2f}%, "
            f"CosSim={attn_stats['cos_sim']:.6f}\n"
        )

    for cfg in args.configs:
        print(f"=== Evaluating config: {cfg} ===")
        use_exp_free = cfg == "log8_fp4_expfree"
        linear_mode = "log8_fp4" if use_exp_free else cfg
        kv_fn = QUANT_FNS[linear_mode]

        replaced = apply_linear_quantization(model, linear_mode, quant_lm_head=args.quant_lm_head)
        row = {"model": args.model, "config": cfg}

        with patch_attention_sdpa(
            use_exp_free=use_exp_free,
            kv_quant_fn=kv_fn,
            base_scale_mode=args.base_scale_mode,
        ):
            if args.eval_ppl:
                row["ppl_wikitext2"] = round(
                    evaluate_perplexity(model, tokenizer, "wikitext2", args.seq_len, args.max_chunks, args.device), 3
                )
                row["ppl_c4"] = round(
                    evaluate_perplexity(model, tokenizer, "c4", args.seq_len, args.max_chunks, args.device), 3
                )
            if args.eval_zeroshot:
                zs = evaluate_zeroshot(model, tokenizer, args.zeroshot_tasks, limit=args.zeroshot_limit)
                for k, v in zs.items():
                    row[f"zs_{k}"] = round(v, 2)

        restore_linear_modules(replaced)
        all_rows.append(row)
        print(f"Result [{cfg}]: {row}")

    # Save CSV
    model_tag = args.model.replace("/", "_")
    csv_path = Path(args.out_dir) / f"eval_{model_tag}.csv"
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved summary CSV to: {csv_path}")


if __name__ == "__main__":
    main()
