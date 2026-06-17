#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def add_matgptq_to_path(matgptq_dir: str) -> None:
    matgptq_path = Path(matgptq_dir).expanduser().resolve()
    sys.path.insert(0, str(matgptq_path))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect layer x token quantization sensitivity rows for H1."
    )
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--matgptq_dir", default=os.environ.get("MATGPTQ_DIR", "~/MatGPTQ"))
    parser.add_argument("--quant_weights_path", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--dataset", default="c4", choices=["c4", "wikitext2", "fineweb_edu"])
    parser.add_argument("--prompts_jsonl", default=None)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--eval_tokens", type=int, default=8192)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--module_regex", default=r".*layers.*((q|k|v|o|gate|up|down)_proj)$")
    parser.add_argument("--max_modules", type=int, default=0, help="0 means all matching modules.")
    parser.add_argument("--sample_shard_count", type=int, default=1)
    parser.add_argument("--sample_shard_index", type=int, default=0)
    parser.add_argument("--high_bit", type=int, default=8)
    parser.add_argument("--low_bits", nargs="+", type=int, default=[3, 4, 6])
    parser.add_argument("--master_bitwidth", type=int, default=8)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--attn_implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str, device: torch.device):
    if device.type == "cpu":
        return torch.float32
    return getattr(torch, name)


def natural_key(name: str):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def layer_id_from_name(name: str) -> int:
    match = re.search(r"layers\.(\d+)", name)
    return int(match.group(1)) if match else -1


def load_prompt_samples(path: str, tokenizer, sequence_length: int, max_samples: int):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("prompt")
            if not text:
                continue
            ids = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=sequence_length,
            ).input_ids
            if ids.numel() >= 2:
                samples.append((ids, obj.get("category", "prompt")))
            if len(samples) >= max_samples:
                break
    return samples


def load_dataset_samples(args, tokenizer):
    from src.data_utils import get_data

    data = get_data(
        args.dataset,
        args.eval_tokens,
        args.sequence_length,
        tokenizer,
        train=False,
    )
    return [(sample, args.dataset) for sample in data[: args.max_samples] if sample.numel() >= 2]


def load_slice_weight(quant_weights_path: Path, module_name: str, master_bit: int, slice_bit: int, dtype, device):
    from src.quant_utils_matgpqt import dequantize

    data = torch.load(quant_weights_path / module_name / "data.pt", map_location="cpu")
    qweight = data["qweight"]
    scale = data["scale"]
    zero = data["zero"]
    perm = data.get("perm")
    invperm = perm.argsort() if perm is not None else torch.arange(qweight.shape[1])
    weight = dequantize(
        qweight.view(qweight.shape[0], scale.shape[1], -1),
        scale.view(qweight.shape[0], -1, 1),
        zero.view(qweight.shape[0], -1, 1),
        master_bit,
        slice_bit,
    ).view_as(qweight)[:, invperm]
    return weight.to(device=device, dtype=dtype)


def token_metrics(logits, input_ids):
    shift_logits = logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view(labels.shape)
    probs = shift_logits.softmax(dim=-1)
    log_probs = shift_logits.log_softmax(dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top = shift_logits.topk(k=5, dim=-1).indices
    margin = shift_logits.topk(k=2, dim=-1).values
    margin = margin[..., 0] - margin[..., 1]
    return {
        "loss": losses.squeeze(0).detach().cpu(),
        "entropy": entropy.squeeze(0).detach().cpu(),
        "margin": margin.squeeze(0).detach().cpu(),
        "top5": top.squeeze(0).detach().cpu(),
    }


def collect_activation_stats(module, input_ids):
    stats = {}

    def hook(_module, inputs, _output):
        x = inputs[0].detach().float()
        x = x[:, :-1, :]
        rms = x.pow(2).mean(dim=-1).sqrt().squeeze(0)
        absmax = x.abs().amax(dim=-1).squeeze(0)
        stats["norm"] = rms.cpu()
        stats["absmax"] = absmax.cpu()
        stats["outlier"] = (absmax / (rms + 1e-8)).cpu()

    handle = module.register_forward_hook(hook)
    return stats, handle


def write_rows(writer, sample_id, category, bit, module_name, high, low, activation):
    high_top1 = high["top5"][:, 0]
    low_top1 = low["top5"][:, 0]
    for pos in range(high["loss"].shape[0]):
        high_set = set(high["top5"][pos].tolist())
        low_set = set(low["top5"][pos].tolist())
        writer.writerow(
            {
                "sample_id": sample_id,
                "category": category,
                "bit": bit,
                "layer_id": layer_id_from_name(module_name),
                "module_name": module_name,
                "token_pos": pos,
                "activation_norm": float(activation["norm"][pos]),
                "activation_absmax": float(activation["absmax"][pos]),
                "activation_outlier": float(activation["outlier"][pos]),
                "high_entropy": float(high["entropy"][pos]),
                "high_margin": float(high["margin"][pos]),
                "high_loss": float(high["loss"][pos]),
                "low_loss": float(low["loss"][pos]),
                "loss_delta": float(low["loss"][pos] - high["loss"][pos]),
                "top1_flip": int(high_top1[pos] != low_top1[pos]),
                "top5_overlap": len(high_set & low_set) / 5.0,
                "high_top1": int(high_top1[pos]),
                "low_top1": int(low_top1[pos]),
            }
        )


def main():
    args = parse_args()
    if args.sample_shard_count < 1:
        raise SystemExit("--sample_shard_count must be >= 1.")
    if not 0 <= args.sample_shard_index < args.sample_shard_count:
        raise SystemExit("--sample_shard_index must be in [0, sample_shard_count).")
    add_matgptq_to_path(args.matgptq_dir)

    from src.model_utils import load_mat_gptq_weights, select_layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype, device)
    quant_weights_path = Path(args.quant_weights_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )
    samples = (
        load_prompt_samples(args.prompts_jsonl, tokenizer, args.sequence_length, args.max_samples)
        if args.prompts_jsonl
        else load_dataset_samples(args, tokenizer)
    )
    if not samples:
        raise SystemExit("No samples loaded.")
    samples = [
        (sample_id, input_ids, category)
        for sample_id, (input_ids, category) in enumerate(samples)
        if sample_id % args.sample_shard_count == args.sample_shard_index
    ]
    if not samples:
        raise SystemExit("No samples assigned to this shard.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model = load_mat_gptq_weights(
        model,
        quant_weights_path,
        master_bitwidth=args.master_bitwidth,
        slice_bitwidth=args.high_bit,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False

    modules = select_layers(model, layer_regex=args.module_regex, layer_classes=torch.nn.Linear)
    module_names = sorted(
        [name for name in modules if (quant_weights_path / name / "data.pt").exists()],
        key=natural_key,
    )
    if args.max_modules:
        module_names = module_names[: args.max_modules]
    if not module_names:
        raise SystemExit("No matching quantized modules found.")

    fieldnames = [
        "sample_id",
        "category",
        "bit",
        "layer_id",
        "module_name",
        "token_pos",
        "activation_norm",
        "activation_absmax",
        "activation_outlier",
        "high_entropy",
        "high_margin",
        "high_loss",
        "low_loss",
        "loss_delta",
        "top1_flip",
        "top5_overlap",
        "high_top1",
        "low_top1",
    ]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for sample_id, input_ids_cpu, category in tqdm(samples, desc="samples"):
                input_ids = input_ids_cpu.to(device)
                high_logits = model(input_ids).logits
                high = token_metrics(high_logits, input_ids)
                del high_logits

                for module_name in tqdm(module_names, desc="modules", leave=False):
                    module = modules[module_name]
                    activation, handle = collect_activation_stats(module, input_ids)
                    _ = model(input_ids).logits
                    handle.remove()

                    original_weight = module.weight.data
                    for bit in args.low_bits:
                        module.weight.data = load_slice_weight(
                            quant_weights_path,
                            module_name,
                            args.master_bitwidth,
                            bit,
                            original_weight.dtype,
                            original_weight.device,
                        )
                        low_logits = model(input_ids).logits
                        low = token_metrics(low_logits, input_ids)
                        write_rows(writer, sample_id, category, bit, module_name, high, low, activation)
                        del low_logits
                    module.weight.data = original_weight
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
