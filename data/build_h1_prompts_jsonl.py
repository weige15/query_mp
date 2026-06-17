#!/usr/bin/env python
import argparse
import json
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def take(rows, n):
    out = []
    for row in rows:
        if len(out) >= n:
            break
        text = row["text"].strip()
        if text:
            out.append(row)
    return out


def gsm8k(n):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = (
        {
            "category": "math",
            "source": "openai/gsm8k",
            "source_id": str(i),
            "text": f"Question: {r['question'].strip()}\nAnswer: {r['answer'].strip()}",
        }
        for i, r in enumerate(ds)
    )
    return take(rows, n)


def mbpp(n):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    def prompt_text(r):
        return (r.get("prompt") or r.get("text") or "").strip()

    rows = (
        {
            "category": "code",
            "source": "google-research-datasets/mbpp",
            "source_id": str(r.get("task_id", i)),
            "text": (
                "Write a Python function for this task.\n"
                f"Task: {prompt_text(r)}\n"
                f"Solution:\n{r.get('code', '').strip()}"
            ),
        }
        for i, r in enumerate(ds)
    )
    return take(rows, n)


def hellaswag(n):
    ds = load_dataset("Rowan/hellaswag", split="validation")

    def row_text(r):
        label = int(r["label"])
        return f"Context: {r['ctx'].strip()}\nContinuation: {r['endings'][label].strip()}"

    rows = (
        {
            "category": "commonsense",
            "source": "Rowan/hellaswag",
            "source_id": str(r.get("ind", i)),
            "text": row_text(r),
        }
        for i, r in enumerate(ds)
    )
    return take(rows, n)


def arc_challenge(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")

    def answer_text(r):
        choices = r["choices"]
        labels = choices["label"]
        texts = choices["text"]
        by_label = dict(zip(labels, texts))
        return by_label.get(r["answerKey"], "")

    def choices_text(r):
        choices = r["choices"]
        return "\n".join(
            f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])
        )

    rows = (
        {
            "category": "science_reasoning",
            "source": "allenai/ai2_arc/ARC-Challenge",
            "source_id": str(r.get("id", i)),
            "text": (
                f"Question: {r['question'].strip()}\n"
                f"Choices:\n{choices_text(r)}\n"
                f"Answer: {answer_text(r).strip()}"
            ),
        }
        for i, r in enumerate(ds)
    )
    return take(rows, n)


def longbench_retrieval(n):
    zip_path = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path) as archive:
        ds = [
            json.loads(line)
            for line in archive.read("data/passage_retrieval_en.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]

    def first_answer(r):
        answers = r.get("answers") or []
        return answers[0] if answers else ""

    rows = (
        {
            "category": "retrieval",
            "source": "THUDM/LongBench/passage_retrieval_en",
            "source_id": str(r.get("_id", i)),
            "text": (
                f"Context:\n{r['context'].strip()}\n\n"
                f"Question: {r['input'].strip()}\n"
                f"Answer: {first_answer(r).strip()}"
            ),
        }
        for i, r in enumerate(ds)
    )
    return take(rows, n)


LOADERS = [gsm8k, mbpp, hellaswag, arc_challenge, longbench_retrieval]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/h1_prompts.jsonl")
    parser.add_argument("--per_category", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    for loader in LOADERS:
        rows.extend(loader(args.per_category))

    if not rows:
        raise SystemExit("No rows produced.")
    missing = [i for i, row in enumerate(rows) if not row.get("text") or not row.get("category")]
    if missing:
        raise SystemExit(f"Rows missing text/category: {missing[:10]}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    print(json.dumps({"output": str(out), "rows": len(rows), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
