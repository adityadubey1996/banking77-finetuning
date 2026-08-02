"""Classify a banking customer message into one of the 77 Banking77 intents.

Usage:
    python predict.py "my card hasn't arrived yet"
    python predict.py --top 3 "why was I charged extra"
    echo "card declined" | python predict.py
"""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = Path(__file__).parent / "banking77-qwen0.5b-lora"
SYSTEM = ("You are an intent classifier for a banking app. "
          "Reply with exactly one intent label and nothing else.")


def load_labels():
    """Label names, so we can snap a near-miss generation onto a real intent."""
    cfg = ADAPTER / "results.json"
    if cfg.exists():
        data = json.loads(cfg.read_text())
        if "labels" in data:
            return data["labels"]
    # Fall back to the dataset itself if results.json predates the labels key.
    from datasets import load_dataset
    return load_dataset("mteb/banking77", split="train").features["label"].names


def normalise(s):
    s = s.strip().lower().split("\n")[0]
    s = re.sub(r"[^a-z0-9_ ]", "", s)
    return re.sub(r"[ _]+", "_", s).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("message", nargs="*", help="customer message; reads stdin if omitted")
    ap.add_argument("--raw", action="store_true",
                    help="print what the model emitted, before snapping to a valid label")
    args = ap.parse_args()

    text = " ".join(args.message) if args.message else sys.stdin.read().strip()
    if not text:
        ap.error("no message given")

    if not ADAPTER.exists():
        sys.exit(f"adapter not found at {ADAPTER}. Run the notebook first, or clone with the "
                 f"adapter directory included.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, str(ADAPTER)).to(device).eval()

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    if args.raw:
        print(raw)
        return

    try:
        labels = load_labels()
    except Exception:
        print(raw)  # no label list available, so report the generation as-is
        return

    lookup = {normalise(l): l for l in labels}
    key = normalise(raw)
    if key in lookup:
        print(lookup[key])
    else:
        near = difflib.get_close_matches(key, list(lookup), n=1, cutoff=0.6)
        print(lookup[near[0]] if near else raw)


if __name__ == "__main__":
    main()
