# Banking77 intent classification, LoRA fine-tune of Qwen2.5-0.5B

A 0.5B parameter model fine-tuned to sort customer banking messages into 77 intents. Accuracy went
from 25.7% zero-shot to 92.7%, and the fine-tuned model does it on a 30 token prompt instead of 700.

Trained in about 20 minutes on a free Colab T4.

| | baseline (zero-shot) | fine-tuned |
|---|---|---|
| accuracy | 25.7% | **92.7%** |
| emitted a valid label | 56.0% | **100%** |
| unparseable output | 22.3% | **0%** |
| prompt length | ~700 tokens | ~30 tokens |

Measured on 300 held-out test examples. Full write-up with error analysis in [RESULTS.md](RESULTS.md).

## Why the two prompts differ

The baseline needs all 77 label names pasted into its prompt or it has no idea what the classes are.
The fine-tuned model learned the label space during training, so it doesn't. That is the practical
argument for fine-tuning a small model rather than prompting it: you pay a one-off training cost and
get shorter prompts, lower latency, and cheaper inference from then on.

The accuracy jump and the 23x prompt reduction come together, not as a tradeoff.

## Try it

```bash
pip install -r requirements.txt
python predict.py "my card hasn't arrived yet"
# card_arrival

python predict.py "why was I charged extra for a payment abroad"
# card_payment_fee_charged
```

Or in code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE), "./banking77-qwen0.5b-lora"
).eval()

prompt = tok.apply_chat_template(
    [{"role": "system", "content": "You are an intent classifier for a banking app. "
                                   "Reply with exactly one intent label and nothing else."},
     {"role": "user", "content": "my card hasn't arrived yet"}],
    tokenize=False, add_generation_prompt=True,
)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
out = model.generate(**ids, max_new_tokens=12, do_sample=False)
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

## Reproducing the training run

Open `banking77_lora_finetune.ipynb` in [Colab](https://colab.research.google.com/), set the runtime
to a T4 GPU, and run all cells. Nothing needs configuring.

The knobs are in the second code cell. Setting `N_TRAIN = 3000` finishes in under ten minutes at a
few points lower accuracy, which is useful if you just want to watch it work.

## What's in here

| file | |
|---|---|
| `banking77_lora_finetune.ipynb` | the whole pipeline: data, baseline, training, eval, error analysis |
| `predict.py` | load the adapter and classify a message |
| `banking77-qwen0.5b-lora/` | the trained LoRA adapter, 45MB |
| `RESULTS.md` | full results, error analysis, and the bugs hit along the way |

## Setup

| | |
|---|---|
| Base model | [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Dataset | [Banking77](https://huggingface.co/datasets/PolyAI/banking77), 13k messages, 77 intents |
| Method | [LoRA](https://arxiv.org/abs/2106.09685), r=16, alpha=32, dropout=0.05 |
| Adapters on | q, k, v, o, gate, up, down projections |
| Trainable | 8.8M of 503M parameters (1.75%) |
| Training | 2 epochs, effective batch 16, lr 2e-4, cosine schedule, fp16 |
| Hardware | one Colab T4, about 20 minutes |

The [original Banking77 paper](https://arxiv.org/abs/2003.04807) (Casanueva et al., 2020) introduced
the dataset.

## Limitations

Most of the remaining 7% of errors are cases where the label taxonomy is ambiguous rather than cases
where the model is confused. `get_disposable_virtual_card` against `getting_virtual_card`, or
`pending_transfer` against `balance_not_updated_after_bank_transfer`, are pairs a human annotator
could reasonably disagree on. That puts a ceiling on accuracy that more training won't lift, and it
suggests the better next move is merging the overlapping intents rather than reaching for a larger
model. Examples are in [RESULTS.md](RESULTS.md).

Beyond that: 300 eval examples gives roughly plus or minus 3% of noise on the accuracy figure, the
model has only ever seen English, and it will confidently emit a label for input that belongs to no
intent at all, so anything production-facing needs an abstain path.
