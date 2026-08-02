# Banking77 LoRA fine-tune: results

Run date: 2 August 2026. Single T4 on Colab free tier, about 20 minutes of training.

## Setup

| | |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Dataset | Banking77, 77-class intent classification |
| Train / eval | 9,993 / 300 held out |
| Method | LoRA, r=16, alpha=32, dropout=0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | 8,798,208 of 502,830,976 (1.75%) |
| Epochs | 2, effective batch 16, lr 2e-4, cosine schedule |
| Precision | fp16 (T4 is sm_75, no real bf16) |

## Before and after

| | baseline | fine-tuned |
|---|---|---|
| accuracy | 25.7% | 92.7% |
| emitted a valid label | 56.0% | 100% |
| unparseable output | 22.3% | 0% |

Final training loss: 0.179. Adapter size on disk: 45MB.

The baseline had all 77 labels pasted into its prompt, roughly 700 tokens per request. The
fine-tuned model doesn't need them and runs on about 30. So the accuracy gain comes alongside a
23x smaller prompt, not at the cost of one.

The format numbers matter as much as the accuracy. The base model failed to produce a usable label
string 44% of the time and produced something completely unparseable in 22% of cases, which means
any downstream system consuming its output would need defensive parsing. After fine-tuning that
goes to zero.

## Where it still gets things wrong

22 errors out of 300. Every single confused pair appears exactly once, which is worth noting on its
own: there is no systematic failure mode, no pair of intents the model reliably mixes up. The
errors are scattered one-offs.

Most of them are cases where the label taxonomy itself is ambiguous rather than cases where the
model is confused:

| customer message | gold label | predicted |
|---|---|---|
| how does a virtual card work | get_disposable_virtual_card | getting_virtual_card |
| Cannot access my top up. | topping_up_by_card | top_up_failed |
| I transferred money yesterday, but it still isn't available? | pending_transfer | balance_not_updated_after_bank_transfer |
| Is a non-electronic card available as well | order_physical_card | get_disposable_virtual_card |
| I'm really stuck. I don't know why but my card payment has not gone through. | declined_card_payment | pending_card_payment |
| how long do money transfers take? | transfer_not_received_by_recipient | pending_transfer |

For several of these a human annotator could reasonably have gone either way. "How long do money
transfers take" being labelled as a complaint about a transfer not arriving is a stretch. That puts
a practical ceiling on accuracy that no amount of extra training fixes, and it suggests the more
useful next move is merging or clarifying the overlapping intents rather than reaching for a bigger
model.

The scattering matters for what you'd do next. A model with one dominant confusion pair is worth
more training data on that pair. A model with 22 unique one-off errors, most of them on genuinely
ambiguous inputs, is close to the ceiling this label set allows.

## Things that broke along the way

- `datasets` 5.x removed support for dataset loading scripts, and the canonical `PolyAI/banking77`
  repo still ships a `banking77.py`. Had to fall back to a parquet mirror.
- That mirror stores the label as an integer id with the human-readable string in a separate
  `label_text` column. First attempt at normalising it produced `LABELS = [0, 1, 2, ...]`, which
  passed a `len(LABELS) == 77` check and then blew up four cells later inside `"\n".join(LABELS)`.
  The assert now checks the type rather than the count.
- `torch.cuda.is_bf16_supported()` returns True on a T4 because it counts emulated bf16. Real bf16
  needs compute capability 8.0 or higher. The notebook gates on capability instead, which is why it
  correctly selects fp16.
- Colab preinstalls `torchao` 0.10.0 and `peft` 0.20 raises an outright `ImportError` on anything
  below 0.16, triggered by the `get_peft_model()` call even though nothing here quantizes. Simplest
  fix was to uninstall it.

## Reproducing

Open `banking77_lora_finetune.ipynb` in Colab, set the runtime to a T4 GPU, and run all cells.
Nothing needs configuring. The knobs are in the second code cell if you want a faster run: set
`N_TRAIN = 3000` and it finishes in under ten minutes with a few points less accuracy.
