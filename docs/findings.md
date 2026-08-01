# Findings & Diagnostic Notes

This document is the honest narrative behind the numbers: what the project set
out to measure, what went wrong the first time, how each problem was diagnosed,
and what can be defended as a result. It is written to be *read alongside the
code* — every claim points at a file you can run.

---

## 1. What the project set out to test

Small instruction-tuned LMs are increasingly used as cheap classifiers. This
project asks a robustness question on **RumourEval-2019 stance classification**
(support / deny / query / comment):

> When we vary the *context* around a reply — from genuinely useful, to
> irrelevant, to actively contradictory — how does a small LM's accuracy and
> **confidence** change, and does adversarial fine-tuning help?

The design (see [`methodology.md`](methodology.md)) holds the target reply fixed
and swaps only the surrounding context across six conditions. That part of the
project is sound and is the reusable idea.

---

## 2. The first result — and why it was not trustworthy

The initial write-up reported that the models were **more confident when wrong
than when right**, especially under corrupted context, and that adversarial
fine-tuning partially fixed this. Recomputing everything from the raw dumps
(`scripts/analyze_results.py`) shows why that headline could not be trusted.

### 2a. A missing baseline hid a class collapse

On the dev split, `comment` is **81.6%** of examples. So the first question for
*any* accuracy number is “does it beat 0.816?”. The original notebooks never
computed that baseline. Once you do:

| Pipeline | Model | acc | comment F1 | classes predicted |
|---|---|---|---|---|
| confidence-scoring | Qwen1.5B-ft | **0.10** | — | 1 (always `deny`) |
| confidence-scoring | Qwen1.5B-adv | **0.10** | — | ~2 |
| generation (correct) | Qwen1.5B-adv | **0.82** | 0.90 | 3 |

The *same adversarial model* scored 0.10 in the confidence pipeline and 0.82 in
the generation pipeline. A model cannot be both. Something about the
**confidence pipeline** — not the model — was broken.

### 2b. Three bugs in the confidence pipeline

Reading `notebooks/confidence_scores.ipynb` against the tokenizer revealed three
independent problems, each verified and each now fixed in
`src/llm_robustness/calibrate.py`:

1. **No chat template.** The fine-tuned chat model was fed a bare
   `"Rumor: …\nAnswer:"` string. It was *trained* on the full
   `<|im_start|>system …` format. Feeding a different input distribution than
   training is enough on its own to wreck predictions.

2. **Wrong label-token ids.** Confidence was defined as the softmax over
   `tokenizer.encode(label)[0]` for the four labels. That is the
   **no-leading-space** token. In an actual sequence the label appears with a
   leading space, which is a *different* byte-level-BPE token:

   | label | `encode(label)[0]` (used) | `" "+label` (emitted in context) |
   |---|---|---|
   | support | 23362 | 1824 |
   | deny | 89963 | 23101 |
   | query | 1631 | 3239 |
   | comment | 6182 | 3980 |

   So the code read logits for tokens the model does not generate here. The
   resulting “confidence ≈ 0.90” is a softmax over four arbitrary logits — it
   does not measure the model's answer distribution.

3. **Train/dev contamination.** The confidence notebook re-parsed the corpus
   with a *merged* dev+train key map and a `'dev' if id in label_map` check,
   which labels essentially every thread `dev`. Its “dev” set had ≈5,669 rows;
   the true dev split is 1,447. The fine-tuned model was thus “evaluated” partly
   on its own training data.

Any one of these invalidates a calibration claim; together they mean the
“more confident when wrong” pattern is a **measurement artifact**. For the
record, ECE on those flawed dumps is ≈0.80 with mean confidence-when-wrong
(0.91) above confidence-when-right (0.84) — exactly the shape you get from a
collapsed classifier scored on the wrong tokens.

---

## 3. What can be defended (correct generation pipeline)

The generation-based evaluation (`experiment_results_*` dumps → recomputed by
`analyze_results.py`) *does* use the chat template and the proper dev split.
Its numbers are internally consistent and hold up against baselines. Per
condition (accuracy):

| condition | 0.5B zero-shot | 3B zero-shot | 1.5B adv-FT | 3B adv-FT |
|---|---|---|---|---|
| reply_only | 0.096 | 0.628 | 0.823 | 0.825 |
| useful | 0.090 | 0.635 | 0.848 | 0.850 |
| irrelevant | 0.080 | 0.697 | 0.846 | 0.845 |
| conflicting | 0.088 | 0.700 | 0.854 | 0.858 |
| mixed | 0.087 | 0.644 | 0.850 | 0.852 |
| **lexical** | 0.093 | **0.598** | 0.841 | **0.851** |
| majority baseline | — | 0.816 | 0.816 | 0.816 |

Defensible conclusions:

- **The 0.5B model fails the task**: 2-class collapse, `comment` F1 = 0.00. This
  is a genuine capacity/imbalance failure, not a bug (it uses the same correct
  pipeline as the working models).
- **Zero-shot 3B is context-sensitive and weakest under the adversarial lexical
  distractor** (0.628 → 0.598, and it drops from 4 predicted classes to 3).
- **Adversarial fine-tuning restores robustness**: adv-FT accuracy is flat at
  0.82–0.86 across all six conditions, and the 3B lexical vulnerability closes
  (**0.598 → 0.851**).
- **Caveat:** macro-F1 stays ~0.30–0.40 — the models lean on the majority class.
  Beating majority *accuracy* while only reaching ~0.35 macro-F1 is the honest
  characterization; it is not a strong minority-class classifier.

---

## 4. What I would do next

1. **Re-run the corrected `calibrate.py`** on the ft/adv adapters and report
   ECE + reliability diagrams on the clean dev split. Only then is any
   calibration claim on the table.
2. **Report a held-out test split.** Everything above is dev; RumourEval ships a
   test set that should be touched once, at the end.
3. **Push minority-class F1**: focal loss or class-balanced loss rather than
   only inverse-frequency sampling; report per-class F1 as the headline metric.
4. **Significance tests as artifacts**: McNemar between conditions/models, saved
   to `results/`, rather than quoted from memory.

---

## 5. Why this is written down

Finding that your own headline result was a measurement artifact — and being
able to point to the exact tokenizer ids and the exact split-construction line
that caused it — is the part of this project worth discussing. The corrected
pipeline is in the package; the flawed dumps are kept only as evidence of the
artifact.
