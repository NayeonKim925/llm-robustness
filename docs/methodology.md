# Methodology

Reference for the data, the six context conditions, the models, and the metrics.
For the honest read of the results see [`findings.md`](findings.md).

## Task & data

RumourEval-2019 (SemEval-2019 Task 7), subtask A: classify the **stance** of a
reply toward a rumour as one of `support`, `deny`, `query`, `comment`. Data are
Twitter and Reddit conversation trees. Parsing (`src/llm_robustness/data.py`)
flattens each tree into reply records with `text`, `label`, `parent_id`,
`thread_id`, `platform`, `split`.

- Splits are assigned **per `reply_id`** from `dev-key.json` / `train-key.json`.
- Deterministic build (`make dataset`): **6,337** examples — train **4,890**,
  dev **1,447**.
- Class balance is heavily skewed toward `comment` (dev: 81.6%). This dominates
  the interpretation of every metric, hence the baselines in `metrics.py`.

## The six context conditions

Built by `data.build_conditions`. The target reply is constant; only the
context changes.

| Condition | Construction |
|---|---|
| `reply_only` | target reply text only |
| `useful` | `[Source] rumour` + `[Context] parent reply` + `[Target] reply` |
| `irrelevant` | source + a same-thread `comment` with **lowest Jaccard overlap** to the target and no stance keywords |
| `conflicting` | source + a real same-thread reply of the **opposing** stance (most stance keywords) |
| `mixed` | source + parent + a conflicting reply |
| `lexical` | `[Context]` = a template sentence asserting the **opposite** of the target's gold stance (`data.LD_TEMPLATES`), + target |

`conflicting`/`mixed` are undefined when no opposing reply exists in the thread
(recorded as `valid_cc=False` and skipped in evaluation, keeping denominators
honest). `lexical` is **label-aware** — a controlled stress test, not a natural
attack.

## Models & training

- Base models: `Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct`.
- Fine-tuning: LoRA (`r=8`, `alpha=16`, dropout `0.05`, targets `q_proj,v_proj`)
  via TRL `SFTTrainer`, 3 epochs, lr `2e-4`, cosine schedule. Config in
  `configs/experiment.yaml`.
- **Class-imbalance handling** (`train.py`): optional inverse-frequency
  `WeightedRandomSampler` (`imbalance_strategy: weighted_sampler`).
- **Adversarial fine-tuning** (`--adversarial`): mixes the `conflicting`,
  `mixed`, and `lexical` conditions into the training set alongside `useful`, so
  the model sees distractor context during training.
- All formatting flows through `prompts.py`, so training text and inference
  prompts are byte-for-byte the same format (a class of bug the original
  notebooks hit; see findings §2b).

## Metrics (`src/llm_robustness/metrics.py`, pure stdlib)

- **Classification**: accuracy, per-class F1, macro-F1; plus **majority-class**
  and **stratified-random** baselines on the same gold set. Invalid/unparsable
  generations are counted as wrong rather than dropped.
- **Calibration**: Expected Calibration Error (ECE, 10 bins), reliability-diagram
  bins, and mean confidence on correct vs incorrect predictions. Confidence is
  the softmax over the four label tokens' next-token logits at the generation
  position (`calibrate.py`), using the **in-context (leading-space)** token ids.

## Data splits (train / validation / test)

RumourEval ships `train` and `dev` key files (the separate official *test* set
is released later and is not included here). The protocol is:

| Fold | Source | Role | n | threads |
|---|---|---|---|---|
| `fit` | `train` minus `val` | fit LoRA adapters | 4,468 | 293 |
| `val` | thread-level slice of `train` | model/checkpoint selection | 422 | 33 |
| `test` | the `dev` split | final reporting, touched once | 1,447 | 38 |

The split is produced deterministically by `data.train_val_split`
(`make splits` → `results/split_manifest.json`) and is **thread-level, not
example-level**. This matters here specifically: the context conditions splice
in text from *other replies in the same thread*, so an example-level split would
leak that context from `fit` into `val`. Splitting whole threads (seeded shuffle,
`val_fraction=0.1`) removes that path; `test_data.py` asserts the fit/val thread
sets are disjoint.

> The numbers in the README were produced by models trained on the **full**
> `train` split and evaluated on `dev` (= `test`), so they are already held-out
> test numbers. The `fit`/`val` protocol above formalises model selection for
> re-runs; if you plug in the official RumourEval test set, point
> `--split test` at it.

## Evaluation protocol

`evaluate.py` renders each example with the chat template, greedy-decodes a
short completion, and parses the label with the deterministic, word-boundary
matcher in `labels.py`. Reported per condition: n, accuracy, macro-F1, per-class
F1, prediction distribution, and the majority baseline.
