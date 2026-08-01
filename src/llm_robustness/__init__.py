"""llm_robustness: controlled context-perturbation evaluation of small LMs.

Modules
-------
labels      Label set and robust label parsing.
metrics     Classification + calibration metrics (pure stdlib).
data        RumourEval-2019 parsing and context-condition construction.
prompts     Single source of truth for prompt / chat formatting.
evaluate    Generation-based evaluation harness (requires transformers/torch).
calibrate   Confidence / calibration measurement (requires transformers/torch).
train       LoRA SFT with class-imbalance handling (requires trl/peft/torch).
"""

__version__ = "0.2.0"
