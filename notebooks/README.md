# Notebooks

## Start here

- **[`00_results.ipynb`](00_results.ipynb)** — the canonical, guided tour of the
  project: task, data splits, results vs. baselines, the confidence-bug
  diagnosis, figures, and significance tests. It runs **CPU-only**, imports the
  `llm_robustness` package (no copy-pasted logic), and reads the committed
  prediction dumps. Regenerate with:

  ```bash
  pip install -e ".[viz,dev]" nbformat nbclient ipykernel pandas
  jupyter nbconvert --to notebook --execute --inplace notebooks/00_results.ipynb
  ```

## Exploratory record (kept as-is)

The remaining notebooks are the original exploratory work — how the pipeline was
built and the models trained on GPU. They contain hard-coded Colab paths and are
**not** the reference; the reusable logic has been extracted into
`src/llm_robustness/`. They are preserved for transparency:

| notebook | what it explored |
|---|---|
| `dataset_parsing_preprocessing.ipynb` | RumourEval parsing + building the six context conditions |
| `qwen_finetuning_experiment.ipynb` | LoRA SFT of Qwen-0.5B/1.5B |
| `qwen_0_5b_1_5b_experiment.ipynb` | zero-shot / fine-tuned evaluation |
| `qwen2_5_3b_evaluation_experiment.ipynb` | Qwen-3B evaluation |
| `confidence scores.ipynb` | the original (flawed) confidence measurement |
| `baseline_experiment.ipynb` | early baselines |
| `attention_analysis.ipynb` | exploratory attention inspection |
