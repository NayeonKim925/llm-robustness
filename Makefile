# Reproducible entry points. CPU-only targets (dataset, analyze, test) need no
# GPU; the train/eval/calibrate targets require a CUDA machine + requirements.

PY ?= python3
CONFIG ?= configs/experiment.yaml

.PHONY: help setup dataset analyze test train eval calibrate clean

help:
	@echo "Targets:"
	@echo "  setup      pip install -e .[viz,dev]  (CPU) ; add ,gpu on a CUDA box"
	@echo "  dataset    build context_conditions.json from raw data (CPU)"
	@echo "  analyze    recompute metrics + baselines from dumps (CPU)"
	@echo "  figures    render result figures from the analysis JSON (CPU)"
	@echo "  significance  paired McNemar tests across conditions (CPU)"
	@echo "  lint       ruff check src/ scripts/ tests/ (CPU)"
	@echo "  test       run unit tests (CPU)"
	@echo "  train      LoRA fine-tune (GPU)   e.g. make train OUT=result/qwen_1.5b_ft"
	@echo "  eval       generation eval (GPU)  e.g. make eval MODEL=Qwen/Qwen2.5-3B-Instruct OUT=results/eval.json"
	@echo "  calibrate  calibration (GPU)      e.g. make calibrate MODEL=... ADAPTER=... OUT=results/cal.json"

setup:
	$(PY) -m pip install -e ".[viz,dev]"

dataset:
	$(PY) scripts/build_dataset.py

analyze:
	$(PY) scripts/analyze_results.py

figures:
	$(PY) scripts/make_figures.py

significance:
	$(PY) scripts/significance.py

lint:
	ruff check src/ scripts/ tests/

test:
	$(PY) -m pytest tests/ -q

# --- GPU targets (override MODEL / ADAPTER / OUT as needed) ---
OUT ?= result/qwen_1.5b_ft
MODEL ?= Qwen/Qwen2.5-1.5B-Instruct
ADAPTER ?=

train:
	$(PY) scripts/run_finetune.py --config $(CONFIG) --output-dir $(OUT)

eval:
	$(PY) scripts/run_eval.py --config $(CONFIG) --base-model $(MODEL) \
		$(if $(ADAPTER),--adapter $(ADAPTER),) --out $(OUT)

calibrate:
	$(PY) scripts/run_calibration.py --config $(CONFIG) --base-model $(MODEL) \
		$(if $(ADAPTER),--adapter $(ADAPTER),) --out $(OUT)

clean:
	rm -rf figures/*.png results/*.json results/*.csv
