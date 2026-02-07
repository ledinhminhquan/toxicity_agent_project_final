.PHONY: help install dev train tune eval serve agent-demo test lint autopilot benchmark error-analysis fairness report slides clean

help:
	@echo "==== Toxicity Agent Commands ===="
	@echo "  make install        Install package + deps"
	@echo "  make train          Train transformer model"
	@echo "  make tune           Hyperparameter tuning"
	@echo "  make eval           Evaluate baselines + finetuned model"
	@echo "  make serve          Run FastAPI server"
	@echo "  make agent-demo     Run agent demo in CLI"
	@echo "  make benchmark      Run latency benchmark"
	@echo "  make error-analysis Run privacy-preserving error analysis"
	@echo "  make fairness       Run fairness slice evaluation"
	@echo "  make report         Generate PDF report"
	@echo "  make slides         Generate PPTX slides"
	@echo "  make autopilot      Run full pipeline (train+eval+report+slides)"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Compile-check all source files"
	@echo "  make clean          Remove build artifacts"

install:
	pip install -r requirements.txt
	pip install -e .

train:
	toxicity-agent train --config configs/train.yaml

tune:
	toxicity-agent tune --config configs/train.yaml

eval:
	toxicity-agent eval --config configs/train.yaml

serve:
	toxicity-agent serve --config configs/infer.yaml --host 0.0.0.0 --port 8000

agent-demo:
	toxicity-agent demo-agent --config configs/infer.yaml

benchmark:
	toxicity-agent benchmark --config configs/infer.yaml --n 300 --warmup 10

error-analysis:
	toxicity-agent error-analysis --config configs/train.yaml --split test --threshold 0.5

fairness:
	toxicity-agent fairness --config configs/train.yaml --fairness-config configs/fairness_slices.yaml --split test

report:
	toxicity-agent generate-report --train-config configs/train.yaml --infer-config configs/infer.yaml

slides:
	toxicity-agent generate-slides --train-config configs/train.yaml --infer-config configs/infer.yaml

autopilot:
	toxicity-agent autopilot \
	  --train-config configs/train.yaml \
	  --infer-config configs/infer.yaml \
	  --fairness-config configs/fairness_slices.yaml

test:
	pytest -q

lint:
	python -m compileall src tests -q

clean:
	rm -rf build dist src/*.egg-info .pytest_cache __pycache__
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
