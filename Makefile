.PHONY: setup lint format test test-all audit clean run-baseline eval label train-retrieval ablate

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

test:
	$(PYTHON) -m pytest tests -m "not slow"

test-all:
	$(PYTHON) -m pytest tests

eval:
	$(PYTHON) evaluate.py

label:
	$(PYTHON) -m streamlit run src/eval/labeling_app.py

train-retrieval:
	$(PYTHON) train_retrieval.py

ablate:
	$(PYTHON) scripts/run_ablations.py


audit:
	@echo "Review docs/AUDIT.md for technical debt, label circularity analysis, and performance metrics."

run-baseline:
	$(PYTHON) -m src.baseline_v1.search text "8x10 beige traditional rug"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache mlruns

