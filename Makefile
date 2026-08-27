PYTHON ?= python3
PYTHONPATH := src
OUTPUT ?= artifacts/feasibility
SAMPLES ?= 12
SIZE ?= 384

.PHONY: test lint format-check compile wheel-smoke check feasibility clean help

help:
	@echo "make test         Run the complete pytest suite"
	@echo "make lint         Run Ruff static checks"
	@echo "make format-check Verify Ruff formatting"
	@echo "make compile      Compile source, tests, and scripts"
	@echo "make wheel-smoke  Build and test an isolated wheel"
	@echo "make check        Run every local quality gate"
	@echo "make feasibility  Run the end-to-end synthetic feasibility benchmark"
	@echo "make clean        Remove generated feasibility outputs"

test:
	PYTHONPATH=$(PYTHONPATH) PYTEST_ADDOPTS= PYTEST_PLUGINS= PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts

format-check:
	$(PYTHON) -m ruff format --check src tests scripts

compile:
	$(PYTHON) -m compileall -q src tests scripts

wheel-smoke:
	$(PYTHON) scripts/wheel_smoke.py

check: lint format-check test compile wheel-smoke

feasibility:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m foldcrack_qc feasibility --output $(OUTPUT) --samples-per-modality $(SAMPLES) --size $(SIZE) --patch-size 32

clean:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m foldcrack_qc clean --output $(OUTPUT)
