PYTHON ?= python3
PYTHONPATH := src
OUTPUT ?= artifacts/feasibility
SAMPLES ?= 12
SIZE ?= 384

.PHONY: test feasibility clean help

help:
	@echo "make test         Run dependency-light unit tests"
	@echo "make feasibility  Run the end-to-end synthetic feasibility benchmark"
	@echo "make clean        Remove generated feasibility outputs"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

feasibility:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m foldcrack_qc feasibility --output $(OUTPUT) --samples-per-modality $(SAMPLES) --size $(SIZE)

clean:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m foldcrack_qc clean --output $(OUTPUT)
