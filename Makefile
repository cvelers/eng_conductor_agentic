.PHONY: install run test

VENV ?= .venv
PORT ?= 8000
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest

venv:
	python3.11 -m venv $(VENV)
	$(PIP) install -r requirements.txt

install:
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) backend.app:app --reload --host 0.0.0.0 --port $(PORT)

test:
	$(PYTEST) -q
