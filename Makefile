# s38-squeeze — Hyperliquid BB/KC squeeze + funding-fuel scanner (read-only, no orders).
PY := $(shell [ -x .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3)
.PHONY: help setup test once run events scans

help:
	@echo "  make setup    venv + requirements"
	@echo "  make test     offline tests (indicator math + fake /info server pipeline)"
	@echo "  make once     one cycle, print table   (ARGS=\"--coins AAA,BBB\" to restrict)"
	@echo "  make run      loop every 60s (Ctrl-C to stop); hospitality1 unit + installer in deploy/"
	@echo "  make events   last 30 state transitions from state/s38.db"
	@echo "  make scans    latest scan rows ranked by score"

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

test:
	$(PY) -m pytest -q -p no:cacheprovider tests

once:
	$(PY) squeeze_scanner.py --once $(ARGS)

run:
	$(PY) squeeze_scanner.py $(ARGS)

events:
	@$(PY) squeeze_scanner.py --events 30

scans:
	@$(PY) squeeze_scanner.py --scans
