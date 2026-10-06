# s38-squeeze — Hyperliquid BB/KC squeeze + funding-fuel scanner (read-only, no orders).
PY := $(shell [ -x .venv/bin/python3 ] && echo .venv/bin/python3 || echo python3)
.PHONY: help setup test once run events scans signals orders

help:
	@echo "  make setup    venv + requirements"
	@echo "  make test     offline tests (indicator math + fake /info server pipeline)"
	@echo "  make once     one cycle, print table   (ARGS=\"--convex\" adds the breakout engine, dry)"
	@echo "  make run      loop every 60s (Ctrl-C to stop); hospitality1 unit + installer in deploy/"
	@echo "  make events   last 30 state transitions from state/s38.db"
	@echo "  make scans    latest scan rows ranked by score"
	@echo "  make signals  convex signals and rejections (why a breakout did not qualify)"
	@echo "  make orders   order reports: dry plans or live fills, slippage, stops"

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

signals:
	@$(PY) squeeze_scanner.py --signals 30

orders:
	@$(PY) squeeze_scanner.py --orders 30
