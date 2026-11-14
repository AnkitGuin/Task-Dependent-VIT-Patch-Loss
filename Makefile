.PHONY: install test smoke core full report
install:
	python -m pip install -e '.[dev]'
test:
	python -m pytest -q
smoke:
	bash scripts/run_smoke.sh
core:
	bash scripts/run_core_cpu.sh
full:
	bash scripts/run_full.sh
report:
	bash scripts/export_artifacts.sh
