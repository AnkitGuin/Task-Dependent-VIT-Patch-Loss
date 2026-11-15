.PHONY: install test smoke core full report paper
install:
	python -m pip install -e '.[dev]'
test:
	pytest -q
smoke:
	bash scripts/run_smoke.sh
core:
	bash scripts/run_core_cpu.sh
full:
	bash scripts/run_full.sh
report:
	bash scripts/export_artifacts.sh
paper:
	cd paper && latexmk -pdf -e '$$bibtex="bibtex8 %O %B";' -interaction=nonstopmode -halt-on-error main.tex
