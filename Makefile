# HPO-Adapt — one-command entry points.
#
#   make setup     create .venv, install the package + dev dependencies
#   make demo      run the live-dashboard demo (examples/demo_dashboard.py)
#   make exp1..4   run the four claim experiments (see experiments/exp*.py)
#   make experiments  run all four experiments sequentially (tag "full")
#   make figures   aggregate results into per-figure CSVs (+ PNG previews)
#   make smoke     tiny sanity matrix of all experiments + figures
#   make test      run the unit-test suite
#   make clean     remove caches and disposable artifacts
#
# Everything runs inside .venv when present (created by `make setup`).

PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
TAG ?= full

.PHONY: setup demo exp1 exp2 exp3 exp4 experiments figures smoke test lint clean help

setup: ## create .venv and install hpo-adapt + dev dependencies
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e ".[dev]"
	@echo "Done. Activate with: source .venv/bin/activate"

demo: ## run the live-dashboard demo (~5-6 min, real training)
	$(PY) examples/demo_dashboard.py

demo-fast: ## ~1 min version of the demo
	$(PY) examples/demo_dashboard.py --fast

exp1: ## Claim 1: adaptive closed loop vs equal-budget static HPO (180 runs)
	$(PY) experiments/exp1_adaptive_vs_static.py --tag $(TAG)

exp2: ## Claim 2: ablation — guidance vs random intervention (120 runs)
	$(PY) experiments/exp2_ablation_guidance.py --tag $(TAG)

exp3: ## Claim 3: RL controller improves across episodes (60 groups x 20)
	$(PY) experiments/exp3_cross_episode.py --tag $(TAG)

exp4: ## Claim 4: observe-decide layer overhead on the real-ML task
	$(PY) experiments/exp4_overhead.py --tag $(TAG)

experiments: exp1 exp2 exp3 exp4 ## run all four experiments

figures: ## aggregate runs into per-figure CSVs (+ PNG previews)
	$(PY) experiments/make_figures.py --tags $(TAG) --render

smoke: ## tiny sanity matrix of every experiment + figures (tag "smoke")
	$(PY) experiments/exp1_adaptive_vs_static.py --smoke
	$(PY) experiments/exp2_ablation_guidance.py --smoke
	$(PY) experiments/exp3_cross_episode.py --smoke
	$(PY) experiments/exp4_overhead.py --smoke
	$(PY) experiments/make_figures.py --tags smoke --render

test: ## run the unit-test suite
	$(PY) -m pytest

lint: ## type-check with mypy
	$(PY) -m mypy src

clean: ## remove caches and disposable artifacts
	rm -rf .pytest_cache .mypy_cache src/.pytest_cache src/.mypy_cache
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	rm -f examples/hgb_demo_memory.sqlite

help: ## list available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
