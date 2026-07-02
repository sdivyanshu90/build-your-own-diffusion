.PHONY: setup lint format typecheck test smoke check clean train-mnist train-cifar10

setup:            ## Editable install with dev tools + pre-commit hooks
	pip install -e ".[dev]"
	pre-commit install

lint:             ## Static checks (no changes)
	ruff check src tests
	ruff format --check src tests

format:           ## Apply formatting and safe autofixes
	ruff check src tests --fix
	ruff format src tests

typecheck:        ## mypy on src
	mypy

test:             ## Full suite with coverage gate (>= 95%)
	pytest --cov=diffusionlab --cov-report=term -q

smoke:            ## 20-step end-to-end training run on synthetic data
	diffusionlab train --config configs/smoke.yaml

check: lint typecheck test  ## Everything CI runs

train-mnist:      ## Train the MNIST config
	diffusionlab train --config configs/mnist.yaml

train-cifar10:    ## Train the CIFAR-10 config
	diffusionlab train --config configs/cifar10.yaml

clean:            ## Remove caches and build artifacts (keeps runs/ and data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov build dist
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
