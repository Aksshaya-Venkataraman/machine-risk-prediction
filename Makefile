.PHONY: install test lint audit train env up down logs admin

install:            ## local dev environment
	python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

test:
	pytest --cov --cov-report=term-missing

lint:
	ruff check . && ruff format --check .

audit:              ## static analysis + known-vulnerability scan
	bandit -c pyproject.toml -r ml services/model-service/model_service services/gateway/gateway
	pip-audit -r services/model-service/requirements.txt --disable-pip --no-deps
	pip-audit -r services/gateway/requirements.txt --disable-pip --no-deps

train:              ## full GridSearchCV training run (writes artifacts/)
	python -m ml.train --synthetic 6000 --out artifacts
	python -m ml.drift_check --baseline ml/baseline_contract.json --candidate artifacts/contract.json

env:                ## create .env with random secrets
	python scripts/generate_env.py

up:
	docker compose up -d --build --wait

down:
	docker compose down

logs:
	docker compose logs -f gateway model-service

admin:              ## create the first admin user (prompts for the password)
	docker compose exec gateway python -m gateway.cli create-admin --email $(EMAIL)
