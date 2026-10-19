.PHONY: install lint format typecheck test eval run docker

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy src tests

test:
	pytest --cov --cov-report=term-missing

eval:
	rag-eval --output reports/eval.json --min-hit-rate 0.9 --min-faithfulness 0.9

run:
	uvicorn rag_eval.main:app --reload

docker:
	docker compose up --build
