.PHONY: setup demo-data prepare train test run smoke compose-up compose-down package

PYTHON ?= python3

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

demo-data:
	.venv/bin/python scripts/create_demo_data.py

prepare:
	.venv/bin/python -m catsdogs.preprocessing --input data/raw --output data/processed

train:
	.venv/bin/python -m catsdogs.train --data data/processed --model-output models/cats_dogs_logreg.pkl --reports-dir reports --augmentation

test:
	.venv/bin/pytest

run:
	MODEL_PATH=models/cats_dogs_logreg.pkl .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

smoke:
	.venv/bin/python scripts/smoke_test.py --base-url http://localhost:8000

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

package:
	.venv/bin/python scripts/package_submission.py

