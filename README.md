# MLOps Assignment 2 - Cats vs Dogs

This repository implements the complete end-to-end MLOps pipeline. It covers Git and DVC versioning, 224x224 RGB preprocessing, class-stratified 80/10/10 splitting, augmentation, model training, MLflow tracking, FastAPI inference, Docker, GitHub Actions CI/CD, Docker Compose deployment, smoke testing, safe logging, Prometheus metrics, and post-deployment performance measurement.

## Current verified result

The bundled scratch CNN was trained on all 24,998 images from the exact assignment-linked Kaggle dataset. Its metadata says `data_provenance: assignment_kaggle_dataset`, `pretrained: false`, and `transfer_learning: false`. The deterministic split contains 19,998 train, 2,498 validation, and 2,502 test images. The validation-selected model obtained **0.9744 test accuracy**, 0.9662 precision, 0.9832 recall, and 0.9746 F1. The held-out confusion matrix contains 1,208 correctly classified cats, 1,230 correctly classified dogs, and 64 total errors.

The repository also contains a synthetic-data generator strictly for quick software smoke testing. Never replace or report the real metrics with synthetic results.

## Architecture

```text
Kaggle data -> DVC raw pointer -> 224x224 RGB + 80/10/10 split
            -> augmented training -> scratch SimpleCNN -> checkpoint.pt + model.onnx
            -> MLflow experiment -> FastAPI -> Docker/GHCR -> Docker Compose deployment
            -> health/predict smoke tests -> logs + /metrics + labeled feedback evaluation
```

`SimpleCNN` has five convolutional blocks with batch normalization, ReLU activation, max pooling, dropout, and a two-class head. All 3,256,946 parameters start randomly; no pretrained weights or transfer learning are used. PyTorch trains the model, while ONNX Runtime serves the exported artifact efficiently in FastAPI and Docker. The earlier NumPy logistic implementation remains in the source as an auditable baseline comparison but is not the deployed model.

## Repository map

| Path | Purpose |
|---|---|
| `src/catsdogs/preprocessing.py` | Validation, RGB conversion, resize/crop, deterministic split |
| `src/catsdogs/train.py` | Retained NumPy logistic baseline and shared report helpers |
| `src/catsdogs/cnn.py` | Beginner-readable five-block scratch CNN architecture |
| `src/catsdogs/train_cnn.py` | Augmentation, PyTorch training, evaluation, ONNX export, MLflow logging |
| `src/catsdogs/model.py` | Serialization and reusable inference utilities |
| `app/main.py` | Health, prediction, feedback, Prometheus metrics, safe logs |
| `tests/` | Data, model utility, and API tests |
| `dvc.yaml`, `params.yaml`, `data/raw.dvc` | Reproducible pipeline and exact real-data content pointer |
| `Dockerfile`, `docker-compose.yml` | Image and deployment target |
| `.github/workflows/ci-cd.yml` | Tests, image build, GHCR publish, automatic deployment |
| `scripts/smoke_test.py` | Health plus prediction post-deployment test |
| `scripts/evaluate_deployed.py` | Simulated real requests with true labels and accuracy |
| `STUDY_NOTES.md` | Detailed explanation and viva preparation |
| `docs/EXTERNAL_TOOLS_SETUP.md` | Exact setup for Kaggle, DVC, MLflow, Docker, GitHub, GHCR |
| `docs/RECORDING_SCRIPT.md` | A timed screen-recording checklist under five minutes |
| `ASSIGNMENT_COMPLIANCE.md` | Requirement-by-requirement evidence and remaining user evidence |

## Prerequisites

- Python 3.11 or 3.12. Python 3.14 is intentionally excluded because ML libraries may lag new Python releases.
- Git.
- Docker Desktop (macOS/Windows) or Docker Engine plus Compose (Linux).
- A Kaggle account for the real dataset.
- A GitHub repository for CI/CD and GHCR.

## One-time setup

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

Expected test result: `9 passed`.

## Reproduce the current real workflow

With the included DVC data/cache in this working copy, rebuild and inspect the result. CNN training takes several minutes and automatically uses CUDA, Apple Metal, or CPU:

```bash
dvc repro
dvc metrics show
mlflow ui --port 5000
```

Open `http://127.0.0.1:5000`, select `cats-vs-dogs-cnn`, open `scratch-simple-cnn`, and capture the parameters, metrics, loss curve, confusion matrix, PyTorch checkpoint, and ONNX model.

## Real Kaggle workflow on a clean computer

The PDF links to `bhavikjikadara/dog-and-cat-classification-dataset`.

1. Try `dvc pull` if you have access to the configured shared remote. Otherwise complete Kaggle authentication using `docs/EXTERNAL_TOOLS_SETUP.md`.
2. Download the real data only when `data/raw` is absent:

```bash
python scripts/download_kaggle_data.py --output data/raw
dvc add --force data/raw
dvc push
```

3. Run the tracked pipeline in the activated environment:

```bash
dvc repro --force
dvc status
dvc metrics show
```

4. Confirm the artifact is real-data trained:

```bash
python - <<'PY'
from catsdogs.model import load_model
m = load_model("models/cats_dogs_cnn.onnx")
print(m["data_provenance"])
print(m["training"]["pretrained"], m["training"]["transfer_learning"])
print(m["test_metrics"])
PY
```

The first output must be `assignment_kaggle_dataset`, and the next line must be `False False`. Review `reports/figures/loss_curve.png`, `reports/figures/confusion_matrix.png`, and `reports/metrics.json`. Never tune on the test split.

5. Commit the versioned pointers, pipeline state, code, reports, and trained model:

```bash
git add .
git commit -m "Complete cats-dogs MLOps pipeline with real dataset"
git push origin main
```

The raw and processed image folders are intentionally excluded from Git. `data/raw.dvc` versions the raw data, while the small trained model is included directly in the submission and Git repository so Docker CI can build without remote credentials.

## Run the API locally

```bash
MODEL_PATH=models/cats_dogs_cnn.onnx uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST -F "image=@/absolute/path/to/cat.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
curl http://127.0.0.1:8000/metrics
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`.

## Use the browser UI

Open `http://127.0.0.1:8000/` after starting the API or Docker Compose. The responsive **PawSight** interface lets a beginner:

1. Drag and drop or browse for a JPG, PNG, WEBP, or BMP image.
2. Preview the selected image and replace it if needed.
3. Select **Analyze photo**.
4. Read the predicted class, confidence, and separate cat/dog probability bars.

The page calls the same `/predict` endpoint used by automated tests. It validates file type and size in the browser, supports keyboard interaction, shows API errors clearly, and does not store the uploaded image.

## Build and deploy with Docker Compose

```bash
docker compose up --build -d
docker compose ps
python scripts/smoke_test.py --base-url http://127.0.0.1:8000
docker compose logs --tail=30 cats-dogs-api
docker compose down
```

The Compose volume retains non-sensitive feedback across container replacements.

## Post-deployment model performance

With the service running and processed real test data available:

```bash
python scripts/evaluate_deployed.py --base-url http://127.0.0.1:8000 --limit 20
```

This sends labeled test images, calls `/feedback`, and writes:

- `reports/monitoring/deployed_predictions.csv`
- `reports/monitoring/deployed_metrics.json`
- service-side `feedback.csv` in the Compose volume

## CI/CD behavior

Every pull request runs tests, builds the image, starts it, and performs health plus prediction smoke tests. A push to `main` repeats CI, publishes immutable SHA and `latest` multi-platform images to GHCR, then a self-hosted runner with label `mlops-deploy` pulls the new SHA, updates Docker Compose, checks health, and performs a prediction. A failed check stops the workflow.

See `docs/EXTERNAL_TOOLS_SETUP.md` for the exact one-time runner and GitHub Environment setup.


