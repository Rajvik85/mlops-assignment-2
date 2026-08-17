# Assignment compliance and gap audit

This matrix maps each PDF requirement to executable evidence. Status is based on code and local verification. Items marked **user evidence** require the student's accounts, real Kaggle data, GitHub run, or recording and cannot be truthfully fabricated in a local reference build.

| Marks | Requirement | Evidence | Status |
|---|---|---|---|
| M1 | Git source versioning | Git repository, `.gitignore`, structured source/tests/docs | Implemented |
| M1 | DVC raw and processed data | `.dvc/`, `data/raw.dvc`, `dvc.yaml`, `dvc.lock`; prepare output tracked by DVC | Implemented and locally reproduced |
| M1 | 224x224 RGB, 80/10/10 | `preprocess_image`, stratified `split_paths`, manifest/summary | Implemented and tested |
| M1 | Data augmentation | Random horizontal flip and brightness in training | Implemented |
| M1 | Baseline and serialized model | NumPy binary logistic regression, `models/cats_dogs_logreg.pkl` | Implemented and trained on exact Kaggle data |
| M1 | MLflow params, metrics, artifacts | Default training path logs parameters, epoch loss, test metrics, model, plots | Implemented and locally executed |
| M1 | Confusion matrix and loss curves | `reports/figures/` and MLflow artifacts | Implemented with real-data outputs |
| M2 | REST health and prediction | FastAPI `/health`, `/predict`; probabilities and label | Implemented and tested |
| M2 | Pinned dependencies | `requirements.txt`, `requirements-prod.txt`; DVC compatibility pin | Implemented and clean-installed |
| M2 | Container | Non-root `Dockerfile`, health check, small production dependency set | Implemented |
| M2 | Local image/prediction proof | Compose commands and smoke script | Locally verifiable; capture in recording |
| M3 | Preprocessing unit test | `tests/test_preprocessing.py` | Passed |
| M3 | Inference utility unit test | `tests/test_model.py` | Passed |
| M3 | CI checkout/install/test/build | GitHub workflow `test-and-build` job | Implemented; GitHub run is user evidence |
| M3 | Registry publish | GHCR immutable SHA and latest multi-platform tags | Implemented; successful package URL is user evidence |
| M4 | Deployment target/manifests | `docker-compose.yml` and named feedback volume | Implemented |
| M4 | Automatic main deployment | Self-hosted `mlops-deploy` job pulls exact SHA and runs Compose | Implemented; connected runner/run is user evidence |
| M4 | Health and prediction smoke test | CI container test and post-deployment shell checks | Implemented; successful run is user evidence |
| M5 | Request/response logging | Middleware logs request ID, method, route, status, latency; excludes file/image | Implemented and tested through API |
| M5 | Count and latency metrics | Prometheus counters/histogram at `/metrics` | Implemented |
| M5 | Real/simulated requests and labels | `/feedback`, `evaluate_deployed.py`, CSV and accuracy JSON | Implemented; real batch output is user evidence |
| Deliverable | Consolidated ZIP | `scripts/package_submission.py` validates required content | Implemented |
| Deliverable | Recording under five minutes | `docs/RECORDING_SCRIPT.md` | Script prepared; recording is user evidence |

## Mandatory actions before claiming final completion

1. Confirm the current model provenance remains `assignment_kaggle_dataset` and report its real metrics.
2. Push the final repository to GitHub and retain a green CI/CD run URL.
3. Confirm the GHCR image exists and the deployed service is healthy.
4. Run at least 20 labeled post-deployment requests and keep the CSV/JSON results.
5. Capture DVC, MLflow, tests, image publishing, deployment, prediction, metrics, and monitoring in the sub-five-minute recording.
6. Run the packager after those outputs exist and inspect the ZIP.

No document can guarantee marks; the matrix maximizes traceable evidence for every item in the supplied brief.
