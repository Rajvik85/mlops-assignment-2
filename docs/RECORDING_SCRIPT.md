# Screen recording plan - target 4 minutes 30 seconds

Prepare all terminals, browser tabs, a real cat/dog image, MLflow UI, GitHub Actions page, and deployed service before recording. Increase terminal font size. Do not show tokens, email, private repository URLs, or personal files.

## 0:00-0:35 - requirement and repository

- State: end-to-end Cats vs Dogs MLOps, 224x224 RGB, 80/10/10, 50 marks.
- Show repository tree and briefly point to source, tests, DVC, Docker, workflow, deployment, reports, and model.
- Show `git log --oneline -3` and `git status`.

## 0:35-1:10 - DVC, preprocessing, and model

- Run `dvc dag` and `dvc status`.
- Open `data/processed/summary.json`; show real counts and zero/known corrupt images.
- Open `reports/metrics.json`; say the real test sample count and metrics.
- Show loss curve and confusion matrix.
- Explicitly confirm model provenance is `assignment_kaggle_dataset`.

## 1:10-1:45 - MLflow and tests

- Show the latest MLflow run: parameters, metrics, and artifacts.
- Run `pytest` and show all tests pass.
- Explain one preprocessing test and one inference utility test.

## 1:45-2:25 - API, container, and monitoring

- Run `docker compose ps` and show healthy status.
- Run `python scripts/smoke_test.py` and show health plus prediction passed.
- Open `/docs` or issue one prediction; show label and both probabilities.
- Show a request log line with path/status/latency, then `/metrics` with non-zero counters.

## 2:25-3:30 - CI, registry, and CD

- Make a harmless documented code change or trigger the prepared main push.
- Show the GitHub Actions graph: tests, Docker build, container smoke test, publish, deploy.
- Show the GHCR SHA tag.
- Show the deploy job pulling the exact SHA, running Compose, health check, and prediction check.
- Return to the service and make one new prediction to prove availability.

## 3:30-4:05 - post-deployment performance

- Run `python scripts/evaluate_deployed.py --limit 20`.
- Show `deployed_metrics.json` and a few CSV rows with true/predicted labels.
- Mention that request data and filenames are not logged.

## 4:05-4:30 - deliverables

- Run `python scripts/package_submission.py`.
- Show the ZIP listing and mention source, configs, DVC, workflow, deployment manifest, reports, and trained artifact.
- Conclude with the end-to-end flow in one sentence.

## Pre-recording checklist

- Real Kaggle data, not demo data.
- Model provenance is `assignment_kaggle_dataset`.
- Tests and Docker smoke pass.
- MLflow newest run has figures and model.
- GitHub self-hosted runner is online with `mlops-deploy`.
- GHCR permissions work.
- Port 8000 and MLflow port 5000 are free.
- Real metrics and sample counts are visible.
- Recording is under five minutes and readable at normal playback speed.

