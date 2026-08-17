# External tools setup - step by step

Use Python 3.11 or 3.12. Commands below assume the virtual environment from `README.md` is activated.

## 1. Kaggle and KaggleHub

Purpose: download the exact dataset linked in the assignment PDF without manually copying thousands of files.

1. Create/sign in to Kaggle and open `https://www.kaggle.com/settings/api`.
2. Select **Generate New Token** and copy the token.
3. Choose one authentication method:

   - Temporary terminal variable on macOS/Linux:

     ```bash
     export KAGGLE_API_TOKEN="paste-token-here"
     ```

   - Temporary PowerShell variable:

     ```powershell
     $env:KAGGLE_API_TOKEN="paste-token-here"
     ```

   - Interactive login, stored by KaggleHub:

     ```bash
     python -c "import kagglehub; kagglehub.login()"
     ```

4. Visit the linked dataset page once while signed in and accept any displayed usage terms.
5. On a clean checkout where `data/raw` is absent, download:

   ```bash
   python scripts/download_kaggle_data.py --output data/raw
   ```

6. Verify that class directories exist somewhere below `data/raw`:

   ```bash
   find data/raw -type d | head
   ```

The preprocessor accepts `Cat`/`Dog` or `cat`/`dog`, even when nested. Never commit a Kaggle token, `access_token`, or `kaggle.json`.

If the exact dataset handle returns not found after login and accepting terms, capture the linked-page error and ask the instructor to approve an alternative. Do not silently substitute a different dataset because that weakens requirement traceability.

## 2. Git

Purpose: version source, configuration, documentation, tests, DVC pointers, and the small trained artifact.

1. Confirm Git:

   ```bash
   git --version
   ```

2. Set your identity once:

   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your-email@example.com"
   ```

3. Review changes and commit deliberately:

   ```bash
   git status
   git add .
   git commit -m "Complete MLOps assignment pipeline"
   ```

Git stores code; DVC stores large-data content and keeps only pointers in Git.

## 3. DVC

Purpose: reproduce raw-data versions, processed data, parameters, metrics, and pipeline dependencies.

This project already contains `.dvc/`, the real-data `data/raw.dvc` pointer, `dvc.yaml`, `dvc.lock`, and a local demonstration remote.

After downloading real data:

```bash
dvc add --force data/raw
dvc push
dvc repro --force
dvc status
dvc metrics show
dvc dag
```

For a shareable remote, configure one storage provider before `dvc push`. Example with an S3-compatible bucket:

```bash
pip install "dvc[s3]==3.61.0"
dvc remote add -d assignment-storage s3://YOUR-BUCKET/mlops-assignment-2
dvc remote modify --local assignment-storage access_key_id YOUR_ACCESS_KEY
dvc remote modify --local assignment-storage secret_access_key YOUR_SECRET_KEY
dvc push
```

The `--local` secrets go to `.dvc/config.local`, which Git ignores. Commit only `.dvc/config`; do not put credentials in it. If you do not have cloud storage, the included local remote is sufficient for an on-machine demonstration, but a grader on another computer cannot pull it.

To prove a DVC version change during the recording:

```bash
dvc status
dvc metrics diff
git diff data/raw.dvc dvc.lock params.yaml
```

## 4. MLflow

Purpose: compare runs and retain parameters, epoch metrics, figures, and model artifacts.

Training logs to `file:./mlruns` by default:

```bash
dvc repro --force train
mlflow ui --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, then:

1. Select experiment `cats-vs-dogs-baseline`.
2. Open the newest `numpy-logistic-baseline` run.
3. Verify parameters such as feature size, epochs, learning rate, seed, and augmentation.
4. Verify test accuracy, precision, recall, F1, and epoch loss.
5. Open artifacts and show model, confusion matrix, loss curve, and metrics JSON.

`--no-mlflow` exists only for minimal debugging. Do not use it for the final assignment run.

## 5. Docker and Docker Compose

Purpose: package the same inference environment and deploy it reproducibly.

### macOS or Windows

1. Install Docker Desktop from `https://docs.docker.com/desktop/`.
2. Start Docker Desktop and wait until the engine is ready.

### Linux

Install Docker Engine and the Compose plugin using the distribution instructions at `https://docs.docker.com/engine/install/` and `https://docs.docker.com/compose/install/linux/`.

Verify and run:

```bash
docker --version
docker compose version
docker info
docker compose up --build -d
docker compose ps
python scripts/smoke_test.py
docker compose logs --tail=30 cats-dogs-api
```

Useful cleanup that preserves the named monitoring volume:

```bash
docker compose down
```

Only use `docker compose down --volumes` if you intentionally want to delete collected feedback.

## 6. GitHub repository, Actions, and GHCR

Purpose: run CI on every pull request/main push, publish the Docker image, and deploy the exact immutable SHA.

1. Create an empty GitHub repository.
2. Connect and push:

   ```bash
   git remote add origin https://github.com/YOUR-USER/YOUR-REPOSITORY.git
   git push -u origin main
   ```

3. In the repository, open **Settings -> Actions -> General -> Workflow permissions** and allow read/write permissions if your organization policy does not already permit package writes.
4. Open the **Actions** tab. The `test-and-build` job must pass.
5. After a main push, open **Packages** and confirm both the commit-SHA and `latest` tags exist in GHCR.
6. Keep the image linked to the repository so the built-in `GITHUB_TOKEN` has package access.

No Docker Hub password is required because the workflow uses GHCR and the short-lived built-in GitHub token.

## 7. Self-hosted deployment runner

Purpose: provide a real machine where the main-branch CD job can run Docker Compose automatically.

Use a private repository for a long-lived self-hosted runner. Public fork pull requests can expose the machine to untrusted workflow code.

1. Choose a machine that stays online for the demonstration and has Docker plus Compose installed.
2. In GitHub open **Repository Settings -> Actions -> Runners -> New self-hosted runner**.
3. Select the machine operating system and architecture.
4. Run GitHub's displayed download, extraction, and registration commands exactly. The registration token is time-limited.
5. When registration asks for labels, add `mlops-deploy`. If it does not prompt, add the label in the runner's GitHub settings.
6. Install it as a service using the operating-system command GitHub displays, or keep `./run.sh` open during the demonstration.
7. Test Docker under the same operating-system account that runs the runner:

   ```bash
   docker info
   docker compose version
   ```

8. In GitHub create **Settings -> Environments -> New environment -> production**. Optional: add a required reviewer for controlled deployment. For a fully automatic classroom demonstration, leave approval rules off.
9. Push a small code change to `main`. Observe `test-and-build -> publish -> deploy`.
10. On the runner host verify:

   ```bash
   docker compose ps
   curl http://127.0.0.1:8000/health
   ```

If `deploy` remains queued, check that the runner is online and has `self-hosted` and `mlops-deploy` labels. If port 8000 is occupied, stop the conflicting service or set `API_PORT` consistently and adjust the smoke URLs.

## 8. Prometheus-format monitoring

No separate Prometheus server is required by the assignment. The service exposes scrape-ready metrics:

```bash
curl http://127.0.0.1:8000/metrics | grep catsdogs
```

Expected families include request count, request latency histogram, prediction count, and feedback correctness count. For screenshots, make several predictions first so counters are non-zero.

## 9. Postman (optional)

Postman is not required because curl and the automated smoke test already provide proof. If used:

1. Create `GET http://127.0.0.1:8000/health`.
2. Create `POST http://127.0.0.1:8000/predict`.
3. Choose **Body -> form-data**.
4. Add key `image`, change its type to **File**, choose a cat/dog image, and send.
5. Save the response showing `prediction_id`, label, and both probabilities.
