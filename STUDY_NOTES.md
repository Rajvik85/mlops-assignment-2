# Study notes - understanding the complete assignment solution

## 1. What the assignment is testing

This is not only an image-classification task. The model is one component inside a controlled software delivery lifecycle. The examiner is checking five connected capabilities:

1. Reproducible model development and experiment evidence.
2. A portable inference service.
3. Automatic quality checks and image creation.
4. Automatic deployment and post-deploy validation.
5. Operational visibility and a complete submission.

The key idea is traceability: a code/data/parameter version should lead to a known model, a known container image, a known deployment, and observable predictions.

## 2. End-to-end flow

1. Git versions code and configuration.
2. DVC versions large data and records a two-stage pipeline.
3. Preprocessing validates each image, converts it to RGB, center-crops/resizes it to 224x224, and creates deterministic stratified splits.
4. Training extracts compact pooled-pixel features, augments the training features, fits logistic regression, evaluates the untouched test split, and serializes the model.
5. MLflow records parameters, per-epoch loss, test metrics, the model, and plots.
6. FastAPI loads the artifact once and exposes health, prediction, feedback, and Prometheus metric routes.
7. Docker locks the runtime and runs the API as a non-root user.
8. CI tests the code and a running container.
9. On `main`, CI publishes an immutable image SHA to GHCR.
10. CD tells a self-hosted machine to pull that exact image and replace the Compose service.
11. Smoke tests check health and an actual prediction.
12. Logs, metrics, feedback, and deployed evaluation show operational and model behavior.

## 3. Why Git and DVC are both needed

Git is excellent for small text files and source history. It is inefficient for thousands of changing images because each binary version increases repository size and Git cannot meaningfully diff pixels.

DVC stores a small pointer file such as `data/raw.dvc` in Git. The pointer contains a content hash and file count. The large files live in a DVC cache/remote. Therefore:

- Git commit identifies the code/configuration/pointer version.
- DVC hash identifies exact data content.
- `dvc pull` restores data for that Git revision.
- `dvc repro` reruns only stages whose dependencies, parameters, or outputs changed.
- `dvc.lock` records the exact dependency/output hashes used by the last run.

`dvc.yaml` is a directed acyclic graph:

```text
data/raw + preprocessing.py + prepare params
                    -> data/processed
data/processed + model.py + train.py + train params
                    -> reports/figures + reports/metrics + model artifact
```

The small model is also committed directly here so a credential-free GitHub runner can build the inference image. The raw and processed datasets remain DVC-managed.

## 4. Image preprocessing

### RGB conversion

Images may be grayscale, RGB, RGBA, palette-based, or rotated through EXIF metadata. The function:

1. Applies EXIF orientation.
2. Converts to exactly three RGB channels.
3. Center-crops while preserving aspect ratio.
4. Resizes to 224x224.
5. Converts integer pixels from `[0,255]` to floating values in `[0,1]`.

The output invariant is shape `(224, 224, 3)`, type `float32`, range `[0,1]`.

### Split logic

Splits are performed separately for cat and dog, so each split contains both classes. A fixed seed makes the shuffle reproducible. With sufficiently large data, each class is divided approximately 80% train, 10% validation, and 10% test.

- Training data fits weights.
- Validation data selects the best epoch/hyperparameters.
- Test data is used once for final unbiased reporting.

Splitting must happen at an entity level when multiple images show the same animal. Otherwise the same animal may appear in train and test, creating leakage and inflated accuracy. The provided dataset structure may not expose animal identity, so this limitation belongs in the model card.

### Corrupt images

Real image collections often contain truncated/non-image files. Preprocessing skips failures and records the path and reason in `summary.json`. Silent skipping is bad because the dataset size may unexpectedly change.

## 5. Data augmentation

Augmentation creates label-preserving variants only for training. This solution randomly applies:

- Horizontal flip.
- Brightness scaling between 0.85 and 1.15.

Validation and test data are never augmented. Augmenting evaluation data would make metrics inconsistent and could leak tuning decisions.

Augmentation helps a model rely less on accidental orientation and lighting. It does not create new semantic information and cannot fix biased or incorrect labels.

## 6. Baseline model mathematics

Each 224x224 RGB image contains 150,528 values. Using every pixel directly would make a CPU baseline unnecessarily slow. Average pooling divides it into 32x32 spatial blocks, giving:

```text
32 x 32 x 3 = 3,072 features
```

Features are standardized using training-only statistics:

```text
z_j = (x_j - mean_j) / std_j
```

The binary logistic model computes a logit:

```text
s = w dot z + b
```

The dog probability is the sigmoid:

```text
P(dog | x) = 1 / (1 + exp(-s))
P(cat | x) = 1 - P(dog | x)
```

At threshold 0.5, dog is predicted when `P(dog) >= 0.5`; otherwise cat.

Training minimizes binary cross-entropy plus L2 regularization:

```text
loss = -mean[y log(p) + (1-y) log(1-p)] + lambda * ||w||^2
```

Mini-batch gradient descent updates weights after each batch. L2 discourages extreme weights and helps generalization. The model keeps weights from the epoch with lowest validation loss.

This baseline is explainable and CPU-friendly but cannot learn complex translation-invariant visual features like a CNN. A good extension would compare it with transfer learning, but the assignment requires at least one baseline, not necessarily the highest possible accuracy.

## 7. Metrics and confusion matrix

For dog as the positive class:

- True positive: dog predicted as dog.
- True negative: cat predicted as cat.
- False positive: cat predicted as dog.
- False negative: dog predicted as cat.

Metrics:

```text
accuracy  = (TP + TN) / all samples
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

Accuracy alone can mislead on imbalanced data. Precision answers “when the model says dog, how often is it right?” Recall answers “of all real dogs, how many did it find?” The confusion matrix shows error direction.

Always report the test sample size. The bundled real-data result uses 2,502 held-out images and achieves 0.6155 accuracy. The modest score is consistent with a simple linear pooled-pixel baseline and leaves clear room for a CNN extension.

## 8. Why MLflow is different from DVC

DVC answers: which data, dependencies, parameters, and outputs formed a reproducible pipeline state?

MLflow answers: across many training attempts, which parameter/metric/artifact combination performed best?

This solution logs:

- Parameters: feature size, epochs, learning rate, batch size, L2, seed, augmentation.
- Metrics: train and validation loss by epoch; test accuracy, precision, recall, F1, and loss.
- Artifacts: serialized model, loss curve, confusion matrix, metrics JSON.

MLflow local storage is suitable for demonstration. A team would run a shared tracking server with durable artifact storage and access controls.

## 9. Model serialization

The `.pkl` file stores a schema-controlled dictionary:

- Format version and model type.
- Weights and bias.
- Feature mean and standard deviation.
- Image and feature sizes.
- Ordered class names and threshold.
- Training settings, time, provenance, and test metrics.

Loading validates required fields and array shapes so corruption or an incompatible artifact fails early. Pickle can execute code during loading; the service must load only the project-generated trusted artifact.

## 10. API design

### `GET /health`

Confirms the process started and the artifact loaded. It returns health status, model type, and artifact format version. Docker and CD use this route.

### `POST /predict`

Accepts multipart file field `image`. It validates MIME type, rejects empty/oversized inputs, preprocesses bytes, performs inference, and returns:

```json
{
  "prediction_id": "uuid",
  "label": "cat",
  "probabilities": {"cat": 0.82, "dog": 0.18}
}
```

The probabilities sum to one. A unique prediction ID links later ground truth without logging image content.

### `POST /feedback`

Accepts `prediction_id`, predicted label, and true label. It stores a minimal CSV row and increments correct/incorrect counters. In production, authentication, a database, retention rules, consent, and deletion policy would be needed.

### `GET /metrics`

Returns Prometheus text with bounded labels for request count, latency histogram, predicted-class count, and feedback correctness count.

## 11. Logging and privacy

The middleware logs request ID, HTTP method, route template, status, and latency. It does not log image bytes, file names, probabilities, or personally identifying data.

Using route templates instead of arbitrary URLs prevents high-cardinality metric labels. High cardinality can make monitoring systems expensive or unstable.

Logging and metrics are related but different:

- Logs are event records useful for debugging individual failures.
- Metrics are numeric time series useful for dashboards and alerts.
- Traces follow a request across distributed services; this assignment does not require them.

## 12. Docker concepts

The Dockerfile starts from a Python 3.11 slim image, installs pinned production-only dependencies, copies source and the trained model, creates a non-root user, exposes port 8000, defines a health check, and starts Uvicorn.

Why separate dependency files?

- `requirements.txt` includes training, DVC, MLflow, tests, and download tools.
- `requirements-prod.txt` includes only inference dependencies.

This reduces image size, build time, vulnerability surface, and unnecessary packages in production.

The Docker build context excludes datasets, caches, Git history, docs, tests, and secrets. The model remains inside the image, so an image SHA identifies both code and model.

Docker Compose is the selected deployment target. It maps port 8000, applies restart and health policies, and mounts a named volume for feedback persistence.

## 13. CI versus CD

Continuous Integration validates every change before it is accepted:

1. Checkout.
2. Install pinned dependencies.
3. Run unit/API tests.
4. Build the Docker image.
5. Start it and run health plus prediction tests.

Continuous Delivery/Deployment moves a validated artifact into an environment:

1. On main, publish SHA and latest tags to GHCR.
2. A self-hosted runner pulls the exact SHA.
3. Compose replaces the service.
4. Health and prediction smoke tests run.
5. Any failed smoke test fails the deployment job.

The immutable SHA is safer than deploying only `latest`: the exact deployed content is traceable and rollback can reference a previous SHA.

The workflow builds `linux/amd64` and `linux/arm64`, so it supports typical cloud/Linux machines and Apple Silicon Docker hosts.

## 14. Why a self-hosted runner is required here

A GitHub-hosted runner is temporary and cannot directly update a service on the student's laptop after the job ends. A self-hosted runner is installed on the deployment machine, so the workflow can run Docker Compose there.

Security consequences:

- Prefer a private repository.
- Do not allow untrusted pull request code on a privileged runner.
- Keep the runner and Docker patched.
- Use short-lived GitHub tokens and least privilege.
- Never store Kaggle/cloud secrets in source.

## 15. Registry and tags

GHCR stores container images. The workflow publishes:

- `ghcr.io/owner/repository:<commit-sha>` for immutable traceability.
- `ghcr.io/owner/repository:latest` for convenience.

Deployment uses the SHA output, not `latest`. If a deployment fails, inspect logs and redeploy the last known good SHA.

## 16. Smoke tests versus unit tests

A unit test checks a small function in isolation, such as preprocessing shape or inference probability normalization.

A smoke test checks that major deployed components connect successfully. Here it verifies the running server, loaded model, multipart upload, preprocessing, inference, and JSON response.

Neither proves high model quality. That requires representative labeled evaluation and ongoing monitoring.

## 17. Post-deployment performance

Production inputs can drift away from training data. The evaluator sends labeled images to the deployed service, records true and predicted labels, and calculates accuracy. This demonstrates the collection loop required by M5.

Important distinction:

- Service health: is the API responding?
- System performance: latency/error rate/throughput.
- Model performance: accuracy/precision/recall on labeled recent traffic.

A service can be healthy and fast while making poor predictions.

## 18. Reproducibility controls in this repository

- Fixed Python range and exact package pins.
- Explicit DVC stage dependencies and parameters.
- Random seed 42 for split, augmentation sequence, and training.
- Training-only normalization statistics stored with the model.
- Validated artifact schema.
- Immutable image SHA.
- Automated tests and smoke tests.
- Model/data provenance marker.

Perfect bit-for-bit reproducibility may still vary across operating systems or NumPy implementations. Functional reproducibility means equivalent inputs, process, and metrics within justified tolerance.

## 19. Common mistakes and how this solution prevents them

| Mistake | Prevention |
|---|---|
| Committing thousands of images to Git | DVC pointer and Git ignore rules |
| Different random split each run | Fixed class-specific seed |
| Grayscale/RGBA crashes | Forced RGB conversion |
| Corrupt file stops entire run | Skip and audit reason |
| Augmenting test images | Augmentation restricted to training |
| Data leakage in normalization | Mean/std computed only on train |
| Reporting demo accuracy as real | Artifact provenance and warnings |
| API starts without model | Lifespan load fails fast |
| Logging sensitive image data | Metadata-only logging |
| Container runs as root | Dedicated non-root user |
| CI only builds but never runs image | Live container smoke test |
| CD deploys mutable latest | Exact SHA is deployed |
| Health test only | Prediction smoke test also required |
| Metrics endpoint with unbounded paths | Route-template labels |
| DVC dependency suddenly breaks | Compatible `pathspec` pin |

## 20. Viva questions and concise answers

**Why 224x224?** It is the assignment requirement and a common input size for standard CNN backbones, giving a consistent tensor shape.

**Why logistic regression?** The brief explicitly allows it as a baseline. It is fast, explainable, CPU-only, and useful as a reference before a CNN.

**Why pool to 32x32?** It reduces 150,528 pixel values to 3,072 while retaining coarse color/spatial information, making laptop training practical.

**Why validation and test sets?** Validation guides model selection; test estimates final generalization. Reusing test data for tuning biases the estimate.

**What does DVC add beyond Git?** Content-addressed storage and reproducible pipelines for large data/artifacts without bloating Git.

**What does MLflow add beyond DVC?** Run-by-run experiment comparison and centralized parameters, metrics, and artifacts.

**Why are probabilities returned?** They expose model confidence for ranking, thresholding, monitoring, and transparent downstream decisions.

**Why is a health endpoint insufficient as a smoke test?** The process may respond while file parsing or model inference is broken. A prediction exercises the full serving path.

**Why use a non-root container user?** It limits damage if the service is compromised.

**Why publish two tags?** SHA provides immutable audit/rollback; latest is convenient for humans.

**Why is the CD runner self-hosted?** It has direct access to the persistent Docker Compose deployment target.

**How is latency measured?** A middleware records monotonic elapsed time for each request and observes a Prometheus histogram.

**How is post-deployment model quality measured?** Store prediction IDs with predicted/true labels, calculate metrics over representative labeled traffic, and compare over time.

**What would you improve next?** Train a transfer-learning CNN, add calibration and drift checks, move MLflow/DVC to shared storage, authenticate endpoints, add Grafana alerts, scan images/dependencies, and implement automatic rollback.

## 21. Final evidence checklist

- Real data provenance printed from the model.
- `dvc status` says up to date and `dvc dag` shows both stages.
- Real `summary.json` shows split counts and corrupt count.
- Real loss curve, confusion matrix, metrics, sample size.
- MLflow run includes parameters, metrics, model, and both figures.
- Six tests pass.
- Docker/Compose service reports healthy.
- Health and prediction smoke test passes.
- GitHub CI, GHCR publish, CD deploy jobs are green.
- `/metrics` contains non-zero counts and latency buckets.
- Deployed evaluation CSV/JSON contains true labels and metrics.
- ZIP contains required source/config/model/report files.
- Recording is readable and under five minutes.
