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
4. Training augments only the training images, fits a five-block CNN from random weights, selects the lowest-validation-loss checkpoint, evaluates the untouched test split once, and exports PyTorch and ONNX artifacts.
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
data/processed + cnn.py + train_cnn.py + train params
                    -> reports/figures + reports/metrics + CNN artifacts
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

- Resized crop with 78%-100% retained area.
- Horizontal flip.
- Small brightness, contrast, and saturation changes.
- Rotation up to 8 degrees.

Validation and test data are never augmented. Augmenting evaluation data would make metrics inconsistent and could leak tuning decisions.

Augmentation helps a model rely less on accidental orientation and lighting. It does not create new semantic information and cannot fix biased or incorrect labels.

## 6. Scratch CNN architecture and mathematics

The deployed model is a convolutional neural network trained entirely from random initialization. It does not download, freeze, or reuse external model weights. The metadata makes this auditable with `pretrained: false` and `transfer_learning: false`.

Each of the five convolutional blocks contains:

```text
3x3 convolution -> batch normalization -> ReLU
3x3 convolution -> batch normalization -> ReLU -> 2x2 max pool
```

The channel counts are 16, 32, 64, 128, and 192. Max pooling changes the spatial dimensions as follows:

```text
224 -> 112 -> 56 -> 28 -> 14 -> 7
```

The final 192x7x7 feature map is flattened. Dropout, a 256-unit dense layer, another dropout layer, and a two-unit output layer produce cat and dog logits. The model has 3,256,946 trainable parameters.

A convolution learns a small filter and slides it across the image. The same filter detects a pattern wherever it appears, which is called weight sharing. Early layers tend to learn edges and colors; deeper layers combine them into textures, parts, and animal-level patterns. ReLU adds non-linearity, batch normalization stabilizes activations, pooling reduces spatial size, and dropout discourages dependence on individual neurons.

The output logits `z_cat` and `z_dog` become probabilities through softmax:

```text
P(class i | x) = exp(z_i) / [exp(z_cat) + exp(z_dog)]
```

Training minimizes cross-entropy with small label smoothing. AdamW updates weights with decoupled weight decay. OneCycle first raises and then lowers the learning rate, enabling rapid learning followed by fine refinement. The checkpoint with the lowest validation loss is retained; the test set is not used for epoch selection.

The old NumPy logistic-regression implementation remains as a comparison baseline. Its 61.55% score explains why spatial feature learning is important, but it is not the deployed artifact.

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

Always report the test sample size. The bundled real-data scratch CNN uses 2,502 held-out images and achieves 0.974420 accuracy, 0.966222 precision, 0.983213 recall, and 0.974643 F1. It correctly classifies 1,208 cats and 1,230 dogs, with 43 cats predicted as dogs and 21 dogs predicted as cats. The separately executed ONNX artifact reproduces the same classification metrics.

## 8. Why MLflow is different from DVC

DVC answers: which data, dependencies, parameters, and outputs formed a reproducible pipeline state?

MLflow answers: across many training attempts, which parameter/metric/artifact combination performed best?

This solution logs:

- Parameters: architecture, no-pretraining flags, parameter count, epochs, learning-rate bounds, batch size, weight decay, dropout, seed, augmentation, device, and sample counts.
- Metrics: train/validation loss and accuracy by epoch; test accuracy, precision, recall, F1, and loss.
- Artifacts: PyTorch checkpoint, ONNX model, JSON metadata, loss curve, confusion matrix, and metrics JSON.

MLflow local storage is suitable for demonstration. A team would run a shared tracking server with durable artifact storage and access controls.

## 9. Model serialization

Two complementary artifacts are saved:

- `cats_dogs_cnn.pt` contains the PyTorch state dictionary and architecture settings for reproducible research or continued training.
- `cats_dogs_cnn.onnx` contains the framework-neutral inference graph used by FastAPI and Docker.
- `cats_dogs_cnn.json` stores ordered classes, input size, normalization, provenance, training settings, no-transfer-learning flags, and test metrics.

At startup, the service loads the ONNX graph into ONNX Runtime and validates the metadata schema. Serving ONNX avoids shipping the much larger training framework in the production container and avoids the arbitrary-code risk associated with loading untrusted pickle files.

## 10. API design

### Browser UX at `/`

The PawSight page is a thin presentation layer served by FastAPI. It provides drag-and-drop selection, image preview, client-side type/size checks, a loading state, accessible status messages, and probability bars. JavaScript sends the selected file to `/predict` using multipart form data. Keeping UI and API on the same origin avoids extra cross-origin configuration and gives beginners one URL to remember.

The UI does not contain a second model and does not calculate its own prediction. The API remains the single source of truth, so command-line, Swagger, automated tests, and browser users all exercise the same inference code.

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
- Fixed RGB normalization constants stored with the model.
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
| Data leakage into tuning | Test set is evaluated only after validation checkpoint selection |
| Accidentally using transfer learning | Metadata records both pretraining flags as false |
| Training/runtime mismatch | Exported ONNX is independently evaluated on all test images |
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

**Why a simple CNN?** Convolutions learn local and translation-tolerant spatial patterns such as edges, fur, ears, and faces. The assignment explicitly permits a simple CNN, and this one reaches 97.44% without external weights.

**Did you use transfer learning?** No. All 3,256,946 parameters start randomly. The code creates `SimpleCNN` directly, and artifact metadata records `pretrained=false` and `transfer_learning=false`.

**Why five blocks?** Repeated convolution and pooling grows the receptive field while reducing 224x224 inputs to compact 7x7 feature maps. This is deep enough for the dataset but still easy to explain.

**Why batch normalization and dropout?** Batch normalization stabilizes learning; dropout regularizes the dense head and reduces overfitting.

**Why export ONNX?** PyTorch is convenient for training, while ONNX Runtime is smaller and framework-neutral for production inference. Export parity was checked on all 2,502 test images.

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

**What would you improve next?** Add probability calibration and drift checks, use identity-aware splitting if animal IDs become available, move MLflow/DVC to shared storage, authenticate endpoints, add Grafana alerts, scan images/dependencies, and implement automatic rollback.

## 21. Final evidence checklist

- Real data provenance printed from the model.
- `dvc status` says up to date and `dvc dag` shows both stages.
- Real `summary.json` shows split counts and corrupt count.
- Real loss curve, confusion matrix, metrics, sample size.
- MLflow run includes parameters, metrics, model, and both figures.
- Nine tests pass, including CNN shape and inference tests.
- Docker/Compose service reports healthy.
- Health and prediction smoke test passes.
- GitHub CI, GHCR publish, CD deploy jobs are green.
- `/metrics` contains non-zero counts and latency buckets.
- Deployed evaluation CSV/JSON contains true labels and metrics.
- ZIP contains required source/config/model/report files.
- Recording is readable and under five minutes.
