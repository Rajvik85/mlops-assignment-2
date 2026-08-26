# Model card - Cats vs Dogs scratch CNN

## Intended use

Educational binary image classification for a pet-adoption MLOps demonstration. It is not suitable for veterinary, animal-welfare, identity, or safety decisions.

## Model

- Input: JPEG/PNG/WEBP/BMP converted to center-cropped 224x224 RGB.
- Architecture: five convolutional blocks with batch normalization, ReLU, max pooling, dropout, and a two-class fully connected head.
- Training: all 3,256,946 trainable parameters initialized randomly and trained from scratch.
- External weights: none. `pretrained=false` and `transfer_learning=false` are stored in the artifact metadata.
- Augmentation: random resized crop, horizontal flip, small color jitter, and small rotation on training images only.
- Optimization: AdamW, OneCycle learning-rate schedule, weight decay, label smoothing, and validation-loss checkpoint selection.
- Serialization: PyTorch checkpoint for reproducibility and ONNX model plus JSON metadata for deployment.
- Output: cat and dog softmax probabilities plus the largest-probability label.

The NumPy logistic-regression implementation remains in source as a comparison baseline, but the API and Docker image serve the CNN.

## Data and split

The training source is the exact assignment-linked Kaggle dataset. Splitting is deterministic and class-stratified at 80% train, 10% validation, and 10% test with seed 42. Preprocessing produced 24,998 valid images: 19,998 training, 2,498 validation, and 2,502 test. Corrupt images are skipped and recorded in `data/processed/summary.json`.

## Verified bundled artifact

The validation-selected scratch CNN was evaluated once on the 2,502-image held-out test set:

| Metric | Value |
|---|---:|
| Accuracy | 0.974420 |
| Precision (dog) | 0.966222 |
| Recall (dog) | 0.983213 |
| F1 (dog) | 0.974643 |
| Correct cats | 1,208 / 1,251 |
| Correct dogs | 1,230 / 1,251 |

The ONNX deployment artifact independently reproduced the same accuracy, precision, recall, F1, and confusion matrix.

## Evaluation evidence

Use `reports/metrics.json`, `reports/figures/confusion_matrix.png`, `reports/figures/loss_curve.png`, and the `cats-vs-dogs-cnn` MLflow run. Always report the test sample count with the scores.

## Limitations

- The Kaggle collection may contain background, capture-device, duplicate-animal, and label biases.
- No animal-identity metadata was available for group-aware splitting, so visually related images could create optimistic estimates.
- Confidence is not calibrated for high-stakes decisions.
- Out-of-distribution images can receive unjustified high confidence.
- Cats and dogs outside the dataset's visual distribution may perform worse than the reported test score.
- Post-deployment quality still requires representative recent labels and drift monitoring.

## Security and privacy

The service limits uploads to 10 MB, validates supported media types, does not log filenames or image bytes, runs as a non-root container user, and loads an ONNX graph rather than an executable pickle. Only project-generated model artifacts should be deployed.
