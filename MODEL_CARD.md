# Model card - Cats vs Dogs baseline

## Intended use

Educational binary image classification for a pet-adoption MLOps demonstration. It is not suitable for animal welfare, veterinary, identity, or safety decisions.

## Model

- Input: JPEG/PNG/WEBP/BMP converted to center-cropped 224x224 RGB.
- Features: 32x32 average-pooled RGB pixels, flattened to 3,072 values and standardized.
- Estimator: binary logistic regression trained with mini-batch gradient descent.
- Output: cat and dog probabilities plus a label at a 0.5 threshold.
- Augmentation: random horizontal flip and brightness variation on training features.

## Data and split

The required production training source is the assignment-linked Kaggle dataset. Splitting is deterministic and class-stratified at 80% train, 10% validation, and 10% test with seed 42. Corrupt images are skipped and recorded in `data/processed/summary.json`.

## Current bundled artifact

The bundled artifact is trained on the exact assignment-linked Kaggle dataset. Its `data_provenance` field is `assignment_kaggle_dataset`. The split contains 19,998 training, 2,498 validation, and 2,502 test images. Test accuracy is 0.6155, precision 0.6180, recall 0.6051, and F1 0.6115. This is a deliberately lightweight linear baseline, not a production-quality vision model.

## Evaluation

Use `reports/metrics.json`, the confusion matrix, loss curve, MLflow run, and `reports/monitoring/deployed_metrics.json`. Report sample sizes with metrics.

## Limitations

- A linear baseline cannot learn the shapes and textures a CNN learns.
- Dataset bias, backgrounds, duplicate animals, label errors, and class imbalance can distort results.
- Confidence is not calibrated for high-stakes use.
- Images unlike the training distribution may produce unjustified high confidence.
- Post-deployment accuracy requires representative true labels, not only test images.

## Security and privacy

The service limits uploads to 10 MB, validates supported media, does not log filenames or image bytes, and loads only the trusted local pickle. Never load an untrusted pickle file.
