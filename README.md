# Peacock vs Sparrow Classifier (mobile-friendly)

A GitHub-ready image classification project:
- Train a lightweight MobileNetV2 transfer-learning model in GitHub Actions.
- Evaluate it and save accuracy, precision, recall, F1, a confusion matrix, and a CSV report.
- Convert the model to TensorFlow.js so GitHub Pages can run predictions in the browser.
- Share the Pages URL with your teacher; no app installation is needed.

## 1. Add your images

In this repository, use **Add file → Upload files** on GitHub. Add clear photos into these folders:

```text
data/
  train/
    peacock/
    sparrow/
  test/
    peacock/
    sparrow/
```

Create the folders by uploading at least one image into each folder, or add a `.gitkeep` placeholder first. Git doesn't keep empty folders.

Suggested starting point: aim for at least 50–100 varied images per class for a school demo, and more if possible. Use different backgrounds, distances, lighting, and poses. Only use images you have permission to use. Avoid near-duplicates.

**Important:** keep the `test` images separate from training images. They should be photos the model never sees during training. If you don't have test images yet, you can leave the test folders empty; the workflow will clearly label its evaluation as validation-only.

## 2. Create the GitHub repository

1. On GitHub, create a repository named `peacock-sparrow-classifier`.
2. Upload the files from this ZIP, preserving the folder structure.
3. Add your images under `data/train/peacock`, `data/train/sparrow`, and (when available) `data/test/peacock`, `data/test/sparrow`.
4. Commit to the `main` branch.

## 3. Run training and publish the website

1. Open the repository → **Actions**.
2. If GitHub asks, enable Actions for the repository.
3. Choose **Train, evaluate, and deploy Pages** → **Run workflow**.
4. Wait for the run to finish green. The first run can take several minutes because it downloads pretrained model weights and trains the classifier.
5. Open **Settings → Pages**. The deployment workflow uses GitHub Actions as the Pages source.
6. Open the Pages URL shown there. Share that URL with your teacher.

The workflow runs on pushes that change training/test images or model code, and can also be started manually. GitHub Actions usage limits apply.

## 4. Test the site

Open the published site, select a photo, and tap **Predict**. It displays the predicted class and the model's confidence score. The image is processed in your browser; the webpage does not upload it to a prediction server.

Confidence is **not** the same as accuracy. Accuracy and other metrics are in `docs/evaluation/` after a successful workflow run.

## Repository structure

```text
.github/workflows/train-evaluate-deploy.yml
data/train/peacock/
data/train/sparrow/
data/test/peacock/
data/test/sparrow/
src/train.py
docs/index.html
docs/style.css
docs/app.js
requirements.txt
```

## Troubleshooting

- **No model found on the website:** training/deployment did not finish successfully. Open Actions and inspect the run log.
- **Dataset folder error:** make sure images are inside the exact class folders above. Supported formats include JPG, JPEG, PNG, and BMP.
- **Poor predictions:** add more diverse, correctly labelled images and a genuinely separate test set; rerun the workflow.
- **Pages 404:** check Settings → Pages and wait a minute after deployment.
- **Training fails:** check the Actions log. GitHub-hosted runner images and package compatibility can change over time.

## Honest evaluation note

When `data/test` contains at least one usable image for each class, the report evaluates on that separate test folder. If not, the workflow evaluates on the held-out validation subset and explicitly marks the report as **validation-only**. Never present validation accuracy as independent test accuracy.
