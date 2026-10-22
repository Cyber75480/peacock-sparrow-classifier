(() => {
  const input = document.getElementById("imageInput");
  const preview = document.getElementById("preview");
  const button = document.getElementById("predictButton");
  const status = document.getElementById("status");
  const result = document.getElementById("result");
  let model = null;
  let imageReady = false;
  const labels = ["Peacock", "Sparrow"];

  async function loadModel() {
    try {
      model = await tf.loadLayersModel("./model/model.json");
      status.textContent = "Model ready. Choose a photo to begin.";
      button.disabled = !imageReady;
    } catch (err) {
      status.textContent = "Model isn't published yet. Run the GitHub Actions training workflow, then refresh this page.";
      console.error(err);
    }
  }

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    result.hidden = true;
    if (!file) {
      imageReady = false;
      button.disabled = true;
      preview.hidden = true;
      return;
    }
    if (!file.type.startsWith("image/")) {
      status.textContent = "Please select an image file.";
      imageReady = false;
      button.disabled = true;
      return;
    }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    preview.onload = () => {
      imageReady = true;
      button.disabled = !model;
      status.textContent = model ? "Photo ready. Tap Predict bird." : "Photo ready; waiting for model…";
    };
  });

  button.addEventListener("click", async () => {
    if (!model || !imageReady) return;
    button.disabled = true;
    status.textContent = "Analysing photo…";
    try {
      const tensor = tf.tidy(() => {
        const pixels = tf.browser.fromPixels(preview)
          .resizeBilinear([224, 224])
          .toFloat()
          .expandDims(0);
        return model.predict(pixels);
      });
      const values = await tensor.data();
      tensor.dispose();
      const best = values.indexOf(Math.max(...values));
      document.getElementById("prediction").textContent = labels[best] || "Unknown";
      document.getElementById("confidence").textContent = `${(values[best] * 100).toFixed(1)}%`;
      result.hidden = false;
      status.textContent = "Prediction complete.";
    } catch (err) {
      status.textContent = "Prediction failed. Try another image or check the Actions log.";
      console.error(err);
    } finally {
      button.disabled = !model || !imageReady;
    }
  });

  async function loadMetrics() {
    try {
      const response = await fetch("./evaluation/summary.json");
      if (!response.ok) throw new Error("Metrics not found");
      const m = await response.json();
      const pct = x => `${(x * 100).toFixed(1)}%`;
      document.getElementById("accuracy").textContent = pct(m.accuracy);
      document.getElementById("precision").textContent = pct(m.macro_precision);
      document.getElementById("recall").textContent = pct(m.macro_recall);
      document.getElementById("f1").textContent = pct(m.macro_f1);
      document.getElementById("evaluationStatus").textContent =
        `${m.evaluation_type} · ${m.evaluated_images} images evaluated`;
      document.getElementById("evalNote").textContent =
        "Metrics describe the images used for this evaluation and do not guarantee performance on every real-world photo.";
    } catch (_) {
      document.getElementById("evaluationStatus").textContent =
        "No report published yet. Run the training workflow first.";
    }
  }

  loadModel();
  loadMetrics();
})();
