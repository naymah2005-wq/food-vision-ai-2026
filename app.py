
from pathlib import Path
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "food_model.keras"
LABELS_PATH = BASE / "models" / "labels.json"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

model = None
labels = []

def load_model():
    global model, labels

    if not MODEL_PATH.exists():
        print("ERROR: food_model.keras not found!")
        return

    if not LABELS_PATH.exists():
        print("ERROR: labels.json not found!")
        return

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_PATH)

        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)

        print("AI MODEL LOADED SUCCESSFULLY!")
        print("Classes:", labels)

    except Exception as exc:
        model = None
        print("MODEL LOADING ERROR:", exc)


load_model()


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model_ready": model is not None,
        "classes": labels
    })


@app.post("/predict")
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "Please capture or upload a photo."
        }), 400

    if model is None:
        return jsonify({
            "model_ready": False,
            "message": "AI model is not available."
        }), 503

    try:
        image = Image.open(
            request.files["image"].stream
        ).convert("RGB").resize((224, 224))

        arr = np.asarray(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        predictions = model.predict(arr, verbose=0)[0]

        idx = int(np.argmax(predictions))
        label = labels[idx]
        confidence = float(predictions[idx]) * 100

        parts = label.rsplit("_", 1)

        if len(parts) == 2 and parts[1] in ("fresh", "spoiled"):
            fruit = parts[0].replace("-", " ").title()
            condition = parts[1].title()
        else:
            fruit = label.replace("_", " ").title()
            condition = "Unknown"

        return jsonify({
            "model_ready": True,
            "fruit": fruit,
            "condition": condition,
            "confidence": round(confidence, 1),
            "label": label,
            "message": "AI prediction only. Do not rely on this app alone to decide whether food is safe to eat."
        })

    except Exception as exc:
        return jsonify({
            "error": f"Image analysis failed: {exc}"
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
