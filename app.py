
import io
import json
import hashlib
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from gtts import gTTS


# =====================================
# FOOD VISION AI - APP CONFIGURATION
# =====================================

st.set_page_config(
    page_title="FOOD VISION AI",
    page_icon="🍎",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

# Model and labels are in the main GitHub folder
MODEL_PATH = BASE_DIR / "food_model.keras"
LABELS_PATH = BASE_DIR / "labels.json"


# =====================================
# LOAD TRAINED AI MODEL
# =====================================

@st.cache_resource
def load_ai():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("food_model.keras is missing.")

    if not LABELS_PATH.exists():
        raise FileNotFoundError("labels.json is missing.")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    return model, labels


# =====================================
# MAIN APP INTERFACE
# =====================================

st.title("🍎 FOOD VISION AI")

st.subheader(
    "AI-Powered Fresh & Spoiled Fruit Scanner"
)

st.markdown("""
Welcome! Identify fresh and spoiled apples and bananas.

### How to use

1. Select Camera or Upload.
2. Keep the fruit clearly visible.
3. Press SCAN & SPEAK.
4. Read and listen to your AI result.

**Supported fruits:** Apples and Bananas.
""")

# Load model
try:
    with st.spinner("Loading FOOD VISION AI..."):
        model, labels = load_ai()

except Exception as error:
    st.error(f"AI Model Loading Error: {error}")
    st.stop()


# =====================================
# CAMERA AND IMAGE UPLOAD
# =====================================

choice = st.radio(
    "Choose Input Method",
    ["Camera", "Upload"],
    horizontal=True
)

if choice == "Camera":

    photo = st.camera_input(
        "📷 Take a Fruit Photo",
        key="camera_photo"
    )

else:

    photo = st.file_uploader(
        "📁 Upload a Fruit Image",
        type=["jpg", "jpeg", "png"],
        key="uploaded_photo"
    )


# =====================================
# FRUIT SCANNING
# =====================================

if photo is not None:

    image_bytes = photo.getvalue()

    photo_id = hashlib.sha256(
        image_bytes
    ).hexdigest()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    st.image(
        image,
        caption="Selected Fruit",
        use_container_width=True
    )

    if st.button(
        "🔍 SCAN & SPEAK",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Scanning your fruit..."):

                # Prepare image for AI model
                resized = image.resize((224, 224))

                image_array = np.array(
                    resized,
                    dtype=np.float32
                )

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                # AI prediction
                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(predictions) != len(labels):
                    raise ValueError(
                        "Model output and labels do not match."
                    )

                index = int(np.argmax(predictions))

                confidence = (
                    float(predictions[index]) * 100
                )

                predicted_label = labels[index]

                fruit, condition = (
                    predicted_label.rsplit("_", 1)
                )

                fruit = fruit.replace("-", " ").title()
                condition = condition.title()

                result = (
                    f"Fruit: {fruit}\n"
                    f"Condition: {condition}\n"
                    f"Confidence: {confidence:.1f}%"
                )

                if confidence < 60:
                    result += (
                        "\nLow confidence. "
                        "Please try another clear photo."
                    )

                # Generate English voice
                spoken_text = (
                    f"Food Vision AI scan result. "
                    f"Fruit: {fruit}. "
                    f"Condition: {condition}. "
                    f"Confidence: {confidence:.1f} percent. "
                    "This is an experimental AI prediction. "
                    "It cannot guarantee food safety."
                )

                audio_data = None

                try:
                    audio_buffer = io.BytesIO()

                    gTTS(
                        text=spoken_text,
                        lang="en"
                    ).write_to_fp(audio_buffer)

                    audio_data = audio_buffer.getvalue()

                except Exception:
                    audio_data = None

                # Save scan result
                st.session_state["scan_result"] = {
                    "photo_id": photo_id,
                    "result": result,
                    "audio": audio_data
                }

        except Exception as error:

            st.error(
                f"Image scanning failed: {error}"
            )


# =====================================
# DISPLAY SCAN RESULT AND VOICE
# =====================================

saved_result = st.session_state.get(
    "scan_result"
)

if (
    photo is not None
    and saved_result is not None
    and saved_result["photo_id"] == photo_id
):

    st.markdown("---")

    st.subheader("📋 AI SCAN RESULT")

    st.text(saved_result["result"])

    if saved_result["audio"]:

        st.subheader("🔊 Listen to Your Result")

        st.audio(
            saved_result["audio"],
            format="audio/mp3"
        )

    else:

        st.warning(
            "Voice generation is temporarily unavailable. "
            "Please check your internet connection."
        )


# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.markdown("""
### FOOD VISION AI

**Experimental Artificial Intelligence Project**

Important: AI predictions may be incorrect.
This application supports apples and bananas only.
It cannot guarantee food safety.
""")
