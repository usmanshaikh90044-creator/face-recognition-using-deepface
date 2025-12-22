import os
import numpy as np
import streamlit as st
from PIL import Image
from deepface import DeepFace

st.set_page_config(page_title="Private Face Recognition Demo", layout="centered")

st.title("🔐 Private Face Recognition Demo")
st.write(
    "This demo recognizes only pre-consented identities and does not store uploaded images."
)

REFERENCE_DIR = "reference_faces"
MODEL_NAME = "Facenet"

# 🔒 Stricter threshold to avoid false positives
THRESHOLD = 0.48
MIN_CONFIDENCE = 70  # percentage

@st.cache_resource
def load_reference_embeddings():
    embeddings_db = {}

    for person in os.listdir(REFERENCE_DIR):
        person_path = os.path.join(REFERENCE_DIR, person)
        if not os.path.isdir(person_path):
            continue

        person_embeddings = []

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            try:
                embedding = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=True
                )[0]["embedding"]

                person_embeddings.append(embedding)

            except Exception:
                pass

        if person_embeddings:
            embeddings_db[person] = person_embeddings

    return embeddings_db


def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


st.info("Loading reference faces...")
reference_embeddings = load_reference_embeddings()
st.success("Reference faces loaded.")

uploaded_file = st.file_uploader(
    "Upload an image for recognition",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    try:
        test_embedding = DeepFace.represent(
            img_path=np.array(image),
            model_name=MODEL_NAME,
            enforce_detection=True
        )[0]["embedding"]

        best_match = None
        min_distance = float("inf")

        for person, embeddings in reference_embeddings.items():
            for ref_emb in embeddings:
                dist = cosine_distance(test_embedding, ref_emb)
                if dist < min_distance:
                    min_distance = dist
                    best_match = person

        confidence = round((1 - min_distance) * 100, 2)

        if min_distance < THRESHOLD and confidence >= MIN_CONFIDENCE:
            st.success(f"✅ Match Found: **{best_match}**")
            st.write(f"📊 Confidence: **{confidence}%**")
        else:
            st.warning("❌ No matching identity found (Unknown)")

    except Exception:
        st.error("No face detected. Please upload a clear face image.")
