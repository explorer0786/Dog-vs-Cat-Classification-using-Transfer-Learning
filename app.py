import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import tf_keras

# Load model using tf.keras
@st.cache_resource
def load_model():
    # Load feature extractor from TensorFlow Hub
    mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    feature_extractor = hub.KerasLayer(mobilenet_url, input_shape=(224, 224, 3), trainable=False)

    # Define the model
    model = tf_keras.Sequential([
        feature_extractor,
        tf_keras.layers.Dense(2)  # Output: 2 classes (Cat, Dog)
    ])

    # Load trained weights
    model.load_weights("cat_dog_model.h5")  # Ensure this file exists
    return model

# Load model once
model = load_model()

# Streamlit UI
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image, and this app will predict whether it's a **cat** or a **dog**.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_batch = np.expand_dims(image_array, axis=0)  # Shape: (1, 224, 224, 3)

    # Make prediction
    prediction = model.predict(image_batch)
    predicted_class = np.argmax(prediction)
    confidence = tf.nn.softmax(prediction[0])[predicted_class]

    # Show prediction
    label = "Cat" if predicted_class == 0 else "Dog"
    st.markdown(f"### 🧠 Prediction: **{label}**")
    st.markdown(f"### 🔎 Confidence: **{confidence * 100:.2f}%**")
