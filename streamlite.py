import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="all_model.tflite")
interpreter.allocate_tensors()

# Define class labels
class_labels = ["acne", "acne_scars", "hyperPigmentation", "white_patches"]  # Replace with your actual class labels

def preprocess_image(image_path):
    try:
        # Use PIL to open the image
        img = cv2.imread(image_path)
        if img is None:
            st.error("Failed to read the image.")
            return None

        img = cv2.resize(img, (224, 224))
        img = np.array(img, dtype=np.float32)  # Convert to FLOAT32

        # Print the image information for debugging
        print(f"Image shape before conversion: {img.shape}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def make_prediction(img):
    preprocessed_img = preprocess_image(img)
    if preprocessed_img is not None:
        # Perform inference with the TensorFlow Lite model
        input_tensor_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_tensor_index, preprocessed_img)
        interpreter.invoke()
        output_tensor_index = interpreter.get_output_details()[0]['index']
        prediction = interpreter.get_tensor(output_tensor_index)

        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]
        confidence = prediction[0][predicted_class_index]

        st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
        st.image(img, caption="Scanned Image", use_column_width=True)
    else:
        st.error("Failed to process the image.")

st.title("Image Classification App")

use_camera = st.checkbox("Use Camera")

if use_camera:
    st.warning("Please note that the camera functionality is experimental and might not work in all environments.")
    camera = cv2.VideoCapture(0)

    if st.button("Scan Face"):
        _, frame = camera.read()
        make_prediction(frame)

    st.stop()
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if st.button("Make Prediction"):
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img = np.array(img)
            make_prediction(img)
        else:
            st.warning("Please upload an image.")

