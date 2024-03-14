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

def preprocess_image(img):
    try:
        # Use PIL to open the image from BytesIO object
        img = Image.open(img).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)  # Convert to FLOAT32
        img_array /= 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def make_prediction(img):
    if img:
        preprocessed_img = preprocess_image(img)
        if preprocessed_img is not None:
            try:
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
                st.image(img, caption="Captured Image", use_column_width=True)
            except Exception as e:
                st.error(f"Error during inference: {e}")
        else:
            st.error("Failed to process the image.")
    else:
        st.warning("Please capture an image using the camera.")

st.title("Image Classification App with Camera")

captured_image = st.camera_input("Capture an image")

if st.button("Make Prediction"):
    make_prediction(captured_image)
