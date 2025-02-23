import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Load the TFLite model correctly in Streamlit Cloud
model_path = os.path.join(os.path.dirname(__file__), "Gmodel_compressed.tflite")

if os.path.exists(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
else:
    st.error("Model file not found! Please upload 'Gmodel_compressed.tflite' to the repository.")

# Function to preprocess and predict using TFLite
def import_and_predict(image_data, interpreter):
    image = ImageOps.fit(image_data, (100, 100), Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image, dtype=np.float32) / 255.0  # Normalize
    img_reshape = np.expand_dims(image, axis=0)  # Add batch dimension

    # Get model input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_reshape)
    interpreter.invoke()  # Run inference

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

# Streamlit app header
st.write("# ***Glaucoma Detector***")
st.write("This is a simple image classification web app to predict glaucoma through a fundus image of the eye.")

# File uploader
file = st.file_uploader("Please upload a JPG image file", type=["jpg", "jpeg"])

if file is None:
    st.text("You haven't uploaded a JPG image file.")
else:
    try:
        imageI = Image.open(file)
        prediction = import_and_predict(imageI, interpreter)
        pred = prediction[0][0]  # Assuming binary classification
        
        if pred > 0.5:
            st.write("## **Prediction:** Your eye is Healthy. Great!!")
            st.balloons()
        else:
            st.write("## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
