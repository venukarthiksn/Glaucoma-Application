import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Function to preprocess and predict
def import_and_predict(image_data, model):
    # Resize the image to match model input
    image = ImageOps.fit(image_data, (100, 100), Image.Resampling.LANCZOS)
    image = image.convert('RGB')
    image = np.asarray(image)
    
    # Display uploaded image
    st.image(image, channels='RGB')
    
    # Normalize the image
    image = image.astype(np.float32) / 255.0
    img_reshape = image[np.newaxis, ...]  # Add batch dimension
    prediction = model.predict(img_reshape)  # Predict with the model
    return prediction

# Load the trained model
model_path = r"C:\Users\kadgekar venukarthik\Downloads\MINI_PROJECT_GLAUCOMA\my_model2.h5"
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found! Please check the file path.")

# Streamlit app header
st.write("""
         # ***Glaucoma Detector***
         """
         )
st.write("This is a simple image classification web app to predict glaucoma through a fundus image of the eye.")

# File uploader
file = st.file_uploader("Please upload a JPG image file", type=["jpg", "jpeg"])

if file is None:
    st.text("You haven't uploaded a JPG image file.")
else:
    try:
        imageI = Image.open(file)
        prediction = import_and_predict(imageI, model)
        pred = prediction[0][0]  # Assuming binary classification
        
        if pred > 0.5:
            st.write("""
                     ## **Prediction:** Your eye is Healthy. Great!!
                     """)
            st.balloons()
        else:
            st.write("""
                     ## **Prediction:** You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                     """)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
