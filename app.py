import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set the working directory to the directory of the script
working_dir = os.path.dirname(os.path.abspath(__file__))

# Set the model path relative to the script
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

# Load the model
model = tf.keras.models.load_model(model_path)

# Load the class names
with open(os.path.join(working_dir, 'class_indices.json')) as f:
    class_indices = json.load(f)

# Function to load and preprocess the image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to predict the class of an image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.title('Plant Health Monitoring System')

# Introduction with detailed subtopics
st.markdown("""
### Welcome to the Plant Health Monitoring System!

Our app uses advanced **Artificial Intelligence (AI)** to help identify and classify diseases in plant leaves through image recognition. Whether you're a farmer, gardener, or plant enthusiast, early detection of plant diseases is crucial for ensuring healthy crops and preventing losses. Below are some key points to help you understand the importance of this tool and how to make the most of it.

---

#### 🌱 Overview of Plant Disease Detection

Plant diseases can severely affect crop yield and quality, impacting food supply and livelihoods. By leveraging **Deep Learning** models trained on thousands of plant images, this app can recognize common diseases from just a photograph of a leaf. With this tool, users can quickly diagnose issues and take timely action.

---

#### 🌍 Why Early Diagnosis is Important

Early detection of plant diseases is the first line of defense against potential outbreaks. Identifying and managing these diseases at an early stage can:
- Improve crop yield and quality.
- Reduce the need for harmful pesticides.
- Lower overall farming costs.
- Help farmers focus on prevention rather than cure.

The goal is to minimize damage to crops, leading to more sustainable farming practices and healthier produce.

---

#### 🤖 The Role of AI in Agriculture

AI is transforming the agricultural industry. By combining **Machine Learning** with image processing, we can automate the identification of diseases and even monitor crop health over time. Our model has been trained on a wide variety of plant images to accurately classify diseases, making it a powerful tool in the field of **precision agriculture**.

---

#### 🚀 How to Use This Application

1. **Upload an Image:** Use the uploader below to submit a clear image of a plant leaf. The image should ideally be a close-up with the leaf centered.
2. **Analyze the Results:** Once the image is uploaded, click the **Classify** button. The app will process the image and provide a prediction of the plant's health.
3. **Act on the Information:** Based on the prediction, you can take necessary steps to treat the disease if detected or simply monitor your plant's health if it is classified as healthy.

Using this app will allow you to understand your plants better and protect them from common diseases. The prediction model covers various common plant diseases, so you can rely on it for a range of crops.

---

""")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
