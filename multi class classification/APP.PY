import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite  # Use tflite_runtime for lighter dependency; alternatively, import tensorflow as tf and use tf.lite

# Load the TFLite model
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")  # Ensure model.tflite is in the same directory or provide full path
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names based on the dataset (assuming from the notebook: apple, banana, orange)
class_names = ['apple', 'banana', 'orange']

# Streamlit app
st.title("Fruit Classifier")
st.write("Upload an image of an apple, banana, or orange to classify it using the trained TensorFlow Lite model.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image: resize to 32x32, convert to numpy array, normalize to [0,1], add batch dimension
    image = image.resize((32, 32))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 32, 32, 3)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the predicted class
    predicted_index = np.argmax(output_data[0])
    predicted_class = class_names[predicted_index]
    confidence = output_data[0][predicted_index] * 100

    # Display the result
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")