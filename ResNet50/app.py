import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Plant Pathogen Identifier", page_icon="🌿")

@st.cache_resource
def load_model():
    # Load the Keras model
    # Ensure 'best_model.keras' is in the same directory
    model = tf.keras.models.load_model('best_model.keras')
    return model

@st.cache_data
def load_class_names():
    # Load class names from the text file
    with open('pathogen_labels_final.txt', 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def predict(image, model, class_names):
    # 1. Resize image to 224x224 as required by ResNet50
    img = image.resize((224, 224))
    
    # 2. Convert to array
    img_array = np.array(img)
    
    # 3. Expand dimensions to match batch size (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Preprocess input (ResNet50 specific preprocessing)
    # This subtracts the mean RGB channels of the ImageNet dataset
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # 5. Make Prediction
    predictions = model.predict(img_array)
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# --- UI Interface ---
st.title("🌿 Plant Pathogen Identifier")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# Load resources
try:
    model = load_model()
    class_names = load_class_names()
    st.success("System ready!")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Leaf'):
        with st.spinner('Analyzing pathogen signatures...'):
            try:
                label, confidence = predict(image, model, class_names)
                
                # Display Results
                st.markdown(f"### Result: **{label}**")
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence * 100:.2f}%")
                
                # Simple advice based on confidence
                if confidence < 0.6:
                    st.warning("⚠️ Confidence is low. The image might not be clear or the disease is not in the database.")
            except Exception as e:
                st.error(f"Error processing image: {e}")