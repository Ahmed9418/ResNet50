import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Config
st.set_page_config(page_title="Plant Pathogen Identifier", page_icon="🌿")

@st.cache_resource
def load_model():
    # Load the TFLite model
    # Ensure 'pathogen_model_final.tflite' is in the same directory
    interpreter = tf.lite.Interpreter(model_path="pathogen_model_ResNet50.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_class_names():
    with open('pathogen_labels.txt', 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def predict(image, interpreter, class_names):
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get expected input shape (e.g., 224, 224) from the model itself
    input_shape = input_details[0]['shape']
    target_height = input_shape[1]
    target_width = input_shape[2]
    
    # 1. Resize image to match model's expected input
    img = image.resize((target_width, target_height))
    
    # 2. Convert to array
    img_array = np.array(img).astype(np.float32)
    
    # 3. Expand dimensions (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. Preprocess (Match ResNet50 training)
    # ResNet50 expects zero-centered BGR, not just 0-1 scaling.
    # We use the standard TF function to ensure accuracy matches your notebook.
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    # 5. Run Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    # 6. Process Results
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# --- UI Interface ---
st.title("🌿 Plant Pathogen Identifier (Lite)")
st.write("Upload an image of a plant leaf to detect potential diseases.")

# Load resources
try:
    interpreter = load_model()
    class_names = load_class_names()
    st.success("System ready! (Running on TFLite)")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Leaf'):
        with st.spinner('Analyzing pathogen signatures...'):
            try:
                label, confidence = predict(image, interpreter, class_names)
                
                st.markdown(f"### Result: **{label}**")
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence * 100:.2f}%")
                
                if confidence < 0.6:
                    st.warning("⚠️ Confidence is low. The image might not be clear.")
            except Exception as e:
                st.error(f"Error processing image: {e}")
