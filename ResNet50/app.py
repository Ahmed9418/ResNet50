import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite # Using standard TF-CPU Lite interpreter

# Page Config
st.set_page_config(page_title="Plant Pathogen Identifier", page_icon="🌿")

@st.cache_resource
def load_model():
    # Load TFLite model
    interpreter = tflite.Interpreter(model_path="ResNet50/pathogen_model_ResNet50.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_class_names():
    with open('ResNet50/pathogen_labels.txt', 'r') as f:
        class_names = f.read().splitlines()
    return class_names

def preprocess_image(image, target_size=(224, 224)):
    # 1. Resize
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(target_size)
    
    # 2. Convert to Numpy Array
    img_array = np.array(img).astype(np.float32)
    
    # 3. Manual ResNet50 Preprocessing
    # ResNet50 expects BGR format, not RGB
    # It also subtracts the mean values of ImageNet
    
    # Convert RGB to BGR
    img_array = img_array[..., ::-1]
    
    # Mean subtraction (ImageNet means: [103.939, 116.779, 123.68])
    mean = [103.939, 116.779, 123.68]
    img_array[..., 0] -= mean[0]
    img_array[..., 1] -= mean[1]
    img_array[..., 2] -= mean[2]
    
    # 4. Add Batch Dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image, interpreter, class_names):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Set Tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    
    # Run Inference
    interpreter.invoke()
    
    # Get Output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    return predicted_class, confidence

# --- UI Interface ---
st.title("🌿 Plant Pathogen Identifier (ResNet50)/(Plant_Village)")
st.write("Upload a leaf image to detect diseases.")

try:
    interpreter = load_model()
    class_names = load_class_names()
    st.success("System ready!")
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Leaf'):
        with st.spinner('Processing...'):
            try:
                label, confidence = predict(image, interpreter, class_names)
                
                st.markdown(f"### Result: **{label}**")
                st.progress(float(confidence))
                st.write(f"Confidence: {confidence * 100:.2f}%")
                
            except Exception as e:
                st.error(f"Error: {e}")


