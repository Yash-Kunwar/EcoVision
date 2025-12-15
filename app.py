import streamlit as st
from PIL import Image
from src.classifier import AnimalClassifier
from src.genai_service import fetch_animal_info

# --- Page Config ---
st.set_page_config(
    page_title="EcoVision AI",
    page_icon="🐾",
    layout="wide"
)

# --- Load Classifier (Cached) ---
# This @st.cache_resource is CRITICAL. 
# It prevents reloading the heavy model on every user interaction.
@st.cache_resource
def get_classifier():
    return AnimalClassifier()

try:
    classifier = get_classifier()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- UI Header ---
st.title("🐾 EcoVision")
st.markdown("Use the **Sidebar** to upload an image or use your live camera.")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input:", ("Upload Image", "Live Camera"))

image_input = None

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an animal image...", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image_input = Image.open(uploaded_file)
else:
    # This automatically accesses your webcam
    camera_file = st.sidebar.camera_input("Take a picture")
    if camera_file:
        image_input = Image.open(camera_file)

# --- Main Processing Logic ---
if image_input:
    # Layout: Image on left, AI Insights on right
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image_input, caption="Captured Image", use_container_width=True)
        
        with st.spinner("Analyzing Image..."):
            # Call our classifier logic
            label, confidence = classifier.predict(image_input)
            raw_preds = classifier.model.predict(classifier.preprocess_image(image_input))
            st.write("Raw Probabilities:", raw_preds) 
        
        # Display Prediction
        st.metric(label="Detected Species", value=label.title())
        st.progress(int(confidence))
        st.caption(f"Confidence: {confidence:.2f}%")

    # GenAI Insights Section
    with col2:
        # Only ask Gemini if the model is fairly sure (prevent hallucination)
        if confidence > 50: 
            st.subheader(f"Insights: {label.title()}")
            
            with st.spinner(f"Asking Gemini about {label}..."):
                info = fetch_animal_info(label)
            
            # Display Scientific Name
            st.markdown(f"**Scientific Name:** *{info.get('scientific_name', 'Unknown')}*")
            
            # Display Facts
            st.info("🧠 **Did you know?**")
            for fact in info.get('fun_facts', []):
                st.write(f"- {fact}")
                
            # Display Related Species
            st.warning("👀 **Similar Animals (Same Genus):**")
            st.write(", ".join(info.get('genus_members', [])))
            
        else:
            st.error(f"⚠️ Low Confidence ({confidence:.1f}%). I am not sure what this is. Please try a clearer image.")

else:
    st.info("👈 Waiting for image input... Upload a file or use the camera!")