import streamlit as st
import os
from Fabric_profiling import GeminiFabricRecognizer
import tempfile
from PIL import Image

st.set_page_config(
    page_title="Fabric Profiling Analysis",
    page_icon="🧵",
    layout="wide"
)

st.title("🧵 Fabric Profiling Analysis")
st.markdown("Upload a fabric image for detailed analysis using Google's Gemini Vision AI")

# Initialize API key from environment or Streamlit secrets
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)

if not api_key:
    st.error("⚠️ No API key found. Please set the GOOGLE_API_KEY environment variable or add it to your Streamlit secrets.")
    st.stop()

# Initialize the recognizer
@st.cache_resource
def get_recognizer():
    return GeminiFabricRecognizer(api_key=api_key)

try:
    recognizer = get_recognizer()
    st.success("✅ Connected to Gemini API successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to initialize Gemini API: {str(e)}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded Fabric Image", use_column_width=True)
    
    with col2:
        st.markdown("### Analysis Options")
        analysis_type = st.selectbox(
            "Choose analysis type:",
            ["comprehensive", "crimp", "type", "quality"],
            help="Select the type of analysis to perform"
        )
        
        # Optional scale input
        include_scale = st.checkbox("Include scale information?")
        scale_mm_per_px = None
        if include_scale:
            scale_mm_per_px = st.number_input(
                "Scale (mm per pixel):",
                min_value=0.0001,
                max_value=10.0,
                value=0.1,
                format="%f"
            )
    
    # Analyze button
    if st.button("🔍 Analyze Fabric"):
        with st.spinner("Analyzing fabric image..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                
                # Run analysis
                result = recognizer.analyze_fabric(
                    temp_path,
                    analysis_type=analysis_type,
                    scale_mm_per_px=scale_mm_per_px
                )
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Display results
                st.markdown("## 📊 Analysis Results")
                
                # Main findings
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fabric Type", result.fabric_type)
                with col2:
                    st.metric("Confidence", f"{result.confidence:.1%}")
                with col3:
                    st.metric("Weave Pattern", result.weave_pattern)
                
                # Detailed findings
                st.markdown("### 🔍 Detailed Analysis")
                
                tabs = st.tabs(["Composition", "Texture & Quality", "Crimp Analysis", "Defects"])
                
                with tabs[0]:
                    st.markdown("#### 🧵 Fiber Composition")
                    for fiber in result.fiber_composition:
                        st.markdown(f"- {fiber}")
                
                with tabs[1]:
                    st.markdown("#### 📝 Texture Description")
                    st.write(result.texture_description)
                    st.markdown("#### ⭐ Quality Assessment")
                    st.write(result.quality_assessment)
                    if result.suggested_thread_count:
                        st.markdown("#### 🔢 Thread Count Estimate")
                        st.write(result.suggested_thread_count)
                
                with tabs[2]:
                    st.markdown("#### 📐 Crimp Analysis")
                    for key, value in result.crimp_observations.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
                
                with tabs[3]:
                    st.markdown("#### ⚠️ Defects Detected")
                    if result.defects_detected:
                        for defect in result.defects_detected:
                            st.markdown(f"- {defect}")
                    else:
                        st.success("No significant defects detected")
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

# Instructions when no file is uploaded
else:
    st.info("👆 Upload a fabric image to begin analysis")
    
# Add footer with helpful information
st.markdown("---")
st.markdown("""
### 📝 Notes
- Supported image formats: JPG, JPEG, PNG
- For best results, ensure good lighting and focus in your fabric images
- The comprehensive analysis provides the most detailed results
""")