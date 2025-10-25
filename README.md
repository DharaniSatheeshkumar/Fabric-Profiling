# Fabric-Profiling

AI-powered fabric analysis using Google's Gemini Vision API and Streamlit

## 🚀 Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up your Google API key:
   - Option 1: Set environment variable:
     ```bash
     # Windows PowerShell
     $env:GOOGLE_API_KEY="your-api-key-here"
     
     # Linux/MacOS
     export GOOGLE_API_KEY="your-api-key-here"
     ```
   - Option 2: Create `.streamlit/secrets.toml`:
     ```toml
     GOOGLE_API_KEY = "your-api-key-here"
     ```

3. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## 📱 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your GitHub repository
4. Add your `GOOGLE_API_KEY` in Streamlit Cloud's secrets management

### Local Development

For local development, you can create a `.env` file:
```
GOOGLE_API_KEY=your-api-key-here
```

## 🔧 Features

- Upload and analyze fabric images
- Multiple analysis types:
  - Comprehensive analysis
  - Crimp analysis
  - Type classification
  - Quality assessment
- Detailed results including:
  - Fabric type identification
  - Weave pattern analysis
  - Fiber composition
  - Quality assessment
  - Defect detection
  
## 📝 API Key

Get your Google Gemini API key from: https://makersuite.google.com/app/apikey

## 📈 Sample Usage

1. Launch the Streamlit app
2. Upload a fabric image
3. Choose analysis type
4. Click "Analyze Fabric"
5. View detailed results