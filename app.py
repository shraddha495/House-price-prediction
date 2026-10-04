import numpy as np
import pickle
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Real Estate Price Predictor", page_icon="🏡", layout="centered"
)

# Custom CSS for Natural Homes Background and Clear Dark Text
st.markdown(
    """
    <style>
    /* Background with a beautiful natural neighborhood/homes image and dark overlay */
    .stApp {
        background: linear-gradient(rgba(10, 25, 47, 0.65), rgba(10, 25, 47, 0.65)), 
                    url('https://images.unsplash.com/photo-1564013799919-ab600027ffc6?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Inter', sans-serif;
    }

    /* Glassmorphism Container for Content */
    .main-container {
        background: rgba(255, 255, 255, 0.92);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Explicitly forcing all headings, text, and labels inside the app to be dark and clearly visible */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, span, div {
        color: #1e293b !important;
    }
    
    /* Input Field Labels specific styling */
    .stNumberInput label, .stSlider label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Modern Button Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white !important;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.6);
    }

    /* Result Box Styling */
    .result-box {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    .result-text {
        color: #065f46 !important;
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Load the Pickled Linear Regression Model
@st.cache_resource
def load_model():
  with open("linear.pkl", "rb") as file:
    model = pickle.load(file)
  return model


model = load_model()

# Wrap app content in a styled container card
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Application Title & Subtitle
st.title("🏡 Real Estate Price Predictor")
st.caption(
    "Enter the property details below to estimate the housing price using your"
    " Linear Regression model."
)

st.markdown("---")

# Input Form Fields matching model schema features
st.write("### 📝 Property Details")

col1, col2 = st.columns(2)

with col1:
  square_footage = st.number_input(
      "Square Footage (sq ft)",
      min_value=300,
      max_value=15000,
      value=2000,
      step=50,
  )
  num_bedrooms = st.number_input(
      "Number of Bedrooms", min_value=1, max_value=10, value=3, step=1
  )
  num_bathrooms = st.number_input(
      "Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5
  )
  year_built = st.number_input(
      "Year Built", min_value=1800, max_value=2026, value=2010, step=1
  )

with col2:
  lot_size = st.number_input(
      "Lot Size (sq ft)", min_value=200, max_value=100000, value=5000, step=100
  )
  garage_size = st.number_input(
      "Garage Size (Cars)", min_value=0, max_value=6, value=2, step=1
  )
  neighborhood_quality = st.slider(
      "Neighborhood Quality (1-10)", min_value=1, max_value=10, value=7, step=1
  )

# Prediction Action
if st.button("🔮 Predict House Price"):
  # Format features matching model schema order:
  # ['Square_Footage', 'Num_Bedrooms', 'Num_Bathrooms', 'Year_Built', 'Lot_Size', 'Garage_Size', 'Neighborhood_Quality']
  features = np.array([[
      square_footage,
      num_bedrooms,
      num_bathrooms,
      year_built,
      lot_size,
      garage_size,
      neighborhood_quality,
  ]])

  # Perform Prediction
  prediction = model.predict(features)[0]

  # Display Pretty Result Output formatted as currency
  st.markdown(
      f"""
        <div class="result-box">
            <span style="color: #047857 !important; font-weight: 600; font-size: 16px;">Estimated Property Price:</span>
            <div class="result-text">${prediction:,.2f}</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

  st.balloons()

st.markdown("</div>", unsafe_allow_html=True)
