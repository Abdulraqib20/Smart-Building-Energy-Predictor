import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Smart Building Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        background-color: #1f477c;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #0083B8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏢 Smart Building Energy Predictor")
st.markdown("""
This application uses a deep learning LSTM model to predict building energy consumption 
based on various environmental and temporal factors.
""")

# Sidebar configuration
st.sidebar.title("Input Params")

@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Minutes'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df['Time'], format='%H:%M').dt.minute
    df = pd.get_dummies(df, columns=['Day'])
    return df

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('first_model.keras')

# Load data and model
try:
    df = load_and_preprocess_data('generated_two.csv')
    model = load_model()
    
    # Initialize scalers
    features_to_use = ['Minutes', 'Temp', 'Humidity', 'Light_Intensity', 'Occupancy'] + [col for col in df.columns if col.startswith('Day_')]
    X = df[features_to_use].values
    y = df['Energy'].values
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1))
    
    # Main content
    st.subheader("📊 Real-time Prediction")
    # Create two columns for input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sidebar
        st.sidebar.subheader("Environmental Parameters")
        temperature = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.1)
        humidity = st.sidebar.slider("Humidity (%)", 20.0, 80.0, 50.0, 0.1)
        light_intensity = st.sidebar.slider("Light Intensity (lux)", 0, 1000, 500)
        occupancy = st.sidebar.radio(
            "Occupancy Status",
            options=[0, 1],
            format_func=lambda x: "Unoccupied" if x == 0 else "Occupied",
            horizontal=True,
            help="0 = Unoccupied, 1 = Occupied"
        )
    
    with col2:
        st.sidebar.subheader("Temporal Parameters")
        selected_time = st.sidebar.time_input("Time of Day", datetime.strptime("09:00", "%H:%M"))
        selected_day = st.sidebar.selectbox("Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    
    # Create prediction button
    if st.button("Predict Energy Consumption"):
        # Convert time to minutes
        minutes = selected_time.hour * 60 + selected_time.minute
        
        # Create day one-hot encoding
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_encoding = [1 if day == selected_day else 0 for day in days]
        
        # Create input feature array
        input_features = np.array([[
            minutes, temperature, humidity, light_intensity, occupancy, *day_encoding
        ]])
        
        # Scale features
        input_scaled = feature_scaler.transform(input_features)
        
        # Create sequence for LSTM (using the last 9 timestamps)
        sequence = np.tile(input_scaled, (9, 1))
        sequence = sequence.reshape(1, 9, input_scaled.shape[1])
        
        # Make prediction
        prediction_scaled = model.predict(sequence)
        prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]
        
        # Display prediction in a nice card
        st.markdown("""
            <div class="metric-card">
                <h3 style="color: #1f477c;">Predicted Energy Consumption</h3>
                <h2 style="color: #2ecc71;">{:.2f} kWh</h2>
            </div>
        """.format(prediction), unsafe_allow_html=True)
        
        # Confidence interval
        confidence_interval = 2.40
        st.info(f"95% Confidence Interval: {(prediction - confidence_interval):.2f} kWh - {(prediction + confidence_interval):.2f} kWh")
    
    # Historical Data Analysis
    st.header("Historical Data Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Energy Patterns", "Feature Relationships", "🎯 Model Performance"])
    
    with tab1:
        # Daily energy pattern
        daily_energy = df.groupby('Minutes')['Energy'].mean().reset_index()
        fig = px.line(daily_energy, x='Minutes', y='Energy',
                     title='Average Daily Energy Consumption Pattern',
                     labels={'Minutes': 'Time (minutes from midnight)',
                            'Energy': 'Energy Consumption (kWh)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['Minutes', 'Temp', 'Humidity', 'Light_Intensity', 'Occupancy', 'Energy']
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr,
                       labels=dict(color="Correlation"),
                       x=numeric_cols,
                       y=numeric_cols,
                       color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        metrics_cols = st.columns(2)
    with metrics_cols[0]:
        st.metric("RMSE", "1.34 kWh")
        st.metric("R² Score", "0.613")
    with metrics_cols[1]:
        st.metric("MAE", "1.07 kWh")
        st.metric("MAPE", "16.24%")

except Exception as e:
    st.error(f"Error loading data or model: {str(e)}")
    st.info("Please ensure that 'generated_two.csv' and 'first_model.keras' are in the same directory as this script.")

# Footer
st.markdown("---")
# Footer
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Developed with ❤️ for Smart Building Energy Management</p>
    </div>
    """, unsafe_allow_html=True)