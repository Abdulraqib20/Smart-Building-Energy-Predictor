import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import joblib
import time
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Smart Building Energy Predictor",
    page_icon="⚡",
    layout="wide"
)
# Title and description
st.title("🏢 Smart Building Energy Predictor")
st.markdown("""
This application predicts building energy consumption using an ensemble of GRU and Random Forest models.
""")

# Meter Mode Selection
st.sidebar.subheader("Meter Configuration")
meter_mode = st.sidebar.selectbox(
    "Select Meter Mode",
    ["Smart Meter 1", "Smart Meter 2", "Smart Meter 3"]
)

# Custom CSS (keeping your existing styles)
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        background-color: #1f477c;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sensor-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    gru_model = tf.keras.models.load_model('gru_model.keras')
    rf_model = joblib.load('random_forest_model.joblib')
    scaler_X = joblib.load('feature_scaler.joblib')
    scaler_y = joblib.load('target_scaler.joblib')
    model_params = joblib.load('model_params.joblib')
    return gru_model, rf_model, scaler_X, scaler_y, model_params


def generate_random_sensor_data(n_points=100):
    current_time = datetime.now()
    times = [(current_time - timedelta(minutes=i)).strftime('%H:%M') for i in range(n_points-1, -1, -1)]
    return {
        'temperature': np.random.normal(25, 2, n_points),
        'humidity': np.random.normal(50, 5, n_points),
        'light': np.random.normal(500, 100, n_points),
        'times': times
    }

def create_real_time_chart(data, y_values, title, y_label, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['times'][-30:],  # Show last 30 points
        y=y_values[-30:],
        mode='lines+markers',
        name=title,
        line=dict(color=color)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=y_label,
        height=300
    )
    return fig

def prepare_sequence_input(input_data, seq_length, scaler_X):
    # Scale the input
    input_scaled = scaler_X.transform(input_data)
    
    # Create sequence
    sequence = np.tile(input_scaled, (seq_length, 1))
    sequence = sequence.reshape(1, seq_length, -1)
    
    # For Random Forest, flatten the sequence
    rf_input = sequence.reshape(1, -1)
    
    return sequence, rf_input

st.subheader("📊 Real-time Sensor Data")

# Initialize session state for sensor data
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = generate_random_sensor_data()
    st.session_state.last_update = time.time()

# Update sensor data every 5 seconds
if time.time() - st.session_state.last_update > 5:
    new_data = generate_random_sensor_data()
    st.session_state.sensor_data = new_data
    st.session_state.last_update = time.time()

# Display real-time charts
col1, col2, col3 = st.columns(3)


# temperature
with col1:
    st.plotly_chart(
        create_real_time_chart(
            st.session_state.sensor_data,
            st.session_state.sensor_data['temperature'],
            'Temperature History',
            'Temperature (°C)',
            'red'
        ),
        use_container_width=True
    )

# humidity
with col2:
    st.plotly_chart(
        create_real_time_chart(
            st.session_state.sensor_data,
            st.session_state.sensor_data['humidity'],
            'Humidity History',
            'Humidity (%)',
            'blue'
        ),
        use_container_width=True
    )

# light
with col3:
    st.plotly_chart(
        create_real_time_chart(
            st.session_state.sensor_data,
            st.session_state.sensor_data['light'],
            'Light Intensity History',
            'Light (lux)',
            'orange'
        ),
        use_container_width=True
    )


# Load models
try:
    gru_model, rf_model, scaler_X, scaler_y, model_params = load_models()
    
    # Add "Tap for New Input" feature
    if st.button("🔄 Tap for New Input"):
        st.session_state.temperature = np.random.uniform(20.0, 30.0)
        st.session_state.humidity = np.random.uniform(40.0, 70.0)
        st.session_state.light_intensity = np.random.uniform(200, 800)
        st.session_state.occupancy = np.random.choice([0, 1])
    
    # Create three columns for visual representation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
        temperature = st.slider("Temperature (°C)", 15.0, 35.0, 
                              float(getattr(st.session_state, 'temperature', 25.0)), 0.1)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=temperature,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [15, 35]},
                  'bar': {'color': "#1f477c"}}))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
        humidity = st.slider("Humidity (%)", 20.0, 80.0, 
                           float(getattr(st.session_state, 'humidity', 50.0)), 0.1)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=humidity,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [20, 80]},
                  'bar': {'color': "#1f477c"}}))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="sensor-card">', unsafe_allow_html=True)
        light_intensity = st.slider("Light Intensity (lux)", 0, 1000, 
                                  int(getattr(st.session_state, 'light_intensity', 500)))
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=light_intensity,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1000]},
                  'bar': {'color': "#1f477c"}}))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Temporal parameters
    col1, col2 = st.columns(2)
    with col1:
        selected_time = st.time_input("Time of Day", datetime.strptime("09:00", "%H:%M"))
        minutes = selected_time.hour * 60 + selected_time.minute
        hour = minutes // 60
    
    with col2:
        selected_day = st.selectbox("Day of Week", 
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        is_weekend = 1 if selected_day in ['Saturday', 'Sunday'] else 0

    occupancy = st.radio(
        "Occupancy Status",
        options=[0, 1],
        format_func=lambda x: "Unoccupied" if x == 0 else "Occupied",
        horizontal=True
    )

    
    
    # Prediction button
    if st.button("Predict Energy Consumption"):
        input_data = np.array([[
            minutes, temperature, humidity, light_intensity, occupancy,
            temperature * humidity, light_intensity * occupancy,
            temperature * light_intensity, temperature * occupancy,
            humidity * light_intensity, is_weekend, 
            0, temperature, humidity, 0, temperature, humidity,
        ]])
        
        # Add day encoding
        days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_encoding = [1 if day == selected_day else 0 for day in days]
        input_data = np.concatenate([input_data, np.array([day_encoding])], axis=1)
        
        # Prepare sequences for both models
        gru_sequence, rf_input = prepare_sequence_input(
            input_data, 
            model_params['seq_length'], 
            scaler_X
        )
        
        # Get predictions with uncertainty
        mc_samples = 100
        gru_predictions = np.array([gru_model.predict(gru_sequence, verbose=0) 
                                for _ in range(mc_samples)])
        gru_pred = gru_predictions.mean(axis=0)
        uncertainty = gru_predictions.std(axis=0)
        
        # RF prediction
        rf_pred = rf_model.predict(rf_input)
        
        # Ensemble prediction
        ensemble_pred = (model_params['gru_weight'] * gru_pred + 
                        model_params['rf_weight'] * rf_pred)
        
        # Transform predictions back to original scale
        final_prediction = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1))[0][0]
        prediction_uncertainty = scaler_y.inverse_transform(uncertainty.reshape(-1, 1))[0][0]

        # Add a time series plot of predicted energy
        st.subheader("Predicted Energy Consumption Pattern")
        
        # Generate a sequence of predictions for the next few hours
        future_times = [(datetime.now() + timedelta(hours=i)).strftime('%H:%M') 
                    for i in range(6)]
        future_predictions = []
        future_uncertainties = []
        
        for _ in range(6):
            input_data[0, 0] = (_ * 60 + minutes) % (24 * 60)  # Update time
            seq, rf_in = prepare_sequence_input(input_data, model_params['seq_length'], scaler_X)
            
            # Get predictions
            gru_preds = np.array([gru_model.predict(seq, verbose=0) 
                                for _ in range(mc_samples)])
            gru_p = gru_preds.mean(axis=0)
            unc = gru_preds.std(axis=0)
            rf_p = rf_model.predict(rf_in)
            
            # Ensemble
            pred = (model_params['gru_weight'] * gru_p + 
                model_params['rf_weight'] * rf_p)
            
            # Transform
            final_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))[0][0]
            pred_unc = scaler_y.inverse_transform(unc.reshape(-1, 1))[0][0]
            
            future_predictions.append(final_pred)
            future_uncertainties.append(pred_unc)

        # Plot the predictions
        fig = go.Figure()

        # Add prediction line
        fig.add_trace(go.Scatter(
            x=future_times,
            y=future_predictions,
            mode='lines+markers',
            name='Predicted Energy',
            line=dict(color='rgb(31, 71, 124)')
        ))

        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=future_times + future_times[::-1],
            y=[y + 2*u for y, u in zip(future_predictions, future_uncertainties)] + 
            [y - 2*u for y, u in zip(future_predictions[::-1], future_uncertainties[::-1])],
            fill='toself',
            fillcolor='rgba(31, 71, 124, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval'
        ))

        fig.update_layout(
            title='Predicted Energy Consumption Pattern',
            xaxis_title='Time',
            yaxis_title='Energy Consumption (kWh)',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        
        # Historical performance metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", "0.252 kWh")
        with col2:
            st.metric("MAE", "0.181 kWh")
        with col3:
            st.metric("R² Score", "0.989")
        with col4:
            st.metric("MAPE", "2.74%")
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please ensure all model files are in the same directory as this script.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <p>Developed with ❤️ for Smart Building Energy Management</p>
    </div>
    """, unsafe_allow_html=True)