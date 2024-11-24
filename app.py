import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tensorflow as tf
import altair as alt


st.set_page_config(
    page_title="Building Energy Prediction",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #0083B8;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #0083B8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and scalers
@st.cache_resource
def load_prediction_model():
    return load_model('first_model.keras')

@st.cache_data
def load_and_preprocess_data():
    # Load data
    df = pd.read_csv('generated_two.csv')
    
    # Convert Time to minutes
    df['Minutes'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(df['Time'], format='%H:%M').dt.minute
    
    # Create feature scaler
    feature_scaler = MinMaxScaler()
    features_to_use = ['Minutes', 'Temp', 'Humidity', 'Light_Intensity', 'Occupancy'] + [col for col in df.columns if col.startswith('Day_')]

    X = df[features_to_use].values
    feature_scaler.fit_transform(X)
    
    # Create target scaler
    target_scaler = MinMaxScaler()
    target_scaler.fit(df['Energy'].values.reshape(-1, 1))
    
    return df, feature_scaler, target_scaler


# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Load model and data
try:
    model = load_prediction_model()
    sample_data, feature_scaler, target_scaler = load_and_preprocess_data()
    # st.success("Model and data loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()
    
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Main title
st.title("🏢 Smart Building Energy Predictor")

# Sidebar
st.sidebar.title("Input Parameters")

# Time selection
time_str = st.sidebar.time_input("Select Time", value=datetime.strptime("09:00", "%H:%M"))
minutes = time_str.hour * 60 + time_str.minute

# Day selection
day = st.sidebar.selectbox("Select Day", 
                          ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])


# Other inputs
temp = st.sidebar.slider("Temperature (°C)", min_value=15.0, max_value=35.0, value=22.0, step=0.1)
humidity = st.sidebar.slider("Humidity (%)", min_value=30.0, max_value=90.0, value=60.0, step=0.1)
light_intensity = st.sidebar.slider("Light Intensity (lux)", min_value=0, max_value=2000, value=500)
occupancy = st.sidebar.selectbox("Occupancy", [0, 1], format_func=lambda x: "Occupied" if x == 1 else "Unoccupied")

if not day or minutes is None or temp is None or humidity is None or light_intensity is None or occupancy is None:
    st.error("Please provide all input parameters before making a prediction.")
    st.stop()


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Real-time Prediction")
    
    if st.button("Predict Energy Consumption"):
        # Prepare input data
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_encoding = [1 if d == day else 0 for d in days]
        input_data = np.array([[minutes, temp, humidity, light_intensity, occupancy] + day_encoding])
        input_scaled = feature_scaler.transform(input_data[:, :5])

        try:
            # Create sequence for LSTM with proper timesteps and features
            timesteps = 9  # Match the expected timesteps of your model
            sequence = np.repeat(input_data, timesteps, axis=0)  # Repeat the full input data
            sequence = sequence.reshape(1, timesteps, input_data.shape[1])  # Ensure shape (1, timesteps, features)

            print("Input data shape:", input_data.shape)  # Should be (1, features)
            print("Scaled sequence shape:", sequence.shape)  # Should be (1, timesteps, features)
            print("Model input shape:", model.input_shape)  # Should match sequence shape



            # Predict
            prediction_scaled = model.predict(sequence)
            prediction = target_scaler.inverse_transform(prediction_scaled)[0][0]

            # Add to history
            timestamp = datetime.now()
            st.session_state.predictions_history.append({
                'timestamp': timestamp,
                'time': time_str.strftime('%H:%M'),
                'day': day,
                'prediction': prediction,
                'temperature': temp,
                'humidity': humidity,
                'light_intensity': light_intensity,
                'occupancy': occupancy
            })

            # Display prediction with confidence interval
            st.metric("Predicted Energy Consumption", f"{prediction:.2f} kWh")

            # Confidence interval
            confidence_interval = 2.63  # From your previous results
            st.info(f"95% Confidence Interval: {(prediction - confidence_interval):.2f} - {(prediction + confidence_interval):.2f} kWh")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")


with col2:
    st.subheader("🎯 Model Performance")
    metrics_cols = st.columns(2)
    with metrics_cols[0]:
        st.metric("RMSE", "1.34 kWh")
        st.metric("R² Score", "0.613")
    with metrics_cols[1]:
        st.metric("MAE", "1.07 kWh")
        st.metric("MAPE", "16.24%")

# if st.session_state.predictions_history:
#     st.subheader("📈 Prediction History")
    
#     # Convert history to DataFrame
#     history_df = pd.DataFrame(st.session_state.predictions_history)
    
#     # Safety check for required columns
#     required_columns = ['time', 'day', 'prediction', 'temperature', 'humidity', 'light_intensity', 'occupancy', 'timestamp']
#     if not all(col in history_df.columns for col in required_columns):
#         st.warning("Some columns are missing. Clearing old predictions.")
#         st.session_state.predictions_history = []
#     else:
#         # Interactive line chart of predictions
#         fig = px.line(history_df, x='timestamp', y='prediction',
#                       title='Energy Consumption Predictions Over Time',
#                       labels={'prediction': 'Energy (kWh)', 'timestamp': 'Time'})
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Show history table
#         st.dataframe(
#             history_df[['time', 'day', 'prediction', 'temperature', 'humidity', 'light_intensity', 'occupancy']]
#             .sort_values('timestamp', ascending=False)
#             .head(5)
#             .style.format({'prediction': '{:.2f}', 'temperature': '{:.1f}', 'humidity': '{:.1f}'})
#         )




################################################ Data Analysis Section ################################################
st.subheader("📊 Data Analysis")
tab1, tab2, tab3 = st.tabs(["Energy Patterns", "Feature Relationships", "Statistics"])

with tab1:
    # Daily energy pattern
    daily_energy = sample_data.groupby('Time')['Energy'].mean().reset_index()
    fig_daily = px.line(daily_energy, x='Time', y='Energy',
                        title='Average Daily Energy Consumption Pattern',
                        labels={'Energy': 'Energy (kWh)', 'Time': 'Time of Day'})
    st.plotly_chart(fig_daily, use_container_width=True)

with tab2:
    # Correlation heatmap
    correlation = sample_data[['Energy', 'Temp', 'Humidity', 'Light_Intensity', 'Occupancy']].corr()
    fig_corr = px.imshow(correlation, 
                         title='Feature Correlations',
                         color_continuous_scale='RdBu')
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    # Summary statistics
    st.write("Summary Statistics:")
    st.dataframe(sample_data[['Energy', 'Temp', 'Humidity', 'Light_Intensity']].describe())

# Footer
st.markdown("""
    <div style='text-align: center; color: grey; padding: 20px;'>
        <p>Developed with ❤️ for Smart Building Energy Management</p>
    </div>
    """, unsafe_allow_html=True)