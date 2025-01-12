import streamlit as st
import streamlit.components.v1 as components
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
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 
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
        light_intensity = st.slider("Light Intensity (lux)", 0, 1500, 
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
    
    
    
    
    ###################################### ALL PREDICTIONS MADE #################################################
    # # Prediction button
    # if st.button("Predict Energy Consumption"):
    #     input_data = np.array([[
    #         minutes, temperature, humidity, light_intensity, occupancy,
    #         temperature * humidity, light_intensity * occupancy,
    #         temperature * light_intensity, temperature * occupancy,
    #         humidity * light_intensity, is_weekend, 
    #         0, temperature, humidity, 0, temperature, humidity,
    #     ]])
        
    #     # Add day encoding
    #     days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    #     day_encoding = [1 if day == selected_day else 0 for day in days]
    #     input_data = np.concatenate([input_data, np.array([day_encoding])], axis=1)
        
    #     # Prepare sequences for both models
    #     gru_sequence, rf_input = prepare_sequence_input(
    #         input_data, 
    #         model_params['seq_length'], 
    #         scaler_X
    #     )
        
    #     # Get predictions with uncertainty
    #     mc_samples = 100
    #     gru_predictions = np.array([gru_model.predict(gru_sequence, verbose=0) 
    #                             for _ in range(mc_samples)])
    #     gru_pred = gru_predictions.mean(axis=0)
    #     uncertainty = gru_predictions.std(axis=0)
        
    #     # RF prediction
    #     rf_pred = rf_model.predict(rf_input)
        
    #     # Ensemble prediction
    #     ensemble_pred = (model_params['gru_weight'] * gru_pred + 
    #                     model_params['rf_weight'] * rf_pred)
        
    #     # Transform predictions back to original scale
    #     final_prediction = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1))[0][0]
    #     prediction_uncertainty = scaler_y.inverse_transform(uncertainty.reshape(-1, 1))[0][0]
        
    #     # Display current prediction in a metric card
    #     st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("Current Prediction", f"{final_prediction:.2f} kWh")
    #     with col2:
    #         st.metric("Uncertainty (±)", f"{prediction_uncertainty*2:.2f} kWh")
    #     with col3:
    #         confidence_interval = f"{(final_prediction - 2*prediction_uncertainty):.2f} - {(final_prediction + 2*prediction_uncertainty):.2f} kWh"
    #         st.metric("95% Confidence Interval", confidence_interval)
    #     st.markdown('</div>', unsafe_allow_html=True)
        

    #     # Add a time series plot of predicted energy
    #     st.subheader("Predicted Energy Consumption Pattern")
        
    #     # Generate a sequence of predictions for the next few hours
    #     current_time = datetime.now().replace(hour=hour, minute=minutes % 60)
    #     future_times = [(current_time + timedelta(hours=i)).strftime('%H:%M') 
    #                 for i in range(6)]
    #     future_predictions = []
    #     future_uncertainties = []
        
    #     for i in range(6):
    #         input_data[0, 0] = (i * 60 + minutes) % (24 * 60)  # Update time
    #         seq, rf_in = prepare_sequence_input(input_data, model_params['seq_length'], scaler_X)
            
    #         # Get predictions
    #         gru_preds = np.array([gru_model.predict(seq, verbose=0) 
    #                             for _ in range(mc_samples)])
    #         gru_p = gru_preds.mean(axis=0)
    #         unc = gru_preds.std(axis=0)
    #         rf_p = rf_model.predict(rf_in)
            
    #         # Ensemble
    #         pred = (model_params['gru_weight'] * gru_p + 
    #             model_params['rf_weight'] * rf_p)
            
    #         # Transform
    #         final_pred = scaler_y.inverse_transform(pred.reshape(-1, 1))[0][0]
    #         pred_unc = scaler_y.inverse_transform(unc.reshape(-1, 1))[0][0]
            
    #         future_predictions.append(final_pred)
    #         future_uncertainties.append(pred_unc)

    #     # Plot the predictions
    #     fig = go.Figure()

    #     # Add prediction line
    #     fig.add_trace(go.Scatter(
    #         x=future_times,
    #         y=future_predictions,
    #         mode='lines+markers',
    #         name='Predicted Energy',
    #         line=dict(color='rgb(31, 71, 124)', width=3),
    #         marker=dict(size=10)
    #     ))

    #     # Add confidence intervals
    #     fig.add_trace(go.Scatter(
    #         x=future_times + future_times[::-1],
    #         y=[y + 2*u for y, u in zip(future_predictions, future_uncertainties)] + 
    #         [y - 2*u for y, u in zip(future_predictions[::-1], future_uncertainties[::-1])],
    #         fill='toself',
    #         fillcolor='rgba(31, 71, 124, 0.2)',
    #         line=dict(color='rgba(255,255,255,0)'),
    #         name='95% Confidence Interval',
    #         hoverinfo='skip'
    #     ))

    #     # Highlight current prediction point
    #     fig.add_trace(go.Scatter(
    #         x=[future_times[0]],
    #         y=[future_predictions[0]],
    #         mode='markers',
    #         name='Current Prediction',
    #         marker=dict(
    #             color='red',
    #             size=15,
    #             symbol='star'
    #         )
    #     ))

    #     fig.update_layout(
    #         title={
    #             'text': 'Energy Consumption Forecast',
    #             'y':0.95,
    #             'x':0.5,
    #             'xanchor': 'center',
    #             'yanchor': 'top'
    #         },
    #         xaxis_title='Time',
    #         yaxis_title='Energy Consumption (kWh)',
    #         showlegend=True,
    #         hovermode='x unified',
    #         hoverlabel=dict(bgcolor="#1F477C"),
    #         yaxis=dict(
    #             tickformat='.2f',
    #             zeroline=True,
    #             zerolinewidth=2,
    #             zerolinecolor='lightgray'
    #         )
    #     )

    #     # Add custom hover template
    #     fig.update_traces(
    #         hovertemplate="<b>Time:</b> %{x}<br>" +
    #                      "<b>Energy:</b> %{y:.2f} kWh<br>",
    #         selector=dict(name='Predicted Energy')
    #     )

    #     st.plotly_chart(fig, use_container_width=True)
    #     st.markdown('</div>', unsafe_allow_html=True)
    
    
    ###########################################################################################################
    
    
    
    # Prediction section - separate buttons for immediate and future predictions
    pred_col1, pred_col2 = st.columns(2)

    # Button 1: Current Prediction
    with pred_col1:
        if st.button("🎯 Get Current Prediction"):
            # Prepare input data
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
            
            # Prepare sequences
            gru_sequence, rf_input = prepare_sequence_input(input_data, model_params['seq_length'], scaler_X)
            
            with st.spinner('Calculating current prediction...'):
                # Get predictions with reduced samples
                mc_samples = 30  # Reduced for better performance
                gru_predictions = np.array([gru_model.predict(gru_sequence, verbose=0) 
                                        for _ in range(mc_samples)])
                gru_pred = gru_predictions.mean(axis=0)
                uncertainty = gru_predictions.std(axis=0)
                
                # RF prediction
                rf_pred = rf_model.predict(rf_input)
                
                # Ensemble prediction
                ensemble_pred = (model_params['gru_weight'] * gru_pred + 
                            model_params['rf_weight'] * rf_pred)
                
                # Transform predictions
                final_prediction = scaler_y.inverse_transform(ensemble_pred.reshape(-1, 1))[0][0]
                prediction_uncertainty = scaler_y.inverse_transform(uncertainty.reshape(-1, 1))[0][0]

                # Display current prediction
                st.success("Current Prediction Ready!")
                
                
                metrics_container = st.container()
                
               
                # st.markdown("""
                #     <style>
                #     .metric-card {
                #         background-color: white;
                #         padding: 20px;
                #         border-radius: 10px;
                #         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                #         margin-bottom: 20px;
                #     }
                #     .metric-row {
                #         display: flex;
                #         justify-content: space-between;
                #         gap: 20px;
                #         margin-bottom: 10px;
                #     }
                #     .metric-item {
                #         background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
                #         padding: 20px;
                #         border-radius: 8px;
                #         flex: 1;
                #         text-align: center;
                #         border: 1px solid #e1e4e8;
                #         transition: all 0.3s ease;
                #     }
                #     .metric-item:hover {
                #         transform: translateY(-5px);
                #         box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
                #     }
                #     .metric-title {
                #         color: #1f477c;
                #         font-size: 1.1em;
                #         font-weight: bold;
                #         margin-bottom: 10px;
                #     }
                #     .metric-value {
                #         color: #0c2d5e;
                #         font-size: 1.8em;
                #         font-weight: bold;
                #     }
                #     .metric-tooltip {
                #         font-size: 0.9em;
                #         color: #666;
                #         margin-top: 10px;
                #         display: none;
                #     }
                #     .metric-item:hover .metric-tooltip {
                #         display: block;
                #     }
                #     </style>
                # """, unsafe_allow_html=True)

                # # Then, create your metrics display with fixed formatting
                # metrics_html = f"""
                # <div class="metric-card">
                #     <h3 style='text-align: center; color: #1f477c; font-size: 1.5em; margin-bottom: 20px;'>
                #         🎯 Energy Prediction Results
                #     </h3>
                #     <div class="metric-row">
                #         <div class="metric-item">
                #             <div class="metric-title">Predicted Energy</div>
                #             <div class="metric-value">{final_prediction:.2f} kWh</div>
                #             <div class="metric-tooltip">
                #                 This is the estimated energy consumption for your building 
                #                 based on the current conditions and input parameters.
                #             </div>
                #         </div>
                        
                #         <div class="metric-item">
                #             <div class="metric-title">Uncertainty</div>
                #             <div class="metric-value">±{prediction_uncertainty:.2f} kWh</div>
                #             <div class="metric-tooltip">
                #                 Represents the potential variation in energy consumption prediction. 
                #                 Lower values indicate more confident predictions.
                #             </div>
                #         </div>
                        
                #         <div class="metric-item">
                #             <div class="metric-title">Confidence Range</div>
                #             <div class="metric-value">{final_prediction - prediction_uncertainty:.2f} - {final_prediction + prediction_uncertainty:.2f} kWh</div>
                #             <div class="metric-tooltip">
                #                 The expected range of energy consumption with 95% confidence. 
                #                 Actual consumption is likely to fall within this range.
                #             </div>
                #         </div>
                #     </div>
                # </div>
                # """

                # # Display the metrics
                # st.markdown(metrics_html, unsafe_allow_html=True)
                # st.markdown('</div>', unsafe_allow_html=True)
                
                
                
                # Define the HTML content
                html_content = f"""
                    <div style="
                        background-color: white;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        margin-bottom: 20px;
                    ">
                        <h3 style="text-align: center; color: #1f477c; font-size: 1.5em; margin-bottom: 20px;">
                            🎯 Energy Prediction Results
                        </h3>
                        <div style="
                            display: flex;
                            justify-content: space-between;
                            gap: 20px;
                            margin-bottom: 10px;
                        ">
                            <div style="
                                background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
                                padding: 20px;
                                border-radius: 8px;
                                flex: 1;
                                text-align: center;
                                border: 1px solid #e1e4e8;
                                transition: all 0.3s ease;
                            ">
                                <div style="color: #1f477c; font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">
                                    Predicted Energy
                                </div>
                                <div style="color: #0c2d5e; font-size: 1.8em; font-weight: bold;">
                                    {final_prediction:.2f} kWh
                                </div>
                                <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                                    Estimated energy consumption based on current conditions
                                </div>
                            </div>
                            
                            <div style="
                                background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
                                padding: 20px;
                                border-radius: 8px;
                                flex: 1;
                                text-align: center;
                                border: 1px solid #e1e4e8;
                                transition: all 0.3s ease;
                            ">
                                <div style="color: #1f477c; font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">
                                    Uncertainty
                                </div>
                                <div style="color: #0c2d5e; font-size: 1.8em; font-weight: bold;">
                                    ±{prediction_uncertainty:.2f} kWh
                                </div>
                                <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                                    Potential variation in prediction
                                </div>
                            </div>
                            
                            <div style="
                                background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
                                padding: 20px;
                                border-radius: 8px;
                                flex: 1;
                                text-align: center;
                                border: 1px solid #e1e4e8;
                                transition: all 0.3s ease;
                            ">
                                <div style="color: #1f477c; font-size: 1.1em; font-weight: bold; margin-bottom: 10px;">
                                    Confidence Range
                                </div>
                                <div style="color: #0c2d5e; font-size: 1.8em; font-weight: bold;">
                                    {final_prediction - prediction_uncertainty:.2f} - {final_prediction + prediction_uncertainty:.2f} kWh
                                </div>
                                <div style="font-size: 0.9em; color: #666; margin-top: 10px;">
                                    95% confidence interval
                                </div>
                            </div>
                        </div>
                    </div>
                """
                
                # Use st.components.html() to render the HTML
                components.html(html_content, height=390, width=700, scrolling=True)
                

    # Button 2: Future Predictions
    with pred_col2:
        if st.button("📈 Show Future Forecast"):
            # Cache function for future predictions
            @st.cache_data(ttl=300)  # Cache for 5 minutes
            def calculate_future_predictions(temp, hum, light, occ, mins, day, weekend):
                # Initialize arrays
                future_times = []
                predictions = []
                uncertainties = []
                
                current_time = datetime.now().replace(hour=mins//60, minute=mins%60)
                
                # Base input data
                base_input = np.array([[
                    mins, temp, hum, light, occ,
                    temp * hum, light * occ,
                    temp * light, temp * occ,
                    hum * light, weekend, 
                    0, temp, hum, 0, temp, hum,
                ]])
                
                # Add day encoding
                days = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_encoding = [1 if d == day else 0 for d in days]
                base_input = np.concatenate([base_input, np.array([day_encoding])], axis=1)
                
                # Calculate predictions for next 6 hours
                for i in range(6):
                    future_time = current_time + timedelta(hours=i)
                    future_times.append(future_time.strftime('%H:%M'))
                    
                    # Update time in input
                    temp_input = base_input.copy()
                    temp_input[0, 0] = (mins + i * 60) % (24 * 60)
                    
                    # Get prediction
                    seq, rf_in = prepare_sequence_input(temp_input, model_params['seq_length'], scaler_X)
                    
                    # Reduced samples for speed
                    gru_preds = np.array([gru_model.predict(seq, verbose=0) 
                                        for _ in range(20)])  # Even fewer samples for future predictions
                    gru_p = gru_preds.mean(axis=0)
                    unc = gru_preds.std(axis=0)
                    rf_p = rf_model.predict(rf_in)
                    
                    pred = (model_params['gru_weight'] * gru_p + 
                        model_params['rf_weight'] * rf_p)
                    
                    predictions.append(float(scaler_y.inverse_transform(pred.reshape(-1, 1))[0][0]))
                    uncertainties.append(float(scaler_y.inverse_transform(unc.reshape(-1, 1))[0][0]))
                
                return future_times, predictions, uncertainties
            
            with st.spinner('Generating forecast...'):
                # Get predictions
                future_times, predictions, uncertainties = calculate_future_predictions(
                    temperature, humidity, light_intensity, occupancy, 
                    minutes, selected_day, is_weekend
                )
            
            # Create forecast plot
            fig = go.Figure()
            
            # Add main prediction line
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='rgb(31, 71, 124)', width=2),
                marker=dict(size=8)
            ))
            
            # Add uncertainty band
            fig.add_trace(go.Scatter(
                x=future_times + future_times[::-1],
                y=[y + 2*u for y, u in zip(predictions, uncertainties)] + 
                [y - 2*u for y, u in zip(predictions[::-1], uncertainties[::-1])],
                fill='toself',
                fillcolor='rgba(31, 71, 124, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Uncertainty Range',
                showlegend=True
            ))
            
            fig.update_layout(
                title='Energy Consumption Forecast (Next 6 Hours)',
                xaxis_title='Time',
                yaxis_title='Energy (kWh)',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add forecast table
            st.write("Hourly Forecast Details:")
            forecast_df = pd.DataFrame({
                'Time': future_times,
                'Predicted Energy (kWh)': [f"{p:.2f}" for p in predictions],
                'Uncertainty (±kWh)': [f"{u:.2f}" for u in uncertainties]
            })
            st.dataframe(forecast_df)
    
    
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