import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Forest Microclimate Predictor",
    layout="centered"
)

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 20% 20%, rgba(34,197,94,0.08), transparent 30%),
        radial-gradient(circle at 80% 70%, rgba(16,185,129,0.08), transparent 35%),
        #06110b;
    overflow-x: hidden;
}

/* Keep main app content above effects */
.block-container {
    position: relative;
    z-index: 3;
}

/* Glowing title */
h1 {
    color: #7fffc1 !important;
    text-shadow:
        0 0 8px rgba(74,222,128,0.45),
        0 0 20px rgba(16,185,129,0.25);
}

/* Floating background particles */
.particle {
    position: fixed;
    width: 5px;
    height: 5px;
    background: rgba(110,255,190,0.85);
    border-radius: 50%;
    box-shadow:
        0 0 6px rgba(110,255,190,0.9),
        0 0 14px rgba(110,255,190,0.5);
    z-index: 1;
    pointer-events: none;
    animation: floatParticle 12s ease-in-out infinite alternate;
}

.p1 { left: 8%; top: 20%; animation-delay: 0s; }
.p2 { left: 18%; top: 65%; animation-delay: 2s; }
.p3 { left: 33%; top: 35%; animation-delay: 4s; }
.p4 { left: 52%; top: 75%; animation-delay: 1s; }
.p5 { left: 68%; top: 28%; animation-delay: 3s; }
.p6 { left: 82%; top: 58%; animation-delay: 5s; }
.p7 { left: 91%; top: 18%; animation-delay: 2.5s; }

@keyframes floatParticle {
    0% {
        transform: translate(0px, 0px) scale(0.8);
        opacity: 0.25;
    }
    50% {
        transform: translate(20px, -35px) scale(1.2);
        opacity: 0.9;
    }
    100% {
        transform: translate(-15px, 25px) scale(0.9);
        opacity: 0.4;
    }
}

/* Faint glowing vine curves */
.vine-left,
.vine-right {
    position: fixed;
    width: 220px;
    height: 900px;
    border-left: 2px solid rgba(74,222,128,0.18);
    border-radius: 50%;
    filter: drop-shadow(0 0 8px rgba(74,222,128,0.2));
    z-index: 1;
    pointer-events: none;
}

.vine-left {
    left: -120px;
    top: -100px;
    transform: rotate(-8deg);
    animation: vineSwayLeft 10s ease-in-out infinite alternate;
}

.vine-right {
    right: -120px;
    bottom: -150px;
    transform: rotate(170deg);
    animation: vineSwayRight 12s ease-in-out infinite alternate;
}

@keyframes vineSwayLeft {
    from { transform: rotate(-8deg) translateY(0px); }
    to   { transform: rotate(-3deg) translateY(25px); }
}

@keyframes vineSwayRight {
    from { transform: rotate(170deg) translateY(0px); }
    to   { transform: rotate(175deg) translateY(-25px); }
}

/* Prediction card glow */
div[data-testid="stMetric"] {
    background: rgba(8, 30, 19, 0.72);
    border: 1px solid rgba(74,222,128,0.28);
    border-radius: 18px;
    padding: 18px;
    box-shadow:
        0 0 18px rgba(34,197,94,0.10),
        inset 0 0 18px rgba(34,197,94,0.04);
}

/* Slightly softer controls */
div[data-baseweb="select"] > div {
    border-radius: 12px;
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(74,222,128,0.35);
    box-shadow: 0 0 12px rgba(34,197,94,0.08);
}

</style>

<div class="particle p1"></div>
<div class="particle p2"></div>
<div class="particle p3"></div>
<div class="particle p4"></div>
<div class="particle p5"></div>
<div class="particle p6"></div>
<div class="particle p7"></div>

<div class="vine-left"></div>
<div class="vine-right"></div>
""", unsafe_allow_html=True)

st.title("Forest Microclimate Predictor")

st.write(
    "Adjust forest conditions below and see how much temperature buffering the model predicts."
)
model = joblib.load("forest_model.pkl")

canopy = st.slider(
    "Canopy Cover (%)",
    min_value=0,
    max_value=100,
    value=50
)

sensor_height = st.selectbox(
    "Sensor Height",
    ["Low", "High"]
)

elevation = st.selectbox(
    "Elevation",
    ["Low", "High"]
)

month_name = st.selectbox(
    "Month",
    ["June", "July", "August", "September", "October"]
)

month_map = {
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10
}
month = month_map[month_name]

if sensor_height == "High":
    sensor_height_encoded = 0
else:
    sensor_height_encoded = 1

if elevation == "High":
    elevation_encoded = 0
else:
    elevation_encoded = 1

input_data = pd.DataFrame({
    "Canopy": [canopy],
    "SensorHeightEncoded": [sensor_height_encoded],
    "ElevationEncoded": [elevation_encoded],
    "Month": [month]
})

if st.button("Predict Temperature Buffering"):
    prediction = model.predict(input_data)
    st.success(
        f"Predicted Temperature Buffering: {prediction[0]:.2f} Celsius"
    )

st.write(
    "Explore how forest conditions may influence temperature buffering. "
    "The model estimates the differences between open-area maximum temperature "
    "and the maximum temperature measured under forest conditions."
)
st.caption(
    "Note: This model is an experimental tool based on the study datset. "
    "Testing showed lower performance on completely unseen sensors, so predictions "
    "should not be treated as accurate forecasts for every forests."
)