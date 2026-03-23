import streamlit as st
import pandas as pd
import numpy as np
import os

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Tesla Dashboard", layout="wide")

# -------------------------------
# CSS (BLUE UI)
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3 {
    color: #00c6ff;
}
.card {
    background: #1e3c72;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
section[data-testid="stSidebar"] {
    background-color: #0b1f3a;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🚀 Tesla Stock Prediction Dashboard")

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, 'data', 'TSLA.csv')
model_path = os.path.join(BASE_DIR, 'outputs', 'models', 'lstm_model.h5')

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')   # FIX latest price issue
    return df

df = load_data()

# -------------------------------
# LOAD MODEL (FIXED)
# -------------------------------
@st.cache_resource
def load_my_model():
    return load_model(model_path, compile=False)   # 🔥 FIX

model = load_my_model()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Controls")

days = st.sidebar.selectbox(
    "Prediction Days",
    [1, 5, 10]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Model Info")
st.sidebar.write("Model: LSTM")
st.sidebar.write("Window Size: 60 days")
st.sidebar.write("Units: 50")
st.sidebar.write("Dropout: 0.2")
st.sidebar.write("Optimizer: Adam")

# -------------------------------
# PREPROCESSING
# -------------------------------
data = df[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -------------------------------
# PREDICTION
# -------------------------------
last_data = scaled_data[-60:]
temp_input = last_data.tolist()

predictions = []

for i in range(days):
    x_input = np.array(temp_input[-60:])
    x_input = x_input.reshape(1, 60, 1)

    pred = model.predict(x_input, verbose=0)
    temp_input.append(pred[0].tolist())
    predictions.append(pred[0][0])

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))

# -------------------------------
# METRICS
# -------------------------------
latest_price = df['Close'].iloc[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
        <h3>Latest Price</h3>
        <h2>${latest_price:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
        <h3>Next Day</h3>
        <h2>${predictions[0][0]:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
        <h3>{days}-Day Forecast</h3>
        <h2>${predictions[-1][0]:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# STOCK TREND
# -------------------------------
st.markdown("## 📉 Stock Price Trend")
df_chart = df.set_index('Date')
st.line_chart(df_chart['Close'])

# -------------------------------
# ACTUAL VS PREDICTED
# -------------------------------
st.markdown("## 📊 Actual vs Predicted")

actual = df['Close'].values[-len(predictions):]

compare_df = pd.DataFrame({
    "Actual": actual,
    "Predicted": predictions.flatten()
})

st.line_chart(compare_df)

# -------------------------------
# PREDICTIONS TABLE
# -------------------------------
st.markdown("## 📌 Future Predictions")

pred_df = pd.DataFrame({
    "Day": [f"Day {i+1}" for i in range(days)],
    "Predicted Price": predictions.flatten()
})

st.dataframe(pred_df, use_container_width=True)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.markdown("## 📂 Dataset Preview")
st.dataframe(df.tail(), use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.markdown("## 💼 Insights")

st.write("""
- LSTM performs better than RNN due to long-term dependency handling  
- Useful for stock forecasting and investment decisions  
- Model captures trends but cannot predict sudden market fluctuations  
""")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🚀 Built using Deep Learning (LSTM) | Streamlit Dashboard")