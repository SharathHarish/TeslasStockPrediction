# **Tesla Stock Price Prediction using Deep Learning (LSTM)**

---

## **1. Introduction**

Stock price prediction is an important problem in the field of finance and data science. It helps investors make informed decisions by analyzing historical trends and forecasting future prices. In this project, we aim to predict the future stock prices of Tesla using deep learning techniques, specifically Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models.

LSTM is particularly suitable for time-series data as it can capture long-term dependencies and patterns in sequential data.

---

## **2. Dataset Description**

The dataset used in this project contains historical stock price data of Tesla. The dataset includes the following columns:

* Date
* Open
* High
* Low
* Close
* Adj Close
* Volume

For this project, we focused on the **closing price (Adj Close)** as it reflects the adjusted value of the stock after accounting for splits and dividends.

---

## **3. Data Cleaning**

The following steps were performed to clean the data:

* Converted the `Date` column into datetime format
* Sorted the dataset based on date
* Set the `Date` column as index
* Handled missing values using forward fill (`ffill`)

These steps ensured that the dataset was consistent and ready for analysis.

---

## **4. Data Preprocessing**

To prepare the data for model training:

* Selected **Adj Close** as the target feature
* Applied **MinMaxScaler** to normalize data between 0 and 1
* Created sequences of **60 time steps** for time-series prediction
* Split the dataset into training (80%) and testing (20%) sets

This preprocessing step is essential for improving model performance.

---

## **5. Data Visualization**

A line plot was created to visualize Tesla’s stock price over time. This helped in identifying trends, fluctuations, and overall movement in stock prices.

---

## **6. Feature Engineering**

Feature engineering was performed by converting the time-series data into supervised learning format using a sliding window approach:

* Each input consists of the previous **60 days of prices**
* Output is the next day’s price

This enables the model to learn temporal dependencies.

---

## **7. Deep Learning Models**

Two models were implemented:

### **1. RNN Model**

* Uses SimpleRNN layers
* Captures short-term dependencies

### **2. LSTM Model**

* Uses LSTM layers
* Captures long-term dependencies
* Better performance compared to RNN

### **Model Configuration:**

* Units: 50 / 64
* Dropout: 0.2 (to prevent overfitting)
* Optimizer: Adam
* Loss Function: Mean Squared Error

---

## **8. Model Evaluation**

The performance of the model was evaluated using:

* **Mean Squared Error (MSE)**
* **Actual vs Predicted graph**

The LSTM model showed better performance due to its ability to retain long-term dependencies.

---

## **9. Results**

* The LSTM model successfully captured the trend of Tesla stock prices
* Predictions for future days (1, 5, 10) were generated
* The model demonstrated reasonable accuracy for trend prediction

---

## **10. Future Predictions**

The model was used to predict stock prices for:

* Next 1 day
* Next 5 days
* Next 10 days

These predictions were visualized in the dashboard.

---

## **11. Streamlit Dashboard**

A user-friendly dashboard was developed using Streamlit which includes:

* Stock price trend visualization
* Actual vs predicted comparison
* Future predictions
* Key metrics (latest price, forecast)
* Interactive controls

---

## **12. Limitations**

* Stock prices are influenced by external factors (news, economy, politics)
* Model relies only on historical data
* Cannot predict sudden market crashes

---

## **13. Conclusion**

In this project, we successfully built a deep learning model to predict Tesla stock prices using LSTM. The model demonstrated the ability to learn patterns from historical data and generate future predictions. The integration of a Streamlit dashboard made the project interactive and user-friendly.

---

## **14. Future Improvements**

* Use more features (volume, indicators)
* Apply advanced models (GRU, Transformer)
* Incorporate real-time data

---

## **15. Project Timeline**

| Phase | Description                         |
| ----- | ----------------------------------- |
| Day 1 | Data Cleaning & Visualization       |
| Day 2 | Preprocessing & Feature Engineering |
| Day 3 | Model Training & Evaluation         |
| Day 4 | Deployment using Streamlit          |

---

## **16. Tools & Technologies**

* Python
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* TensorFlow / Keras
* Streamlit

---
## **⚙️ 🚀 INSTALLATION & RUNNING INSTRUCTIONS**
🖥️ 1️⃣ CREATE PROJECT FOLDER
clone TeslaStockPrediction
cd TeslaStockPrediction
🧪 2️⃣ CREATE VIRTUAL ENVIRONMENT
✅ Windows:
python -m venv venv
▶️ 3️⃣ ACTIVATE VENV
✅ Windows (PowerShell):
venv\Scripts\activate

👉 You should see:

(venv)
📦 4️⃣ INSTALL REQUIREMENTS
pip install -r requirements.txt
📄 5️⃣ REQUIREMENTS.TXT 
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
scikit-learn==1.4.2
tensorflow==2.15.0
h5py==3.10.0


📓 7️⃣ RUN NOTEBOOK
jupyter notebook

👉 Then:

Open model.ipynb
Run all cells
Ensure model is saved
▶️ 8️⃣ RUN STREAMLIT APP
streamlit run app/app.py

## **17. Deployment**

The project was successfully deployed using Streamlit, allowing users to interact with the model and visualize predictions in real-time.
https://teslasstockprediction-jpkgzq78xzqx7cchzjnqoe.streamlit.app/

---

## **18. Technical Tags**

#finance #stockprediction #deeplearning #lstm

---
