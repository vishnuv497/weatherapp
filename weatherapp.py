import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

st.title("Weather Prediction App")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    
    # Check if the necessary columns are present
    if 'Date' in df.columns and 'Temperature' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Step 2: Data Visualization
        st.subheader("Temperature over Time")
        plt.figure(figsize=(10, 5))
        plt.plot(df['Date'], df['Temperature'], label='Temperature')
        plt.xlabel('Date')
        plt.ylabel('Temperature')
        plt.title('Temperature Trend')
        plt.legend()
        st.pyplot(plt)
        
        # Step 3: Train Test Split
        X = df[['DayOfYear']]
        y = df['Temperature']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Step 4: Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Model Evaluation
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        st.subheader("Model Performance")
        st.write(f"Root Mean Squared Error: {rmse:.2f}")
        
        # Step 5: Prediction
        st.subheader("Make a Prediction")
        d
