import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction")

@st.cache_data
def load_data():
    return pd.read_csv("house_prices_sample.csv")

df = load_data()

@st.cache_resource
def train_model():
    X = df[['SquareFeet', 'NumBedrooms', 'NumBathrooms', 'YearBuilt', 'NeighborhoodQuality']]
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    return model, scaler

model, scaler = train_model()

# ส่วนการทำนาย
st.subheader(" ทำนายราคาบ้านใหม่")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📝 กรอกข้อมูลบ้าน")
    input_sqft = st.slider("ขนาดพื้นที่ (Square Feet)", 1000, 4000, 2000)
    input_bedrooms = st.slider("จำนวนห้องนอน", 1, 6, 3)
    input_bathrooms = st.slider("จำนวนห้องน้ำ", 1.0, 5.0, 2.5, 0.5)
    input_year = st.slider("ปีที่สร้าง", 1990, 2022, 2010)
    input_quality = st.slider("คุณภาพย่าน (1-5)", 1, 5, 3)

with col2:
    st.markdown("### 💰 ผลการทำนาย")
    
    input_data = np.array([[input_sqft, input_bedrooms, input_bathrooms, input_year, input_quality]])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)[0]
    
    st.metric("ราคาที่ทำนายได้", f"${predicted_price:,.0f}")
    
    lower_bound = predicted_price * 0.9
    upper_bound = predicted_price * 1.1
    st.info(f"ช่วงราคาประมาณ: ${lower_bound:,.0f} - ${upper_bound:,.0f}")