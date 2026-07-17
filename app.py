import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="House Price Prediction ML", layout="wide")
st.title("🏠 House Price Prediction with Machine Learning")

# โหลดข้อมูล
@st.cache_data
def load_data():
    return pd.read_csv("house_prices_sample.csv")

df = load_data()

# ฝึกโมเดล
@st.cache_resource
def train_model():
    X = df[['SquareFeet', 'NumBedrooms', 'NumBathrooms', 'YearBuilt', 'NeighborhoodQuality']]
    y = df['Price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return model, r2, mae, X.columns

model, r2, mae, feature_names = train_model()

# แสดงผลการฝึกโมเดล
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("R² Score", f"{r2:.3f}")
with col2:
    st.metric("Mean Absolute Error", f"${mae:,.0f}")
with col3:
    st.metric("จำนวนข้อมูลฝึก", len(df))

# แสดง Feature Importance
st.subheader("📊 ความสำคัญของแต่ละฟีเจอร์")
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})
fig = px.bar(importance_df, x='Feature', y='Coefficient', 
             title='Impact of Each Feature on Price')
st.plotly_chart(fig, use_container_width=True)

# ส่วนทำนายราคา
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
    
    # ทำนาย
    input_data = np.array([[input_sqft, input_bedrooms, input_bathrooms, input_year, input_quality]])
    predicted_price = model.predict(input_data)[0]
    
    st.metric("ราคาที่ทำนายได้", f"${predicted_price:,.0f}")
    
    # แสดงช่วงราคา
    lower_bound = predicted_price * 0.9
    upper_bound = predicted_price * 1.1
    st.info(f"ช่วงราคาประมาณ: ${lower_bound:,.0f} - ${upper_bound:,.0f}")

# แสดงข้อมูลเปรียบเทียบ
st.subheader("📈 เปรียบเทียบกับข้อมูลที่มี")
comparison_df = df.copy()
comparison_df['Predicted'] = model.predict(df[['SquareFeet', 'NumBedrooms', 'NumBathrooms', 'YearBuilt', 'NeighborhoodQuality']])

fig_compare = px.scatter(comparison_df, x='Price', y='Predicted', 
                         hover_data=['SquareFeet', 'NumBedrooms'],
                         title='Actual vs Predicted Prices')
fig_compare.add_traces(px.line(x=[200000, 600000], y=[200000, 600000]).data)
st.plotly_chart(fig_compare, use_container_width=True)

# แสดงข้อมูลทั้งหมด
with st.expander("📋 ดูข้อมูลทั้งหมด"):
    st.dataframe(df, use_container_width=True)