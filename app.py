import streamlit as st
import pandas as pd

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="House Price Prediction", layout="wide")
st.title("🏠 House Price Prediction App")

# โหลดข้อมูล
@st.cache_data
def load_data():
    return pd.read_csv("house_prices_sample.csv")

try:
    df = load_data()
    st.success("✅ โหลดข้อมูลสำเร็จ!")
    
    # แสดงข้อมูล
    st.subheader("📊 ข้อมูลบ้านตัวอย่าง")
    st.dataframe(df, use_container_width=True)
    
    # สรุปข้อมูล
    st.subheader(" สรุปข้อมูล")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("จำนวนตัวอย่าง", len(df))
    with col2:
        st.metric("ราคาเฉลี่ย", f"${df['Price'].mean():,.0f}")
    with col3:
        st.metric("พื้นที่เฉลี่ย", f"{df['SquareFeet'].mean():,.0f} sqft")
    
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {e}")