# ============================================
# Streamlit Web Application for Loan Prediction
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Loan Status Prediction - SVM Model",
    page_icon="🏦",
    layout="wide"
)

# Load Model and Preprocessing Objects
@st.cache_resource
def load_model():
    model = joblib.load('svm_loan_model.pkl')
    scaler = joblib.load('scaler.pkl')
    education_encoder = joblib.load('education_encoder.pkl')
    home_encoder = joblib.load('home_encoder.pkl')
    intent_encoder = joblib.load('intent_encoder.pkl')
    return model, scaler, education_encoder, home_encoder, intent_encoder

model, scaler, education_encoder, home_encoder, intent_encoder = load_model()

# ============================================
# Header
# ============================================
st.title("🏦 Loan Status Prediction System")
st.markdown("### ทำนายสถานะการกู้ยืมด้วย Support Vector Machine (SVM)")
st.markdown("---")

# ============================================
# Sidebar - Input Form
# ============================================
st.sidebar.header("📝 กรอกข้อมูลผู้กู้ยืม")

with st.sidebar.form("loan_form"):
    st.subheader("ข้อมูลส่วนตัว")
    
    # Person Info
    person_age = st.number_input("อายุ (ปี)", min_value=18, max_value=100, value=25, step=1)
    person_gender = st.selectbox("เพศ", ["Male", "Female"])
    person_education = st.selectbox(
        "ระดับการศึกษา", 
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"]
    )
    person_income = st.number_input("รายได้ต่อปี ($)", min_value=0, value=50000, step=1000)
    person_emp_exp = st.number_input("ประสบการณ์ทำงาน (ปี)", min_value=0, value=3, step=1)
    person_home_ownership = st.selectbox(
        "สถานะที่อยู่อาศัย", 
        ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    
    st.markdown("---")
    st.subheader("ข้อมูลการกู้ยืม")
    
    loan_amnt = st.number_input("จำนวนเงินกู้ ($)", min_value=500, value=10000, step=500)
    loan_intent = st.selectbox(
        "วัตถุประสงค์การกู้", 
        ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", 
         "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"]
    )
    loan_int_rate = st.slider("อัตราดอกเบี้ย (%)", min_value=5.0, max_value=25.0, value=12.0, step=0.1)
    loan_percent_income = st.slider("สัดส่วนเงินกู้ต่อรายได้", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    
    st.markdown("---")
    st.subheader("ประวัติเครดิต")
    
    cb_person_cred_hist_length = st.number_input("ประวัติเครดิต (ปี)", min_value=0, value=5, step=1)
    credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=10)
    previous_loan_defaults_on_file = st.selectbox("เคยผิดนัดชำระหนี้หรือไม่", ["No", "Yes"])
    
    submitted = st.form_submit_button("🔮 ทำนายผลลัพธ์", use_container_width=True)

# ============================================
# Main Content
# ============================================
if submitted:
    # Prepare input data
    input_data = {
        'person_age': [person_age],
        'person_gender': [0 if person_gender == "Male" else 1],
        'person_education': [education_encoder.transform([person_education])[0]],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [home_encoder.transform([person_home_ownership])[0]],
        'loan_amnt': [loan_amnt],
        'loan_intent': [intent_encoder.transform([loan_intent])[0]],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [1 if previous_loan_defaults_on_file == "Yes" else 0]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # Display Results
    st.markdown("---")
    st.header("📊 ผลการทำนาย")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 0:
            st.success("✅ ผ่านการอนุมัติ")
            st.metric("สถานะ", "ไม่ผิดนัดชำระ", delta="Low Risk")
        else:
            st.error("❌ ไม่ผ่านการอนุมัติ")
            st.metric("สถานะ", "มีความเสี่ยงผิดนัด", delta="High Risk")
    
    with col2:
        st.metric("ความมั่นใจ (ไม่ผิดนัด)", f"{probability[0]*100:.2f}%")
    
    with col3:
        st.metric("ความมั่นใจ (ผิดนัด)", f"{probability[1]*100:.2f}%")
    
    # Probability Visualization
    st.markdown("---")
    st.subheader("📈 ความน่าจะเป็นของการทำนาย")
    
    fig = go.Figure(data=[
        go.Bar(
            name='Probability',
            x=['ไม่ผิดนัด (0)', 'ผิดนัด (1)'],
            y=[probability[0], probability[1]],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f'{probability[0]*100:.2f}%', f'{probability[1]*100:.2f}%'],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Input Summary
    st.markdown("---")
    st.subheader("📋 สรุปข้อมูลที่กรอก")
    
    summary_data = {
        'Feature': [
            'อายุ', 'เพศ', 'การศึกษา', 'รายได้ต่อปี', 'ประสบการณ์ทำงาน',
            'สถานะที่อยู่อาศัย', 'จำนวนเงินกู้', 'วัตถุประสงค์', 'อัตราดอกเบี้ย',
            'สัดส่วนเงินกู้ต่อรายได้', 'ประวัติเครดิต (ปี)', 'Credit Score',
            'เคยผิดนัดชำระ'
        ],
        'Value': [
            f"{person_age} ปี",
            person_gender,
            person_education,
            f"${person_income:,.0f}",
            f"{person_emp_exp} ปี",
            person_home_ownership,
            f"${loan_amnt:,.0f}",
            loan_intent,
            f"{loan_int_rate}%",
            f"{loan_percent_income*100:.1f}%",
            f"{cb_person_cred_hist_length} ปี",
            credit_score,
            previous_loan_defaults_on_file
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    # Welcome message
    st.info("👈 กรุณากรอกข้อมูลในฟอร์มด้านซ้ายเพื่อทำนายสถานะการกู้ยืม")
    
    # Show some statistics
    st.markdown("---")
    st.subheader("📊 ข้อมูลโมเดล")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Algorithm", "SVM (RBF Kernel)")
    
    with col2:
        st.metric("Features", "13 ตัวแปร")
    
    with col3:
        st.metric("Model Type", "Classification")
    
    # Feature importance explanation
    st.markdown("---")
    st.subheader("🔍 ตัวแปรที่ใช้ในการทำนาย")
    
    features_info = pd.DataFrame({
        'ตัวแปร': [
            'person_age', 'person_gender', 'person_education', 'person_income',
            'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'previous_loan_defaults_on_file'
        ],
        'คำอธิบาย': [
            'อายุของผู้กู้ยืม',
            'เพศ (Male=0, Female=1)',
            'ระดับการศึกษา',
            'รายได้ต่อปี',
            'จำนวนปีประสบการณ์ทำงาน',
            'สถานะที่อยู่อาศัย (RENT/OWN/MORTGAGE/OTHER)',
            'จำนวนเงินที่ต้องการกู้',
            'วัตถุประสงค์การกู้ยืม',
            'อัตราดอกเบี้ย',
            'สัดส่วนเงินกู้ต่อรายได้',
            'จำนวนปีที่มีประวัติเครดิต',
            'คะแนนเครดิต (300-850)',
            'ประวัติการผิดนัดชำระหนี้ (Yes=1, No=0)'
        ]
    })
    
    st.dataframe(features_info, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏦 Loan Status Prediction System | Powered by SVM & Streamlit</p>
        <p>Developed with ❤️ using Python</p>
    </div>
    """, 
    unsafe_allow_html=True
)