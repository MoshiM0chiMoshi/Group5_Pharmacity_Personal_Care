import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Hệ thống Pharmacity", layout="wide")

st.title("Hệ thống dự đoán tổng tiền và phân khúc Khách hàng")
st.markdown("---")

# 1. Tải mô hình hồi quy
@st.cache_resource
def load_model():
    reg_path = "model/regression_model.pkl" if os.path.exists("model/regression_model.pkl") else "regression_model.pkl"
    return joblib.load(reg_path) if os.path.exists(reg_path) else None

reg_model = load_model()
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.header("Nhập thông tin đầu vào")
    
    age = st.slider("Độ tuổi (Age)", 18, 80, 30)
    unit_price = st.number_input("Đơn giá sản phẩm (UnitPrice - VNĐ)", min_value=10000, max_value=5000000, value=150000, step=10000)
    quantity = st.slider("Số lượng mua (Quantity)", 1, 10, 1)
    rating = st.slider("Đánh giá sản phẩm (Rating)", 1.0, 5.0, 4.5, 0.1)
    
    st.write("")
    predict_button = st.button("Thực hiện Dự đoán & Phân khúc", type="primary")

with col_result:
    st.header("Kết quả phân tích")
    
    if predict_button:
        if reg_model is not None:
            try:
                # Tạo bảng dữ liệu đầu vào với đúng 4 cột
                input_data = pd.DataFrame([[age, unit_price, quantity, rating]], 
                                          columns=['Age', 'UnitPrice', 'Quantity', 'Rating'])
                
                # Dự đoán tổng chi tiêu (Hồi quy)
                pred_amount = reg_model.predict(input_data)[0]
                
                # Phân loại phân khúc khách hàng tự động thông minh dựa trên kết quả chi tiêu và độ tuổi
                if pred_amount > 1000000:
                    segment = "Nhóm Khách hàng VIP / Chi tiêu cao"
                elif pred_amount > 500000:
                    segment = "Nhóm Khách hàng Tiềm năng / Chi tiêu trung bình"
                else:
                    segment = "Nhóm Khách hàng Phổ thông / Mua sắm tiết kiệm"
                
                st.success("Phân tích thành công!")
                
                # Hiển thị kết quả 1: Tổng chi tiêu
                st.metric(label="1. Tổng chi tiêu dự kiến (Total Amount)", value=f"{pred_amount:,.2f} VNĐ")
                
                st.write("")
                
                # Hiển thị kết quả 2: Phân khúc khách hàng
                st.info(f"**2. Phân khúc Khách hàng Dự đoán:**\n\n {segment}")
                
                # Tóm tắt thông số đã chọn
                st.write("")
                st.subheader("Tóm tắt thông tin đầu vào:")
                st.write(f"- Độ tuổi: {age}")
                st.write(f"- Đơn giá: {unit_price:,.0f} VNĐ")
                st.write(f"- Số lượng: {quantity}")
                st.write(f"- Đánh giá: {rating}")
                
            except Exception as e:
                st.error(f"Lỗi xử lý: {e}")
        else:
            st.warning("Không tìm thấy file mô hình Regression.")
    else:
        st.info("Vui lòng nhập thông tin bên cột trái và nhấn nút để xem kết quả dự đoán và phân khúc.")

st.markdown("---")