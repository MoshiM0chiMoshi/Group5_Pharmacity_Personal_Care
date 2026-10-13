from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Cài đặt giao diện và đường dẫn file
# --------------------------------------------------
st.set_page_config(
    page_title="Pharmacity - Dự đoán đơn hàng",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent
# --------------------------------------------------
# 2. Đọc dữ liệu và các mô hình đã lưu
# --------------------------------------------------

@st.cache_resource
def load_models():
    regression_model = joblib.load(BASE_DIR / "model" / "regression_model.pkl")
    classification_info = joblib.load(BASE_DIR / "model" / "classification_model.pkl")
    pca_model = joblib.load(BASE_DIR / "model" / "pca.pkl")
    cluster_scaler = joblib.load(BASE_DIR / "model" / "scaler.pkl")
    kmeans_model = joblib.load(BASE_DIR / "model" / "kmeans_pca.pkl")
    return regression_model, classification_info, pca_model, cluster_scaler, kmeans_model
try:
    df = pd.read_csv(BASE_DIR / "eda_completed_master_dataset.csv")
    regression_model, classification_info, pca_model, cluster_scaler, kmeans_model = load_models()
except FileNotFoundError as error:
    st.error("Không tìm thấy file dữ liệu hoặc mô hình. Hãy kiểm tra các thư mục data/ và model/.")
    st.code(str(error))
    st.stop()

# --------------------------------------------------
# 3. Chuẩn bị dữ liệu đầu vào cho mô hình phân loại
# --------------------------------------------------
def prepare_classification_data(age, quantity, rating, gender, city, payment_method, category):
    """Tao mot dong du lieu dung thu tu cot da luu trong classification_model.pkl."""
    # File .pkl cua nhom co khoa "columns", khong co khoa "category_values".
    columns = classification_info["columns"]

    # Tao truoc tat ca cot voi gia tri 0 de dung du so cot mo hinh da hoc.
    input_row = pd.DataFrame(0.0, index=[0], columns=columns)
    input_row.at[0, "Age"] = age
    input_row.at[0, "Quantity"] = quantity
    input_row.at[0, "Rating"] = rating

    selected_categories = {
        "Gender": gender,
        "City": city,
        "PaymentMethod": payment_method,
        "Category": category,
    }

    # Danh dau 1 cho cot One-Hot tuong ung neu cot do co trong mo hinh.
    # Cot khong co trong danh sach la nhom tham chieu do get_dummies(drop_first=True) tao ra.
    for column_name, selected_value in selected_categories.items():
        dummy_column = f"{column_name}_{selected_value}"
        if dummy_column in input_row.columns:
            input_row.at[0, dummy_column] = 1.0

    return input_row


# --------------------------------------------------
# 4. Tiêu đề ứng dụng
# --------------------------------------------------
st.title("💊 Hệ thống dự đoán đơn hàng Pharmacity")
st.write(
    "Ứng dụng minh họa các mô hình đã xây dựng ở Task 1–4: "
    "dự đoán giá trị đơn hàng, phân loại đơn hàng giá trị cao và phân cụm khách hàng."
)
# Các mục chính của ứng dụng
tab_order, tab_cluster, tab_chart, tab_info = st.tabs(
    ["Dự đoán đơn hàng", "Phân cụm khách hàng", "Biểu đồ PCA + K-Means", "Thông tin mô hình"]
)


# --------------------------------------------------
# TAB 1: Dự đoán tổng tiền và phân loại đơn hàng
# --------------------------------------------------
with tab_order:
    st.subheader("Nhập thông tin đơn hàng")
    st.write("Điền các thông tin bên dưới, sau đó nhấn **Dự đoán**.")

    gender_options = sorted(df["Gender"].dropna().astype(str).unique().tolist())
    city_options = sorted(df["City"].dropna().astype(str).unique().tolist())
    payment_options = sorted(df["PaymentMethod"].dropna().astype(str).unique().tolist())
    category_options = sorted(df["Category"].dropna().astype(str).unique().tolist())

    with st.form("order_prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.slider(
                "Tuổi khách hàng",
                min_value=int(df["Age"].min()),
                max_value=int(df["Age"].max()),
                value=30
            )
            unit_price = st.number_input(
                "Đơn giá sản phẩm (VND)",
                min_value=10000,
                max_value=400000,
                value=90000,
                step=1000
            )
            quantity = st.slider(
                "Số lượng sản phẩm",
                min_value=int(df["Quantity"].min()),
                max_value=int(df["Quantity"].max()),
                value=2
            )
            rating = st.slider(
                "Điểm đánh giá sản phẩm",
                min_value=4.50,
                max_value=5.00,
                value=4.75,
                step=0.01
            )

        with right:
            gender = st.selectbox("Giới tính", gender_options)
            city = st.selectbox("Thành phố", city_options)
            payment_method = st.selectbox("Phương thức thanh toán", payment_options)
            category = st.selectbox("Nhóm sản phẩm", category_options)

        submitted = st.form_submit_button("Dự đoán", type="primary")

    if submitted:
        # Task 3: dự đoán giá trị đơn hàng bằng mô hình hồi quy đã lưu.
        regression_input = pd.DataFrame([{
            "Age": age,
            "UnitPrice": unit_price,
            "Quantity": quantity,
            "Rating": rating,
        }])
        predicted_amount = float(regression_model.predict(regression_input)[0])
        predicted_amount = max(0, predicted_amount)

        # Task 4: dự đoán đơn thường hay đơn giá trị cao.
        classification_input = prepare_classification_data(
            age, quantity, rating, gender, city, payment_method, category
        )

        classification_model = classification_info["model"]
        if classification_info["use_scaler"]:
            classification_input_ready = classification_info["scaler"].transform(classification_input)
        else:
            classification_input_ready = classification_input

        predicted_class = int(classification_model.predict(classification_input_ready)[0])
        # Tim dung cot xac suat cua lop 1 (don gia tri cao).
        probabilities = classification_model.predict_proba(classification_input_ready)[0]
        high_value_index = list(classification_model.classes_).index(1)
        high_value_probability = float(probabilities[high_value_index])
        threshold_vnd = float(classification_info.get("threshold_vnd", 199000))

        st.divider()
        st.subheader("Kết quả dự đoán")
        result_left, result_right = st.columns(2)

        with result_left:
            st.metric("Tổng tiền dự đoán", f"{predicted_amount:,.0f} VND")
            st.caption("Kết quả từ mô hình Regression của Task 3.")

        with result_right:
            if predicted_class == 1:
                st.success("Phân loại: Đơn giá trị cao")
            else:
                st.info("Phân loại: Đơn thường")
            st.write(f"Xác suất dự đoán là đơn giá trị cao: **{high_value_probability:.1%}**")
            st.caption(f"Ngưỡng tạo nhãn trong Task 4: {threshold_vnd:,.0f} VND.")

        st.warning(
            "Đây là dự đoán của mô hình đã học từ dữ liệu. Kết quả có thể khác giá trị thực tế; "
            "đặc biệt, mô hình phân loại sử dụng thông tin số lượng và nhóm sản phẩm, "
            "không sử dụng đơn giá làm đầu vào."
        )


# --------------------------------------------------
# TAB 2: Dự đoán cụm khách hàng
# --------------------------------------------------
with tab_cluster:
    st.subheader("Phân cụm khách hàng bằng PCA + K-Means")
    st.write(
        "Nhập các thống kê của một khách hàng. Các biến đầu vào giống Task 1: "
        "tuổi, số giao dịch, tổng chi tiêu, giá trị đơn trung bình, số lượng trung bình và điểm đánh giá trung bình."
    )

    with st.form("customer_cluster_form"):
        left, right = st.columns(2)
        with left:
            customer_age = st.number_input("Tuổi trung bình", min_value=18.0, max_value=69.0, value=30.0, step=1.0)
            order_count = st.number_input("Số giao dịch", min_value=1, max_value=100, value=2, step=1)
            total_spending = st.number_input("Tổng chi tiêu (VND)", min_value=0, max_value=50000000, value=300000, step=10000)
        with right:
            average_order_value = st.number_input("Giá trị đơn trung bình (VND)", min_value=0, max_value=5000000, value=150000, step=10000)
            average_quantity = st.number_input("Số lượng trung bình mỗi đơn", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
            average_rating = st.number_input("Điểm đánh giá trung bình", min_value=4.5, max_value=5.0, value=4.75, step=0.01)

        cluster_submitted = st.form_submit_button("Tìm cụm khách hàng", type="primary")

    if cluster_submitted:
        customer_input = pd.DataFrame([{
            "Age": customer_age,
            "Order_Count": order_count,
            "Total_Spending": total_spending,
            "Avg_Order_Value": average_order_value,
            "Avg_Quantity": average_quantity,
            "Avg_Rating": average_rating,
        }])

        # Thực hiện đúng thứ tự: chuẩn hóa -> PCA -> K-Means.
        customer_scaled = cluster_scaler.transform(customer_input)
        customer_pca = pca_model.transform(customer_scaled)
        cluster_label = int(kmeans_model.predict(customer_pca)[0])

        st.success(f"Khách hàng được xếp vào **Cụm {cluster_label}**.")
        st.caption(
            "Nhãn Cụm 0/Cụm 1 do K-Means tạo ra; nhãn số không tự mang nghĩa tốt hoặc xấu. "
            "Nhóm có thể phân tích đặc điểm từng cụm để đặt tên theo hành vi khách hàng."
        )


# --------------------------------------------------
# TAB 3: Biểu đồ minh họa kết quả PCA + K-Means
# --------------------------------------------------
with tab_chart:
    st.subheader("Dữ liệu khách hàng trên hai thành phần PCA đầu tiên")
    st.write("Mỗi điểm tương ứng với một khách hàng; màu sắc biểu thị cụm K-Means được dự đoán.")

    customer_features = df.groupby("CustomerID").agg(
        Age=("Age", "mean"),
        Order_Count=("OrderID", "count"),
        Total_Spending=("TotalAmount", "sum"),
        Avg_Order_Value=("TotalAmount", "mean"),
        Avg_Quantity=("Quantity", "mean"),
        Avg_Rating=("Rating", "mean")
    ).reset_index()

    cluster_columns = [
        "Age", "Order_Count", "Total_Spending",
        "Avg_Order_Value", "Avg_Quantity", "Avg_Rating"
    ]
    chart_scaled = cluster_scaler.transform(customer_features[cluster_columns])
    chart_pca = pca_model.transform(chart_scaled)
    chart_labels = kmeans_model.predict(chart_pca)

    chart_df = pd.DataFrame({
        "PC1": chart_pca[:, 0],
        "PC2": chart_pca[:, 1],
        "Cluster": chart_labels.astype(str)
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    for cluster_name in sorted(chart_df["Cluster"].unique()):
        one_cluster = chart_df[chart_df["Cluster"] == cluster_name]
        ax.scatter(
            one_cluster["PC1"],
            one_cluster["PC2"],
            alpha=0.7,
            label=f"Cụm {cluster_name}"
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Customer Segmentation - PCA + K-Means")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"Số khách hàng trong dữ liệu: **{len(customer_features)}**")
    st.write(f"Số thành phần PCA được sử dụng: **{pca_model.n_components_}**")
    st.write(f"Phương sai được giữ lại bởi PCA: **{pca_model.explained_variance_ratio_.sum():.2%}**")


# --------------------------------------------------
# TAB 4: Thông tin mô hình
# --------------------------------------------------
with tab_info:
    st.subheader("Các mô hình sử dụng trong ứng dụng")

    st.markdown(
        "**Task 1 – PCA:** chuẩn hóa 6 đặc trưng khách hàng và giảm chiều dữ liệu. "
        "Mô hình trong bộ file này giữ lại 4 thành phần chính."
    )
    st.markdown(
        "**Task 2 – K-Means:** phân cụm khách hàng trên dữ liệu sau PCA. "
        "Ứng dụng hiển thị nhãn cụm mà mô hình dự đoán được."
    )
    st.markdown(
        "**Task 3 – Regression:** dùng `Age`, `UnitPrice`, `Quantity`, `Rating` để dự đoán `TotalAmount`."
    )
    st.markdown(
        "**Task 4 – Classification:** dùng `Age`, `Quantity`, `Rating`, `Gender`, `City`, "
        "`PaymentMethod`, `Category` để dự đoán đơn thường hay đơn giá trị cao."
    )

    st.caption(
        "Trong CSV của nhóm, cột `Price` và `Price_VND_ref` có cùng giá trị. "
        "Khi tạo đầu vào cho Regression, ứng dụng dùng `Price` với tên `UnitPrice` "
        "để khớp với đặc trưng trong notebook Task 3."
    )

    st.subheader("Một vài thông tin về bộ dữ liệu")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Số giao dịch", f"{len(df):,}")
    metric2.metric("Số khách hàng", f"{df['CustomerID'].nunique():,}")
    metric3.metric("Số nhóm sản phẩm", f"{df['Category'].nunique():,}")
