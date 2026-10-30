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

def find_data_file():
    """Tim file CSV o thu muc app hoac thu muc data/ (cac notebook cua nhom dung ca hai cach)."""
    candidates = [
        "eda_completed_master_dataset.csv",
        "data/eda_completed_master_dataset.csv",
        "data/eda_dataset_with_cluster_labels.csv",
    ]
    for relative_path in candidates:
        if (BASE_DIR / relative_path).exists():
            return BASE_DIR / relative_path
    raise FileNotFoundError("Không thấy file CSV trong: " + ", ".join(candidates))


# Sai số RMSE của mô hình Regression (Task 3) trên tập test, lấy từ notebook 02_Regression.
REGRESSION_RMSE = 48444


@st.cache_resource
def load_models():
    regression_model = joblib.load(BASE_DIR / "model" / "regression_model.pkl")
    classification_info = joblib.load(BASE_DIR / "model" / "classification_model.pkl")
    pca_model = joblib.load(BASE_DIR / "model" / "pca.pkl")
    cluster_scaler = joblib.load(BASE_DIR / "model" / "scaler_cluster.pkl")
    kmeans_model = joblib.load(BASE_DIR / "model" / "kmeans_pca.pkl")
    return regression_model, classification_info, pca_model, cluster_scaler, kmeans_model
try:
    df = pd.read_csv(find_data_file())
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
# 3b. Hồ sơ và ý nghĩa của từng cụm khách hàng
# --------------------------------------------------
CLUSTER_COLUMNS = [
    "Age", "Order_Count", "Total_Spending",
    "Avg_Order_Value", "Avg_Quantity", "Avg_Rating"
]


def build_cluster_profile():
    """Tính đặc điểm trung bình của từng cụm và đặt tên theo mức chi tiêu."""
    customer_table = df.groupby("CustomerID").agg(
        Age=("Age", "mean"),
        Order_Count=("OrderID", "count"),
        Total_Spending=("TotalAmount", "sum"),
        Avg_Order_Value=("TotalAmount", "mean"),
        Avg_Quantity=("Quantity", "mean"),
        Avg_Rating=("Rating", "mean")
    ).reset_index()

    scaled = cluster_scaler.transform(customer_table[CLUSTER_COLUMNS])
    customer_table["Cluster"] = kmeans_model.predict(pca_model.transform(scaled))

    profile = customer_table.groupby("Cluster")[CLUSTER_COLUMNS].mean()
    profile["Num_Customers"] = customer_table["Cluster"].value_counts().sort_index()
    profile["Share"] = profile["Num_Customers"] / profile["Num_Customers"].sum()

    # Nhãn cụm do K-Means tạo ngẫu nhiên, nên đặt tên dựa trên tổng chi tiêu trung bình.
    ranked = profile["Total_Spending"].sort_values().index.tolist()
    names = {}
    for position, cluster_id in enumerate(ranked):
        if position == 0:
            names[cluster_id] = "Khách hàng phổ thông (chi tiêu thấp)"
        elif position == len(ranked) - 1:
            names[cluster_id] = "Khách hàng giá trị cao (chi tiêu cao)"
        else:
            names[cluster_id] = "Khách hàng chi tiêu trung bình"
    profile["Name"] = pd.Series(names)
    return profile


cluster_profile = build_cluster_profile()


# --------------------------------------------------
# 4. Tiêu đề ứng dụng
# --------------------------------------------------
st.title("💊 Hệ thống dự đoán đơn hàng Pharmacity")
st.write(
    "Ứng dụng minh họa các mô hình đã xây dựng ở Task 1–4: "
    "ước tính giá sản phẩm, phân loại đơn hàng giá trị cao và phân cụm khách hàng."
)
# Các mục chính của ứng dụng
tab_order, tab_cluster, tab_chart, tab_info = st.tabs(
    ["Dự đoán đơn hàng", "Phân cụm khách hàng", "Biểu đồ PCA + K-Means", "Thông tin mô hình"]
)


# --------------------------------------------------
# TAB 1: Dự đoán giá sản phẩm (Task 3) và phân loại đơn hàng (Task 4)
# --------------------------------------------------
with tab_order:
    st.subheader("Nhập thông tin đơn hàng")
    st.write("Điền các thông tin bên dưới, sau đó nhấn **Dự đoán**.")

    gender_options = sorted(df["Gender"].dropna().astype(str).unique().tolist())
    city_options = sorted(df["City"].dropna().astype(str).unique().tolist())
    payment_options = sorted(df["PaymentMethod"].dropna().astype(str).unique().tolist())
    category_options = sorted(df["Category"].dropna().astype(str).unique().tolist())
    brand_options = sorted(df["Brand"].dropna().astype(str).unique().tolist())

    with st.form("order_prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.slider(
                "Tuổi khách hàng",
                min_value=int(df["Age"].min()),
                max_value=int(df["Age"].max()),
                value=30,
                help="Dùng cho phân loại đơn hàng (Task 4)."
            )
            quantity = st.slider(
                "Số lượng sản phẩm",
                min_value=int(df["Quantity"].min()),
                max_value=int(df["Quantity"].max()),
                value=2,
                help="Dùng cho phân loại đơn hàng (Task 4)."
            )
            rating = st.slider(
                "Điểm đánh giá sản phẩm",
                min_value=4.50,
                max_value=5.00,
                value=4.75,
                step=0.01,
                help="Dùng cho cả dự đoán giá (Task 3) và phân loại đơn hàng (Task 4)."
            )
            gender = st.selectbox("Giới tính", gender_options)

        with right:
            category = st.selectbox(
                "Nhóm sản phẩm", category_options,
                help="Dùng cho cả dự đoán giá (Task 3) và phân loại đơn hàng (Task 4)."
            )
            brand = st.selectbox(
                "Thương hiệu", brand_options,
                help="Dùng cho dự đoán giá sản phẩm (Task 3)."
            )
            city = st.selectbox("Thành phố", city_options)
            payment_method = st.selectbox("Phương thức thanh toán", payment_options)

        submitted = st.form_submit_button("Dự đoán", type="primary")

    if submitted:
        # Task 3: dự đoán giá sản phẩm (Price) từ Category, Brand, Rating.
        # Pipeline đã lưu tự one-hot Category và Brand, nên chỉ cần đưa đúng 3 cột này.
        regression_input = pd.DataFrame([{
            "Category": category,
            "Brand": brand,
            "Rating": rating,
        }])
        predicted_price = float(regression_model.predict(regression_input)[0])
        predicted_price = max(0, predicted_price)

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
            st.metric("Giá sản phẩm dự đoán", f"{predicted_price:,.0f} VND")
            st.caption(
                "Kết quả từ mô hình Regression của Task 3 (dựa trên nhóm sản phẩm, thương hiệu "
                f"và điểm đánh giá). Sai số trung bình (RMSE) khoảng {REGRESSION_RMSE:,.0f} VND "
                "nên chỉ dùng làm giá tham khảo."
            )

        with result_right:
            if predicted_class == 1:
                st.success("Phân loại: Đơn giá trị cao")
            else:
                st.info("Phân loại: Đơn thường")
            st.write(f"Xác suất dự đoán là đơn giá trị cao: **{high_value_probability:.1%}**")
            st.caption(f"Ngưỡng tạo nhãn trong Task 4: {threshold_vnd:,.0f} VND.")

        st.warning(
            "Đây là dự đoán của mô hình đã học từ dữ liệu. Kết quả có thể khác giá trị thực tế. "
            "Hai mô hình độc lập với nhau: mô hình hồi quy ước tính giá một sản phẩm, "
            "còn mô hình phân loại đánh giá cả đơn hàng dựa trên số lượng và nhóm sản phẩm "
            "(không dùng đơn giá làm đầu vào)."
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

        info = cluster_profile.loc[cluster_label]
        st.success(f"Khách hàng được xếp vào **Cụm {cluster_label} – {info['Name']}**.")
        st.write(
            f"Đặc điểm trung bình của cụm này: **{info['Order_Count']:.1f}** giao dịch, "
            f"tổng chi tiêu **{info['Total_Spending']:,.0f} VND**, "
            f"giá trị đơn trung bình **{info['Avg_Order_Value']:,.0f} VND**, "
            f"**{info['Avg_Quantity']:.1f}** sản phẩm mỗi đơn. "
            f"Cụm gồm **{int(info['Num_Customers'])}** khách hàng ({info['Share']:.0%} tổng số)."
        )

    st.markdown("#### So sánh các cụm")
    profile_view = cluster_profile.rename(columns={
        "Name": "Tên cụm",
        "Age": "Tuổi TB",
        "Order_Count": "Số giao dịch TB",
        "Total_Spending": "Tổng chi tiêu TB (VND)",
        "Avg_Order_Value": "Giá trị đơn TB (VND)",
        "Avg_Quantity": "Số lượng TB/đơn",
        "Avg_Rating": "Điểm đánh giá TB",
        "Num_Customers": "Số khách hàng",
    })[["Tên cụm", "Số khách hàng", "Tuổi TB", "Số giao dịch TB",
        "Tổng chi tiêu TB (VND)", "Giá trị đơn TB (VND)",
        "Số lượng TB/đơn", "Điểm đánh giá TB"]]
    profile_view.index.name = "Cụm"
    st.dataframe(
        profile_view.style.format({
            "Tuổi TB": "{:.1f}",
            "Số giao dịch TB": "{:.2f}",
            "Tổng chi tiêu TB (VND)": "{:,.0f}",
            "Giá trị đơn TB (VND)": "{:,.0f}",
            "Số lượng TB/đơn": "{:.2f}",
            "Điểm đánh giá TB": "{:.2f}",
        }),
        use_container_width=True
    )
    st.caption(
        "Nhãn Cụm 0/Cụm 1 do K-Means tạo ra. Tên cụm được đặt dựa trên tổng chi tiêu trung bình "
        "của các khách hàng trong cụm."
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

    fig, ax = plt.subplots(figsize=(6, 3.3))
    for cluster_name in sorted(chart_df["Cluster"].unique()):
        one_cluster = chart_df[chart_df["Cluster"] == cluster_name]
        ax.scatter(
            one_cluster["PC1"],
            one_cluster["PC2"],
            alpha=0.7,
            label=f"Cụm {cluster_name} – {cluster_profile.loc[int(cluster_name), 'Name']}"
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
        "**Task 3 – Regression:** dùng `Category`, `Brand`, `Rating` để dự đoán giá sản phẩm `Price` "
        "(Linear Regression, huấn luyện trên bảng 392 sản phẩm)."
    )
    st.markdown(
        "**Task 4 – Classification:** dùng `Age`, `Quantity`, `Rating`, `Gender`, `City`, "
        "`PaymentMethod`, `Category` để dự đoán đơn thường hay đơn giá trị cao."
    )

    st.caption(
        "Task 3 dự đoán giá của một sản phẩm, còn Task 4 phân loại giá trị của cả đơn hàng "
        "(`TotalAmount = Price × Quantity`). Vì vậy hai mô hình dùng bộ đặc trưng khác nhau "
        "và không đưa `Price` hay `TotalAmount` vào đầu vào để tránh rò rỉ dữ liệu."
    )

    st.subheader("Một vài thông tin về bộ dữ liệu")
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Số giao dịch", f"{len(df):,}")
    metric2.metric("Số khách hàng", f"{df['CustomerID'].nunique():,}")
    metric3.metric("Số nhóm sản phẩm", f"{df['Category'].nunique():,}")