"""
Step 2 - Dynamic Data Generation (Part A: Traditional Python Libraries)
Bai tap: Phan tich du lieu va A.I - Pharmacity Cham soc ca nhan

Muc tieu:
    Du lieu crawl (pharmacity_personal_care.csv) chi la du lieu SAN PHAM (tinh, static).
    Buoc nay mo phong du lieu GIAO DICH / KHACH HANG (dong, dynamic) de co mot bo du lieu
    ban hang day du hon cho cac buoc phan tich phia sau (VD: outlier detection tren
    Quantity / Total Spent, RFM, ...).

Cong cu su dung dung theo yeu cau de bai:
    - numpy.random           -> sinh Quantity theo phan phoi xac suat (Poisson) + outlier co chu dich
    - random                 -> chon ngau nhien San pham, Phuong thuc thanh toan, Trang thai don hang
    - Faker                  -> sinh thong tin ca nhan khach hang (ho ten, email, sdt, ngay sinh)
    - pandas.date_range      -> sinh chuoi thoi gian (khung ngay ban hang) de lay mau OrderDate

Dau ra:
    - pharmacity_customers_simulated.csv   (bang khach hang gia lap)
    - pharmacity_orders_simulated.csv      (bang don hang gia lap, da noi voi san pham + khach hang)
"""

import random
import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# 0. Cau hinh & tai du lieu san pham da crawl (Step 1)
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker("vi_VN")
Faker.seed(SEED)

products = pd.read_csv("pharmacity_personal_care.csv").drop(columns=["URL"])
print(f"Da nap {len(products)} san pham tu du lieu crawl.")

N_CUSTOMERS = 250
N_ORDERS = 1500

# Faker 'vi_VN' chua ho tro tot provider dia chi (city() tra ket qua loi),
# nen dung mot danh sach tinh/thanh pho thuc te de gan cho khach hang.
VN_CITIES = [
    "Ho Chi Minh", "Ha Noi", "Da Nang", "Hai Phong", "Can Tho",
    "Bien Hoa", "Nha Trang", "Hue", "Vung Tau", "Buon Ma Thuot",
    "Quy Nhon", "Thai Nguyen", "Nam Dinh", "Vinh", "Da Lat",
]

# ---------------------------------------------------------------------------
# 1. Sinh bang KHACH HANG (dung Faker cho thong tin ca nhan)
# ---------------------------------------------------------------------------
genders = np.random.choice(["Nam", "Nu"], size=N_CUSTOMERS, p=[0.45, 0.55])

customers = pd.DataFrame({
    "CustomerID": [f"CUS{i:05d}" for i in range(1, N_CUSTOMERS + 1)],
    "FullName": [
        fake.name_male() if g == "Nam" else fake.name_female() for g in genders
    ],
    "Gender": genders,
    "Email": [fake.unique.email() for _ in range(N_CUSTOMERS)],
    "PhoneNumber": [fake.phone_number() for _ in range(N_CUSTOMERS)],
    "City": np.random.choice(VN_CITIES, size=N_CUSTOMERS),
    "DateOfBirth": [
        fake.date_of_birth(minimum_age=16, maximum_age=65) for _ in range(N_CUSTOMERS)
    ],
    "MembershipLevel": np.random.choice(
        ["Thanh vien moi", "Silver", "Gold", "Diamond"],
        size=N_CUSTOMERS, p=[0.4, 0.3, 0.2, 0.1],
    ),
})

# ---------------------------------------------------------------------------
# 2. Sinh khung thoi gian ban hang (dung pandas.date_range)
# ---------------------------------------------------------------------------
date_pool = pd.date_range(start="2025-01-01", end="2026-09-27", freq="D")

# Trong ngay cung random gio phut giay de OrderDate co ca timestamp
order_dates = pd.to_datetime(
    np.random.choice(date_pool, size=N_ORDERS)
) + pd.to_timedelta(np.random.randint(0, 24 * 3600, size=N_ORDERS), unit="s")

# ---------------------------------------------------------------------------
# 3. Sinh bang DON HANG
#    Quantity: mo phong theo phan phoi Poisson (numpy.random) + tiem outlier
#    de phuc vu buoc phat hien outlier (IQR) o cau hoi sau.
# ---------------------------------------------------------------------------
chosen_products = products.sample(n=N_ORDERS, replace=True, random_state=SEED).reset_index(drop=True)

quantity = np.random.poisson(lam=2, size=N_ORDERS) + 1          # phan phoi thuc te: da so mua 1-4 san pham
outlier_mask = np.random.rand(N_ORDERS) < 0.015                  # ~1.5% don hang la outlier (mua si)
quantity[outlier_mask] = np.random.randint(20, 50, size=outlier_mask.sum())
quantity = np.clip(quantity, 1, None)

payment_methods = random.choices(
    ["Tien mat", "Momo", "ZaloPay", "The ngan hang", "VNPay"],
    weights=[0.30, 0.25, 0.20, 0.15, 0.10],
    k=N_ORDERS,
)
order_status = random.choices(
    ["Hoan thanh", "Da huy", "Doi tra"],
    weights=[0.88, 0.08, 0.04],
    k=N_ORDERS,
)

orders = pd.DataFrame({
    "OrderID": [f"OD{i:06d}" for i in range(1, N_ORDERS + 1)],
    "CustomerID": np.random.choice(customers["CustomerID"], size=N_ORDERS),
    "OrderDate": order_dates,
    "ProductID": chosen_products["ProductID"].values,
    "ProductName": chosen_products["ProductName"].values,
    "Category": chosen_products["Category"].values,
    "Brand": chosen_products["Brand"].values,
    "UnitPrice": chosen_products["Price"].values,
    "Quantity": quantity,
    "PaymentMethod": payment_methods,
    "OrderStatus": order_status,
})
orders["TotalSpent"] = orders["UnitPrice"] * orders["Quantity"]
orders = orders.sort_values("OrderDate").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 4. Xuat file ket qua
# ---------------------------------------------------------------------------
customers.to_csv("pharmacity_customers_simulated.csv", index=False, encoding="utf-8-sig")
orders.to_csv("pharmacity_orders_simulated.csv", index=False, encoding="utf-8-sig")

print(f"\nDa sinh {len(customers)} khach hang -> pharmacity_customers_simulated.csv")
print(f"Da sinh {len(orders)} don hang   -> pharmacity_orders_simulated.csv")
print(f"So don hang outlier (mua si, Quantity 20-49): {outlier_mask.sum()}")
print("\n5 dong dau cua bang don hang:")
print(orders.head())
