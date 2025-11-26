import pandas as pd
import numpy as np
import random

print("Đang load file...")
# Load file events hiện tại (có group_id)
df = pd.read_csv("events_with_groups.csv")
n_rows = len(df)
print(f"Đã load {n_rows} dòng")

# Danh mục sản phẩm
categories_list = ["Business", "Entertainment", "Cooking", "History", "Music",
                   "Comics", "Travel", "Arts", "Sports", "Psychology"]

print("Đang tạo dữ liệu state (vectorized)...")

# Random các cột chung cho TẤT CẢ các dòng cùng lúc (vectorized)
df['gender'] = np.random.choice(["Male", "Female", "Other"], size=n_rows)
df['age_group'] = np.random.choice(["U20","U30","U40","U50","U60"], size=n_rows)
df['day_of_week'] = np.random.randint(1, 8, size=n_rows)
df['position'] = np.random.choice(["cart","search","home"], size=n_rows)

# Khởi tạo các cột với giá trị mặc định = 0 hoặc rỗng
print("Khởi tạo các cột...")
df['num_products'] = 0
df['total_value'] = 0
df['avg_value'] = 0
df['cart_item_ids'] = ''
df['order_ids'] = ''
df['total_recent_purchases'] = 0
df['category'] = ''

# Tạo mask cho từng position
cart_mask = df['position'] == 'cart'
search_mask = df['position'] == 'search'
home_mask = df['position'] == 'home'

print(f"  - Cart: {cart_mask.sum()} dòng")
print(f"  - Search: {search_mask.sum()} dòng")
print(f"  - Home: {home_mask.sum()} dòng")

# Hàm sinh total_recent_purchases với phân phối tập trung 200k-600k
def generate_purchase_amount():
    """70% trong khoảng 200k-600k, 30% còn lại phân bố 50k-1.5M"""
    if random.random() < 0.7:
        # 70% tập trung 200k-600k
        return random.randint(200000, 600000)
    else:
        # 30% phân bố rộng hơn
        return random.randint(50000, 1500000)

print("Sinh dữ liệu lịch sử mua hàng và categories...")
# Sinh lịch sử mua hàng cho TẤT CẢ các dòng (80% có lịch sử)
for idx in df.index:
    if random.random() < 0.8:  # 80% có lịch sử mua hàng
        # Sinh order_ids (1-5 đơn hàng, giao động mạnh ở 1-3)
        # 70% có 1-3 đơn, 30% có 4-5 đơn
        if random.random() < 0.7:
            num_orders = random.randint(1, 4)  # 1-3 đơn
        else:
            num_orders = random.randint(4, 6)  # 4-5 đơn
        df.at[idx, 'order_ids'] = str(random.sample(range(1, 101), num_orders))
        
        # Sinh total_recent_purchases theo phân phối
        df.at[idx, 'total_recent_purchases'] = generate_purchase_amount()
        
        # Sinh categories (2-4 danh mục)
        num_cats = random.randint(2, 5)  # 2-4 danh mục
        df.at[idx, 'category'] = str(random.sample(categories_list, num_cats))
    else:
        # 20% chưa mua hàng
        df.at[idx, 'order_ids'] = ''
        df.at[idx, 'total_recent_purchases'] = 0
        df.at[idx, 'category'] = ''

print("Xử lý cart_item_ids và giá trị cho TẤT CẢ positions...")
# Tất cả positions đều có thể có dữ liệu giỏ hàng (70% có, 30% không)
for idx in df.index:
    if random.random() < 0.7:  # 70% có sản phẩm trong giỏ
        # Sinh số lượng sản phẩm (1-6)
        num_prod = random.randint(1, 7)  # 1-6 sản phẩm
        
        # Sinh cart_item_ids
        df.at[idx, 'cart_item_ids'] = str(random.sample(range(1, 51), num_prod))
        
        # num_products dựa trên số lượng cart_item_ids
        df.at[idx, 'num_products'] = num_prod
        
        # Sinh total_value và avg_value tương ứng
        total_val = random.randint(50000, 2000001)  # 50k-2M
        df.at[idx, 'total_value'] = total_val
        df.at[idx, 'avg_value'] = total_val / num_prod
    # else: 30% không có sản phẩm → giữ nguyên giá trị 0 mặc định

print("Đang lưu file...")
# Lưu file mới
df.to_csv("events_with_states.csv", index=False)
print("Hoàn tất: events_with_states.csv đã có dữ liệu state thô random.")
