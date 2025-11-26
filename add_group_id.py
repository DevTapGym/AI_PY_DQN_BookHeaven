import pandas as pd
import numpy as np

# Đường dẫn file gốc
file_path = "events.csv"
output_path = "events_with_groups.csv"

# Load toàn bộ dữ liệu
df = pd.read_csv(file_path)

# Đếm số lần xuất hiện của mỗi itemid
item_counts = df["itemid"].value_counts().reset_index()
item_counts.columns = ["itemid","count"]

# Sắp xếp theo count giảm dần
item_counts = item_counts.sort_values("count", ascending=False).reset_index(drop=True)

# Tạo 50 nhóm từ 1 → 50
num_groups = 50
item_counts["group_id"] = (np.arange(len(item_counts)) * num_groups) // len(item_counts) + 1  # +1 để bắt đầu từ 1

# Tạo dict để map itemid -> group_id
item_to_group = dict(zip(item_counts["itemid"], item_counts["group_id"]))

# Thêm cột group_id vào dataframe gốc
df["group_id"] = df["itemid"].map(item_to_group)

# Lưu file mới
df.to_csv(output_path, index=False)
print(f"Hoàn tất, 50 nhóm đã được gán vào {output_path}")
print("Tổng số item:", len(item_counts))
print("Nhóm tối thiểu:", df["group_id"].min(), "Nhóm tối đa:", df["group_id"].max())




#235,061