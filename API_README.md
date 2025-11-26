# DQN Product Recommendation API

API để gợi ý sản phẩm sử dụng mô hình DQN đã huấn luyện.

## 🚀 Cài đặt

```bash
pip install -r requirements_api.txt
```

## 📦 Chạy API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: `http://localhost:8000`

## 📌 Endpoints

### 1. POST `/recommend` - Gợi ý sản phẩm (Epsilon-Greedy ε=0.5)

**Chiến lược gợi ý:**

- **50% Exploit**: Dựa trên model DQN đã học (Q-values cao nhất)
- **50% Explore**: Random recommendations (khám phá sở thích mới)

**Request:**

```json
{
  "user_state": {
    "gender": "Male",
    "age_group": "U40",
    "position": "cart",
    "day_of_week": 7,
    "num_products": 4,
    "total_value": 850000,
    "avg_value": 212500,
    "cart_item_ids": "[12, 13, 27, 36]",
    "order_ids": "[5, 15, 28]",
    "total_recent_purchases": 450000,
    "category": "['Business', 'Entertainment', 'Music']"
  },
  "top_k": 5
}
```

**Response:**

```json
{
  "recommended": [15, 23, 8, 42, 31],
  "confidence_scores": [0.35, 0.25, 0.18, 0.12, 0.1]
}
```

**Lưu ý:**

- `recommended`: Danh sách group IDs được gợi ý (có thể là model-based hoặc random)
- `confidence_scores`: Điểm tin cậy dựa trên Q-values (softmax)

### 2. POST `/feedback` - Gửi phản hồi (RL format: s, a, r, s', done)

**Request:**

```json
{
  "state": {
    "gender": "Male",
    "age_group": "U40",
    "position": "cart",
    "day_of_week": 7,
    "num_products": 4,
    "total_value": 850000,
    "avg_value": 212500,
    "cart_item_ids": "[12, 13, 27, 36]",
    "order_ids": "[5, 15, 28]",
    "total_recent_purchases": 450000,
    "category": "['Business', 'Entertainment', 'Music']"
  },
  "action": 15,
  "event_type": "transaction",
  "reward": 100.0,
  "next_state": {
    "gender": "Male",
    "age_group": "U40",
    "position": "home",
    "day_of_week": 7,
    "num_products": 0,
    "total_value": 0,
    "avg_value": 0,
    "cart_item_ids": "",
    "order_ids": "[5, 15, 28, 36]",
    "total_recent_purchases": 1300000,
    "category": ""
  },
  "done": true
}
```

**Tham số:**

- `state`: Trạng thái hiện tại (bắt buộc)
- `action`: Group ID user tương tác (bắt buộc)
- `event_type`: "view" / "addtocart" / "transaction" (bắt buộc)
- `reward`: Reward tùy chỉnh (optional - tự động tính từ event_type)
  - `view` → 5.0
  - `addtocart` → 30.0
  - `transaction` → 100.0
- `next_state`: Trạng thái tiếp theo (optional - mặc định = state)
- `done`: Episode kết thúc? (optional - mặc định = true)

**Auto-retrain:**

- **Transaction**: Train ngay lập tức (cần ≥32 samples)
- **View/Addtocart**: Train mỗi 50 samples

**Response:**

```json
{
  "status": "success",
  "message": "Feedback received",
  "reward": 100.0,
  "buffer_size": 85,
  "retrain": {
    "status": "success",
    "samples_used": 85,
    "epochs": 5,
    "avg_loss": 0.0234,
    "trigger": "transaction"
  }
}
```

### 3. POST `/retrain` - Trigger retrain thủ công

**Request:** Không cần body

**Response:**

```json
{
  "status": "success",
  "samples_used": 150,
  "epochs": 5,
  "avg_loss": 0.0189
}
```

hoặc

```json
{
  "status": "skipped",
  "reason": "Not enough samples (need 32+)"
}
```

### 4. GET `/health` - Kiểm tra trạng thái

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "num_actions": 50,
  "feedback_buffer_size": 45
}
```

### 5. GET `/stats` - Thống kê

**Response:**

```json
{
  "model_info": {
    "state_size": 18,
    "num_actions": 50,
    "total_groups": 50
  },
  "feedback_buffer": {
    "size": 45,
    "max_size": 1000,
    "accuracy": 0.68,
    "avg_reward": 35.2,
    "total_correct": 31
  }
}
```

## 📊 Giải thích các trường

### User State (Format giống CSV gốc)

- `gender`: "Male", "Female", "Other"
- `age_group`: "U20", "U30", "U40", "U50", "U60"
- `position`: "cart" (giỏ hàng), "home" (trang chủ), "search" (tìm kiếm)
- `day_of_week`: 1-7 (1=Monday, 7=Sunday)
- `num_products`: Số sản phẩm (1-6)
- `total_value`: Tổng giá trị session (50,000 - 2,000,000)
- `avg_value`: Giá trị trung bình (total_value / num_products)
- `cart_item_ids`: Danh sách item IDs trong giỏ - string "[12, 13, 27]" (70% có, 30% rỗng)
- `order_ids`: Danh sách order IDs đã mua - string "[1, 5, 10]" (80% có, 20% rỗng)
- `total_recent_purchases`: Tổng tiền mua gần đây - 50,000-1,500,000 (70% tập trung 200k-600k)
- `category`: Danh sách categories - string "['Business', 'Music']" (80% có)

### Event Type (Feedback)

- `view`: Người dùng chỉ xem (reward=5.0)
- `addtocart`: Thêm vào giỏ (reward=30.0)
- `transaction`: Mua hàng (reward=100.0)

## 🔄 Online Learning

API tự động retrain model từ feedback buffer:

**Smart Retrain Logic:**

- 🔴 **Transaction**: Train ngay lập tức (quan trọng nhất!)
- 🟡 **View/Addtocart**: Gom batch 50 samples → Train 1 lần
- ⚙️ **Manual**: Gọi `/retrain` bất cứ lúc nào

**Training Config:**

- Batch size: 32
- Epochs: 5
- Learning rate: 0.0001
- Gamma (γ): 0.99
- Target Q: `reward + (1 - done) * γ * max(Q(next_state))`

**Mở rộng:**

1. A/B testing giữa model cũ và mới
2. Personalization cho từng user segment
3. Lưu feedback buffer vào database để persistent learning

## ⚠️ Lưu ý

- Model file cần tồn tại: `dqn_product_recommendation.pth`
- API chạy trên CPU, có thể chuyển sang GPU nếu cần
- Feedback buffer reset khi restart API
- Nên setup logging và monitoring cho production
