# DQN Product Recommendation System

## 📁 Cấu trúc thư mục

```
DQN/
├── src/
│   ├── __init__.py
│   ├── config.py              # Cấu hình và hyperparameters
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py   # Xử lý dữ liệu
│   ├── models/
│   │   ├── __init__.py
│   │   └── dqn.py            # Kiến trúc DQN
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # Logic training
│   │   └── evaluator.py      # Đánh giá model
│   └── utils/
│       ├── __init__.py
│       ├── visualizer.py     # Vẽ đồ thị
│       └── helpers.py        # Các hàm tiện ích
├── outputs/                   # Thư mục lưu kết quả
├── main.py                   # Script chính
├── add_state.py              # Script tạo state
├── download_dataset.py       # Script download dữ liệu
└── events_with_states.csv    # Dữ liệu đầu vào

```

## 🚀 Cách sử dụng

### 1. Tạo dữ liệu state

```bash
python add_state.py
```

### 2. Huấn luyện model

```bash
python main.py
```

## 📊 Kết quả

Sau khi huấn luyện, các file sau sẽ được tạo trong thư mục gốc:

- `dqn_product_recommendation.pth` - Model đã train
- `dqn_training_results.png` - Đồ thị kết quả training

## ⚙️ Cấu hình

Chỉnh sửa các tham số trong `src/config.py`:

- `NUM_EPISODES`: Số episode huấn luyện
- `BATCH_SIZE`: Kích thước batch
- `LEARNING_RATE`: Learning rate
- `REWARD_MAP`: Mapping reward theo event type
- ...

## 📦 Module

### src/config.py

Chứa tất cả cấu hình và hyperparameters

### src/data/preprocessing.py

- Load và xử lý dữ liệu
- Encode categorical features
- Normalize numerical features
- Tạo state features và action labels

### src/models/dqn.py

- Định nghĩa kiến trúc DQN
- 4 fully-connected layers với dropout

### src/training/trainer.py

- DQNTrainer class với Experience Replay
- Epsilon-greedy exploration
- Target network updates

### src/training/evaluator.py

- Đánh giá Top-1 và Top-5 accuracy
- Hiển thị ví dụ predictions

### src/utils/visualizer.py

- Vẽ đồ thị Loss, Reward, Accuracy

### src/utils/helpers.py

- Save/Load model
- Print summary
