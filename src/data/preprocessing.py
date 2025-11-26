"""
Data preprocessing module
"""
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ..config import *


def load_and_preprocess_data(file_path):
    """Load và tiền xử lý dữ liệu"""
    print("=" * 60)
    print("LOAD VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    
    print(f"\n[1] Đang load dữ liệu từ {file_path}...")
    df = pd.read_csv(file_path)
    print(f"   - Tổng số dòng: {len(df)}")
    print(f"   - Các cột: {list(df.columns)}")
    
    # Xử lý missing values
    print("\n[2] Xử lý missing values...")
    df['order_ids'] = df['order_ids'].fillna('')
    df['category'] = df['category'].fillna('')
    df['cart_item_ids'] = df['cart_item_ids'].fillna('')
    df['total_recent_purchases'] = df['total_recent_purchases'].fillna(0)
    df['num_products'] = df['num_products'].fillna(0)
    df['total_value'] = df['total_value'].fillna(0)
    df['avg_value'] = df['avg_value'].fillna(0)
    
    return df


def encode_categorical_features(df):
    """Encode các features categorical"""
    print("\n[3] Encoding categorical features...")
    
    # One-hot encode gender (Male, Female, Other)
    gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
    # Đảm bảo có đủ 3 cột
    for gender in ['gender_Male', 'gender_Female', 'gender_Other']:
        if gender not in gender_dummies.columns:
            gender_dummies[gender] = 0
    
    df = pd.concat([df, gender_dummies[['gender_Male', 'gender_Female', 'gender_Other']]], axis=1)
    print(f"   - Gender one-hot encoded: Male, Female, Other")
    
    # Label encode age_group (U20, U30, U40, U50, U60)
    le_age = LabelEncoder()
    df['age_group_encoded'] = le_age.fit_transform(df['age_group'])
    print(f"   - Age group classes: {list(le_age.classes_)}")
    
    # Label encode position (cart, home, search)
    le_position = LabelEncoder()
    df['position_encoded'] = le_position.fit_transform(df['position'])
    print(f"   - Position classes: {list(le_position.classes_)}")
    
    label_encoders = {
        'age_group': le_age,
        'position': le_position
    }
    
    return df, label_encoders


def parse_list_columns(df):
    """Parse các cột dạng list"""
    print("\n[4] Parsing list columns...")
    
    def safe_parse_list(x):
        if isinstance(x, str) and x != '':
            try:
                return ast.literal_eval(x)
            except:
                return []
        return []
    
    df['order_ids_list'] = df['order_ids'].apply(safe_parse_list)
    df['category_list'] = df['category'].apply(safe_parse_list)
    df['cart_item_ids_list'] = df['cart_item_ids'].apply(safe_parse_list)
    
    # Tính features từ lists và convert sang numeric
    df['num_orders'] = df['order_ids_list'].apply(len).astype(np.int32)
    df['num_categories'] = df['category_list'].apply(len).astype(np.int32)
    df['num_cart_items'] = df['cart_item_ids_list'].apply(len).astype(np.int32)
    
    print(f"   - Parsed orders: min={df['num_orders'].min()}, max={df['num_orders'].max()}")
    
    return df


def normalize_numerical_features(df):
    """Normalize các features số"""
    print("\n[5] Normalizing numerical features...")
    
    # Đảm bảo các cột là numeric
    df['total_recent_purchases'] = pd.to_numeric(df['total_recent_purchases'], errors='coerce').fillna(0)
    df['total_value'] = pd.to_numeric(df['total_value'], errors='coerce').fillna(0)
    df['avg_value'] = pd.to_numeric(df['avg_value'], errors='coerce').fillna(0)
    
    df['total_recent_purchases_norm'] = (df['total_recent_purchases'] / TOTAL_RECENT_PURCHASES_MAX).clip(0, 1)
    df['total_value_norm'] = (df['total_value'] / TOTAL_VALUE_MAX).clip(0, 1)
    df['avg_value_norm'] = (df['avg_value'] / AVG_VALUE_MAX).clip(0, 1)
    
    print(f"   - Mẫu dữ liệu sau normalization:")
    print(df[['gender_Male', 'gender_Female', 'age_group_encoded', 'position_encoded', 
              'num_orders', 'total_recent_purchases_norm']].head())
    
    return df


def build_features_and_labels(df):
    """Xây dựng features và labels"""
    print("\n[6] Xây dựng state features và action labels...")
    
    # Bỏ qua các cột không cần thiết
    for col in COLS_TO_DROP:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"   - Đã bỏ cột: {col}")
    
    # Action: group_id
    if 'group_id' not in df.columns:
        raise ValueError("Không tìm thấy cột 'group_id' trong dữ liệu!")
    
    df['action'] = df['group_id']
    y = df['action'].values
    print(f"   - Action (group_id) shape: {y.shape}")
    print(f"   - Unique actions: {len(np.unique(y))}")
    print(f"   - Action range: [{y.min()}, {y.max()}]")
    
    # Reward: dựa trên event
    if 'event' not in df.columns:
        raise ValueError("Không tìm thấy cột 'event' trong dữ liệu!")
    
    df['reward'] = df['event'].map(REWARD_MAP)
    if df['reward'].isna().any():
        print(f"    Cảnh báo: Có {df['reward'].isna().sum()} giá trị event không khớp với reward_map")
        print(f"   - Các event unique: {df['event'].unique()}")
        df['reward'] = df['reward'].fillna(0.1)
    
    print(f"   - Reward mapping: {REWARD_MAP}")
    print(f"   - Reward distribution:")
    for event, reward in REWARD_MAP.items():
        count = (df['reward'] == reward).sum()
        print(f"     • {event}: {count} samples (reward={reward})")
    
    # Tạo derived features (đảm bảo là numeric)
    df['has_purchase_history'] = (df['num_orders'] > 0).astype(np.int32)
    df['has_cart_items'] = (df['num_cart_items'] > 0).astype(np.int32)
    df['has_categories'] = (df['num_categories'] > 0).astype(np.int32)
    df['is_high_value'] = (df['total_recent_purchases'] > 500000).astype(np.int32)  # Giảm ngưỡng xuống 500k
    
    # Đảm bảo day_of_week và num_products là numeric
    df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').fillna(1).astype(np.int32)
    df['num_products'] = pd.to_numeric(df['num_products'], errors='coerce').fillna(0).astype(np.int32)
    
    # State features
    X = df[STATE_FEATURES].values
    
    # Đảm bảo X là numeric và không có giá trị object
    X = X.astype(np.float32)
    
    print(f"   - State shape: {X.shape}")
    print(f"   - State dtype: {X.dtype}")
    print(f"   - State features ({len(STATE_FEATURES)}): {STATE_FEATURES}")
    print(f"   - Sample state values: {X[0]}")
    
    # Map action to index (0-based)
    action_to_idx = {action: idx for idx, action in enumerate(sorted(np.unique(y)))}
    idx_to_action = {idx: action for action, idx in action_to_idx.items()}
    df['action_idx'] = df['action'].map(action_to_idx)
    y_idx = df['action_idx'].values
    
    print(f"   - Action mapping: {len(action_to_idx)} unique groups")
    print(f"   - Action index range: [0, {len(action_to_idx)-1}]")
    
    rewards = df['reward'].values
    
    return X, y_idx, rewards, action_to_idx, idx_to_action


def split_data(X, y, rewards):
    """Chia dữ liệu train/val/test"""
    print("\n[7] Chia dữ liệu train/val/test...")
    
    # Chia train/test
    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
        X, y, rewards, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    
    # Chia validation từ train (10% của train)
    X_train, X_val, y_train, y_val, r_train, r_val = train_test_split(
        X_train, y_train, r_train, test_size=VAL_SIZE, random_state=RANDOM_SEED
    )
    
    print(f"   - Train set: {X_train.shape[0]} samples")
    print(f"   - Validation set: {X_val.shape[0]} samples")
    print(f"   - Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test


def preprocess_pipeline(file_path):
    """Pipeline đầy đủ cho preprocessing"""
    df = load_and_preprocess_data(file_path)
    df, label_encoders = encode_categorical_features(df)
    df = parse_list_columns(df)
    df = normalize_numerical_features(df)
    X, y, rewards, action_to_idx, idx_to_action = build_features_and_labels(df)
    X_train, X_val, X_test, y_train, y_val, y_test, r_train, r_val, r_test = split_data(X, y, rewards)
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'r_train': r_train,
        'r_val': r_val,
        'r_test': r_test,
        'action_to_idx': action_to_idx,
        'idx_to_action': idx_to_action,
        'label_encoders': label_encoders,
        'num_actions': len(action_to_idx)
    }
