"""
Configuration file cho DQN model
"""

# Dataset Config
DATASET_FILE = "events_with_states.csv"
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # 10% validation set riêng
RANDOM_SEED = 42

# Model Hyperparameters
HIDDEN_SIZE = 256  # Giảm model size để tránh overfitting
LEARNING_RATE = 0.0001  # Giảm LR để tránh dao động
GAMMA = 0.99  # Tăng discount factor cho long-term
DROPOUT_RATE = 0.4  # Tăng dropout để generalize tốt hơn

# Training Hyperparameters
EPSILON_START = 1.0
EPSILON_END = 0.05  # Giảm để exploit nhiều hơn
EPSILON_DECAY = 0.95  # Decay chậm hơn
BATCH_SIZE = 2048  # Tăng batch size để stable hơn
MEMORY_SIZE = 100000
NUM_EPISODES = 100  # Tăng lên 100 episodes
SAMPLES_PER_EPISODE = 50000  # Tăng samples/episode
WARMUP_EPISODES = 5  # Warm-up phase
EARLY_STOP_PATIENCE = 10  # Dừng nếu acc drop 10 episodes liên tiếp

# Reward Mapping
REWARD_MAP = {
    'view': 0.3,
    'addtocart': 0.5,
    'transaction': 1.0
}

# Normalization Constants (based on new data generation)
TOTAL_RECENT_PURCHASES_MAX = 1500000
TOTAL_VALUE_MAX = 2000000
AVG_VALUE_MAX = 2000000

# Columns to Drop
COLS_TO_DROP = ['timestamp', 'visitorid', 'transactionid', 'itemid']

# State Features (17 features - bỏ recent_searches)
STATE_FEATURES = [
    'gender_Male', 'gender_Female', 'gender_Other',  # 3 features từ one-hot
    'age_group_encoded', 'position_encoded', 'day_of_week',
    'num_orders', 'num_categories', 'num_cart_items',  # num_orders thay vì num_purchased_products
    'total_recent_purchases_norm', 'total_value_norm', 'avg_value_norm',
    'num_products',
    'has_purchase_history', 'has_cart_items', 'has_categories', 'is_high_value'
]

# Output Files
MODEL_FILE = "dqn_product_recommendation.pth"
PLOT_FILE = "dqn_training_results.png"
