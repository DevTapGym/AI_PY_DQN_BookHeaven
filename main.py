"""
Main training script for DQN Product Recommendation
"""
import torch
import numpy as np
from src.config import DATASET_FILE, NUM_EPISODES, STATE_FEATURES, MODEL_FILE, PLOT_FILE
from src.data.preprocessing import preprocess_pipeline
from src.models.dqn import create_model
from src.training.trainer import DQNTrainer, train_dqn
from src.training.evaluator import evaluate_final_metrics, show_prediction_examples
from src.utils.visualizer import plot_training_results
from src.utils.helpers import save_model, print_summary


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH DQN - GỢI Ý SẢN PHẨM")
    print("=" * 60)
    
    # 1. Preprocessing
    print("\n🔍 KIỂM TRA VÀ TIỀN XỬ LÝ DỮ LIỆU...")
    data = preprocess_pipeline(DATASET_FILE)
    
    # Validate data types
    print("\n✓ Kiểm tra kiểu dữ liệu:")
    print(f"  - X_train dtype: {data['X_train'].dtype}, shape: {data['X_train'].shape}")
    print(f"  - y_train dtype: {data['y_train'].dtype}, shape: {data['y_train'].shape}")
    print(f"  - r_train dtype: {data['r_train'].dtype}, shape: {data['r_train'].shape}")
    
    # Check for NaN or Inf
    if np.isnan(data['X_train']).any():
        print("  ⚠️ Cảnh báo: X_train có giá trị NaN!")
        # Replace NaN with 0
        data['X_train'] = np.nan_to_num(data['X_train'], nan=0.0)
        print("  ✓ Đã thay thế NaN bằng 0")
    
    if np.isinf(data['X_train']).any():
        print("  ⚠️ Cảnh báo: X_train có giá trị Inf!")
        # Replace Inf with large number
        data['X_train'] = np.nan_to_num(data['X_train'], posinf=1e6, neginf=-1e6)
        print("  ✓ Đã thay thế Inf")
    
    print("✓ Dữ liệu hợp lệ, tiếp tục huấn luyện...\n")
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    r_train = data['r_train']
    r_val = data['r_val']
    r_test = data['r_test']
    action_to_idx = data['action_to_idx']
    idx_to_action = data['idx_to_action']
    label_encoders = data['label_encoders']
    num_actions = data['num_actions']
    
    # 2. Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_size = len(STATE_FEATURES)
    
    policy_net, target_net = create_model(state_size, num_actions, device)
    
    # 3. Create trainer
    trainer = DQNTrainer(policy_net, target_net, device)
    
    # 4. Train
    losses, episode_rewards, train_accuracies, val_accuracies = train_dqn(
        trainer, X_train, y_train, r_train, X_val, y_val, X_test, y_test
    )
    
    # 5. Final evaluation
    accuracy, top5_accuracy = evaluate_final_metrics(
        policy_net, X_test, y_test, idx_to_action, device
    )
    
    # 6. Show examples
    show_prediction_examples(
        policy_net, X_test, y_test, idx_to_action, device, num_examples=10
    )
    
    # 7. Plot results (dùng val_accuracies để monitor generalization)
    plot_training_results(losses, episode_rewards, val_accuracies)
    
    # 8. Save model
    save_model(
        policy_net, trainer.optimizer, label_encoders, action_to_idx,
        state_size, num_actions, accuracy, top5_accuracy,
        losses[-1] if losses else 0
    )
    
    # 9. Print summary
    print_summary(
        state_size, num_actions, len(X_train),
        NUM_EPISODES, accuracy, top5_accuracy
    )


if __name__ == "__main__":
    main()
