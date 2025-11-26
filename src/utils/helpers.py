"""
Utility functions
"""
import torch
from ..config import *


def save_model(model, optimizer, label_encoders, action_to_idx, 
               state_size, action_size, accuracy, top5_accuracy, 
               final_loss, save_path=None):
    """Lưu model và metadata"""
    print(f"\n[12] Lưu mô hình...")
    
    if save_path is None:
        save_path = MODEL_FILE
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'state_features': STATE_FEATURES,
        'label_encoders': label_encoders,
        'action_to_idx': action_to_idx,
        'hyperparameters': {
            'state_size': state_size,
            'action_size': action_size,
            'hidden_size': HIDDEN_SIZE,
            'learning_rate': LEARNING_RATE,
            'gamma': GAMMA
        },
        'metrics': {
            'final_accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'final_loss': final_loss
        }
    }, save_path)
    
    print(f"   - Đã lưu mô hình: {save_path}")


def load_model(model, optimizer, load_path=None):
    """Load model từ file"""
    if load_path is None:
        load_path = MODEL_FILE
    
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def print_summary(state_size, action_size, num_train_samples, 
                  num_episodes, accuracy, top5_accuracy):
    """In tóm tắt kết quả"""
    print("\n" + "=" * 60)
    print("✅ HOÀN TẤT TẤT CẢ CÁC BƯỚC!")
    print("=" * 60)
    print(f"\nTóm tắt:")
    print(f"   - Mô hình DQN với {state_size} state features và {action_size} actions")
    print(f"   - Huấn luyện trên {num_train_samples} samples trong {num_episodes} episodes")
    print(f"   - Top-1 Accuracy: {accuracy*100:.2f}%")
    print(f"   - Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    print(f"   - Files đã tạo:")
    print(f"     • {MODEL_FILE} (mô hình)")
    print(f"     • {PLOT_FILE} (đồ thị)")
    print("=" * 60)
