"""
Visualization module
"""
import matplotlib.pyplot as plt
from ..config import *


def plot_training_results(losses, episode_rewards, accuracies, save_path=None):
    """Vẽ đồ thị kết quả training"""
    print(f"\n[11] Vẽ đồ thị học tập...")
    
    plt.figure(figsize=(15, 5))
    
    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Reward curve
    plt.subplot(1, 3, 2)
    plt.plot(episode_rewards)
    plt.title('Average Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Avg Reward')
    plt.grid(True)
    
    # Accuracy curve
    plt.subplot(1, 3, 3)
    plt.plot(accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOT_FILE
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   - Đã lưu đồ thị: {save_path}")
    plt.close()
