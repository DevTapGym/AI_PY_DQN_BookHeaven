"""
DQN Model definition
"""
import torch
import torch.nn as nn
from ..config import *


class DQN(nn.Module):
    """Deep Q-Network for product recommendation với Batch Normalization"""
    
    def __init__(self, state_size, action_size, hidden_size=None):
        super(DQN, self).__init__()
        
        if hidden_size is None:
            hidden_size = HIDDEN_SIZE
        
        # Giảm xuống 3 layers để tránh overfitting
        # Layer 1
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        if x.size(0) > 1:  # Chỉ dùng BN khi batch_size > 1
            x = self.bn1(x)
        x = torch.nn.functional.leaky_relu(x, 0.01)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = torch.nn.functional.leaky_relu(x, 0.01)
        x = self.dropout(x)
        
        # Output layer (no activation, no dropout)
        x = self.fc3(x)
        return x


def create_model(state_size, action_size, device):
    """Tạo policy và target networks"""
    print(f"\n[8] Xây dựng mô hình DQN...")
    print(f"   - State size: {state_size}")
    print(f"   - Action size: {action_size}")
    print(f"   - Hidden size: {HIDDEN_SIZE}")
    print(f"   - Learning rate: {LEARNING_RATE}")
    print(f"   - Device: {device}")
    
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print(f"\n   Kiến trúc mô hình:")
    print(policy_net)
    
    return policy_net, target_net
