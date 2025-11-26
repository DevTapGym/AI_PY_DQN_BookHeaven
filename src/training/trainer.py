"""
Training module for DQN
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from ..config import *


class DQNTrainer:
    """DQN Trainer với Experience Replay"""
    
    def __init__(self, policy_net, target_net, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device
        
        # Optimizer với weight decay
        self.optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Learning rate scheduler - Cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPISODES, eta_min=1e-6)
        
        # Huber Loss thay vì MSE (robust hơn với outliers)
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = deque(maxlen=MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.action_size = policy_net.fc3.out_features
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection với boltzmann exploration"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                # Thêm softmax sampling thay vì greedy (exploration tốt hơn)
                if training and random.random() < 0.1:  # 10% sử dụng softmax
                    probs = torch.softmax(q_values / 0.5, dim=1)  # temperature=0.5
                    return torch.multinomial(probs, 1).item()
                else:
                    return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Lưu transition vào replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def optimize_model(self):
        """Optimize policy network sử dụng experience replay"""
        if len(self.memory) < BATCH_SIZE:
            return 0
        
        # Sample batch
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Double DQN: sử dụng policy net để select action, target net để evaluate
        with torch.no_grad():
            # Chọn action từ policy network
            next_actions = self.policy_net(next_states).argmax(1)
            # Đánh giá action từ target network
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        # Compute loss (Huber loss đã được set)
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def update_target_network(self):
        """Cập nhật target network từ policy network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())


def train_dqn(trainer, X_train, y_train, r_train, X_val, y_val, X_test, y_test):
    """Training loop chính với early stopping và validation monitoring"""
    print(f"\n[9] Bắt đầu huấn luyện...")
    print(f"\nHuấn luyện {NUM_EPISODES} episodes")
    print(f"Dữ liệu: {len(X_train)} samples train, {len(X_val)} samples val")
    print(f"Sử dụng {SAMPLES_PER_EPISODE} samples/episode")
    print("-" * 60)
    
    losses = []
    episode_rewards = []
    train_accuracies = []
    val_accuracies = []
    best_val_accuracy = 0.0
    patience = EARLY_STOP_PATIENCE
    patience_counter = 0
    
    for episode in range(NUM_EPISODES):
        # Shuffle và lấy subset data
        indices = np.random.permutation(len(X_train))[:SAMPLES_PER_EPISODE]
        
        episode_loss = 0
        episode_reward = 0
        steps = 0
        
        for i in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[i:i+BATCH_SIZE]
            
            for idx in batch_indices:
                state = X_train[idx]
                action = trainer.select_action(state)
                
                # Get true action and reward
                true_action = y_train[idx]
                base_reward = r_train[idx]  # 0.3, 0.5, hoặc 1.0
                
                # Class-weighted rewards để cân bằng imbalanced data
                # Class distribution: view=96.7%, addtocart=2.5%, transaction=0.8%
                if action == true_action:
                    # Correct: reward cao hơn cho class hiếm
                    if base_reward == 1.0:  # transaction (0.8% - rất hiếm)
                        reward = 100.0
                    elif base_reward == 0.5:  # addtocart (2.5% - hiếm)
                        reward = 30.0
                    else:  # view (96.7% - phổ biến)
                        reward = 5.0
                else:
                    # Wrong: penalty nhẹ để không discourage exploration
                    reward = -0.1
                
                # Next state = current state (simplification)
                next_state = state.copy()
                done = 1  # Always done (episodic task)
                
                # Store transition
                trainer.store_transition(state, action, reward, next_state, done)
                
                episode_reward += reward
                steps += 1
            
            # Optimize model
            loss = trainer.optimize_model()
            if loss > 0:
                episode_loss += loss
        
        # Decay epsilon
        trainer.decay_epsilon()
        
        # Step learning rate scheduler
        trainer.scheduler.step()
        
        # Update target network every 3 episodes
        if episode % 3 == 0:
            trainer.update_target_network()
        
        # Evaluate on both train and validation
        train_acc = evaluate_model(trainer.policy_net, X_train, y_train, trainer.device, num_samples=1000)
        val_acc = evaluate_model(trainer.policy_net, X_val, y_val, trainer.device, num_samples=1000)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        avg_loss = episode_loss / max(1, steps // BATCH_SIZE)
        losses.append(avg_loss)
        episode_rewards.append(episode_reward / steps)
        
        # Early stopping dựa trên validation accuracy
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            patience_counter = 0
            # Save best model
            best_model_state = trainer.policy_net.state_dict().copy()
        else:
            patience_counter += 1
        
        # In kết quả mỗi episode
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"Ep {episode+1:3d}/{NUM_EPISODES} | "
              f"Loss: {avg_loss:.4f} | "
              f"Rew: {episode_reward/steps:.3f} | "
              f"Eps: {trainer.epsilon:.3f} | "
              f"TrainAcc: {train_acc:.3f} | "
              f"ValAcc: {val_acc:.3f} | "
              f"Best: {best_val_accuracy:.3f} | "
              f"LR: {current_lr:.6f} | "
              f"Wait: {patience_counter}/{patience}")
        
        # Mỗi 10 episodes: in stability metrics (std deviation)
        if (episode + 1) % 10 == 0 and episode > 0:
            recent_rewards = episode_rewards[-10:]
            recent_val_accs = val_accuracies[-10:]
            print(f"   📊 Last 10 eps | Reward std: {np.std(recent_rewards):.3f} | ValAcc std: {np.std(recent_val_accs):.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at episode {episode+1}! Best val acc: {best_val_accuracy:.3f}")
            # Restore best model
            trainer.policy_net.load_state_dict(best_model_state)
            break
    else:
        # Nếu hoàn thành tất cả episodes
        print(f"\n✓ Hoàn thành {NUM_EPISODES} episodes! Best val acc: {best_val_accuracy:.3f}")
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH HUẤN LUYỆN!")
    print("=" * 60)
    
    return losses, episode_rewards, train_accuracies, val_accuracies


def evaluate_model(model, X_test, y_test, device, num_samples=1000):
    """Đánh giá model trên tập test"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for idx in range(min(num_samples, len(X_test))):
            state = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(device)
            q_values = model(state)
            predicted = q_values.argmax().item()
            true_action = y_test[idx]
            
            if predicted == true_action:
                correct += 1
            total += 1
    
    model.train()
    return correct / total if total > 0 else 0
