"""
Model evaluation module
"""
import torch
import numpy as np


def evaluate_final_metrics(model, X_test, y_test, idx_to_action, device):
    """Đánh giá đầy đủ metrics trên tập test"""
    print("\n[10] Đánh giá mô hình trên tập test...")
    
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    top5_correct = 0
    action_size = model.fc3.out_features
    
    with torch.no_grad():
        for idx in range(len(X_test)):
            state = torch.FloatTensor(X_test[idx]).unsqueeze(0).to(device)
            q_values = model(state)
            
            # Top-1 accuracy
            predicted = q_values.argmax().item()
            true_action = y_test[idx]
            
            if predicted == true_action:
                correct_predictions += 1
            
            # Top-5 accuracy
            top5 = q_values.topk(min(5, action_size))[1].cpu().numpy().flatten()
            if true_action in top5:
                top5_correct += 1
            
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    top5_accuracy = top5_correct / total_predictions
    
    print(f"\n📊 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"   - Tổng số samples test: {total_predictions}")
    print(f"   - Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   - Top-5 Accuracy: {top5_accuracy:.4f} ({top5_accuracy*100:.2f}%)")
    print(f"   - Số dự đoán đúng: {correct_predictions}/{total_predictions}")
    
    return accuracy, top5_accuracy


def show_prediction_examples(model, X_test, y_test, idx_to_action, device, num_examples=10):
    """Hiển thị một số ví dụ dự đoán"""
    print(f"\n📋 MỘT SỐ VÍ DỤ DỰ ĐOÁN:")
    print("-" * 60)
    
    model.eval()
    with torch.no_grad():
        for i in range(min(num_examples, len(X_test))):
            state = torch.FloatTensor(X_test[i]).unsqueeze(0).to(device)
            q_values = model(state)
            predicted_idx = q_values.argmax().item()
            true_action_idx = y_test[i]
            
            # Convert back to original group_id
            predicted_group = idx_to_action[predicted_idx]
            true_group = idx_to_action[true_action_idx]
            
            confidence = torch.softmax(q_values, dim=1).max().item()
            
            print(f"   Sample {i+1}: Predicted group={predicted_group}, True group={true_group}, "
                  f"Confidence={confidence:.3f}, Match={'✓' if predicted_idx==true_action_idx else '✗'}")
