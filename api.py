"""
FastAPI for DQN Product Recommendation System
2 endpoints: /recommend (get recommendations), /feedback (learn from user feedback)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
import pickle
import ast
import uvicorn
from src.models.dqn import DQN
from src.config import STATE_FEATURES

app = FastAPI(title="DQN Product Recommendation API", version="1.0.0")

# Global variables
model = None
device = None
label_encoders = None
action_to_idx = None
idx_to_action = None
state_size = None
num_actions = None

# Feedback buffer for online learning
feedback_buffer = []
MAX_FEEDBACK_BUFFER = 1000


class UserState(BaseModel):
    """User state input - format giống CSV"""
    gender: str  # "Male", "Female", "Other"
    age_group: str  # "U20", "U30", "U40", "U50", "U60"
    position: str  # "cart", "home", "search"
    day_of_week: int  # 1-7 (1=Monday)
    num_products: float  # Số sản phẩm
    total_value: float  # Tổng giá trị session
    avg_value: float  # Giá trị trung bình
    cart_item_ids: Optional[str] = ""  # VD: "[12, 13, 27, 36]"
    order_ids: Optional[str] = ""  # Danh sách order IDs (đơn hàng đã mua)
    total_recent_purchases: float = 0  # Tổng tiền mua gần đây (50k-1.5M)
    category: Optional[str] = ""  # Danh mục
    
    
class RecommendRequest(BaseModel):
    """Request for recommendations"""
    user_state: UserState
    top_k: Optional[int] = 5
    

class RecommendResponse(BaseModel):
    """Response with recommendations"""
    recommended: List[int]  # Top-K group IDs
    confidence_scores: List[float]
    

class FeedbackRequest(BaseModel):
    """Feedback from user (RL format: state, action, reward, next_state, done)"""
    state: UserState  # Current state
    action: int  # Recommended group_id
    reward: Optional[float] = None  # Có thể tính tự động
    event_type: str  # "view", "addtocart", "transaction"
    next_state: Optional[UserState] = None  # Next state (optional)
    done: bool = True  # Episode done? (default True)


def load_model():
    """Load trained model and encoders"""
    global model, device, label_encoders, action_to_idx, idx_to_action, state_size, num_actions
    
    try:
        import os
        if not os.path.exists('dqn_product_recommendation.pth'):
            print("⚠️  Model file not found. Please train the model first: python main.py")
            print("   API will start but /recommend endpoint will return 503 error")
            return
            
        # Load model checkpoint (weights_only=False for sklearn objects)
        checkpoint = torch.load('dqn_product_recommendation.pth', map_location='cpu', weights_only=False)
        
        state_size = checkpoint.get('state_size', 17)  # Default 17 features
        num_actions = checkpoint.get('num_actions', 50)  # Default 50 actions
        label_encoders = checkpoint.get('label_encoders', {})
        action_to_idx = checkpoint.get('action_to_idx', {})
        idx_to_action = checkpoint.get('idx_to_action', {})
        
        # Nếu idx_to_action rỗng, tạo mapping 1:1 (index -> group_id)
        if not idx_to_action:
            print("idx_to_action empty, creating default mapping (1-50)")
            idx_to_action = {i: i+1 for i in range(num_actions)}
        
        # Convert keys to int (đề phòng string keys)
        idx_to_action = {int(k): int(v) for k, v in idx_to_action.items()}
        
        # Create and load model
        device = torch.device('cpu')
        model = DQN(state_size, num_actions).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"   - State size: {state_size}")
        print(f"   - Actions: {num_actions}")
        print(f"   - Action mapping: {len(idx_to_action)} groups")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


def normalize_user_state(user_state: UserState) -> UserState:
    """Normalize user state với giá trị mặc định cho người dùng mới"""
    # Gán giá trị mặc định nếu trống hoặc None
    if not user_state.gender or user_state.gender.strip() == "":
        user_state.gender = "Other"
    
    if not user_state.age_group or user_state.age_group.strip() == "":
        user_state.age_group = "U20"
    
    if not user_state.position or user_state.position.strip() == "":
        user_state.position = "home"
    
    return user_state


def state_to_vector(user_state: UserState) -> np.ndarray:
    """Convert user state to feature vector (17 features)"""
    # Normalize user state trước khi xử lý
    user_state = normalize_user_state(user_state)
    
    # One-hot encode gender
    gender_male = 1.0 if user_state.gender == "Male" else 0.0
    gender_female = 1.0 if user_state.gender == "Female" else 0.0
    gender_other = 1.0 if user_state.gender == "Other" else 0.0
    
    # Encode age_group
    age_mapping = {"U20": 0, "U30": 1, "U40": 2, "U50": 3, "U60": 4}
    age_encoded = age_mapping.get(user_state.age_group, 0)
    
    # Encode position
    position_mapping = {"cart": 0, "home": 1, "search": 2}
    position_encoded = position_mapping.get(user_state.position, 0)
    
    # Parse cart_item_ids để đếm số items
    import ast
    try:
        cart_items = ast.literal_eval(user_state.cart_item_ids) if user_state.cart_item_ids else []
        num_cart_items = len(cart_items) if isinstance(cart_items, list) else 0
    except:
        num_cart_items = 0
    
    # Parse order_ids để đếm số đơn hàng
    try:
        order_ids = ast.literal_eval(user_state.order_ids) if user_state.order_ids else []
        num_orders = len(order_ids) if isinstance(order_ids, list) else 0
    except:
        num_orders = 0
    
    # Parse category để đếm số categories
    try:
        categories = ast.literal_eval(user_state.category) if user_state.category else []
        num_categories = len(categories) if isinstance(categories, list) else 0
    except:
        num_categories = 0
    
    # Normalize numerical features
    total_recent_purchases_norm = min(user_state.total_recent_purchases / 1500000.0, 1.0)  # Max 1.5M
    total_value_norm = min(user_state.total_value / 2000000.0, 1.0)  # Max 2M
    avg_value_norm = min(user_state.avg_value / 2000000.0, 1.0)  # Max 2M
    
    # Derived features
    has_purchase_history = 1.0 if num_orders > 0 else 0.0
    has_cart_items = 1.0 if num_cart_items > 0 else 0.0
    has_categories = 1.0 if num_categories > 0 else 0.0
    is_high_value = 1.0 if user_state.total_value > 500000 else 0.0
    
    # Build feature vector (17 features - bỏ recent_searches)
    features = np.array([
        gender_male,
        gender_female,
        gender_other,
        age_encoded,
        position_encoded,
        user_state.day_of_week,
        num_orders,  # Số đơn hàng
        num_categories,
        num_cart_items,
        total_recent_purchases_norm,
        total_value_norm,
        avg_value_norm,
        user_state.num_products,
        has_purchase_history,
        has_cart_items,
        has_categories,
        is_high_value
    ], dtype=np.float32)
    
    return features


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get product group recommendations based on user state
    Epsilon-greedy: 0.8 explore cho user mới, 0.5 cho user có đủ thông tin
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Kiểm tra xem user có đầy đủ thông tin không
        is_new_user = (
            not request.user_state.gender or request.user_state.gender.strip() == "" or
            not request.user_state.age_group or request.user_state.age_group.strip() == "" or
            not request.user_state.position or request.user_state.position.strip() == "" or
            not request.user_state.order_ids or request.user_state.order_ids.strip() == "" or
            request.user_state.total_recent_purchases == 0
        )
        
        # Convert state to vector (sẽ tự động normalize với giá trị mặc định)
        state_vector = state_to_vector(request.user_state)
        
        # Get Q-values from model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            q_values = model(state_tensor).squeeze(0).cpu().numpy()
        
        # Epsilon-greedy động: 80% explore cho user mới, 50% cho user có đủ thông tin
        epsilon = 0.8 if is_new_user else 0.5
        top_k = min(request.top_k, num_actions)
        
        if np.random.rand() < epsilon:
            # Explore: Random recommendations (50%)
            top_k_indices = np.random.choice(num_actions, size=top_k, replace=False)
        else:
            # Exploit: Model-based recommendations (50%)
            top_k_indices = np.argsort(q_values)[-top_k:][::-1]
        
        # Convert to group IDs and confidence scores
        recommended_groups = [idx_to_action[int(idx)] for idx in top_k_indices]
        
        # Normalize scores to 0-1 using softmax
        q_values_top_k = q_values[top_k_indices]
        exp_scores = np.exp(q_values_top_k - np.max(q_values_top_k))
        confidence_scores = (exp_scores / exp_scores.sum()).tolist()
        
        # Log strategy cho debugging
        strategy = "explore (new user)" if is_new_user else "mixed (epsilon=0.5)"
        print(f"[INFO] Recommendation strategy: {strategy}, epsilon={epsilon}")
        
        return RecommendResponse(
            recommended=recommended_groups,
            confidence_scores=confidence_scores
        )
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Recommendation failed: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


def retrain_model_from_buffer():
    """Retrain model using feedback buffer (online learning)"""
    global model
    
    if len(feedback_buffer) < 32:  # Minimum batch size
        return {"status": "skipped", "reason": "Not enough samples (need 32+)"}
    
    try:
        # Prepare training data from buffer
        states = torch.FloatTensor([fb["state"] for fb in feedback_buffer]).to(device)
        actions = torch.LongTensor([fb["action"] for fb in feedback_buffer]).to(device)
        rewards = torch.FloatTensor([fb["reward"] for fb in feedback_buffer]).to(device)
        next_states = torch.FloatTensor([fb["next_state"] for fb in feedback_buffer]).to(device)
        dones = torch.FloatTensor([fb["done"] for fb in feedback_buffer]).to(device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        
        model.train()
        
        # Train for a few epochs
        num_epochs = 5
        batch_size = 32
        total_loss = 0
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for i in range(0, len(states), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_rewards = rewards[batch_indices]
                batch_next_states = next_states[batch_indices]
                batch_dones = dones[batch_indices]
                
                # Forward pass
                q_values = model(batch_states)
                q_values_for_actions = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Target Q-value: Q_target = reward + (1 - done) * gamma * max(Q(next_state))
                with torch.no_grad():
                    next_q_values = model(batch_next_states).max(1)[0]
                    target_q_values = batch_rewards + (1 - batch_dones) * 0.99 * next_q_values
                
                # Loss: MSE between predicted Q and target Q
                loss = criterion(q_values_for_actions, target_q_values)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        model.eval()
        avg_loss = total_loss / (num_epochs * (len(states) // batch_size + 1))
        
        return {
            "status": "success",
            "samples_used": len(feedback_buffer),
            "epochs": num_epochs,
            "avg_loss": round(avg_loss, 4)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """
    Submit user feedback for online learning
    Tự động normalize user state với giá trị mặc định cho người dùng mới
    
    Example request:
    {
        "state": { ... },
        "action": 15,
        "event_type": "transaction",
        "reward": 100.0,  # Optional - auto-calculate from event_type
        "next_state": { ... },  # Optional - if not provided, use current state
        "done": true  # Optional - default True
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert state to vector (tự động normalize)
        state_vector = state_to_vector(request.state)
        
        # Convert next_state to vector (nếu không có thì dùng state hiện tại)
        if request.next_state is not None:
            next_state_vector = state_to_vector(request.next_state)
        else:
            next_state_vector = state_vector.copy()
        
        # Calculate reward if not provided
        if request.reward is None:
            reward_map = {
                "view": 5.0,
                "addtocart": 30.0,
                "transaction": 100.0
            }
            reward = reward_map.get(request.event_type, 1.0)
        else:
            reward = request.reward
        
        # Store feedback with next_state and done
        feedback_data = {
            "state": state_vector,
            "action": action_to_idx.get(request.action, 0),
            "reward": reward,
            "next_state": next_state_vector,
            "done": 1.0 if request.done else 0.0,
            "event_type": request.event_type
        }
        
        feedback_buffer.append(feedback_data)
        
        # Keep buffer size manageable
        if len(feedback_buffer) > MAX_FEEDBACK_BUFFER:
            feedback_buffer.pop(0)
        
        # Smart retrain logic:
        # - transaction: Train ngay lập tức (quan trọng!)
        # - view/addtocart: Gom đủ 50 samples mới train
        retrain_info = None
        
        if request.event_type == "transaction":
            # Transaction: Train ngay
            if len(feedback_buffer) >= 32:  # Cần ít nhất 32 samples
                retrain_info = retrain_model_from_buffer()
                retrain_info["trigger"] = "transaction"
        else:
            # View/Addtocart: Train khi đủ 50 samples
            if len(feedback_buffer) >= 50 and len(feedback_buffer) % 50 == 0:
                retrain_info = retrain_model_from_buffer()
                retrain_info["trigger"] = f"{request.event_type}_batch"
        
        response = {
            "status": "success",
            "message": "Feedback received",
            "reward": reward,
            "buffer_size": len(feedback_buffer)
        }
        
        if retrain_info:
            response["retrain"] = retrain_info
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")


@app.post("/retrain")
async def retrain():
    """
    Manually trigger model retraining from feedback buffer
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = retrain_model_from_buffer()
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING DQN PRODUCT RECOMMENDATION API")
    print("=" * 60)
    print("\n📌 Endpoints:")
    print("   POST /recommend - Get recommendations (ε=0.8 new users, ε=0.5 existing)")
    print("   POST /feedback  - Submit feedback (auto-normalize user state)")
    print("                     → transaction: instant retrain")
    print("                     → view/cart: batch retrain @50 samples")
    print("   POST /retrain   - Manually trigger retraining")
    print("   GET  /health    - Health check")
    print("   GET  /stats     - Model statistics")
    print("\n🌐 Access at: http://localhost:8000")
    print("📖 Docs at: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
