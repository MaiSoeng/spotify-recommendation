"""
NeuMF Inference Script for SageMaker Endpoint
Handles real-time recommendation requests with user feature integration.

Features:
- Load trained NeuMF model
- Real-time user embedding lookup
- Top-K recommendation generation
- Integration with DynamoDB for user features
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===============================
# NeuMF Model Definition (must match training)
# ===============================

class GMF(nn.Module):
    """Generalized Matrix Factorization"""
    
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb * item_emb


class MLP(nn.Module):
    """Multi-Layer Perceptron path"""
    
    def __init__(self, num_users, num_items, embedding_dim, layers=[128, 64, 32], dropout=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        mlp_layers = []
        input_dim = embedding_dim * 2
        
        for output_dim in layers:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = output_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        self.output_dim = layers[-1]
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        concat = torch.cat([user_emb, item_emb], dim=-1)
        return self.mlp(concat)


class NeuMF(nn.Module):
    """Neural Matrix Factorization"""
    
    def __init__(self, num_users, num_items, gmf_embedding_dim=32, 
                 mlp_embedding_dim=32, mlp_layers=[128, 64, 32], dropout=0.2):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        self.gmf = GMF(num_users, num_items, gmf_embedding_dim)
        self.mlp = MLP(num_users, num_items, mlp_embedding_dim, mlp_layers, dropout)
        
        final_input_dim = gmf_embedding_dim + self.mlp.output_dim
        self.predict_layer = nn.Linear(final_input_dim, 1)
    
    def forward(self, user_ids, item_ids):
        gmf_output = self.gmf(user_ids, item_ids)
        mlp_output = self.mlp(user_ids, item_ids)
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.predict_layer(concat))
        return prediction.squeeze()


# ===============================
# Model Loading
# ===============================

def model_fn(model_dir):
    """
    Load trained NeuMF model for inference.
    Called once when the endpoint container starts.
    """
    logger.info(f"Loading model from {model_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model metadata: {metadata}")
    
    # Load encodings
    with open(os.path.join(model_dir, 'user_to_idx.json'), 'r') as f:
        user_to_idx = json.load(f)
    
    with open(os.path.join(model_dir, 'item_to_idx.json'), 'r') as f:
        item_to_idx = json.load(f)
    
    with open(os.path.join(model_dir, 'idx_to_item.json'), 'r') as f:
        idx_to_item = json.load(f)
    
    # Try to load TorchScript model first
    model_pt_path = os.path.join(model_dir, 'model.pt')
    model_pth_path = os.path.join(model_dir, 'model_state.pth')
    
    try:
        if os.path.exists(model_pt_path):
            logger.info("Loading TorchScript model...")
            model = torch.jit.load(model_pt_path, map_location=device)
        else:
            raise FileNotFoundError("model.pt not found")
    except Exception as e:
        logger.warning(f"Failed to load TorchScript model: {e}")
        logger.info("Falling back to state dict loading...")
        
        # Recreate model architecture
        mlp_layers = [int(x) for x in metadata.get('mlp_layers', '128,64,32').split(',')]
        
        model = NeuMF(
            num_users=metadata['num_users'],
            num_items=metadata['num_items'],
            gmf_embedding_dim=metadata.get('gmf_embedding_dim', 32),
            mlp_embedding_dim=metadata.get('mlp_embedding_dim', 32),
            mlp_layers=mlp_layers,
            dropout=0.0
        )
        
        # Load state dict with map_location
        model.load_state_dict(torch.load(model_pth_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Users: {metadata['num_users']}, Items: {metadata['num_items']}")
    
    return {
        'model': model,
        'device': device,
        'metadata': metadata,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'num_users': metadata['num_users'],
        'num_items': metadata['num_items']
    }


# ===============================
# Input Processing
# ===============================

def input_fn(request_body, request_content_type='application/json'):
    """
    Parse and validate input request.
    
    Expected input format:
    {
        "user_id": "user123",           # External user ID
        "user_idx": 42,                  # Optional: pre-encoded user index
        "n_recommendations": 10,         # Number of recommendations to return
        "exclude_items": ["track1"],     # Optional: items to exclude
        "include_scores": true,          # Optional: include prediction scores
        "user_history": ["track1", ...]  # Optional: recent user history
    }
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Validate required fields
        if 'user_id' not in input_data and 'user_idx' not in input_data:
            raise ValueError("Request must include 'user_id' or 'user_idx'")
        
        # Set defaults
        input_data.setdefault('n_recommendations', 10)
        input_data.setdefault('exclude_items', [])
        input_data.setdefault('include_scores', True)
        input_data.setdefault('user_history', [])
        
        return input_data
    
    raise ValueError(f"Unsupported content type: {request_content_type}")


# ===============================
# Prediction
# ===============================

def predict_fn(input_data, model_artifacts):
    """
    Generate recommendations for the given user.
    """
    model = model_artifacts['model']
    device = model_artifacts['device']
    user_to_idx = model_artifacts['user_to_idx']
    item_to_idx = model_artifacts['item_to_idx']
    idx_to_item = model_artifacts['idx_to_item']
    num_items = model_artifacts['num_items']
    
    user_id = input_data.get('user_id', 'unknown')
    n_recommendations = input_data.get('n_recommendations', 10)
    exclude_items = set(input_data.get('exclude_items', []))
    include_scores = input_data.get('include_scores', True)
    user_history = input_data.get('user_history', [])
    
    # Get user index
    if 'user_idx' in input_data:
        user_idx = input_data['user_idx']
    else:
        user_idx = user_to_idx.get(str(user_id), 0)
    
    # Validate user index
    if user_idx >= model_artifacts['num_users']:
        logger.warning(f"User index {user_idx} out of range, using cold start strategy")
        user_idx = 0  # Fallback to first user (could be improved with better cold start)
    
    # Add user history to exclude list
    for item in user_history:
        if item in item_to_idx:
            exclude_items.add(item)
    
    # Convert exclude items to indices
    exclude_indices = set()
    for item in exclude_items:
        if str(item) in item_to_idx:
            exclude_indices.add(item_to_idx[str(item)])
    
    logger.info(f"Generating recommendations for user {user_id} (idx: {user_idx})")
    logger.info(f"Excluding {len(exclude_indices)} items from recommendations")
    
    # Score all items
    with torch.no_grad():
        # Create tensors for all items
        user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long).to(device)
        item_tensor = torch.tensor(list(range(num_items)), dtype=torch.long).to(device)
        
        # Get predictions in batches if necessary
        batch_size = 10000
        if num_items > batch_size:
            scores = []
            for start in range(0, num_items, batch_size):
                end = min(start + batch_size, num_items)
                batch_user = user_tensor[start:end]
                batch_item = item_tensor[start:end]
                batch_scores = model(batch_user, batch_item).cpu().numpy()
                scores.extend(batch_scores)
            scores = np.array(scores)
        else:
            scores = model(user_tensor, item_tensor).cpu().numpy()
    
    # Rank items (excluding filtered ones)
    ranked_indices = np.argsort(-scores)
    
    # Generate recommendations
    recommendations = []
    for idx in ranked_indices:
        idx = int(idx)
        
        # Skip excluded items
        if idx in exclude_indices:
            continue
        
        # Get item info
        item_id = idx_to_item.get(str(idx), f"item_{idx}")
        
        rec = {
            'rank': len(recommendations) + 1,
            'track_id': item_id,
        }
        
        if include_scores:
            rec['score'] = float(scores[idx])
        
        recommendations.append(rec)
        
        if len(recommendations) >= n_recommendations:
            break
    
    result = {
        'user_id': user_id,
        'user_idx': user_idx,
        'recommendations': recommendations,
        'num_excluded': len(exclude_indices),
        'model_version': model_artifacts['metadata'].get('training_date', 'unknown')
    }
    
    logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
    
    return result


# ===============================
# Output Formatting
# ===============================

def output_fn(prediction, response_content_type='application/json'):
    """
    Format the prediction response.
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction, default=str)
    
    raise ValueError(f"Unsupported response content type: {response_content_type}")


# ===============================
# Local Testing
# ===============================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test NeuMF inference locally')
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing the trained model')
    parser.add_argument('--user-id', type=str, default='test_user',
                        help='User ID to generate recommendations for')
    parser.add_argument('--n-recs', type=int, default=10,
                        help='Number of recommendations')
    
    args = parser.parse_args()
    
    # Load model
    model_artifacts = model_fn(args.model_dir)
    
    # Create test request
    test_request = {
        'user_id': args.user_id,
        'n_recommendations': args.n_recs,
        'include_scores': True
    }
    
    # Process input
    input_data = input_fn(json.dumps(test_request), 'application/json')
    
    # Generate predictions
    result = predict_fn(input_data, model_artifacts)
    
    # Format output
    output = output_fn(result, 'application/json')
    
    print("\nRecommendation Results:")
    print(json.dumps(json.loads(output), indent=2))
