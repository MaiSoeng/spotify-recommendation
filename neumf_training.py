"""
NeuMF (Neural Matrix Factorization) Training Script for SageMaker
Combines GMF (Generalized Matrix Factorization) and MLP paths for music recommendation.

Based on the paper: "Neural Collaborative Filtering" by He et al. (WWW 2017)

Architecture:
    - GMF Path: Element-wise product of user/item embeddings
    - MLP Path: Concatenated embeddings through deep neural network
    - NeuMF: Combines GMF and MLP for final prediction
"""

import argparse
import os
import json
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===============================
# Data Classes
# ===============================

class SpotifyPlaylistDataset(Dataset):
    """Dataset for Spotify playlist interactions with weighted ratings"""
    
    def __init__(self, interactions, num_items, num_negatives=4, rating_lookup=None):
        """
        Args:
            interactions: List of (user_idx, item_idx) positive pairs
            num_items: Total number of items for negative sampling
            num_negatives: Number of negative samples per positive
            rating_lookup: Dict of (user_idx, item_idx) -> rating (optional)
        """
        self.interactions = interactions
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rating_lookup = rating_lookup or {}
        
        # Build user->items mapping for negative sampling
        self.user_items = defaultdict(set)
        for user_idx, item_idx in interactions:
            self.user_items[user_idx].add(item_idx)
        
        # Pre-generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """Generate positive and negative samples with ratings"""
        samples = []
        
        for user_idx, item_idx in self.interactions:
            # Positive sample - use actual rating if available, else 1.0
            rating = self.rating_lookup.get((user_idx, item_idx), 1.0)
            samples.append((user_idx, item_idx, rating))
            
            # Negative samples - always 0.0
            for _ in range(self.num_negatives):
                neg_item = np.random.randint(0, self.num_items)
                while neg_item in self.user_items[user_idx]:
                    neg_item = np.random.randint(0, self.num_items)
                samples.append((user_idx, neg_item, 0.0))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, item_idx, label = self.samples[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(item_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


class EvaluationDataset(Dataset):
    """Dataset for model evaluation"""
    
    def __init__(self, test_interactions, train_user_items, num_items, num_negatives=99):
        """
        Args:
            test_interactions: Test (user_idx, item_idx) pairs
            train_user_items: Dict of user->set of training items
            num_items: Total number of items
            num_negatives: Number of negative samples for evaluation
        """
        self.samples = []
        
        for user_idx, item_idx in test_interactions:
            # Get negative items (not in train or test)
            negatives = []
            while len(negatives) < num_negatives:
                neg_item = np.random.randint(0, num_items)
                if neg_item not in train_user_items.get(user_idx, set()) and neg_item != item_idx:
                    negatives.append(neg_item)
            
            self.samples.append((user_idx, item_idx, negatives))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, pos_item, neg_items = self.samples[idx]
        return user_idx, pos_item, neg_items


# ===============================
# NeuMF Model
# ===============================

class GMF(nn.Module):
    """Generalized Matrix Factorization"""
    
    def __init__(self, num_users, num_items, embedding_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        return user_emb * item_emb  # Element-wise product


class MLP(nn.Module):
    """Multi-Layer Perceptron path"""
    
    def __init__(self, num_users, num_items, embedding_dim, layers=[128, 64, 32], dropout=0.2):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # MLP layers
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
    """
    Neural Matrix Factorization
    Combines GMF and MLP paths for collaborative filtering
    """
    
    def __init__(
        self,
        num_users,
        num_items,
        gmf_embedding_dim=32,
        mlp_embedding_dim=32,
        mlp_layers=[128, 64, 32],
        dropout=0.2
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # GMF path
        self.gmf = GMF(num_users, num_items, gmf_embedding_dim)
        
        # MLP path
        self.mlp = MLP(num_users, num_items, mlp_embedding_dim, mlp_layers, dropout)
        
        # Final prediction layer
        final_input_dim = gmf_embedding_dim + self.mlp.output_dim
        self.predict_layer = nn.Linear(final_input_dim, 1)
        
        # Initialize prediction layer
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')
    
    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_output = self.gmf(user_ids, item_ids)
        
        # MLP path
        mlp_output = self.mlp(user_ids, item_ids)
        
        # Concatenate and predict
        concat = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = torch.sigmoid(self.predict_layer(concat))
        
        return prediction.squeeze()
    
    def get_user_embedding(self, user_ids):
        """Get combined user embedding for inference"""
        gmf_emb = self.gmf.user_embedding(user_ids)
        mlp_emb = self.mlp.user_embedding(user_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)
    
    def get_item_embedding(self, item_ids):
        """Get combined item embedding for inference"""
        gmf_emb = self.gmf.item_embedding(item_ids)
        mlp_emb = self.mlp.item_embedding(item_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=-1)


# ===============================
# Data Loading & Processing
# ===============================

def load_spotify_data(data_path):
    """Load and process Spotify Million Playlist Dataset format"""
    logger.info(f"Loading data from {data_path}")
    
    try:
        # Try to load preprocessed parquet
        if os.path.isdir(data_path):
            df = pd.read_parquet(data_path)
        else:
            # Try JSON format (Spotify MPD format)
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Extract playlist-track interactions
            interactions = []
            for playlist in data.get('playlists', []):
                pid = playlist['pid']
                for track in playlist.get('tracks', []):
                    interactions.append({
                        'playlist_id': pid,
                        'track_uri': track['track_uri'],
                        'track_name': track.get('track_name'),
                        'artist_name': track.get('artist_name'),
                        'position': track.get('pos', 0)
                    })
            
            df = pd.DataFrame(interactions)
        
        logger.info(f"Loaded {len(df)} interactions")
        return df
        
    except Exception as e:
        logger.warning(f"Could not load data: {e}")
        logger.info("Generating sample data for demonstration")
        return generate_sample_data()


def generate_sample_data(n_users=5000, n_items=10000, n_interactions=100000):
    """Generate sample interaction data for testing"""
    logger.info(f"Generating sample data: {n_users} users, {n_items} items, {n_interactions} interactions")
    
    np.random.seed(42)
    
    # Power-law distribution for realistic item popularity
    item_popularity = np.random.power(0.5, n_items)
    item_popularity = item_popularity / item_popularity.sum()
    
    data = {
        'playlist_id': np.random.randint(0, n_users, n_interactions),
        'track_uri': np.random.choice(n_items, n_interactions, p=item_popularity),
        'track_name': [f'Track_{i}' for i in np.random.randint(0, n_items, n_interactions)],
        'artist_name': [f'Artist_{i % 500}' for i in range(n_interactions)],
        'position': np.random.randint(0, 50, n_interactions)
    }
    
    return pd.DataFrame(data)


def prepare_data(df, test_ratio=0.2):
    """
    Prepare data for training and evaluation.
    
    Key improvement: Aggregates repeated interactions into play counts,
    then converts to implicit ratings using log transform.
    This preserves the signal from users who play a song multiple times.
    """
    logger.info("Preparing data for training...")
    
    # Determine column names
    if 'playlist_id' in df.columns:
        user_col = 'playlist_id'
    elif 'user_idx' in df.columns:
        user_col = 'user_idx'
    else:
        user_col = df.columns[0]
    
    if 'track_uri' in df.columns:
        item_col = 'track_uri'
    elif 'item_idx' in df.columns:
        item_col = 'item_idx'
    else:
        item_col = df.columns[1]
    
    logger.info(f"Using columns: user={user_col}, item={item_col}")
    
    # ===============================
    # Step 1: Aggregate Interactions
    # ===============================
    # Count how many times each user interacted with each item
    interaction_counts = df.groupby([user_col, item_col]).size().reset_index(name='play_count')
    
    logger.info(f"Raw interactions: {len(df)}")
    logger.info(f"After aggregation: {len(interaction_counts)} unique user-item pairs")
    logger.info(f"Compression ratio: {len(df)/len(interaction_counts):.2f}x")
    
    # Play count statistics
    logger.info(f"Play count stats: mean={interaction_counts['play_count'].mean():.2f}, "
                f"max={interaction_counts['play_count'].max()}, "
                f"median={interaction_counts['play_count'].median():.0f}")
    
    # ===============================
    # Step 2: Compute Implicit Ratings
    # ===============================
    # Use log transform: rating = log(1 + play_count)
    # This prevents power users from dominating the signal
    interaction_counts['rating'] = np.log1p(interaction_counts['play_count'])
    
    # Normalize to 0-1 range for training stability
    max_rating = interaction_counts['rating'].max()
    interaction_counts['normalized_rating'] = interaction_counts['rating'] / max_rating
    
    logger.info(f"Rating range: 0.0 to 1.0 (normalized from log scale)")
    
    # ===============================
    # Step 3: Create Encodings
    # ===============================
    unique_users = interaction_counts[user_col].unique()
    unique_items = interaction_counts[item_col].unique()
    
    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    item_to_idx = {t: i for i, t in enumerate(unique_items)}
    idx_to_item = {i: t for t, i in item_to_idx.items()}
    
    num_users = len(unique_users)
    num_items = len(unique_items)
    
    logger.info(f"Unique users: {num_users}, Unique items: {num_items}")
    
    # Add encoded indices
    interaction_counts['user_idx'] = interaction_counts[user_col].map(user_to_idx)
    interaction_counts['item_idx'] = interaction_counts[item_col].map(item_to_idx)
    
    # ===============================
    # Step 4: Create Train/Test Split
    # ===============================
    # Create interaction tuples with ratings
    interactions_with_ratings = [
        (row['user_idx'], row['item_idx'], row['normalized_rating'])
        for _, row in interaction_counts.iterrows()
    ]
    
    # Split into train/test
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(
        interactions_with_ratings,
        test_size=test_ratio,
        random_state=42
    )
    
    # Extract just (user, item) pairs for compatibility with existing code
    train_interactions = [(u, i) for u, i, r in train_data]
    test_interactions = [(u, i) for u, i, r in test_data]
    
    # Build rating lookup for weighted training
    rating_lookup = {(u, i): r for u, i, r in interactions_with_ratings}
    
    logger.info(f"Train: {len(train_interactions)}, Test: {len(test_interactions)}")
    
    # Build user->items mapping for training set
    train_user_items = defaultdict(set)
    for user_idx, item_idx in train_interactions:
        train_user_items[user_idx].add(item_idx)
    
    return {
        'train_interactions': train_interactions,
        'test_interactions': test_interactions,
        'num_users': num_users,
        'num_items': num_items,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'train_user_items': train_user_items,
        'rating_lookup': rating_lookup  # NEW: for weighted training
    }


# ===============================
# Training & Evaluation
# ===============================

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for user_ids, item_ids, labels in train_loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, eval_dataset, device, k_values=[5, 10, 20]):
    """Evaluate model using Hit Rate and NDCG"""
    model.eval()
    
    metrics = {f'HR@{k}': [] for k in k_values}
    metrics.update({f'NDCG@{k}': [] for k in k_values})
    
    with torch.no_grad():
        for user_idx, pos_item, neg_items in eval_dataset:
            # Score positive item and negatives
            items = [pos_item] + list(neg_items)
            user_tensor = torch.tensor([user_idx] * len(items), dtype=torch.long).to(device)
            item_tensor = torch.tensor(items, dtype=torch.long).to(device)
            
            scores = model(user_tensor, item_tensor).cpu().numpy()
            
            # Rank items
            ranked_indices = np.argsort(-scores)
            pos_rank = np.where(ranked_indices == 0)[0][0] + 1  # 1-indexed rank
            
            # Calculate metrics
            for k in k_values:
                # Hit Rate
                hit = 1.0 if pos_rank <= k else 0.0
                metrics[f'HR@{k}'].append(hit)
                
                # NDCG
                if pos_rank <= k:
                    ndcg = 1.0 / np.log2(pos_rank + 1)
                else:
                    ndcg = 0.0
                metrics[f'NDCG@{k}'].append(ndcg)
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}


def train_model(args, data_dict):
    """Main training loop"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = SpotifyPlaylistDataset(
        data_dict['train_interactions'],
        data_dict['num_items'],
        num_negatives=args.num_negatives,
        rating_lookup=data_dict.get('rating_lookup')  # Use weighted ratings
    )
    
    eval_dataset = EvaluationDataset(
        data_dict['test_interactions'][:1000],  # Sample for faster evaluation
        data_dict['train_user_items'],
        data_dict['num_items']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Create model
    model = NeuMF(
        num_users=data_dict['num_users'],
        num_items=data_dict['num_items'],
        gmf_embedding_dim=args.gmf_embedding_dim,
        mlp_embedding_dim=args.mlp_embedding_dim,
        mlp_layers=[int(x) for x in args.mlp_layers.split(',')],
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    criterion = nn.BCELoss()
    
    # Training loop
    best_hr = 0
    best_epoch = 0
    training_history = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        metrics = evaluate(model, eval_dataset, device)
        
        epoch_time = time.time() - start_time
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"Loss: {train_loss:.4f} - "
            f"HR@10: {metrics['HR@10']:.4f} - "
            f"NDCG@10: {metrics['NDCG@10']:.4f} - "
            f"Time: {epoch_time:.1f}s"
        )
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics,
            'epoch_time': epoch_time
        })
        
        # Learning rate scheduling
        scheduler.step(metrics['HR@10'])
        
        # Save best model
        if metrics['HR@10'] > best_hr:
            best_hr = metrics['HR@10']
            best_epoch = epoch + 1
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'args': vars(args)
            }, os.path.join(args.model_dir, 'best_model.pth'))
        
        # Early stopping
        if epoch - best_epoch >= args.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    logger.info(f"Best HR@10: {best_hr:.4f} at epoch {best_epoch}")
    
    return model, training_history, {'best_hr@10': best_hr, 'best_epoch': best_epoch}


def save_model_artifacts(model, data_dict, args, training_history, final_metrics):
    """Save model and related artifacts for deployment"""
    logger.info(f"Saving model artifacts to {args.model_dir}")
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Save model for inference (TorchScript)
    model.eval()
    scripted_model = torch.jit.script(model)
    scripted_model.save(os.path.join(args.model_dir, 'model.pt'))
    
    # Save model state dict (backup)
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'model_state.pth'))
    
    # Save encoders and metadata
    metadata = {
        'num_users': data_dict['num_users'],
        'num_items': data_dict['num_items'],
        'gmf_embedding_dim': args.gmf_embedding_dim,
        'mlp_embedding_dim': args.mlp_embedding_dim,
        'mlp_layers': args.mlp_layers,
        'training_date': datetime.now().isoformat(),
        'final_metrics': final_metrics
    }
    
    with open(os.path.join(args.model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save encoders
    with open(os.path.join(args.model_dir, 'user_to_idx.json'), 'w') as f:
        json.dump({str(k): v for k, v in data_dict['user_to_idx'].items()}, f)
    
    with open(os.path.join(args.model_dir, 'item_to_idx.json'), 'w') as f:
        json.dump({str(k): v for k, v in data_dict['item_to_idx'].items()}, f)
    
    with open(os.path.join(args.model_dir, 'idx_to_item.json'), 'w') as f:
        json.dump({str(k): v for k, v in data_dict['idx_to_item'].items()}, f)
    
    # Save training history
    with open(os.path.join(args.model_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Model artifacts saved successfully")

    # ===============================
    # Analytics / Visualization Data
    # ===============================
    output_data_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Save Training Metrics for QuickSight
    # Flat JSON structure is easier for QuickSight/Athena
    metrics_flat = []
    for epoch_data in training_history:
        metrics_flat.append({
            'epoch': epoch_data['epoch'],
            'loss': epoch_data['train_loss'],
            'hr_10': epoch_data['HR@10'],
            'ndcg_10': epoch_data['NDCG@10'],
            'timestamp': datetime.now().isoformat()
        })
    
    with open(os.path.join(output_data_dir, 'training_metrics.json'), 'w') as f:
        # Save as newline-delimited JSON for Athena
        for row in metrics_flat:
            f.write(json.dumps(row) + '\n')

    # Generate Batch Predictions (Snapshot)
    # Predict Top-10 tracks for a sample of users
    logger.info("Generating sample predictions for visualization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    sample_users = list(data_dict['user_to_idx'].keys())[:100] # First 100 users
    num_items = data_dict['num_items']
    all_items_tensor = torch.tensor(list(range(num_items)), dtype=torch.long).to(device)
    idx_to_item = data_dict['idx_to_item']
    
    predictions_export = []
    
    with torch.no_grad():
        for user_id in sample_users:
            user_idx = data_dict['user_to_idx'][user_id]
            user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long).to(device)
            
            scores = model(user_tensor, all_items_tensor).cpu().numpy()
            top_indices = np.argsort(-scores)[:5] # Top 5
            
            for rank, item_idx in enumerate(top_indices):
                item_name = idx_to_item[item_idx]
                predictions_export.append({
                    'user_id': user_id,
                    'track': item_name,
                    'rank': rank + 1,
                    'score': float(scores[item_idx]),
                    'prediction_date': datetime.now().date().isoformat()
                })
                
    with open(os.path.join(output_data_dir, 'sample_predictions.json'), 'w') as f:
        for row in predictions_export:
            f.write(json.dumps(row) + '\n')
            
    logger.info(f"Saved {len(predictions_export)} predictions to {output_data_dir}")


# ===============================
# SageMaker Inference Functions
# ===============================

def model_fn(model_dir):
    """Load model for SageMaker inference"""
    logger.info(f"Loading model from {model_dir}")
    
    # Load metadata
    with open(os.path.join(model_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load encoders
    with open(os.path.join(model_dir, 'item_to_idx.json'), 'r') as f:
        item_to_idx = json.load(f)
    
    with open(os.path.join(model_dir, 'idx_to_item.json'), 'r') as f:
        idx_to_item = json.load(f)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(os.path.join(model_dir, 'model.pt'), map_location=device)
    model.eval()
    
    return {
        'model': model,
        'metadata': metadata,
        'item_to_idx': item_to_idx,
        'idx_to_item': idx_to_item,
        'device': device
    }


def input_fn(request_body, request_content_type):
    """Parse inference request"""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    """Generate recommendations"""
    model = model_artifacts['model']
    device = model_artifacts['device']
    idx_to_item = model_artifacts['idx_to_item']
    num_items = model_artifacts['metadata']['num_items']
    
    user_id = input_data.get('user_id')
    user_idx = input_data.get('user_idx', 0)  # Default to 0 for new users
    n_recommendations = input_data.get('n_recommendations', 10)
    exclude_items = set(input_data.get('exclude_items', []))
    
    # Score all items for this user
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * num_items, dtype=torch.long).to(device)
        item_tensor = torch.tensor(list(range(num_items)), dtype=torch.long).to(device)
        
        scores = model(user_tensor, item_tensor).cpu().numpy()
    
    # Rank and filter
    ranked_indices = np.argsort(-scores)
    
    recommendations = []
    for idx in ranked_indices:
        item_id = idx_to_item.get(str(idx), f"item_{idx}")
        if item_id not in exclude_items:
            recommendations.append({
                'track_id': item_id,
                'score': float(scores[idx]),
                'rank': len(recommendations) + 1
            })
        if len(recommendations) >= n_recommendations:
            break
    
    return {
        'user_id': user_id,
        'recommendations': recommendations
    }


def output_fn(prediction, response_content_type):
    """Format response"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")


# ===============================
# Main
# ===============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuMF Training for Music Recommendation')
    
    # Model hyperparameters
    parser.add_argument('--gmf-embedding-dim', type=int, default=32,
                        help='GMF embedding dimension')
    parser.add_argument('--mlp-embedding-dim', type=int, default=32,
                        help='MLP embedding dimension')
    parser.add_argument('--mlp-layers', type=str, default='128,64,32',
                        help='MLP layer sizes (comma-separated)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--num-negatives', type=int, default=4,
                        help='Number of negative samples per positive')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio')
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--output-data-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    args = parser.parse_args()
    
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load data
    try:
        # Load data directly (supports directory/partitioned parquet)
        data_path = args.train
        df = load_spotify_data(data_path)
    except Exception as e:
        logger.warning(f"Could not load data from {args.train}: {e}")
        df = generate_sample_data()
    
    # Prepare data
    data_dict = prepare_data(df, test_ratio=args.test_ratio)
    
    # Train model
    model, training_history, final_metrics = train_model(args, data_dict)
    
    # Save artifacts
    save_model_artifacts(model, data_dict, args, training_history, final_metrics)
    
    # Save final metrics for SageMaker
    os.makedirs(args.output_data_dir, exist_ok=True)
    with open(os.path.join(args.output_data_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info("Training completed successfully!")
