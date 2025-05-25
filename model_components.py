import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoActivityEmbedding(nn.Module):
    """Learnable no-activity token for days with no CP trading"""

    def __init__(self, embed_dim):
        super().__init__()
        self.no_activity_token = nn.Parameter(torch.randn(embed_dim))

    def forward(self):
        return self.no_activity_token


class CPEmbeddingMLP(nn.Module):
    """Shared MLP for CP feature embedding with no-activity token"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # <--- Additional layer
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # 🔥 NEW: Learnable no-activity embedding
        self.no_activity_embedding = NoActivityEmbedding(output_dim)

    def forward(self, x):
        return self.mlp(x)

    def get_no_activity_embedding(self, batch_size, device):
        """Return no-activity embedding for batch"""
        return self.no_activity_embedding().unsqueeze(0).expand(batch_size, -1).to(device)

class HistoricalAttention(nn.Module):
    """Attention mechanism for aggregating historical CP embeddings"""

    def __init__(self, embed_dim, no_activity_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        no_activity_dim = no_activity_dim or embed_dim

        # Query, Key, Value projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Learnable no-activity token
        self.no_activity_embedding = nn.Parameter(torch.randn(1, no_activity_dim))

        # Output projection
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional mask of shape [batch_size, seq_len]
        """
        batch_size = embeddings.shape[0]

        # Handle empty sequences with no-activity token
        if embeddings.shape[1] == 0:
            return self.no_activity_embedding.expand(batch_size, -1)

        # Multi-head attention
        q = self.query(embeddings)
        k = self.key(embeddings)
        v = self.value(embeddings)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)

        # Apply mask if provided
        if mask is not None:
            # Convert to boolean mask
            bool_mask = mask.bool()
            scores = scores.masked_fill(bool_mask.unsqueeze(1) == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum
        context = torch.matmul(attn_weights, v)

        # Mean pooling over sequence dimension
        context = context.mean(dim=1)

        # Output projection
        output = self.output_projection(context)

        return output


class TransformerEncoder(nn.Module):
    """Transformer encoder for processing daily features"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask of shape [batch_size, seq_len]
        """
        if mask is not None:
            # Convert to boolean mask and invert for transformer (True = masked/padded)
            bool_mask = mask.bool()
            padding_mask = ~bool_mask  # Now this works since bool_mask is boolean
        else:
            padding_mask = None

        output = self.transformer(x, src_key_padding_mask=padding_mask)

        # Global average pooling with mask
        if mask is not None:
            # Apply mask before averaging
            bool_mask = mask.bool()
            mask_expanded = bool_mask.unsqueeze(-1).expand_as(output)
            masked_output = output * mask_expanded.float()
            lengths = bool_mask.sum(dim=1, keepdim=True).float()
            # Avoid division by zero
            lengths = torch.clamp(lengths, min=1.0)
            output = masked_output.sum(dim=1) / lengths
        else:
            output = output.mean(dim=1)

        return output


class FusionModule(nn.Module):
    """Fusion module for combining stock and CP embeddings"""

    def __init__(self, stock_dim, cp_dim, output_dim=128):
        super().__init__()

        self.stock_projection = nn.Linear(stock_dim, output_dim)
        self.cp_projection = nn.Linear(cp_dim, output_dim)

        # Gating mechanism
        self.gate = nn.Linear(stock_dim + cp_dim, output_dim)

    def forward(self, stock_embedding, cp_embedding):
        """
        Args:
            stock_embedding: Tensor of shape [batch_size, stock_dim]
            cp_embedding: Tensor of shape [batch_size, cp_dim]
        """
        # Project both embeddings to the same dimension
        stock_proj = self.stock_projection(stock_embedding)
        cp_proj = self.cp_projection(cp_embedding)

        # Compute gate values
        gate_input = torch.cat([stock_embedding, cp_embedding], dim=1)
        gate = torch.sigmoid(self.gate(gate_input))

        # Gated combination
        output = gate * stock_proj + (1 - gate) * cp_proj

        return output


class PredictionHead(nn.Module):
    """Final MLP for prediction"""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),  # ✅ Fixed
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),

            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, x):
        return torch.sigmoid(self.mlp(x).squeeze(-1))
