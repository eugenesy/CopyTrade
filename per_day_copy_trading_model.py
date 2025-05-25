import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import (
    CPEmbeddingMLP, HistoricalAttention,
    TransformerEncoder, FusionModule, PredictionHead
)


class PerDayCopyTradingModel(nn.Module):
    def __init__(
        self,
        cp_input_dim=387,    # 🔥 CHANGED: 384 + 2 + 1
        cp_hidden_dim=128,
        cp_embed_dim=128,    # 🔥 INCREASED for better capacity
        stock_feature_dim=5,
        daily_token_dim=392, # 🔥 CHANGED: 5 + 387
        transformer_hidden_dim=256,  # 🔥 INCREASED
        transformer_layers=3,        # 🔥 INCREASED
        transformer_heads=8,         # 🔥 INCREASED
        fusion_dim=128,
        prediction_hidden_dim=128,
        dropout=0.2          # 🔥 INCREASED
    ):

        super().__init__()

        # CP embedding MLP (shared for target and historical)
        self.cp_embedding = CPEmbeddingMLP(
            input_dim=cp_input_dim,
            hidden_dim=cp_hidden_dim,
            output_dim=cp_embed_dim,
            dropout=dropout
        )

        # 🔥 NEW: Store dimensions for no-activity token
        self.cp_embed_dim = cp_embed_dim

        # Transformer encoder for daily sequences x_t = [s_t || α(C_t)]
        self.transformer = TransformerEncoder(
            input_dim=daily_token_dim,  # Process full daily tokens directly
            hidden_dim=transformer_hidden_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout
        )

        # Fusion module
        self.fusion = FusionModule(
            stock_dim=daily_token_dim,  # Output from transformer
            cp_dim=cp_embed_dim,  # Target CP embedding
            output_dim=fusion_dim
        )

        # Prediction head
        self.prediction_head = PredictionHead(
            input_dim=fusion_dim,
            hidden_dim=prediction_hidden_dim,
            dropout=dropout
        )

    def forward(self, batch):
        """Your new forward method goes here - REPLACE the existing one"""
        batch_size = batch['target_features'].shape[0]
        device = batch['target_features'].device

        # Process target CP features
        target_cp_embedding = self.cp_embedding(batch['target_features'])

        # Process daily sequences with no-activity token replacement
        daily_sequences = batch['daily_sequences']
        no_activity_masks = batch.get('no_activity_mask', [])

        if daily_sequences:
            # Replace no-activity days with learnable token
            for i, (seq, mask) in enumerate(zip(daily_sequences, no_activity_masks)):
                if seq.shape[0] > 0 and mask.shape[0] > 0:
                    # Get no-activity embedding
                    no_activity_embed = self.cp_embedding.get_no_activity_embedding(1, device).squeeze(0)

                    # Replace CP activity part for no-activity days
                    for day_idx in range(seq.shape[0]):
                        if mask[day_idx]:  # No activity on this day
                            # Replace CP activity portion (last cp_embed_dim elements)
                            seq[day_idx, -self.cp_embed_dim:] = no_activity_embed

        # Continue with existing logic for padding and transformer processing
        if daily_sequences:
            max_len = max(seq.shape[0] for seq in daily_sequences)
            sequence_masks = []
            padded_sequences = []

            for seq in daily_sequences:
                if seq.shape[0] > 0:
                    pad_len = max_len - seq.shape[0]
                    if pad_len > 0:
                        padding = torch.zeros((pad_len, seq.shape[1]), device=device)
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq
                    mask = torch.cat([
                        torch.ones(seq.shape[0], device=device),
                        torch.zeros(pad_len, device=device)
                    ])
                else:
                    padded_seq = torch.zeros((max_len, 393), device=device)
                    mask = torch.zeros(max_len, device=device)

                padded_sequences.append(padded_seq)
                sequence_masks.append(mask)

            padded_sequences = torch.stack(padded_sequences)
            padded_masks = torch.stack(sequence_masks)
        else:
            padded_sequences = torch.zeros((batch_size, 1, 393), device=device)
            padded_masks = torch.zeros((batch_size, 1), device=device)

        # Apply transformer to get stock embedding z
        stock_embedding = self.transformer(padded_sequences, padded_masks)

        # Fusion: h = γ([z || e^target])
        fused_embedding = self.fusion(stock_embedding, target_cp_embedding)

        # Final prediction: ŷ = σ(ω(h))
        prediction = self.prediction_head(fused_embedding)

        return prediction
