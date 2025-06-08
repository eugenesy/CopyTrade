import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import CPEmbeddingMLP, TransformerEncoder, FusionModule, PredictionHead


class PerDayCopyTradingModel(nn.Module):
    def __init__(
            self,
            cp_input_dim=387,
            cp_hidden_dim=128,
            cp_embed_dim=128,
            stock_feature_dim=5,
            daily_token_dim=392,
            transformer_hidden_dim=256,
            transformer_layers=3,
            transformer_heads=8,
            fusion_dim=128,
            prediction_hidden_dim=128,
            dropout=0.2,
            ablation_mode='full'
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.cp_embed_dim = cp_embed_dim

        # CP embedding for target features (always needed)
        self.cp_embedding = CPEmbeddingMLP(
            input_dim=cp_input_dim,
            hidden_dim=cp_hidden_dim,
            output_dim=cp_embed_dim,
            dropout=dropout
        )

        if ablation_mode == 'target_cp_only':
            # No Transformer or Fusion needed, direct prediction from target CP embedding
            self.prediction_head = PredictionHead(
                input_dim=cp_embed_dim,
                hidden_dim=prediction_hidden_dim,
                dropout=dropout
            )
        elif ablation_mode == 'historical_only':
            # Only historical data, no target CP fusion
            self.transformer = TransformerEncoder(
                input_dim=daily_token_dim,
                hidden_dim=transformer_hidden_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=dropout
            )
            self.prediction_head = PredictionHead(
                input_dim=daily_token_dim,
                hidden_dim=prediction_hidden_dim,
                dropout=dropout
            )
        else:
            # Full mode with both historical and target CP data
            self.transformer = TransformerEncoder(
                input_dim=daily_token_dim,
                hidden_dim=transformer_hidden_dim,
                num_layers=transformer_layers,
                num_heads=transformer_heads,
                dropout=dropout
            )
            self.fusion = FusionModule(
                stock_dim=daily_token_dim,
                cp_dim=cp_embed_dim,
                output_dim=fusion_dim
            )
            self.prediction_head = PredictionHead(
                input_dim=fusion_dim,
                hidden_dim=prediction_hidden_dim,
                dropout=dropout
            )

    def forward(self, batch):
        batch_size = batch['target_features'].shape[0]
        device = batch['target_features'].device

        if self.ablation_mode == 'target_cp_only':
            # Process only target CP embedding, no historical data
            target_cp_embedding = self.cp_embedding(batch['target_features'])
            prediction = self.prediction_head(target_cp_embedding)

        elif self.ablation_mode == 'historical_only':
            # Process only historical data, no target CP fusion
            daily_sequences = batch['daily_sequences']
            if daily_sequences:
                max_len = max(seq.shape[0] for seq in daily_sequences)
                padded_sequences = []
                sequence_masks = []
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
                        padded_seq = torch.zeros((max_len, 392), device=device)
                        mask = torch.zeros(max_len, device=device)
                    padded_sequences.append(padded_seq)
                    sequence_masks.append(mask)
                padded_sequences = torch.stack(padded_sequences)
                padded_masks = torch.stack(sequence_masks)
            else:
                padded_sequences = torch.zeros((batch_size, 1, 392), device=device)
                padded_masks = torch.zeros((batch_size, 1), device=device)

            historical_embedding = self.transformer(padded_sequences, padded_masks)
            prediction = self.prediction_head(historical_embedding)

        else:
            # Full mode with both historical and target CP data
            target_cp_embedding = self.cp_embedding(batch['target_features'])
            daily_sequences = batch['daily_sequences']
            if daily_sequences:
                max_len = max(seq.shape[0] for seq in daily_sequences)
                padded_sequences = []
                sequence_masks = []
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
                        padded_seq = torch.zeros((max_len, 392), device=device)
                        mask = torch.zeros(max_len, device=device)
                    padded_sequences.append(padded_seq)
                    sequence_masks.append(mask)
                padded_sequences = torch.stack(padded_sequences)
                padded_masks = torch.stack(sequence_masks)
            else:
                padded_sequences = torch.zeros((batch_size, 1, 392), device=device)
                padded_masks = torch.zeros((batch_size, 1), device=device)

            stock_embedding = self.transformer(padded_sequences, padded_masks)
            fused_embedding = self.fusion(stock_embedding, target_cp_embedding)
            prediction = self.prediction_head(fused_embedding)

        return prediction
