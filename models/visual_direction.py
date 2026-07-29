import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualDirectionModule(nn.Module):

    def __init__(
        self,
        d_img,
        d_hidden,
        d_model,
        num_heading_bins=16,
        num_nodes=1839,
    ):
        super().__init__()

        # Image feature projection
        self.img_proj = nn.Sequential(
            nn.LayerNorm(d_img),
            nn.Linear(d_img, d_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Node ID embedding
        self.node_feat = nn.Embedding(num_nodes, 64)

        # Node coordinate projection
        self.xy_proj = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_hidden + 64 + 64, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heading_bins),
        )

        self.dir_to_model = nn.Linear(num_heading_bins, d_model)

    def forward(self, image_features, obs_node_ids, obs_xy):
        img_feat = self.img_proj(image_features)
        node_feat = self.node_feat(obs_node_ids)
        xy_feat = self.xy_proj(obs_xy)

        fused = torch.cat([img_feat, node_feat, xy_feat], dim=-1)
        dir_logits = self.fusion(fused)          # [B, N, K]
        dir_prob = F.softmax(dir_logits, dim=-1) # [B, N, K]
        dir_embed = self.dir_to_model(dir_prob)  # [B, N, d_model]

        return dir_logits, dir_prob, dir_embed
