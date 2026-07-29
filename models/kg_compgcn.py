import torch
import torch.nn as nn
import torch.nn.functional as F


class CompGCNConv(nn.Module):

    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W_self = nn.Linear(in_channels, out_channels, bias=False)
        self.W_fwd = nn.Linear(in_channels, out_channels, bias=False)
        self.W_inv = nn.Linear(in_channels, out_channels, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        for w in [self.W_self, self.W_fwd, self.W_inv]:
            nn.init.xavier_uniform_(w.weight)

    def forward(self, x, edge_index, edge_type, rel_emb):
        row, col = edge_index  
        num_nodes = x.size(0)
        dim = x.size(1)
       
        rel = rel_emb[edge_type]          
        composed = x[col] - rel            
      
        out = self.W_self(x)              

        num_rel = rel_emb.size(0) // 2
        is_fwd = edge_type < num_rel

        if is_fwd.any():
            fwd_msg = self.W_fwd(composed[is_fwd])  # [E_fwd, out_channels]
            out = out.index_add(0, row[is_fwd], fwd_msg)

        if (~is_fwd).any():
            inv_msg = self.W_inv(composed[~is_fwd])  # [E_inv, out_channels]
            out = out.index_add(0, row[~is_fwd], inv_msg)

        return out


class KGCompGCN(nn.Module):

    def __init__(self, num_entities, num_rel_types, kg_dim=128):
        super().__init__()

        self.num_entities = num_entities
        self.num_rel_types = num_rel_types

        self.entity_emb = nn.Embedding(num_entities, kg_dim)
        self.rel_emb = nn.Embedding(num_rel_types * 2, kg_dim)

        self.conv1 = CompGCNConv(kg_dim, kg_dim, num_rel_types)
        self.conv2 = CompGCNConv(kg_dim, kg_dim, num_rel_types)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, edge_index, edge_type):
        x = self.entity_emb.weight       
        r = self.rel_emb.weight          
        x = self.conv1(x, edge_index, edge_type, r)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type, r)

        return x  

    def extra_repr(self):
        return (
            f"num_entities={self.num_entities}, "
            f"num_rel_types={self.num_rel_types}"
        )