import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch_geometric.nn import GATConv

from transformers import AutoModel, AutoTokenizer

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.layer_norm(x)
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.conv2 = GATConv(out_channels * heads, out_channels, heads=heads, concat=False)

    def forward(self, x, edge_index, edge_weights):
        x = self.conv1(x, edge_index, edge_attr=edge_weights)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_weights)
        return x
    
    
class CrossAttention(nn.Module):
    def __init__(self, D):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Parameter(torch.randn(D, D))
        self.D = D
    
    def forward(self, x):  # x (B, L, D)
        Q, K, V = torch.matmul(x, self.W_q), x, x,
        attention_scores = torch.matmul(Q, K.transpose(1, 2)) / (2 * self.D) ** 0.5  # (B, L, L)      
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, L, L)
        output = torch.matmul(attention_weights, V)  # (B, L, L) * (B, L, D) -> (B, L, D)
        return output


class EmptyHead(nn.Module):
    """空头，内部没有任何参数，forward 直接返回输入"""
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x          # 相似轨迹搜索直接拿 embedding 去做检索，不需要额外变


class GAT_graph(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT_graph, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


class Sentence_Transformer(nn.Module):

    def __init__(self, pretrained_repo):
        super(Sentence_Transformer, self).__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings
