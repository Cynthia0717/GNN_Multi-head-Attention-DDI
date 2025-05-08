import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GATConv, LayerNorm, BatchNorm
from torch_geometric.utils import dense_to_sparse
from sklearn.neighbors import kneighbors_graph


# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GNN_Multihead_Attention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128, heads=4, graph_type='hybrid', k=5, dropout=0.3):
        """
        基于 KNN 图的 GNN 模型（带注意力机制）

        :param input_size: 输入维度
        :param output_size: 输出维度
        :param hidden_size: 隐藏层维度
        :param heads: 注意力头数
        :param graph_type: 图类型 ['fully_connected', 'knn', 'similarity']
        :param k: KNN 的邻居数量
        :param dropout: Dropout 概率
        """
        super(GNN_Multihead_Attention, self).__init__()

        set_seed()

        self.graph_type = graph_type
        self.k = k
        self.static_edge_index = None

        # GAT 注意力卷积层
        self.conv1 = GATConv(input_size, hidden_size, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_size * heads, hidden_size, heads=heads, dropout=dropout, concat=False)

        # 标准化层
        self.norm1 = BatchNorm(hidden_size * heads)
        self.norm2 = BatchNorm(hidden_size)

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # 自适应激活函数 PReLU
        self.prelu = nn.PReLU()

        # 全连接输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def set_static_graph(self, edge_index):
        """设置静态图"""
        self.static_edge_index = edge_index

    def build_graph(self, x):
        """基于不同策略构建 edge_index"""
        num_nodes = x.size(0)
        device = x.device

        if num_nodes < 2:
            print(f"Warning: Not enough nodes to build a graph (num_nodes={num_nodes}). Skipping this batch.")
            return torch.empty(2, 0, dtype=torch.long, device=device)  # 空的邻接矩阵

        if self.graph_type == 'hybrid':
            # 混合图：结合 KNN 图和余弦相似度图
            x_cpu = x.detach().cpu().numpy()

            # 防止 k 大于节点数量
            k = min(self.k, num_nodes - 1)
            k = max(k, 1)

            # 构建 KNN 图
            knn_adj = kneighbors_graph(x_cpu, k, mode='connectivity', include_self=False)

            # 转换为 PyTorch 格式
            knn_edge_index, _ = dense_to_sparse(torch.tensor(knn_adj.toarray(), dtype=torch.float, device=device))

            # 计算余弦相似度矩阵
            cos_sim = cosine_similarity(x_cpu)
            cos_sim = torch.tensor(cos_sim, dtype=torch.float, device=device)

            # 筛选阈值以上的边（提高稀疏性，防止过拟合）
            threshold = cos_sim.mean()
            cos_edge_index = torch.nonzero(cos_sim > threshold, as_tuple=False).T

            # 合并双图
            combined_edges = torch.cat([knn_edge_index, cos_edge_index], dim=1)
            edge_index = torch.unique(combined_edges, dim=1)

        else:
            raise ValueError("Unsupported graph_type. Choose from ['fully_connected', 'knn', 'similarity']")

        return edge_index.to(device).long()

    def forward(self, x):
        # 使用静态图或构建 KNN 图
        edge_index = self.static_edge_index if self.static_edge_index is not None else self.build_graph(x)

        # 第一层 GAT 卷积 + LayerNorm + Dropout
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = self.prelu(x)
        x = self.dropout(x)

        # 第二层 GAT 卷积 + LayerNorm
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)

        # 输出层
        x = self.fc(x)
        return x
