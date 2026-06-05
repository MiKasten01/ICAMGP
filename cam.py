import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

class GPT2Block(nn.Module):
    """
    Standard GPT-2 Block: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, 
                                          dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, return_weights=False):
        # 1. Self-Attention
        residual = x
        x = self.ln1(x)
        attn_output, attn_weights = self.attn(x, x, x, 
                                              key_padding_mask=key_padding_mask, 
                                              need_weights=return_weights)
        x = residual + self.dropout(attn_output)
        
        # 2. Feed Forward
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        
        if return_weights:
            return x, attn_weights
        return x

class ClusterSparseAttentionModel(nn.Module):
    """
    Optimized version of ClusterAttentionModel using batched (sparse) attention.
    Both Local and Global attention use the same GPT-2 style block structure.
    
    Args:
        feature_class (List[int] | None): List of cluster IDs for each feature.
        hidden_dim (int): Hidden dimension size.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        output_dim (int): Output dimension size.
        ablation_type (str | None): Ablation study type.
            - 'woLocal': Skip local (intra-cluster) attention.
            - 'woGlobal': Skip global (inter-cluster) attention.
            - 'woCluster': Treat all features as a single cluster (Global Self-Attention).
    """
    def __init__(self, feature_class: list | None, hidden_dim: int = 8, num_heads: int = 1, 
                 dropout: float = 0.1, output_dim: int = 1, ablation_type: str | None = None,
                 max_cluster: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feature_class = feature_class
        self.ablation_type = ablation_type
        
        # 1. Analyze clusters
        if feature_class is None:
            feature_class = [0] * 1 
            self.num_features = 1
        else:
            self.num_features = len(feature_class)

        # Handle 'woCluster' ablation: Force all features into a single cluster (ID 0)
        if self.ablation_type == 'woCluster':
            self.class_to_indices = {0: list(range(self.num_features))}
        else:
            self.class_to_indices = defaultdict(list)
            for idx, cls in enumerate(feature_class):
                self.class_to_indices[int(cls)].append(idx)
        
        # Truncate clusters exceeding max_cluster
        if max_cluster > 0:
            for cls in list(self.class_to_indices.keys()):
                if len(self.class_to_indices[cls]) > max_cluster:
                    self.class_to_indices[cls] = self.class_to_indices[cls][:max_cluster]
            
        self.clusters = sorted(self.class_to_indices.keys())
        self.num_clusters = len(self.clusters)
        
        # 2. Build padding/gather maps
        max_len = 0
        for cls in self.clusters:
            max_len = max(max_len, len(self.class_to_indices[cls]))
        self.max_len = max_len
        
        indices_map = torch.full((self.num_clusters, max_len), self.num_features, dtype=torch.long)
        padding_mask = torch.ones((self.num_clusters, max_len), dtype=torch.bool)
        
        for i, cls in enumerate(self.clusters):
            idxs = self.class_to_indices[cls]
            length = len(idxs)
            indices_map[i, :length] = torch.tensor(idxs, dtype=torch.long)
            padding_mask[i, :length] = False
            
        self.register_buffer('indices_map', indices_map)
        self.register_buffer('padding_mask', padding_mask)
        
        # 3. Feature Embeddings
        self.feature_embeddings = nn.Parameter(torch.randn(self.num_features + 1, hidden_dim))
        nn.init.xavier_uniform_(self.feature_embeddings)
        with torch.no_grad():
            self.feature_embeddings[self.num_features].fill_(0)

        # 4. Unified Attention Blocks
        if self.ablation_type != 'woLocal':
            self.local_block = GPT2Block(hidden_dim, num_heads, dropout)
        
        if self.ablation_type != 'woGlobal':
            self.global_block = GPT2Block(hidden_dim, num_heads, dropout)
        
        # 5. Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.num_clusters, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        前向传播过程
        输入: x: [B, F]
        输出: output: [B, output_dim]
        """
        B = x.size(0)
        device = x.device
        
        # === 1. Gather Features ===
        pad_col = torch.zeros(B, 1, device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad_col], dim=1)
        
        flat_indices = self.indices_map.view(-1) # [C*L]
        gathered_values = x_pad[:, flat_indices].view(B, self.num_clusters, self.max_len) # [B, C, L]
        
        gathered_weights = self.feature_embeddings[flat_indices].view(self.num_clusters, self.max_len, self.hidden_dim) # [C, L, H]
        
        # tokens: [B, C, L, H]
        tokens = gathered_values.unsqueeze(-1) * gathered_weights.unsqueeze(0)
        
        # === 2. Local Attention (Batched) ===
        # Flatten to [B*C, L, H] for batched attention
        tokens_flat = tokens.view(B * self.num_clusters, self.max_len, self.hidden_dim)
        mask_flat = self.padding_mask.repeat(B, 1) # [B*C, L]
        
        local_weights = None
        # Logic for 'woCluster': Treat as single large local block
        if self.ablation_type == 'woCluster' or self.ablation_type != 'woLocal':
            # [Optimization] Chunked processing to avoid OOM
            # If B*C is very large, splitting into chunks reduces peak memory of Attention Matrix (N_chunk, L, L)
            chunk_size = 4096 
            total_items = tokens_flat.size(0)
            
            if total_items <= chunk_size:
                 # Standard path
                if return_attn:
                    tokens_flat, local_weights = self.local_block(tokens_flat, key_padding_mask=mask_flat, return_weights=True)
                else:
                    tokens_flat = self.local_block(tokens_flat, key_padding_mask=mask_flat)
            else:
                # Chunked path
                out_chunks = []
                weight_chunks = []
                
                for i in range(0, total_items, chunk_size):
                    end = min(i + chunk_size, total_items)
                    t_chunk = tokens_flat[i:end]
                    m_chunk = mask_flat[i:end]
                    
                    if return_attn:
                        res, w = self.local_block(t_chunk, key_padding_mask=m_chunk, return_weights=True)
                        out_chunks.append(res)
                        weight_chunks.append(w)
                    else:
                        res = self.local_block(t_chunk, key_padding_mask=m_chunk)
                        out_chunks.append(res)
                
                tokens_flat = torch.cat(out_chunks, dim=0)
                if return_attn and weight_chunks:
                    local_weights = torch.cat(weight_chunks, dim=0)

        # === 3. Aggregation (Mean Pooling) ===
        mask_unsqueeze = mask_flat.unsqueeze(-1) # [B*C, L, 1]
        tokens_masked = tokens_flat.masked_fill(mask_unsqueeze, 0.0)
        sum_tokens = tokens_masked.sum(dim=1) # [B*C, H]
        valid_counts = (~mask_flat).sum(dim=1, keepdim=True).float().clamp(min=1.0) # [B*C, 1]
        
        cluster_tokens = (sum_tokens / valid_counts).view(B, self.num_clusters, self.hidden_dim) # [B, C, H]
        
        # === 4. Global Attention ===
        # No padding mask needed for global attention (assuming all clusters are valid)
        global_attn = None
        fused = cluster_tokens
        
        if self.ablation_type != 'woGlobal':
            if return_attn:
                fused, global_attn = self.global_block(cluster_tokens, return_weights=True)
            else:
                fused = self.global_block(cluster_tokens)
            
        # === 5. Decode ===
        output = self.decoder(fused.reshape(B, -1))
        
        if return_attn:
            local_attn_dict = {}
            if local_weights is not None:
                local_weights = local_weights.view(B, self.num_clusters, self.max_len, self.max_len)
                for i, cls in enumerate(self.clusters):
                    valid_len = (~self.padding_mask[i]).sum().item()
                    local_attn_dict[cls] = local_weights[:, i, :valid_len, :valid_len].detach()
            
            global_attn_out = global_attn.detach() if global_attn is not None else None
            return output, local_attn_dict, global_attn_out
            
        return output







### ===============================
### 以下是废案
### ===============================

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
        attn_output, attn_weights = self.attn(x, x, x, need_weights=return_weights)
        out = self.norm(x + self.dropout(attn_output))
        if return_weights:
            return out.squeeze(1) if out.size(1) == 1 else out, attn_weights
        else:
            return out.squeeze(1) if out.size(1) == 1 else out

class ClusterAttentionModel(nn.Module):
    def __init__(self, feature_class=None, hidden_dim=8, num_heads=1, dropout=0.1, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feature_class = feature_class

        if feature_class is None:
            self.class_to_indices = {0: None}
        else:
            self.class_to_indices = defaultdict(list)
            for idx, cls in enumerate(feature_class):
                self.class_to_indices[int(cls)].append(idx)

        self.clusters = sorted(self.class_to_indices.keys())
        self.per_class_encoders = nn.ModuleDict()

        for cls in self.clusters:
            dim_in = len(self.class_to_indices[cls]) if self.class_to_indices[cls] is not None else len(feature_class)
            self.per_class_encoders[str(cls)] = nn.Sequential(
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                SelfAttentionBlock(hidden_dim, num_heads, dropout)
            )

        self.global_attention = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * len(self.clusters), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        B = x.size(0)
        class_embeddings = []
        local_attn_weights = {}

        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x if idxs is None else x[:, idxs]
            encoder = self.per_class_encoders[str(cls)]

            # 拆解 attention block 以提取注意力权重
            *base_layers, attn_block = encoder
            cls_feat = cls_input
            for layer in base_layers:
                cls_feat = layer(cls_feat)
            if return_attn:
                cls_feat, attn = attn_block(cls_feat, return_weights=True)
                local_attn_weights[cls] = attn.detach()
            else:
                cls_feat = attn_block(cls_feat)
            class_embeddings.append(cls_feat.unsqueeze(1))

        tokens = torch.cat(class_embeddings, dim=1)  # [B, N_cls, H]
        if return_attn:
            fused, global_attn = self.global_attention(tokens, return_weights=True)
        else:
            fused = self.global_attention(tokens)
        fused_flat = fused.reshape(B, -1)
        output = self.decoder(fused_flat)

        if return_attn:
            return output, local_attn_weights, global_attn.detach()
        return output

class ClusterAttention_woGlobal(nn.Module):
    """Ablation: remove inter-cluster (global) attention"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        base = ClusterAttentionModel(*args, **kwargs)
        self.__dict__.update(base.__dict__)

    def forward(self, x, return_attn=False):
        B = x.size(0)
        class_embeddings = []

        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x if idxs is None else x[:, idxs]
            encoder = self.per_class_encoders[str(cls)]

            *base_layers, attn_block = encoder
            cls_feat = cls_input
            for layer in base_layers:
                cls_feat = layer(cls_feat)
            cls_feat = attn_block(cls_feat)
            class_embeddings.append(cls_feat.unsqueeze(1))

        tokens = torch.cat(class_embeddings, dim=1)  # [B, N_cls, H]
        #  不做 global attention，直接拼接
        fused_flat = tokens.reshape(B, -1)
        output = self.decoder(fused_flat)
        return output

class ClusterAttention_woLocal(nn.Module):
    """Ablation: remove intra-cluster (local) attention"""

    def __init__(self, feature_class=None, hidden_dim=64, num_heads=4,
                 dropout=0.1, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feature_class = feature_class

        # ===== 构建簇索引映射 =====
        if feature_class is None:
            self.class_to_indices = {0: None}
        else:
            self.class_to_indices = defaultdict(list)
            for idx, cls in enumerate(feature_class):
                self.class_to_indices[int(cls)].append(idx)

        self.clusters = sorted(self.class_to_indices.keys())

        # ===== 每个簇一个线性层，用于将簇特征映射到 hidden_dim =====
        # （输出维度与原模型 SelfAttentionBlock 输出一致）
        self.per_class_linear = nn.ModuleDict()
        for cls in self.clusters:
            dim_in = len(self.class_to_indices[cls]) if self.class_to_indices[cls] is not None else len(feature_class)
            self.per_class_linear[str(cls)] = nn.Sequential(
                nn.LayerNorm(dim_in),
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        # ===== 簇间全局注意力（保持不变） =====
        self.global_attention = SelfAttentionBlock(hidden_dim, num_heads, dropout)

        # ===== Decoder 与原模型保持一致 =====
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * len(self.clusters), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        """
        输入:
            x: [B, F]
        输出:
            out: [B, output_dim]
        """
        B = x.size(0)
        class_embeddings = []
        # ===== 簇内线性编码（不含 Self-Attention）=====
        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x if idxs is None else x[:, idxs]       # [B, n_cls_features]
            encoder = self.per_class_linear[str(cls)]           # [Linear to hidden_dim]
            cls_feat = encoder(cls_input)                       # [B, hidden_dim]
            # 聚合为单一 token
            cls_token = cls_feat.unsqueeze(1)                   # [B, 1, hidden_dim]
            class_embeddings.append(cls_token)
        # 拼接所有簇 token
        tokens = torch.cat(class_embeddings, dim=1)             # [B, N_cls, hidden_dim]
        # ===== 簇间 Attention =====
        if return_attn:
            fused, attn = self.global_attention(tokens, return_weights=True)
        else:
            fused = self.global_attention(tokens)

        # ===== 解码预测 =====
        fused_flat = fused.reshape(B, -1)
        out = self.decoder(fused_flat)

        if return_attn:
            return out, attn.detach()
        return out

class ClusterAttention_woCluster(nn.Module):
    """Ablation: treat all features as one cluster (no clustering)"""

    def __init__(self, input_dim, hidden_dim=64, num_heads=4, dropout=0.1, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # 所有特征视为一个簇 -> 线性投影成 hidden_dim
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 单 token 自注意力（可以理解为 self-attention refine）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

        # Decoder 直接映射输出
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, F]  (所有特征)
        Returns:
            out: [B, output_dim]
        """
        B = x.size(0)
        # 线性投影得到一个全局 token
        token = self.feature_proj(x).unsqueeze(1)  # [B, 1, H]
        # self-attention refine（虽然只有1个token，但保持结构一致）
        attn_out, _ = self.self_attention(token, token, token)
        fused = self.norm(token + attn_out)  # [B, 1, H]
        # 解码
        out = self.decoder(fused.squeeze(1))  # [B, output_dim]
        return out




class ClusterModel(nn.Module):
    def __init__(self, feature_class, hidden_dim=64,num_heads=4, dropout=0.1, output_dim=1):
        super().__init__()
        self.feature_class = feature_class
        self.class_to_indices = defaultdict(list)
        for idx, cls in enumerate(feature_class):
            self.class_to_indices[int(cls)].append(idx)
        self.clusters = sorted(self.class_to_indices.keys())
        self.num_clusters = len(self.clusters)

        # ===== 每个簇的卷积编码器 =====
        self.per_class_encoders = nn.ModuleDict({
            str(cls): nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),     # 每个簇输出 1 个 token
                nn.Flatten(),                # [B, hidden_dim]
                nn.Dropout(dropout)
            ) for cls in self.clusters
        })

        # ===== 簇间 Attention (轻量级) =====
        self.inter_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # ===== Decoder =====
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.size(0)
        cluster_tokens = []

        # 簇内卷积编码
        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x[:, idxs].unsqueeze(1)  # [B, 1, n_cls_features]
            cls_token = self.per_class_encoders[str(cls)](cls_input)  # [B, hidden_dim]
            cluster_tokens.append(cls_token.unsqueeze(1))  # [B, 1, hidden_dim]

        cluster_tokens = torch.cat(cluster_tokens, dim=1)  # [B, N_cls, hidden_dim]

        # 簇间 attention
        attn_out, _ = self.inter_attn(cluster_tokens, cluster_tokens, cluster_tokens)
        attn_out = self.norm(cluster_tokens + attn_out)

        # 展平送入 decoder
        out = self.decoder(attn_out.reshape(B, -1))
        return out


class LearnableCluster(nn.Module):
    def __init__(self, input_dim, num_clusters=50, hidden_dim=64, tau=0.5):
        """
        可学习聚类模块
        Args:
            input_dim: 输入特征数 (F)
            num_clusters: 簇个数
            hidden_dim: 隐藏维度
            tau: softmax 温度
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim
        self.tau = tau

        # 每个特征 (标量) -> 向量嵌入
        self.feature_proj = nn.Linear(1, hidden_dim)

        # 簇中心 (K x H)
        self.centers = nn.Parameter(torch.randn(num_clusters, hidden_dim))

    def forward(self, x):
        """
        Args:
            x: [B, F]
        Returns:
            cluster_tokens: [B, K, H]  每个簇的表示
            assign_prob: [B, F, K]     每个特征属于各簇的概率
            hard_assign: [B, F]        每个特征的硬簇标签
        """
        B, Fdim = x.shape

        # === 特征嵌入 [B, F, H] ===
        feat_embed = self.feature_proj(x.unsqueeze(-1))

        # === 计算相似度 [B, F, K] ===
        sim = torch.matmul(feat_embed, self.centers.T)

        # === soft assignment ===
        assign_prob = torch.softmax(sim / self.tau, dim=-1)

        # === 簇 token: 加权和 [B, K, H] ===
        numerator = torch.einsum('bfk,bfh->bkh', assign_prob, feat_embed)
        denominator = assign_prob.sum(dim=1, keepdim=True).transpose(1, 2) + 1e-6  # [B, K, 1]
        cluster_tokens = numerator / denominator

        # === 硬分配 [B, F] ===
        hard_assign = assign_prob.argmax(dim=-1)

        return cluster_tokens, assign_prob, hard_assign


class LearnableClusterAttentionModel(nn.Module):
    def __init__(self, input_dim, num_clusters=50, hidden_dim=64,
                 num_heads=4, dropout=0.1, output_dim=1, tau=0.5):
        """
        聚类 + 簇间注意力 + 解码
        Args:
            input_dim: 输入特征数 (F)
            num_clusters: 簇个数
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            dropout: dropout 比例
            output_dim: 输出维度
            tau: soft assignment 温度
        """
        super().__init__()
        self.clusterer = LearnableCluster(input_dim, num_clusters, hidden_dim, tau)

        # 簇间 Attention
        self.inter_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, return_assign=False):
        """
        Args:
            x: [B, F]
            return_assign: 是否返回 soft/hard assignment
        """
        B = x.size(0)

        # === 聚类 ===
        cluster_tokens, assign_prob, hard_assign = self.clusterer(x)  # [B, K, H]

        # === 簇间 Attention ===
        attn_out, _ = self.inter_attn(cluster_tokens, cluster_tokens, cluster_tokens)
        attn_out = self.norm(cluster_tokens + attn_out)

        # === Decoder ===
        out = self.decoder(attn_out.reshape(B, -1))  # [B, output_dim]

        if return_assign:
            return out, assign_prob, hard_assign
        return out

class ExpertBlock(nn.Module):
    """
    定义一个专家块，其中包含线性层、批量归一化、ReLU 激活函数和 Dropout。

    参数:
    dim_in (int): 输入特征的维度。
    hidden_dim (int): 隐藏层的维度。
    dropout (float): Dropout 正则化的比例。
    """
    def __init__(self, dim_in, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim_in, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播过程，经过线性变换、批量归一化、ReLU 激活和 Dropout。

        参数:
        x (Tensor): 输入特征。

        返回:
        Tensor: 输出特征。
        """
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultiLayerExpertBlock(nn.Module):
    """
    定义一个多层专家块，包含 k + 1 层，每层的维度逐渐减半，第零层是将输入映射到更高维度的一个专家层。

    参数:
    dim_in (int): 输入特征的维度。
    output_dim (int): 输出特征的维度。
    k (int): 专家块的层数。
    dropout (float): Dropout 正则化的比例。
    """

    def __init__(self, dim_in, output_dim, k=3, dropout=0.1):
        super().__init__()

        # 计算零层的输出维度
        high_dim = output_dim * (2 ** k)

        # 定义零层，将dim_in映射到一个更高的维度（作为专家层）
        self.zero_layer = ExpertBlock(dim_in, high_dim, dropout)

        # 定义后续的 k 层专家块
        layers = []
        current_dim = high_dim
        for i in range(k):
            next_dim = current_dim // 2
            layers.append(ExpertBlock(current_dim, next_dim, dropout))
            current_dim = next_dim

        self.expert_layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        前向传播过程，首先通过零层将输入映射到高维，然后逐层经过专家块，每一层的维度逐渐减半，最终得到输出。

        参数:
        x (Tensor): 输入特征。

        返回:
        Tensor: 输出特征。
        """
        x = self.zero_layer(x)  # 经过零层映射到更高维度
        for layer in self.expert_layers:  # 遍历所有 k 层
            x = layer(x)
        return x

class ClusterAttentionModelWithExperts(nn.Module):
    """
    基于多个专家块的集群注意力模型，采用多专家并行编码每个类别的输入。

    参数:
    feature_class (list): 特征类别列表，用于定义每个类别的索引。
    hidden_dim (int): 每个类别的专家块的隐藏层维度。
    num_heads (int): 全局注意力层的头数。
    dropout (float): Dropout 正则化的比例。
    output_dim (int): 模型输出维度。
    num_experts (int): 每个类别的专家块数量。
    adjust_hidden_dim_by_cls (bool): 如果为 True，隐藏层维度将根据类别数量动态调整。
    """

    def __init__(self, feature_class=None, hidden_dim=8, num_heads=1, dropout=0.1, output_dim=1, num_experts=3,
                 adjust_hidden_dim_by_cls=False):
        super().__init__()
        self.dropout = dropout
        self.feature_class = feature_class
        self.num_experts = num_experts
        self.adjust_hidden_dim_by_cls = adjust_hidden_dim_by_cls

        if feature_class is None:
            self.class_to_indices = {0: None}
        else:
            self.class_to_indices = defaultdict(list)
            for idx, cls in enumerate(feature_class):
                self.class_to_indices[int(cls)].append(idx)

        self.clusters = sorted(self.class_to_indices.keys())
        self.per_class_encoders = nn.ModuleDict()
        self.class_hidden_dims = {}

        # 每个类别使用多个专家块
        for cls in self.clusters:
            dim_in = len(self.class_to_indices[cls]) if self.class_to_indices[cls] is not None else len(feature_class)
            # 根据类别数量动态调整隐藏层维度
            if self.adjust_hidden_dim_by_cls:
                self.class_hidden_dims[cls] = 64 * len(self.clusters)  # 动态调整每个类别的隐藏层维度
            else:
                self.class_hidden_dims[cls] = hidden_dim  # 默认每个类别使用相同的 hidden_dim

            # 每个类别使用专家块
            self.per_class_encoders[str(cls)] = MultiLayerExpertBlock(
                dim_in,
                self.class_hidden_dims[cls],
                k=num_experts,  # 这里的 num_experts 是控制层数
                dropout=dropout
            )

        # 全局注意力机制作用于类别级别的特征
        self.global_attention = SelfAttentionBlock(hidden_dim, num_heads, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * len(self.clusters), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn=False):
        """
        前向传播过程，使用多个专家块对每个类别进行编码，之后通过全局注意力层和解码器生成最终输出。

        参数:
        x (Tensor): 输入数据。
        return_attn (bool): 如果为 True，返回注意力权重。

        返回:
        Tensor: 输出的最终预测值。
        """
        B = x.size(0)
        class_embeddings = []

        # 对每个类别进行编码
        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x if idxs is None else x[:, idxs]
            encoder = self.per_class_encoders[str(cls)]
            cls_feat = encoder(cls_input)
            class_embeddings.append(cls_feat.unsqueeze(1))

        # 拼接所有类别的特征
        tokens = torch.cat(class_embeddings, dim=1)

        # 使用全局注意力机制处理类别之间的关系
        if return_attn:
            fused, global_attn = self.global_attention(tokens, return_weights=True)
        else:
            fused = self.global_attention(tokens)

        # 展平并通过解码器输出最终结果
        fused_flat = fused.reshape(B, -1)
        output = self.decoder(fused_flat)

        if return_attn:
            return output, global_attn.detach()
        return output

class SelfAttentionBlockWithBN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        # 输入投影：低维 → hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 注意力模块
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        """
        输入:
            x: Tensor, shape [B, D] 或 [B, N, D]，其中 D 可以小于 hidden_dim
        输出:
            Tensor, shape [B, D] 或 [B, N, D]（D=hidden_dim）
        """
        squeeze_flag = False
        if x.dim() == 2:
            x = x.unsqueeze(1)   # 转换为 [B, 1, D]
            squeeze_flag = True

        x_proj = self.input_proj(x)  # [B, N, hidden_dim]
        attn_output, attn_weights = self.attn(x_proj, x_proj, x_proj, need_weights=return_weights)
        # 残差连接
        out = x_proj + self.dropout(attn_output)
        # BatchNorm1d 需要 [B, C, L] 或 [B, C]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)  # [B, N, hidden_dim]

        if squeeze_flag:
            out = out.squeeze(1)  # [B, hidden_dim]
        if return_weights:
            return out, attn_weights
        else:
            return out


class AttnPoolingWithCLS(nn.Module):
    def __init__(self, hidden_dim=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 可学习 [CLS] 向量
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [B, N, D]  输入簇 token
        return: [B, D]  聚合后的整体表示
        """
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, D]
        x = torch.cat([cls, x], dim=1)           # 拼接 [CLS]

        out, _ = self.attn(x, x, x)              # [B, 1+N, D]
        cls_out = out[:, 0, :]                   # 取 [CLS]
        return self.norm(cls_out)                # [B, D]



class SelfAttentionBlockBN(nn.Module):
    """带 BatchNorm 和残差的 Attention Block"""
    def __init__(self, hidden_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, D]
        residual = x
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)  # 残差连接

        # BN 按 D 维度做 -> 转 [B, D, N]
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)

        residual = x
        ffn_out = self.ffn(x)
        x = residual + self.dropout(ffn_out)

        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        return x




import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, n_cls, 1]
        return: [B, hidden_dim]
        """
        h = torch.tanh(self.proj(x))              # [B, n_cls, hidden_dim]
        attn_scores = self.score(h).squeeze(-1)   # [B, n_cls]
        attn_weights = torch.softmax(attn_scores, dim=1)
        token = torch.bmm(attn_weights.unsqueeze(1), h)  # [B, 1, hidden_dim]
        return token.squeeze(1)


class DNNGPWithClusterAttention(nn.Module):
    def __init__(self, feature_class, in_dim, hidden_dim=64, num_heads=4, dropout=0.1, output_dim=1):
        """
        Args:
            feature_class: 聚类标签
            in_dim: 序列长度
            hidden_dim: 注意力嵌入维度
            num_heads: 多头注意力头数
            dropout: dropout 比例
            output_dim: 输出维度
        """
        super().__init__()
        self.feature_class = feature_class
        self.class_to_indices = defaultdict(list)
        for idx, cls in enumerate(feature_class):
            self.class_to_indices[int(cls)].append(idx)
        self.clusters = sorted(self.class_to_indices.keys())
        self.num_clusters = len(self.clusters)

        # ===== CNN backbone =====
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=7)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv1d(32, hidden_dim, kernel_size=3)

        # ===== 簇内 Attention Pooling =====
        self.pooling = AttentionPooling(input_dim=1, hidden_dim=hidden_dim)

        # ===== Cross-Attention (Q=feats, K/V=cluster_tokens) =====
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.bn_after_attn = nn.BatchNorm1d(hidden_dim)

        # ===== Decoder =====
        L1 = in_dim - 15 + 1
        L2 = L1 - 7 + 1
        L3 = L2 - 3 + 1
        self.L_out = L3
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.L_out, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.size(0)

        # ===== CNN backbone =====
        x_cnn = x.unsqueeze(1)          # [B, 1, L]
        x_cnn = F.relu(self.conv1(x_cnn))
        x_cnn = self.bn1(x_cnn)
        x_cnn = self.dropout1(x_cnn)

        x_cnn = F.relu(self.conv2(x_cnn))
        x_cnn = self.dropout2(x_cnn)

        x_cnn = F.relu(self.conv3(x_cnn))   # [B, hidden_dim, L’]
        feats = x_cnn.transpose(1, 2)       # [B, L’, hidden_dim]

        # ===== 簇内 pooling 得到簇 token =====
        cluster_tokens = []
        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x[:, idxs]          # [B, n_cls_features]
            cls_input = cls_input.unsqueeze(-1)  # [B, n_cls, 1]
            cls_token = self.pooling(cls_input)  # [B, hidden_dim]
            cluster_tokens.append(cls_token.unsqueeze(1))  # [B, 1, hidden_dim]

        cluster_tokens = torch.cat(cluster_tokens, dim=1)  # [B, N_cls, hidden_dim]

        # ===== Cross-Attention (Q=feats, K/V=cluster_tokens) =====
        out, _ = self.cross_attn(feats, cluster_tokens, cluster_tokens)  # [B, L’, hidden_dim]
        out = feats + out

        out = out.transpose(1, 2)  # [B, hidden_dim, L’] 方便 BN
        out = self.bn_after_attn(out).transpose(1, 2)  # [B, L’, hidden_dim]

        # ===== Decoder =====
        out = self.decoder(out.reshape(B, -1))  # [B, output_dim]
        return out

# ========== 簇 Transformer ==========
class ClusterTransformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 一个共享的 learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, x, mask=None):
        """
        x: [B, L, input_dim]
        mask: [B, L] (True 表示 pad 位置)
        """
        B = x.size(0)
        x = self.proj(x)  # [B, L, hidden_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+L, hidden_dim]

        if mask is not None:
            mask = F.pad(mask, (1, 0), value=False)  # 给CLS补个False
        out = self.encoder(x, src_key_padding_mask=mask)  # [B, 1+L, hidden_dim]

        return out[:, 0, :]  # 取CLS


# ========== 总模型 ==========
class ClusterTransformerModel(nn.Module):
    def __init__(self, feature_class, hidden_dim=64, num_heads=1, num_layers=1, dropout=0.1, output_dim=1):
        super().__init__()
        self.feature_class = feature_class
        self.class_to_indices = defaultdict(list)
        for idx, cls in enumerate(feature_class):
            self.class_to_indices[int(cls)].append(idx)
        self.clusters = sorted(self.class_to_indices.keys())
        self.num_clusters = len(self.clusters)

        # 一个共享 Transformer
        self.cluster_transformer = ClusterTransformer(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

        # Decoder: 拼接所有簇 token
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * self.num_clusters, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        B = x.size(0)
        cluster_tokens = []

        for cls in self.clusters:
            idxs = self.class_to_indices[cls]
            cls_input = x[:, idxs].unsqueeze(-1)  # [B, n_cls, 1]
            # 不需要 mask 的简单版本
            cls_token = self.cluster_transformer(cls_input)  # [B, hidden_dim]
            cluster_tokens.append(cls_token)

        cluster_tokens = torch.stack(cluster_tokens, dim=1)  # [B, N_cls, hidden_dim]
        out = self.decoder(cluster_tokens.reshape(B, -1))  # [B, output_dim]
        return out
