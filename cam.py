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

