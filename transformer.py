import torch
import torch.nn.functional as F

import utils

class TransformerLayer(torch.nn.Module):
    def __init__(self,
                 feature_dim,
                 ffn=True,
                 ffn_dim_expansion=4
                 ):
        super(TransformerLayer, self).__init__()

        # multi-head attention
        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        # self.multi_head_attn = torch.nn.MultiheadAttention(feature_dim, 2, batch_first=True, device='cuda')

        self.norm1 = torch.nn.LayerNorm(feature_dim)

        self.ffn = ffn

        if self.ffn:
            in_channels = feature_dim * 2
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                torch.nn.GELU(),
                torch.nn.Linear(in_channels * ffn_dim_expansion, feature_dim, bias=False),
            )

            self.norm2 = torch.nn.LayerNorm(feature_dim)

    def forward(self, source, target):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        message = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

        message = self.merge(message)

        # message, _ = self.multi_head_attn(query, key, value, need_weights=False)
        message = self.norm1(message)

        if self.ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message

class FeatureTransformer(torch.nn.Module):
    def __init__(self, feature_dim, num_layers, bidir=True, ffn=True, ffn_dim_expansion=4, post_norm=False):
        super(FeatureTransformer, self).__init__()

        self.bidir = bidir

        self.layers = torch.nn.ModuleList([
            TransformerLayer(feature_dim, ffn=ffn, ffn_dim_expansion=ffn_dim_expansion
                             )
            for i in range(num_layers)])

        self.post_norm = post_norm

        if self.post_norm:
            self.norm = torch.nn.LayerNorm(feature_dim, eps=1e-06)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1):

        b, c, h, w = feature0.shape

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        if self.bidir:

            concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
            concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

            for layer in self.layers:
                concat0 = layer(concat0, concat1)
                concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

            if self.post_norm:
                concat0 = self.norm(concat0)

            feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

            # reshape back
            feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

            return feature0, feature1

        else:
            for layer in self.layers:
                feature0 = layer(feature0, feature1)
                
            if self.post_norm:
                feature0 = self.norm(feature0)

            feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

            return feature0

class FeatureFlowAttention(torch.nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, feature_dim):
        super(FeatureFlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.v_proj = torch.nn.Linear(feature_dim, feature_dim)

        self.merge = torch.nn.Linear(feature_dim, feature_dim)

        self.norm = torch.nn.LayerNorm(feature_dim, eps=1e-06)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature0.size()

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        concat = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]

        flow = flow.flatten(-2).permute(0, 2, 1)

        query = self.q_proj(concat)  # [B, H*W, C]
        key = self.k_proj(concat)  # [B, H*W, C]
        value = self.v_proj(concat)  # [B, H*W, C]

        attn = query @ key.transpose(-2, -1) / (c ** 0.5)  # [B, H*W, H*W]
        attn = torch.softmax(attn, dim=-1)

        flow = attn @ flow
        value = attn @ value

        value = self.merge(value)
        value = self.norm(value)

        concat = concat + value

        feature0, feature1 = concat.chunk(chunks=2, dim=0)
        flow0, flow1 = flow.chunk(chunks=2, dim=0)

        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        flow0 = flow0.view(b, h, w, 2).permute(0, 3, 1, 2)
        flow1 = flow1.view(b, h, w, 2).permute(0, 3, 1, 2)

        return feature0, feature1, flow0, flow1

class FlowAttention(torch.nn.Module):

    def __init__(self, feature_dim):
        super(FlowAttention, self).__init__()

        self.q_proj = torch.nn.Linear(feature_dim, feature_dim)
        self.k_proj = torch.nn.Linear(feature_dim, feature_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        flow = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        flow = F.scaled_dot_product_attention(query, key, flow)

        flow = flow.view(b, h, w, 2).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return flow
