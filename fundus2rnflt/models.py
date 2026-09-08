"""Glaucoma classifier architectures: single-stream ResNet, dual-branch concat fusion,
dual-branch attention fusion, the RNFLT input transform, and a config-driven builder."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models import (resnet18, resnet50, ResNet18_Weights, ResNet50_Weights)


def _make_resnet_backbone(name: str = "resnet18",
                          in_channels: int = 3,
                          pretrained: bool = True) -> (nn.Module, int):
    """
    Create a ResNet backbone that outputs a pooled feature vector.
    Returns (backbone_module, feature_dim).
    """
    if name == "resnet18":
        # resnet = models.resnet18(pretrained=pretrained)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
    elif name == "resnet50":
        # resnet = models.resnet50(pretrained=pretrained)
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported encoder '{name}'. Choose 'resnet18' or 'resnet50'.")

    # Adapt first conv to arbitrary in_channels while preserving pretrained stats
    if in_channels != resnet.conv1.in_channels:
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=False)
        with torch.no_grad():
            # Average RGB filters and repeat to the new number of channels
            if old_conv.weight.shape[1] == 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # [out,1,k,k]
                resnet.conv1.weight[:] = mean_w.repeat(1, in_channels, 1, 1)
            else:
                # Fallback: Kaiming init if original wasn't 3ch (unlikely here)
                nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Build a feature extractor up to (and including) avgpool
    # Children: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs [B, C, 1, 1]
    feat_dim = resnet.fc.in_features
    return backbone, feat_dim


class _Encoder(nn.Module):
    """Wraps a ResNet backbone to output a flat feature vector."""
    def __init__(self, name: str, in_channels: int, pretrained: bool):
        super().__init__()
        self.backbone, self.out_dim = _make_resnet_backbone(
            name=name, in_channels=in_channels, pretrained=pretrained
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> features: [B, out_dim]
        x = self.backbone(x)         # [B, C, 1, 1]
        x = torch.flatten(x, 1)      # [B, C]
        return x


class DualBranchResNet(nn.Module):
    """
    Dual-branch CNN with mid-level (feature) fusion.

    - Fundus branch: ResNet (e.g., resnet18), input 3x224x224 (ImageNet normalization assumed upstream).
    - RNFLT branch:  ResNet (e.g., resnet18), input 1x224x224 (or 2x224x224 if later adding mask channel).
    - Fusion: concat pooled features -> (optional dropout) -> FC -> 1 logit.

    Compatibility with current train_classifier.py:
      * You can pass EITHER:
          (a) a single concatenated tensor x = cat([fundus(3ch), rnflt(1or2ch)], dim=1), i.e. x:[B,4or5,H,W]
          (b) two tensors separately: model(fundus, rnflt)
      * If only a single tensor with 3ch or 1/2ch is provided, the missing branch
        will be treated as zeros (this lets you reuse it for single-modality runs if desired).

    Notes:
      - This model does NOT require an RNFLT mask right now. If NaNs were present,
        fill them with 0 in the dataset pipeline before batching (as you currently do).
    """
    def __init__(self,
                 fundus_encoder: str = "resnet18",
                 rnflt_encoder: str = "resnet18",
                 fundus_in_channels: int = 3,
                 rnflt_in_channels: int = 1,
                 pretrained_fundus: bool = True,
                 pretrained_rnflt: bool = False,
                 dropout: float = 0.3,
                 hidden_dim: int = None):
        super().__init__()

        # Encoders
        self.fundus_enc = _Encoder(fundus_encoder, fundus_in_channels, pretrained_fundus)
        self.rnflt_enc  = _Encoder(rnflt_encoder,  rnflt_in_channels,  pretrained_rnflt)

        fusion_in = self.fundus_enc.out_dim + self.rnflt_enc.out_dim

        # Simple head: concat -> (optional hidden) -> 1 logit
        if hidden_dim is None:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_in, 1)
            )
        else:
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def _split_or_route(self, x: torch.Tensor, x_rnflt: torch.Tensor = None):
        """
        Accepts either:
          - x: concatenated channels [B, 4or5, H, W]  (first 3 = fundus, rest = rnflt)
          - or x: fundus [B,3,H,W], x_rnflt: rnflt [B,1or2,H,W]
        Returns (fundus, rnflt) where missing branches are set to None.
        """
        if x_rnflt is not None:
            return x, x_rnflt

        assert x.dim() == 4, "Expected 4D tensor [B,C,H,W] as single input."
        c = x.size(1)
        if c >= 4:
            # Concatenated fused input: first 3 are fundus, remainder are rnflt
            fundus = x[:, :3, :, :]
            rnflt  = x[:, 3:, :, :]
            return fundus, rnflt
        elif c == 3:
            # Fundus-only
            return x, None
        elif c in (1, 2):
            # RNFLT-only (1 or 2 channels)
            return None, x
        else:
            raise ValueError(f"Unexpected channel count {c}. "
                             f"Expected 3 (fundus), 1/2 (rnflt), or 4/5 (fused).")

    def forward(self,
                x_fundus: torch.Tensor,
                x_rnflt: torch.Tensor = None) -> torch.Tensor:
        fundus, rnflt = self._split_or_route(x_fundus, x_rnflt)

        # Encode
        if fundus is not None:
            f_feat = self.fundus_enc(fundus)
            batch_size = f_feat.size(0)
            device = f_feat.device
            dtype = f_feat.dtype
        else:
            # Build a placeholder batch size/device from rnflt
            r_probe = self.rnflt_enc(rnflt)  # [B, D_r]
            batch_size = r_probe.size(0)
            device = r_probe.device
            dtype = r_probe.dtype
            # Then we will overwrite r_feat below correctly
            # and set f_feat as zeros of the right shape
            f_feat = torch.zeros(batch_size, self.fundus_enc.out_dim, device=device, dtype=dtype)

        if rnflt is not None:
            r_feat = self.rnflt_enc(rnflt)
        else:
            r_feat = torch.zeros(batch_size, self.rnflt_enc.out_dim, device=device, dtype=dtype)

        fused = torch.cat([f_feat, r_feat], dim=1)  # [B, D_f + D_r]
        logit = self.head(fused)                    # [B, 1]
        return logit


def build_dual_branch_resnet(fundus_encoder: str = "resnet18",
                             rnflt_encoder: str = "resnet18",
                             rnflt_channels: int = 1,
                             pretrained_fundus: bool = True,
                             pretrained_rnflt: bool = False,
                             dropout: float = 0.3,
                             hidden_dim: int = None) -> DualBranchResNet:
    """
    Convenience factory to create a DualBranchResNet with common defaults.
    """
    return DualBranchResNet(
        fundus_encoder=fundus_encoder,
        rnflt_encoder=rnflt_encoder,
        fundus_in_channels=3,
        rnflt_in_channels=rnflt_channels,
        pretrained_fundus=pretrained_fundus,
        pretrained_rnflt=pretrained_rnflt,
        dropout=dropout,
        hidden_dim=hidden_dim
    )

# ================================
# Attention-based dual-branch fusion (Li-style)
# ================================

# ---- Utilities ----
def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Returns [H*W, dim] sinusoidal 2D positional encodings.
    """
    def get_1d_pos_embed(n, d):
        omega = torch.arange(d // 2, device=device, dtype=dtype)
        omega = 1. / (10000 ** (omega / (d / 2)))
        pos = torch.arange(n, device=device, dtype=dtype)
        out = torch.einsum('n,d->nd', pos, omega)
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        if d % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb  # [n, d]

    assert dim % 2 == 0, "positional dim should be even"
    emb_h = get_1d_pos_embed(h, dim // 2)  # [H, dim/2]
    emb_w = get_1d_pos_embed(w, dim // 2)  # [W, dim/2]
    pos = torch.cat([
        emb_h[:, None, :].expand(h, w, -1),
        emb_w[None, :, :].expand(h, w, -1)
    ], dim=-1)  # [H, W, dim]
    return pos.reshape(h * w, dim)  # [H*W, dim]


# ---- ResNet encoders that output spatial feature maps (C x H x W) ----
def _make_resnet_spatial_backbone(name: str = "resnet18",
                                  in_channels: int = 3,
                                  pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Returns a backbone that outputs spatial features (after layer4, BEFORE avgpool),
    and the feature dimension (C).
    """
    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported encoder '{name}'. Choose 'resnet18' or 'resnet50'.")

    # adapt first conv for arbitrary in_channels (same policy as baseline)
    if in_channels != resnet.conv1.in_channels:
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(in_channels,
                                 old_conv.out_channels,
                                 kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride,
                                 padding=old_conv.padding,
                                 bias=False)
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                resnet.conv1.weight[:] = mean_w.repeat(1, in_channels, 1, 1)
            else:
                nn.init.kaiming_normal_(resnet.conv1.weight, mode="fan_out", nonlinearity="relu")

    # keep everything up to layer4 (exclude avgpool and fc)
    spatial_backbone = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )  # -> [B, C, H, W]
    feat_dim = resnet.fc.in_features
    return spatial_backbone, feat_dim


class _SpatialEncoder(nn.Module):
    """
    Wraps a ResNet that outputs spatial features [B, C, H, W].
    Exposes `.backbone` to keep compatibility with external code that may
    access conv1/bn1/layer1 for freezing or weight edits.
    """
    def __init__(self, name: str, in_channels: int, pretrained: bool):
        super().__init__()
        self.backbone, self.out_dim = _make_resnet_spatial_backbone(
            name=name, in_channels=in_channels, pretrained=pretrained
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)  # [B, C, H, W]


# ---- Tiny Transformer pieces (with bias-capable attention) ----
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 4.0, drop: float = 0.1):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiasSelfAttention(nn.Module):
    """
    Multi-head self-attention over a joint sequence with an optional additive bias mask.
    bias: Tensor broadcastable to [B, heads, L_q, L_k], added to attention logits before softmax.
    """
    def __init__(self, dim: int, heads: int = 6, drop: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, L, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each [B, heads, L, Dh]
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B, heads, L, L]

        if bias is not None:
            # bias is added to logits; should be broadcastable to [B, heads, L, L]
            attn = attn + bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B, heads, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.proj_drop(self.out(out))
        return out


class BiasCrossAttention(nn.Module):
    """
    Multi-head cross-attention: queries attend to keys/values from another sequence.
    Supports an additive bias on KEY positions (e.g., to mask/penalize RNFLT tokens).
    """
    def __init__(self, dim: int, heads: int = 6, drop: float = 0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(drop)
        self.proj_drop = nn.Dropout(drop)

    def forward(self,
                qx: torch.Tensor,   # [B, Lq, D]
                kx: torch.Tensor,   # [B, Lk, D]
                bias_k: Optional[torch.Tensor] = None  # broadcastable to [B, heads, Lq, Lk]
                ) -> torch.Tensor:
        B, Lq, D = qx.shape
        Lk = kx.size(1)
        q = self.to_q(qx).reshape(B, Lq, self.heads, D // self.heads).permute(0, 2, 1, 3)  # [B,H,Lq,Dh]
        k = self.to_k(kx).reshape(B, Lk, self.heads, D // self.heads).permute(0, 2, 1, 3)  # [B,H,Lk,Dh]
        v = self.to_v(kx).reshape(B, Lk, self.heads, D // self.heads).permute(0, 2, 1, 3)  # [B,H,Lk,Dh]

        attn = (q * ((D // self.heads) ** -0.5)) @ k.transpose(-2, -1)  # [B,H,Lq,Lk]
        if bias_k is not None:
            attn = attn + bias_k  # add key-position bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v  # [B,H,Lq,Dh]
        out = out.transpose(1, 2).reshape(B, Lq, D)
        out = self.proj_drop(self.out(out))
        return out


class AttnBlock(nn.Module):
    """
    One fusion block:
      - joint self-attention over [CLS + Fundus + RNFLT] with optional RNFLT key bias
      - bidirectional cross-attention (Fundus->RNFLT with RNFLT bias; RNFLT->Fundus no bias by default)
      - MLP (FFN)
    """
    def __init__(self, dim: int, heads: int = 6, mlp_mult: float = 2.0, drop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = BiasSelfAttention(dim, heads=heads, drop=drop)

        self.ln2_f = nn.LayerNorm(dim)
        self.cross_f_to_r = BiasCrossAttention(dim, heads=heads, drop=drop)  # F queries RNFLT
        self.ln2_r = nn.LayerNorm(dim)
        self.cross_r_to_f = BiasCrossAttention(dim, heads=heads, drop=drop)  # RNFLT queries Fundus

        self.ln3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_mult=mlp_mult, drop=drop)

    def forward(self,
                cls_tok: torch.Tensor,   # [B, 1, D]
                f_tok: torch.Tensor,     # [B, Tf, D]
                r_tok: torch.Tensor,     # [B, Tr, D]
                joint_bias: Optional[torch.Tensor],   # [B, H, L, L] or broadcastable
                rnflt_key_bias: Optional[torch.Tensor]  # [B, H, Lq, Tr] or broadcastable
                ):
        B = cls_tok.size(0)
        x = torch.cat([cls_tok, f_tok, r_tok], dim=1)  # [B, 1+Tf+Tr, D]

        # 1) Self-attention with joint bias (applies RNFLT bias to RNFLT key columns)
        x = x + self.self_attn(self.ln1(x), bias=joint_bias)

        # split back
        cls, f, r = x[:, :1, :], x[:, 1:1+f_tok.size(1), :], x[:, 1+f_tok.size(1):, :]

        # 2) Cross: Fundus queries RNFLT (mask-aware on RNFLT keys)
        f = f + self.cross_f_to_r(self.ln2_f(f), self.ln2_r(r), bias_k=rnflt_key_bias)

        # 3) Cross: RNFLT queries Fundus (no bias by default; can add if needed)
        r = r + self.cross_r_to_f(self.ln2_r(r), self.ln2_f(f), bias_k=None)

        # 4) FFN on the re-concatenated sequence
        x = torch.cat([cls, f, r], dim=1)
        x = x + self.mlp(self.ln3(x))
        cls, f, r = x[:, :1, :], x[:, 1:1+f_tok.size(1), :], x[:, 1+f_tok.size(1):, :]

        return cls, f, r


class DualBranchResNetAttn(nn.Module):
    """
    Dual-branch encoders (ResNet) + tiny attention fusion head with mask-aware RNFLT bias.
    API mirrors DualBranchResNet so train_classifier.py remains unchanged.

    RNFLT mask/bias support:
      - If RNFLT channels >= 2, channel index 1 is treated as a binary validity mask (1=valid tissue; 0=invalid/disc/cup).
        -> Hard ignore in attention by applying -inf to those key positions.
      - If RNFLT channels >= 3, channel index 2 is treated as a "newly supervised" softness map in [0..1].
        -> Soft down-weight via additive negative bias: -beta * map.
      - If channels < 2, attention runs without biases (backward-compatible).
    """
    def __init__(self,
                 fundus_encoder: str = "resnet18",
                 rnflt_encoder: str = "resnet18",
                 fundus_in_channels: int = 3,
                 rnflt_in_channels: int = 1,
                 pretrained_fundus: bool = True,
                 pretrained_rnflt: bool = False,
                 d_model: int = 384,
                 num_layers: int = 2,
                 num_heads: int = 6,
                 mlp_mult: float = 2.0,
                 dropout: float = 0.1,
                 beta_soft: float = 1.5,   # strength for 'newly supervised' soft down-weight
                 ):
        super().__init__()

        # Encoders output spatial features
        self.fundus_enc = _SpatialEncoder(fundus_encoder, fundus_in_channels, pretrained_fundus)
        self.rnflt_enc  = _SpatialEncoder(rnflt_encoder,  rnflt_in_channels,  pretrained_rnflt)

        # Per-branch projection to a shared token width
        self.f_proj = nn.Linear(self.fundus_enc.out_dim, d_model)
        self.r_proj = nn.Linear(self.rnflt_enc.out_dim,  d_model)

        # Class token and modality/type embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.f_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.r_type = nn.Parameter(torch.zeros(1, 1, d_model))

        # Fusion blocks
        self.blocks = nn.ModuleList([
            AttnBlock(dim=d_model, heads=num_heads, mlp_mult=mlp_mult, drop=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # bias hyperparameter
        self.beta_soft = float(beta_soft)

        # init params
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.f_type, std=0.02)
        nn.init.trunc_normal_(self.r_type, std=0.02)

    # Keep the same routing helper as in baseline (copy to avoid import gymnastics)
    def _split_or_route(self, x: torch.Tensor, x_rnflt: Optional[torch.Tensor] = None):
        if x_rnflt is not None:
            return x, x_rnflt

        assert x.dim() == 4, "Expected 4D tensor [B,C,H,W] as single input."
        c = x.size(1)
        if c >= 4:
            fundus = x[:, :3, :, :]
            rnflt  = x[:, 3:, :, :]
            return fundus, rnflt
        elif c == 3:
            return x, None
        elif c in (1, 2, 3, 4):
            return None, x
        else:
            raise ValueError(f"Unexpected channel count {c}.")

    def _tokenize(self, fmap: torch.Tensor, proj: nn.Linear,
                  type_emb: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        fmap: [B, C, H, W] -> tokens: [B, H*W, d_model], add type embedding + 2D pos enc.
        """
        B, C, H, W = fmap.shape
        x = fmap.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = proj(x)                           # [B, HW, D]

        # add type and positional encodings
        x = x + type_emb  # broadcast over tokens
        pos = _build_2d_sincos_pos_embed(H, W, x.size(-1), device=x.device, dtype=x.dtype)  # [HW, D]
        x = x + pos.unsqueeze(0)
        return x, H, W

    def _build_biases(self,
                      rnflt_img: Optional[torch.Tensor],
                      f_len: int,
                      r_len: int,
                      num_heads: int,
                      ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build (joint_self_bias, cross_f_to_r_bias).
        - joint_self_bias: [B, H, L, L], where RNFLT key columns get negative bias.
        - cross_f_to_r_bias: [B, H, Lq, Tr], bias on RNFLT keys only.
        Returns (None, None) if no mask information available.
        """
        if rnflt_img is None or rnflt_img.size(1) < 2:
            return None, None

        B, C, Hr, Wr = rnflt_img.shape
        Tr = Hr * Wr

        # channel 1: validity mask (1 = valid tissue; 0 = invalid/disc/cup)
        valid = rnflt_img[:, 1:2, :, :].detach()
        valid = (valid > 0.5).float()  # [B,1,H,W]
        valid = valid.flatten(2)       # [B,1,Tr]

        # hard mask: invalid -> -inf
        hard_bias = (1.0 - valid) * 1e4  # large positive before negation to avoid dtype min overflow
        hard_bias = -hard_bias.squeeze(1)  # [B, Tr]

        # soft bias from "newly supervised" map if present (channel 2)
        if C >= 3 and self.beta_soft > 0:
            soft = rnflt_img[:, 2:3, :, :].detach().clamp(min=0.0, max=1.0).flatten(2).squeeze(1)  # [B, Tr]
            soft_bias = -self.beta_soft * soft
        else:
            soft_bias = torch.zeros_like(hard_bias)

        key_bias = (hard_bias + soft_bias)  # [B, Tr]

        # ---- Build cross-attention bias: [B, heads, Lq=f_len(+CLS? no), Tr] ----
        cross_bias = key_bias[:, None, None, :].expand(B, num_heads, f_len, Tr).contiguous()

        # ---- Build joint self-attention bias over [CLS + F + R] ----
        L = 1 + f_len + r_len
        joint_bias = torch.zeros(B, num_heads, L, L, device=rnflt_img.device, dtype=rnflt_img.dtype)
        # columns corresponding to RNFLT tokens
        start_r = 1 + f_len
        joint_bias[:, :, :, start_r:start_r + r_len] = key_bias[:, None, None, :]

        return joint_bias, cross_bias

    def forward(self,
                x_fundus: torch.Tensor,
                x_rnflt: Optional[torch.Tensor] = None) -> torch.Tensor:
        fundus, rnflt = self._split_or_route(x_fundus, x_rnflt)

        # Encode spatial features
        if fundus is None and rnflt is None:
            raise ValueError("At least one modality must be provided.")

        if fundus is not None:
            f_map = self.fundus_enc(fundus)  # [B, Cf, Hf, Wf] (usually 7x7)
            B = f_map.size(0)
            device, dtype = f_map.device, f_map.dtype
        else:
            r_probe = self.rnflt_enc(rnflt)
            B = r_probe.size(0)
            device, dtype = r_probe.device, r_probe.dtype

        # Tokenize
        if fundus is not None:
            f_tok, Hf, Wf = self._tokenize(f_map, self.f_proj, self.f_type)  # [B, Tf, D]
        else:
            f_tok = torch.zeros(B, 0, self.f_proj.out_features, device=device, dtype=dtype)
            Hf = Wf = 0

        if rnflt is not None:
            r_map = self.rnflt_enc(rnflt)  # features for tokens
            r_tok, Hr, Wr = self._tokenize(r_map, self.r_proj, self.r_type)  # [B, Tr, D]
        else:
            r_tok = torch.zeros(B, 0, self.r_proj.out_features, device=device, dtype=dtype)
            Hr = Wr = 0

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]

        # Biases from RNFLT channels (if available)
        # NOTE: rnflt here is the ORIGINAL input image tensor (not features) to read mask channels
        joint_bias, cross_bias = self._build_biases(
            rnflt_img=rnflt, f_len=f_tok.size(1), r_len=r_tok.size(1), num_heads=self.blocks[0].self_attn.heads
        )

        # Run fusion blocks
        for blk in self.blocks:
            cls, f_tok, r_tok = blk(cls, f_tok, r_tok, joint_bias, cross_bias)

        # Classifier on CLS
        out = self.head(self.norm(cls)).squeeze(1)  # [B, 1]
        return out


def build_dual_branch_resnet_attn(fundus_encoder: str = "resnet18",
                                  rnflt_encoder: str = "resnet18",
                                  rnflt_channels: int = 1,
                                  pretrained_fundus: bool = True,
                                  pretrained_rnflt: bool = False,
                                  d_model: int = 384,
                                  num_layers: int = 2,
                                  num_heads: int = 6,
                                  mlp_mult: float = 2.0,
                                  dropout: float = 0.1,
                                  beta_soft: float = 1.5) -> DualBranchResNetAttn:
    """
    Factory for the attention-fusion model. Signature mirrors build_dual_branch_resnet,
    with extra knobs for the attention head kept optional.
    """
    return DualBranchResNetAttn(
        fundus_encoder=fundus_encoder,
        rnflt_encoder=rnflt_encoder,
        fundus_in_channels=3,
        rnflt_in_channels=rnflt_channels,
        pretrained_fundus=pretrained_fundus,
        pretrained_rnflt=pretrained_rnflt,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_mult=mlp_mult,
        dropout=dropout,
        beta_soft=beta_soft
    )


# ================================
# Single-stream model, RNFLT transform, config-driven builder
# ================================
def build_single_stream_model(name, in_channels, pretrained):
    """ResNet with the first conv replaced for in_channels (pretrained conv1 weights are discarded
    even for 3-channel input; this is what the paper's fundus baseline was trained with)."""
    if name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    elif name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    elif name == 'mlp':
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 224 * 224, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )
    else:
        raise ValueError(f"Unknown model {name}")


class RNFLTTransform:
    """
    Converts an RNFLT array -> torch tensor [1,H,W], fills NaNs, applies normalization.
    Accepts either a NumPy array (HxW) or a torch.Tensor (HxW or 1xHxW).
    Applies light blur/noise when aug='basic'.
    """
    def __init__(self, norm='zscore', mean=None, std=None, vmin=0.0, vmax=350.0,
                 fill_value=0.0, aug='none'):
        self.norm = norm
        self.mean = mean
        self.std = std
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.fill_value = float(fill_value)
        self.aug = aug

    def __call__(self, rnflt_in):
        if isinstance(rnflt_in, torch.Tensor):
            x = rnflt_in.detach().float()
        else:
            x = torch.from_numpy(rnflt_in).float()

        if x.dim() == 3 and x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.dim() == 3 and x.shape[0] > 1:
            x = x[0]
        elif x.dim() != 2:
            raise ValueError(f"RNFLTTransform expected HxW or 1xHxW, got shape {tuple(x.shape)}")

        nan_mask = ~torch.isfinite(x)
        if nan_mask.any():
            x = x.clone()
            x[nan_mask] = self.fill_value

        if self.norm == 'zscore':
            assert (self.mean is not None) and (self.std is not None) and (self.std > 0), \
                "zscore requires mean and std."
            x = (x - self.mean) / self.std
        elif self.norm == 'minmax':
            denom = max(self.vmax - self.vmin, 1e-6)
            x = (x - self.vmin) / denom
        elif self.norm == 'none':
            pass
        else:
            raise ValueError(self.norm)

        x = x.unsqueeze(0)  # 1xHxW

        if self.aug == 'basic':
            if torch.rand(1).item() < 0.5:
                x = TF.gaussian_blur(x, kernel_size=3)
            if torch.rand(1).item() < 0.5:
                x = x + 0.01 * torch.randn_like(x)

        return x


def zero_rnflt_mask_channel_if_present(model, expected_in_channels):
    """Zero the mask-channel (index 1) weights of the RNFLT encoder's first conv when the
    RNFLT input has >= 2 channels. Works for both fusion models."""
    if expected_in_channels < 2:
        return
    enc = getattr(model, 'rnflt_enc', None)
    if enc is None:
        return

    conv = getattr(enc.backbone, 'conv1', None)
    if conv is None:
        for m in enc.backbone.modules():
            if isinstance(m, nn.Conv2d):
                conv = m
                break

    if isinstance(conv, nn.Conv2d) and conv.in_channels == expected_in_channels:
        with torch.no_grad():
            if conv.weight.shape[1] >= 2:
                conv.weight[:, 1, :, :] = 0.0


def build_classifier_from_config(config):
    """Rebuild the classifier a run was trained with from its logged config.json.
    Returns (model, dual_branch): dual_branch models take (fundus, rnflt) separately."""
    input_type = config['input_type']
    model_name = config['model_name']
    pretrained = bool(config['pretrained'])
    rnflt_channels = int(config['rnflt_channels'])

    if input_type in ('fused_real', 'fused_pred'):
        if model_name in ('resnet18_attn', 'resnet50_attn'):
            backbone_name = 'resnet18' if model_name == 'resnet18_attn' else 'resnet50'
            model = build_dual_branch_resnet_attn(
                fundus_encoder=backbone_name,
                rnflt_encoder=backbone_name,
                rnflt_channels=rnflt_channels,
                pretrained_fundus=pretrained,
                pretrained_rnflt=False,
                d_model=config['attn_d_model'],
                num_layers=config['attn_layers'],
                num_heads=config['attn_heads'],
                mlp_mult=config['attn_mlp_mult'],
                dropout=config['attn_dropout'],
                beta_soft=config['attn_beta_soft'],
            )
        else:
            assert model_name in ('resnet18', 'resnet50'), "Fusion baseline expects a ResNet encoder (resnet18/resnet50)."
            model = build_dual_branch_resnet(
                fundus_encoder=model_name,
                rnflt_encoder=model_name,
                rnflt_channels=rnflt_channels,
                pretrained_fundus=pretrained,
                pretrained_rnflt=False,
                dropout=0.3,
                hidden_dim=None,
            )
        zero_rnflt_mask_channel_if_present(model, expected_in_channels=rnflt_channels)
        return model, True

    if input_type == 'fundus':
        in_channels = 3
    elif input_type in ('rnflt_real', 'rnflt_pred'):
        in_channels = rnflt_channels
    else:
        raise ValueError(input_type)
    model = build_single_stream_model(model_name, in_channels, pretrained=pretrained)
    if input_type in ('rnflt_real', 'rnflt_pred') and rnflt_channels == 2:
        with torch.no_grad():
            model.conv1.weight[:, 1, :, :] = 0.0  # channel 1 is mask; channel 0 is rnflt
    return model, False
