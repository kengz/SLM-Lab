"""L1 Being Embedding — Layer 1 of the SLM agent architecture.

Implements the being-time embedding pipeline:
    L0Output → channel attention → hierarchical fusion → temporal integration → (B, 512)

Philosophy: Heidegger's three temporal ecstases (thrownness, falling, projection)
constitute the temporal structure of Dasein. This module operationalizes that structure
as a computational pipeline.

Source: notes/layers/L1-being-embedding.md
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# L0 Interface
# ---------------------------------------------------------------------------

@dataclass
class L0Output:
    """Channel embeddings produced by L0 perception pipeline.

    All present fields are (B, 512) tensors.
    proprioception is always present; others are phase-dependent.
    """
    proprioception: torch.Tensor          # (B, 512) — always present
    vision: torch.Tensor | None = None    # (B, 512) — Phase 3.2b+
    audio: torch.Tensor | None = None     # (B, 512) — Phase 3.2b+
    object_state: torch.Tensor | None = None  # (B, 512) — Phase 3.2a only

    def to_channel_stack(self) -> torch.Tensor:
        """Returns (B, N_channels, 512) in canonical order."""
        channels = [self.proprioception]
        if self.vision is not None:
            channels.append(self.vision)
        if self.audio is not None:
            channels.append(self.audio)
        if self.object_state is not None:
            channels.append(self.object_state)
        return torch.stack(channels, dim=1)

    def get_channel_types(self) -> list[str]:
        """Return ordered list of active channel type names."""
        types = ['proprioception']
        if self.vision is not None:
            types.append('vision')
        if self.audio is not None:
            types.append('audio')
        if self.object_state is not None:
            types.append('object_state')
        return types


@dataclass
class L1Output:
    """Output of Layer 1, consumed by Layer 2 and higher layers."""
    being_embedding: torch.Tensor        # (B, 512) spatial integration (present only)
    being_time_embedding: torch.Tensor   # (B, 512) full temporal integration
    h_t: torch.Tensor                    # (B, 1024) GRU state for next step
    thrownness: torch.Tensor             # (B, 512) past channel
    falling: torch.Tensor                # (B, 512) present channel
    projection: torch.Tensor             # (B, 512) future channel


# ---------------------------------------------------------------------------
# Channel Type Embedding
# ---------------------------------------------------------------------------

class ChannelTypeEmbedding(nn.Module):
    """Learnable per-channel-type position embedding added to channel stack.

    Allows ChannelAttention to distinguish modality types.
    Params: 4 × 512 = 2K.
    """
    CHANNEL_TYPES = ['proprioception', 'vision', 'audio', 'object_state']

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.embeddings = nn.Embedding(len(self.CHANNEL_TYPES), d_model)

    def forward(self, channel_stack: torch.Tensor,
                channel_types: list[str]) -> torch.Tensor:
        """Add type embeddings to channel stack.

        Args:
            channel_stack: (B, N, D)
            channel_types: list of N channel type names

        Returns:
            (B, N, D) with type embeddings added
        """
        type_ids = torch.tensor(
            [self.CHANNEL_TYPES.index(t) for t in channel_types],
            device=channel_stack.device,
        )
        type_emb = self.embeddings(type_ids)  # (N, D)
        return channel_stack + type_emb.unsqueeze(0)


# ---------------------------------------------------------------------------
# Channel Attention
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Cross-channel self-attention transformer block.

    Modalities inform each other before fusion (unified disclosure).
    Input/Output: (B, N_channels, 512) — variable N (1–4).

    Architecture: 1 transformer encoder layer (pre-norm).
    Params: ~2.1M (d_model=512, n_heads=8).
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_head = d_model // n_heads  # 64
        self.n_heads = n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Cross-channel attention with pre-norm residual connections.

        Args:
            x: (B, N, D) where N = N_channels

        Returns:
            (B, N, D) attended channel embeddings
        """
        B, N, D = x.shape

        # Pre-norm multi-head self-attention
        residual = x
        x_norm = self.norm1(x)
        q = self.q_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = residual + self.dropout(out)

        # Pre-norm FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Hierarchical Fusion MLP
# ---------------------------------------------------------------------------

class HierarchicalFusion(nn.Module):
    """Attended channels → being embedding.

    Concatenates all channels (zero-padded to max_channels), projects to 512-dim.
    Preserves full channel identity vs mean pooling.

    Input: (B, N_channels, 512), Output: (B, 512).
    Params: ~3.6M (max_channels=4).
    """

    def __init__(self, max_channels: int = 4, d_model: int = 512):
        super().__init__()
        self.max_channels = max_channels
        self.d_model = d_model

        self.fusion = nn.Sequential(
            nn.Linear(max_channels * d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, attended: torch.Tensor) -> torch.Tensor:
        """Fuse attended channels into single being embedding.

        Args:
            attended: (B, N_channels, 512) — N may be < max_channels

        Returns:
            (B, 512) being embedding
        """
        B, N, D = attended.shape

        if N < self.max_channels:
            pad = torch.zeros(B, self.max_channels - N, D, device=attended.device)
            attended = torch.cat([attended, pad], dim=1)

        flat = attended.reshape(B, self.max_channels * D)  # (B, 2048)
        return self.fusion(flat)  # (B, 512)


# ---------------------------------------------------------------------------
# Thrownness Encoder (GRU)
# ---------------------------------------------------------------------------

class ThrownessEncoder(nn.Module):
    """GRU compresses history into thrownness embedding.

    Hidden state = accumulated experience (Heidegger: Geworfenheit).
    Phase 3.2a: thrownness is informative from the start (proprio + object_state history).
    Projection/falling = zeros in 3.2a (world model untrained).

    Input: being_embedding (B, 512), h_prev (B, 1024)
    Output: thrownness (B, 512), h_t (B, 1024)
    Params: ~4.6M.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024,
                 output_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, being_embedding: torch.Tensor,
                h_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update GRU with current being embedding.

        Args:
            being_embedding: (B, 512) current being embedding
            h_prev: (B, 1024) previous hidden state

        Returns:
            thrownness: (B, 512) projected hidden state
            h_t: (B, 1024) updated hidden state for next step
        """
        h_t = self.gru(being_embedding, h_prev)   # (B, 1024)
        thrownness = self.norm(self.proj(h_t))     # (B, 512)
        return thrownness, h_t

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state to zeros."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ---------------------------------------------------------------------------
# Projection Encoder (World Model Imagination)
# ---------------------------------------------------------------------------

class ProjectionEncoder(nn.Module):
    """Imagination rollout → projection embedding.

    Learnable weighted mean pooling over H imagination steps → project to 512.
    Phase 3.2a: unused (world model untrained; BeingEmbedding passes zeros).

    Input: (B, H, 512), Output: (B, 512).
    Params: ~0.5M.
    """

    def __init__(self, d_model: int = 512, n_steps: int = 15):
        super().__init__()
        self.n_steps = n_steps
        self.step_weights = nn.Parameter(torch.ones(n_steps) / n_steps)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, imagined_states: torch.Tensor) -> torch.Tensor:
        """Pool imagined future states into projection embedding.

        Args:
            imagined_states: (B, H, 512) where H ≤ n_steps

        Returns:
            (B, 512) projection embedding
        """
        H = imagined_states.shape[1]
        weights = self.step_weights[:H].softmax(dim=0)  # (H,)
        weighted = (imagined_states * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.proj(weighted)  # (B, 512)


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Transformer over three temporal ecstases → being-time embedding.

    Attends over: [CLS, thrownness (past), falling (present), projection (future)].
    CLS token output = being-time embedding (aggregates all three temporal modes).

    Architecture: 4-layer transformer, pre-norm, 8 heads.
    Params: ~8.4M.
    """

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Learnable temporal position embeddings (3 temporal channels: past, present, future)
        self.temporal_pos = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # CLS token for output aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, thrownness: torch.Tensor, falling: torch.Tensor,
                projection: torch.Tensor) -> torch.Tensor:
        """Integrate three temporal channels via transformer attention.

        Args:
            thrownness: (B, 512) past — GRU compressed history
            falling: (B, 512) present — current being embedding
            projection: (B, 512) future — world model imagination (zeros in 3.2a)

        Returns:
            (B, 512) being-time embedding (CLS token output)
        """
        B = thrownness.shape[0]

        # Stack temporal channels and add positional embeddings
        temporal_stack = torch.stack([thrownness, falling, projection], dim=1)  # (B, 3, 512)
        temporal_stack = temporal_stack + self.temporal_pos.unsqueeze(0)

        # Prepend CLS token → (B, 4, 512)
        cls = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls, temporal_stack], dim=1)

        encoded = self.encoder(sequence)  # (B, 4, 512)
        return self.out_norm(encoded[:, 0, :])  # CLS token → (B, 512)


# ---------------------------------------------------------------------------
# BeingEmbedding — Top-Level L1 Module
# ---------------------------------------------------------------------------

class BeingEmbedding(nn.Module):
    """L1 complete pipeline: L0 channels → being embedding → being-time embedding.

    Forward:
        L0Output → channel type emb → channel attention → hierarchical fusion
               → GRU thrownness → temporal attention → L1Output

    Phase 3.2a: 2 channels (proprio + object_state), projection = zeros.
    Phase 3.2b: 2 channels (proprio + vision), projection from world model.
    Phase 3.2b+: 3 channels (proprio + vision + audio), full temporal structure.

    Output dim: 512 (being_time_embedding).
    Total params: ~19.2M.
    """

    def __init__(self, max_channels: int = 4, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        self.channel_type_emb = ChannelTypeEmbedding(d_model)
        self.channel_attn = ChannelAttention(d_model, n_heads=8)
        self.fusion = HierarchicalFusion(max_channels, d_model)
        self.thrownness_enc = ThrownessEncoder(d_model, hidden_dim=1024)
        self.projection_enc = ProjectionEncoder(d_model, n_steps=15)
        self.temporal_attn = TemporalAttention(d_model, n_heads=8, n_layers=4)

    def forward(
        self,
        l0_output: L0Output,
        h_prev: torch.Tensor,
        imagined_states: torch.Tensor | None = None,
    ) -> L1Output:
        """Full L1 forward pass.

        Args:
            l0_output: L0Output with per-channel embeddings
            h_prev: (B, 1024) GRU hidden state from previous step
            imagined_states: (B, H, 512) from L2 world model, or None (Phase 3.2a)

        Returns:
            L1Output with being_embedding, being_time_embedding, h_t, and temporal channels
        """
        # 1. Build channel stack from L0
        channel_stack = l0_output.to_channel_stack()      # (B, N, 512)
        channel_types = l0_output.get_channel_types()

        # 2. Add channel type embeddings
        channel_stack = self.channel_type_emb(channel_stack, channel_types)

        # 3. Cross-channel attention
        attended = self.channel_attn(channel_stack)        # (B, N, 512)

        # 4. Hierarchical fusion → being embedding (present moment)
        being_embedding = self.fusion(attended)            # (B, 512)

        # 5. Temporal channels
        thrownness, h_t = self.thrownness_enc(being_embedding, h_prev)

        falling = being_embedding  # present moment — no transform (Ax3)

        if imagined_states is not None:
            projection = self.projection_enc(imagined_states)
        else:
            # Phase 3.2a: world model untrained → projection = zeros
            projection = torch.zeros_like(being_embedding)

        # 6. Temporal attention → being-time embedding
        being_time_embedding = self.temporal_attn(thrownness, falling, projection)

        return L1Output(
            being_embedding=being_embedding,
            being_time_embedding=being_time_embedding,
            h_t=h_t,
            thrownness=thrownness,
            falling=falling,
            projection=projection,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize GRU hidden state for episode start."""
        return self.thrownness_enc.init_hidden(batch_size, device)
