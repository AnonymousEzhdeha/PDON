import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaScratchBlock(nn.Module):
    """Pure PyTorch Mamba-style mixer block with selective scan."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: int = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_model // 16)

        # Mamba-style input projection and output projection.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # Depthwise causal conv over sequence on the x branch.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        # Token-conditioned selection maps s_Delta(x), s_B(x), s_C(x).
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # Continuous parameters: A (D_inner, N) and skip D (D_inner,).
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state) * 0.02)
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D_model)
        x_norm = self.norm(x)
        batch_size, seq_len, _ = x_norm.shape

        # In-proj and gate split, as in Mamba's x/z branches.
        xz = self.in_proj(x_norm)
        x_branch, z_branch = torch.chunk(xz, 2, dim=-1)  # (B, L, D_inner) each

        # Depthwise causal conv + SiLU on x branch.
        x_conv = x_branch.transpose(1, 2)  # (B, D_inner, L)
        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = x_conv.transpose(1, 2)  # (B, L, D_inner)
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)
        delta_raw, b_raw, c_raw = torch.split(
            proj,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )

        # Delta is token-conditioned then projected to per-channel steps.
        delta = F.softplus(self.dt_proj(delta_raw)) + 1e-4        # (B, L, D_inner)
        b_sel = b_raw                                              # (B, L, N)
        c_sel = c_raw                                              # (B, L, N)

        state = torch.zeros(
            batch_size,
            self.d_inner,
            self.d_state,
            dtype=x_norm.dtype,
            device=x_norm.device,
        )
        outputs = []

        # A is constrained negative for stability; shape (D_inner, N).
        A = -torch.exp(self.A_log).to(dtype=x_norm.dtype, device=x_norm.device)
        A_e = A.unsqueeze(0).unsqueeze(0)                         # (1, 1, D_inner, N)

        # ZOH-style discretization for each (t, d, n).
        delta_e = delta.unsqueeze(-1)                             # (B, L, D_inner, 1)
        a_bar = torch.exp(delta_e * A_e)                          # (B, L, D_inner, N)
        b_scaled = b_sel.unsqueeze(2)                             # (B, L, 1, N)
        b_bar = ((a_bar - 1.0) / (A_e + 1e-8)) * b_scaled         # (B, L, D_inner, N)

        for t in range(seq_len):
            u_t = x_conv[:, t, :]

            # Selective scan recurrence: h_t = A_bar h_{t-1} + B_bar x_t.
            state = a_bar[:, t, :, :] * state + b_bar[:, t, :, :] * u_t.unsqueeze(-1)

            # Output projection y_t = C(x_t) h_t + D x_t.
            y_t = torch.einsum("bdn,bn->bd", state, c_sel[:, t, :])
            y_t = y_t + self.D_skip.to(dtype=u_t.dtype, device=u_t.device) * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)                           # (B, L, D_inner)

        # Mamba-style multiplicative gate from z branch.
        y = y * F.silu(z_branch)
        return self.out_proj(y)


class MambaScratch(nn.Module):
    """Stack of MambaScratchBlock modules with residual connections."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        n_layers: int = 1,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dt_rank: int = None,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaScratchBlock(
                    d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    dt_rank=dt_rank,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out_proj(h)
