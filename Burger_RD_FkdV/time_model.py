import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)


def _configure_xla_cuda_data_dir():
    # Must run before importing jax so XLA can find libdevice on GPU nodes.
    if "XLA_FLAGS" in os.environ and "xla_gpu_cuda_data_dir" in os.environ["XLA_FLAGS"]:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []

    if conda_prefix:
        candidates.extend(
            [
                os.path.join(conda_prefix, "lib", pyver, "site-packages", "jaxlib", "cuda"),
                os.path.join(conda_prefix, "lib", pyver, "site-packages", "triton", "third_party", "cuda"),
                os.path.join(conda_prefix, "lib"),
            ]
        )

    for root in list(sys.path):
        if "site-packages" in root:
            candidates.extend(
                [
                    os.path.join(root, "jaxlib", "cuda"),
                    os.path.join(root, "triton", "third_party", "cuda"),
                ]
            )

    checked = set()
    for cand in candidates:
        if not cand or cand in checked:
            continue
        checked.add(cand)

        matches = glob.glob(os.path.join(cand, "**", "libdevice.10.bc"), recursive=True)
        if matches:
            # Point XLA directly at the directory containing libdevice.10.bc.
            libdevice_dir = os.path.dirname(matches[0])
            old_flags = os.environ.get("XLA_FLAGS", "").strip()
            new_flag = f"--xla_gpu_cuda_data_dir={libdevice_dir}"
            os.environ["XLA_FLAGS"] = f"{old_flags} {new_flag}".strip() if old_flags else new_flag

        ptxas_matches = glob.glob(os.path.join(cand, "**", "ptxas"), recursive=True)
        if ptxas_matches:
            ptxas_dir = os.path.dirname(ptxas_matches[0])
            parent_cuda_dir = os.path.dirname(ptxas_dir)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if ptxas_dir not in path_parts:
                os.environ["PATH"] = ptxas_dir + os.pathsep + os.environ.get("PATH", "")
            if not os.environ.get("CUDA_HOME"):
                os.environ["CUDA_HOME"] = parent_cuda_dir
            return


_configure_xla_cuda_data_dir()

try:
    import jax
    from jax import dlpack as jax_dlpack
    import jax.numpy as jnp
    import jax.random as jr
    from linoss.models.LinOSS import LinOSSLayer, apply_linoss_im, apply_linoss_imex
    _LINOSS_SOURCE_AVAILABLE = True
    _LINOSS_SOURCE_IMPORT_ERROR = None
except Exception as e:
    _LINOSS_SOURCE_AVAILABLE = False
    _LINOSS_SOURCE_IMPORT_ERROR = e


def _torch_to_jax(tensor, force_cpu=False):
    use_dlpack = (
        _LINOSS_SOURCE_AVAILABLE
        and (not force_cpu)
        and tensor.is_cuda
        and jax.default_backend() == "gpu"
    )
    if use_dlpack:
        return jax_dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))
    arr = tensor.detach().to("cpu").numpy().astype(np.float32, copy=False)
    return jnp.asarray(arr)


def _jax_to_torch(jax_arr, like_tensor, force_cpu=False):
    use_dlpack = (
        _LINOSS_SOURCE_AVAILABLE
        and (not force_cpu)
        and like_tensor.is_cuda
        and jax.default_backend() == "gpu"
    )
    if use_dlpack:
        t = torch.utils.dlpack.from_dlpack(jax_dlpack.to_dlpack(jax_arr))
        return t.to(device=like_tensor.device, dtype=like_tensor.dtype)
    arr = np.array(jax_arr, dtype=np.float32, copy=True)
    return torch.from_numpy(arr).to(device=like_tensor.device, dtype=like_tensor.dtype)


class GRU(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(GRU, self).__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)  # Output layer to match input dimensions

    def forward(self, x):
        x = self.in_proj(x)
        lstm_out, _ = self.lstm(x)
        out = self.out_proj(lstm_out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, output_dim)  # Output layer to match input dimensions

    def forward(self, x):
        x = self.in_proj(x)
        lstm_out, (_, _) = self.lstm(x)
        out = self.out_proj(lstm_out)
        return out


@torch.jit.script
def _oss_no_rollout_im(
    x: torch.Tensor,
    omega2: torch.Tensor,
    damping: torch.Tensor,
    drive: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    h = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
    v = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
    ys = torch.empty(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)

    # IM update: z_n uses y_n (implicit in position component).
    inv_denom = 1.0 / (1.0 + (dt * dt) * omega2)
    for t in range(seq_len):
        u_t = x[:, t, :]
        rhs = v + dt * (-omega2 * h - damping * v + drive * u_t)
        v = rhs * inv_denom
        h = h + dt * v
        ys[:, t, :] = h

    return ys


@torch.jit.script
def _oss_no_rollout_imex(
    x: torch.Tensor,
    omega2: torch.Tensor,
    damping: torch.Tensor,
    drive: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, d_model = x.shape
    h = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
    v = torch.zeros(batch_size, d_model, device=x.device, dtype=x.dtype)
    ys = torch.empty(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)

    # IMEX update from paper Eq. (5): z_n uses y_{n-1} explicitly.
    for t in range(seq_len):
        u_t = x[:, t, :]
        v = v + dt * (-omega2 * h - damping * v + drive * u_t)
        h = h + dt * v
        ys[:, t, :] = h

    return ys


class OSS_NO(nn.Module):
    """Minimal oscillatory state-space temporal model (LinOSS-style)."""

    def __init__(self, d_model, input_dim, output_dim, discretization="IMEX", dt=1.0, dt_min=1e-3, dt_max=1.0):
        super(OSS_NO, self).__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.out_proj = nn.Linear(d_model, output_dim)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)
        if not (0.0 < self.dt_min < self.dt_max):
            raise ValueError(
                f"Invalid dt bounds for OSS_NO: dt_min={self.dt_min}, dt_max={self.dt_max}. "
                "Require 0 < dt_min < dt_max."
            )

        # Match LinOSS source behavior: learn per-channel steps and constrain via sigmoid.
        dt_init = float(np.clip(float(dt), self.dt_min + 1e-6, self.dt_max - 1e-6))
        dt_ratio = (dt_init - self.dt_min) / (self.dt_max - self.dt_min)
        dt_logit_init = float(np.log(dt_ratio / (1.0 - dt_ratio)))
        self.dt_logits = nn.Parameter(torch.full((d_model,), dt_logit_init, dtype=torch.float32))
        self.discretization = discretization.upper()
        if self.discretization not in {"IM", "IMEX"}:
            raise ValueError(
                f"Invalid discretization '{discretization}' for OSS_NO. Use 'IM' or 'IMEX'."
            )

        # Diagonal oscillator parameters.
        self.omega = nn.Parameter(torch.randn(d_model) * 0.02)
        self.damping = nn.Parameter(torch.randn(d_model) * 0.02)
        self.drive = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x):
        x = self.in_proj(x)

        omega = torch.nn.functional.softplus(self.omega)
        damping = torch.nn.functional.softplus(self.damping)
        omega2 = omega * omega
        dt = self.dt_min + (self.dt_max - self.dt_min) * torch.sigmoid(self.dt_logits)
        dt = dt.to(device=x.device, dtype=x.dtype)

        if self.discretization == "IM":
            y = _oss_no_rollout_im(x, omega2, damping, self.drive, dt)
        else:
            y = _oss_no_rollout_imex(x, omega2, damping, self.drive, dt)

        y = self.out_proj(y)
        return y


class _LinOSSBridgeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A_diag, B, C, D, steps, module):
        x_jax = _torch_to_jax(x.detach(), force_cpu=module._force_cpu_jax)
        y_jax = module._jax_forward_with_params(
            x_jax,
            _torch_to_jax(A_diag.detach(), force_cpu=module._force_cpu_jax),
            _torch_to_jax(B.detach(), force_cpu=module._force_cpu_jax),
            _torch_to_jax(C.detach(), force_cpu=module._force_cpu_jax),
            _torch_to_jax(D.detach(), force_cpu=module._force_cpu_jax),
            _torch_to_jax(steps.detach(), force_cpu=module._force_cpu_jax),
        )
        y = _jax_to_torch(y_jax, x, force_cpu=module._force_cpu_jax)
        ctx.module = module
        ctx.save_for_backward(
            x.detach(),
            A_diag.detach(),
            B.detach(),
            C.detach(),
            D.detach(),
            steps.detach(),
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x_saved, A_saved, B_saved, C_saved, D_saved, S_saved = ctx.saved_tensors
        grad_x_jax, grad_A_jax, grad_B_jax, grad_C_jax, grad_D_jax, grad_S_jax = ctx.module._jax_vjp_all(
            _torch_to_jax(x_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(A_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(B_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(C_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(D_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(S_saved.detach(), force_cpu=ctx.module._force_cpu_jax),
            _torch_to_jax(grad_output.detach(), force_cpu=ctx.module._force_cpu_jax),
        )

        grad_x = _jax_to_torch(jnp.real(grad_x_jax), grad_output, force_cpu=ctx.module._force_cpu_jax)
        grad_A = _jax_to_torch(jnp.real(grad_A_jax), A_saved, force_cpu=ctx.module._force_cpu_jax)
        grad_B = _jax_to_torch(jnp.real(grad_B_jax), B_saved, force_cpu=ctx.module._force_cpu_jax)
        grad_C = _jax_to_torch(jnp.real(grad_C_jax), C_saved, force_cpu=ctx.module._force_cpu_jax)
        grad_D = _jax_to_torch(jnp.real(grad_D_jax), D_saved, force_cpu=ctx.module._force_cpu_jax)
        grad_S = _jax_to_torch(jnp.real(grad_S_jax), S_saved, force_cpu=ctx.module._force_cpu_jax)
        return grad_x, grad_A, grad_B, grad_C, grad_D, grad_S, None


class OSS_source(nn.Module):
    """Source-backed LinOSS temporal model using JAX LinOSS core layer."""

    def __init__(self, d_model, input_dim, output_dim, discretization="IM", seed=0):
        super(OSS_source, self).__init__()
        if not _LINOSS_SOURCE_AVAILABLE:
            raise ImportError(
                "LinOSS source import failed. Ensure JAX/equinox are installed and the cloned "
                "linoss repo is available in the project root. Original error: "
                f"{_LINOSS_SOURCE_IMPORT_ERROR}"
            )

        self.in_proj = nn.Linear(input_dim, d_model)
        self.out_proj = nn.Linear(d_model, output_dim)

        self.discretization = discretization
        self._force_cpu_jax = False

        # Initialize from LinOSS source core block directly (JAX implementation).
        try:
            linoss_init = LinOSSLayer(
                ssm_size=d_model,
                H=d_model,
                discretization=discretization,
                key=jr.PRNGKey(seed),
            )
        except Exception as e:
            # Robust fallback for cluster nodes where GPU-JAX cannot locate libdevice.
            if "libdevice" not in str(e).lower():
                raise
            self._force_cpu_jax = True
            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                linoss_init = LinOSSLayer(
                    ssm_size=d_model,
                    H=d_model,
                    discretization=discretization,
                    key=jr.PRNGKey(seed),
                )
            print("[OSS_source] Falling back to CPU JAX path due to libdevice resolution error.")

        self.linoss_A_diag = nn.Parameter(torch.tensor(np.array(linoss_init.A_diag), dtype=torch.float32))
        self.linoss_B = nn.Parameter(torch.tensor(np.array(linoss_init.B), dtype=torch.float32))
        self.linoss_C = nn.Parameter(torch.tensor(np.array(linoss_init.C), dtype=torch.float32))
        self.linoss_D = nn.Parameter(torch.tensor(np.array(linoss_init.D), dtype=torch.float32))
        self.linoss_steps = nn.Parameter(torch.tensor(np.array(linoss_init.steps), dtype=torch.float32))

    def _jax_forward_with_params(self, x_jax, A_diag, B, C, D, steps):
        # x_jax: (batch, seq_len, d_model)
        def _compute(xx, aa, bb, cc, dd, ss):
            A_act = jax.nn.relu(aa)
            B_complex = bb[..., 0] + 1j * bb[..., 1]
            C_complex = cc[..., 0] + 1j * cc[..., 1]
            steps_act = jax.nn.sigmoid(ss)

            def _single(input_sequence):
                if self.discretization == "IMEX":
                    ys = apply_linoss_imex(A_act, B_complex, C_complex, input_sequence, steps_act)
                else:
                    ys = apply_linoss_im(A_act, B_complex, C_complex, input_sequence, steps_act)
                Du = jax.vmap(lambda u: dd * u)(input_sequence)
                return ys + Du

            return jax.vmap(_single)(xx)

        def _compute_on_cpu(xx, aa, bb, cc, dd, ss):
            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                xx = jax.device_put(xx, cpu_device)
                aa = jax.device_put(aa, cpu_device)
                bb = jax.device_put(bb, cpu_device)
                cc = jax.device_put(cc, cpu_device)
                dd = jax.device_put(dd, cpu_device)
                ss = jax.device_put(ss, cpu_device)
                return _compute(xx, aa, bb, cc, dd, ss)

        if self._force_cpu_jax:
            return _compute_on_cpu(x_jax, A_diag, B, C, D, steps)

        try:
            return _compute(x_jax, A_diag, B, C, D, steps)
        except Exception as e:
            if "libdevice" not in str(e).lower() and "ptxas" not in str(e).lower():
                raise
            self._force_cpu_jax = True
            print("[OSS_source] Switching LinOSS forward to CPU JAX due to CUDA toolchain error.")
            return _compute_on_cpu(x_jax, A_diag, B, C, D, steps)

    def _jax_vjp_all(self, x_jax, A_diag, B, C, D, steps, grad_y_jax):
        try:
            _, pullback = jax.vjp(self._jax_forward_with_params, x_jax, A_diag, B, C, D, steps)
            grad_x, grad_A, grad_B, grad_C, grad_D, grad_S = pullback(grad_y_jax)
            return grad_x, grad_A, grad_B, grad_C, grad_D, grad_S
        except Exception as e:
            if "libdevice" not in str(e).lower() and "ptxas" not in str(e).lower():
                raise
            self._force_cpu_jax = True
            print("[OSS_source] Switching LinOSS backward to CPU JAX due to CUDA toolchain error.")
            cpu_device = jax.devices("cpu")[0]
            with jax.default_device(cpu_device):
                x_jax = jax.device_put(x_jax, cpu_device)
                A_diag = jax.device_put(A_diag, cpu_device)
                B = jax.device_put(B, cpu_device)
                C = jax.device_put(C, cpu_device)
                D = jax.device_put(D, cpu_device)
                steps = jax.device_put(steps, cpu_device)
                grad_y_jax = jax.device_put(grad_y_jax, cpu_device)
                _, pullback = jax.vjp(self._jax_forward_with_params, x_jax, A_diag, B, C, D, steps)
                grad_x, grad_A, grad_B, grad_C, grad_D, grad_S = pullback(grad_y_jax)
                return grad_x, grad_A, grad_B, grad_C, grad_D, grad_S

    def forward(self, x):
        x = self.in_proj(x)
        y = _LinOSSBridgeFn.apply(
            x,
            self.linoss_A_diag,
            self.linoss_B,
            self.linoss_C,
            self.linoss_D,
            self.linoss_steps,
            self,
        )
        y = self.out_proj(y)
        return y


class linoss_pytorch_on(nn.Module):
    """
    PyTorch Implementation of LinOSS Temporal Model.
    
    This class wraps the native PyTorch LinOSS implementation (from linoss_pytorch/)
    instead of the JAX reference implementation. It provides the same interface as
    OSS_source but uses pure PyTorch for all computations.
    
    Key Features:
    - Full PyTorch implementation (no JAX dependency)
    - Supports both IM (Implicit) and IMEX (Implicit-Explicit) discretization schemes
    - Efficient computation with complex-valued state-space matrices
    - Direct gradient computation through PyTorch's autograd
    
    Architecture:
    Input Projection → LinOSS Core (IM or IMEX) → Output Projection
    
    Paper Reference: "Linear Oscillatory State-Space Models: Stable and Efficient Sequence Learning"
    """
    
    def __init__(self, d_model, input_dim, output_dim, discretization="IM", seed=0, num_layers=1, dropout=0.05):
        """
        Initialize PyTorch-based LinOSS temporal model.
        
        Args:
            d_model (int): Internal dimension for LinOSS state-space model (ssm_size = d_model)
            input_dim (int): Input feature dimension (temporal dimension from FNO)
            output_dim (int): Output feature dimension (same as input_dim in temporal processing)
            discretization (str): Either 'IM' (implicit) or 'IMEX' (implicit-explicit)
                                 IM: exponentially stable, dissipative dynamics
                                 IMEX: symplectic, volume-preserving dynamics
            seed (int): Random seed for initialization (PyTorch uses global seed)
            num_layers (int): Number of LinOSS blocks to stack (default: 1)
            dropout (float): Dropout rate for regularization (default: 0.05)
        """
        super(linoss_pytorch_on, self).__init__()
        
        # Import LinOSS PyTorch implementation
        _linoss_pytorch_path = os.path.join(_PROJECT_ROOT, "linoss_pytorch")
        if _linoss_pytorch_path not in sys.path:
            sys.path.insert(0, _linoss_pytorch_path)
        
        try:
            from linoss_pytorch import LinOSS, LinOSSLayer
            self._linoss_pytorch_available = True
        except ImportError as e:
            raise ImportError(
                f"PyTorch LinOSS implementation not found in {_linoss_pytorch_path}. "
                "Ensure linoss_pytorch/ directory exists with linoss_pytorch.py. "
                f"Error: {e}"
            )
        
        # Store configuration
        self.d_model = d_model
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.discretization = discretization
        self.num_layers = num_layers
        
        # Device will be set when the module is moved to a device
        self.device_cache = None
        
        # Input projection: input_dim → d_model
        self.in_proj = nn.Linear(input_dim, d_model)
        
        # Initialize LinOSS core model
        # Paper Eq. 1-4: Full LinOSS state-space model with multiple blocks
        self.linoss_core = LinOSS(
            input_dim=d_model,              # Feature dimension (H in paper)
            output_dim=d_model,             # Output dimension (same as input for temporal)
            ssm_size=d_model,               # State-space size (P in paper, one mode per feature)
            feature_dim=d_model,            # Internal feature dimension
            num_blocks=num_layers,          # Stack multiple LinOSSBlocks
            discretization=discretization,   # IM or IMEX scheme
            dropout=dropout,                # Regularization
            device='cpu'                    # Will be moved to correct device in forward
        )
        
        # Output projection: d_model → output_dim
        self.out_proj = nn.Linear(d_model, output_dim)
    
    def _get_device(self):
        """Get the device from the first parameter."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    def forward(self, x):
        """
        Forward pass through PyTorch LinOSS temporal model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
                            Comes from spatial FNO processing
                            
        Returns:
            y (torch.Tensor): Output tensor of shape (batch_size, seq_len, output_dim)
                            Processed temporal sequence
        
        Computation Flow:
        1. Input projection: (batch, seq_len, input_dim) → (batch, seq_len, d_model)
        2. Process each batch element through LinOSS:
           - Reshape to (seq_len, d_model) for sequence processing
           - Apply LinOSS core with Paper Eq. 3-4 (IM) or Eq. 5-6 (IMEX)
           - Reshape back to (batch, seq_len, d_model)
        3. Output projection: (batch, seq_len, d_model) → (batch, seq_len, output_dim)
        """
        
        # Ensure LinOSS core is on the correct device
        device = self._get_device()
        if self.linoss_core.input_proj.weight.device != device:
            self.linoss_core = self.linoss_core.to(device)
        
        # Step 1: Input projection (batch, seq_len, input_dim) → (batch, seq_len, d_model)
        x_proj = self.in_proj(x)  # Shape: (batch, seq_len, d_model)

        # Step 2: Vectorized LinOSS over batch (avoids Python loop over batch dimension)
        y = self.linoss_core(x_proj)  # Shape: (batch, seq_len, d_model)

        # Step 3: Output projection (batch, seq_len, d_model) → (batch, seq_len, output_dim)
        y = self.out_proj(y)  # Shape: (batch, seq_len, output_dim)
        
        return y


import math
from functools import partial
import json
import copy

from collections import namedtuple

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(
            Mamba2 if ssm_layer == "Mamba2" else Mamba,
            layer_idx=layer_idx,
            **ssm_cfg,
            **factory_kwargs
        )
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs
        )
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to a common residual-scaling scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )
        return hidden_states


class Mamba_NO(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        input_dim: int,
        output_dim: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        self.embedding = nn.Linear(input_dim, d_model)
        self.out_embedding = nn.Linear(d_model, output_dim)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm)
            )

        hidden_states = self.out_embedding(hidden_states)
        return hidden_states


if __name__ == "__main__":
    pass