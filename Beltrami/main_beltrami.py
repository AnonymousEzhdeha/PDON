import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.decomposition import PCA
import json
from datetime import datetime
import shutil

import os
import sys
import debugpy
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
####### DEBUGGER
if os.environ.get("DEBUG", "0") == "1":
    debugpy.listen(("0.0.0.0", 5678))
    print("⏳ Waiting for debugger attach...")
    debugpy.wait_for_client()
    print("Debugger attached! 🚀")
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print("GPU name: CPU-only runtime")
#######

from PODDON_TGV import PODDON_GRU, PODDON_LSTM, PODDON_Mamba, MambaConfig, PODDON_OSS_NO, PODDON_Mamba_Scratch

from PODDON_TGV import GalerkinTransformer, Transformer, GNOT


# ========================================================================
# Experiment Run Directory Management
# ========================================================================
def get_next_run_number(base_dir):
    """Find the next available run number (1, 2, 3, ...) and create directory structure."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Find the next available number
    existing_nums = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path) and item.isdigit():
                existing_nums.append(int(item))
    
    next_num = max(existing_nums) + 1 if existing_nums else 1
    
    # Create run directory and subdirectories
    run_dir = os.path.join(base_dir, str(next_num))
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    return next_num, run_dir, weights_dir


base_log_dir = os.path.join(SCRIPT_DIR, "run")
run_number, log_dir, weights_dir = get_next_run_number(base_log_dir)
print(f"\n{'='*70}")
print(f"Experiment Run #{run_number}")
print(f"Log directory: {log_dir}")
print(f"Weights directory: {weights_dir}")
print(f"{'='*70}\n")


class TeeStream:
    """Redirect output to multiple streams (terminal and log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


# Redirect stdout/stderr to both terminal and log file
terminal_log_file = os.path.join(log_dir, "terminal_log.txt")
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_log_handle = open(terminal_log_file, "w", buffering=1)
sys.stdout = TeeStream(_orig_stdout, _log_handle)
sys.stderr = TeeStream(_orig_stderr, _log_handle)
print("Logging to:", terminal_log_file)

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--grid_x', type=int, default=17, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=1, help="terminal time")

    parser.add_argument('--N_train', type=int, default=900)
    parser.add_argument('--N_test', type=int, default=100)

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

    parser.add_argument('--model', type=str, default="mamba_scratch", choices=["GRU", "LSTM", "Mamba", "MambaScratch", "mamba_scratch", "OSS", "ST", "GT", "FT", "T", "GNOT"], help="model architecture to use")
    parser.add_argument('--profile_timing', type=int, default=0, help="Log avg forward/backward/step time per batch (1/0)")

    parser.add_argument('--n_components_decode', type=int, default=128)
    parser.add_argument('--n_components_encode', type=int, default=128)

    # OSS-specific arguments
    parser.add_argument('--oss_hidden_dim', type=int, default=256,
                       help="OSS hidden dimension")
    parser.add_argument('--oss_num_layers', type=int, default=1,
                       help="Number of stacked OSS cores")
    parser.add_argument('--oss_discretization', type=str, default="IMEX", 
                       choices=["IM", "IMEX"],
                       help="OSS discretization scheme: IM (implicit, dissipative) or IMEX (implicit-explicit, symplectic)")
    parser.add_argument('--oss_dt', type=float, default=1.0, 
                       help="OSS initial timestep value (will be learned and constrained)")
    parser.add_argument('--oss_dt_min', type=float, default=1e-3, 
                       help="OSS minimum timestep constraint")
    parser.add_argument('--oss_dt_max', type=float, default=1.0, 
                       help="OSS maximum timestep constraint")
    parser.add_argument('--oss_use_layernorm', type=int, default=0,
                       help="Apply LayerNorm before OSS rollout (1/0)")
    parser.add_argument('--oss_residual_weight', type=float, default=0.0,
                       help="Residual mixing weight between stacked OSS layers")
    parser.add_argument('--oss_proj_dropout', type=float, default=0.0,
                       help="Dropout used inside OSS projections")
    parser.add_argument('--oss_robust_dt_init', type=int, default=0,
                       help="Use clipped/logit-safe dt initialization (1/0)")

    # Option 1: token-wise adaptive dt.
    parser.add_argument('--oss_use_input_dt', type=int, default=0,
                       help="Use input-conditioned token-wise dt (1/0)")

    # Option 2: input-conditioned drive/damping.
    parser.add_argument('--oss_use_input_drive_damping', type=int, default=0,
                       help="Use input-conditioned drive/damping (1/0)")
    parser.add_argument('--oss_input_drive_scale', type=float, default=1.0,
                       help="Scale for input-driven drive term")
    parser.add_argument('--oss_input_damping_scale', type=float, default=1.0,
                       help="Scale for input-driven damping term")

    # Option 3: D-skip direct path.
    parser.add_argument('--oss_use_d_skip', type=int, default=0,
                       help="Enable OSS D-skip output path (1/0)")
    parser.add_argument('--oss_d_skip_init', type=float, default=1.0,
                       help="Initial value for OSS D-skip parameter")

    # Option 4: oscillator gating branch.
    parser.add_argument('--oss_use_osc_gate', type=int, default=0,
                       help="Enable oscillator gating branch (1/0)")

    # Option 5: causal depthwise prefilter.
    parser.add_argument('--oss_use_causal_prefilter', type=int, default=0,
                       help="Enable causal depthwise Conv1d prefilter (1/0)")
    parser.add_argument('--oss_prefilter_kernel_size', type=int, default=3,
                       help="Kernel size for causal prefilter")

    # Option 6: expand-and-project oscillator core.
    parser.add_argument('--oss_use_expand_project', type=int, default=0,
                       help="Enable expanded oscillator latent core (1/0)")
    parser.add_argument('--oss_expand_factor', type=int, default=2,
                       help="Expansion factor for oscillator latent core")
    parser.add_argument('--oss_expand_init_scale', type=float, default=0.02,
                       help="Initialization scale for expand/project layers")

    # Option 7: low-rank coupled oscillators.
    parser.add_argument('--oss_use_coupled_oscillators', type=int, default=0,
                       help="Enable low-rank coupled oscillator operators (1/0)")
    parser.add_argument('--oss_coupling_rank', type=int, default=4,
                       help="Low-rank dimension for oscillator coupling")
    parser.add_argument('--oss_coupling_scale', type=float, default=0.05,
                       help="Scale for coupled oscillator contribution")

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# ========================================================================
# Setup Loss Logging and Config File
# ========================================================================
loss_log_path = os.path.join(log_dir, "loss_values.csv")
with open(loss_log_path, "w") as f_loss:
    f_loss.write("epoch,train_loss_epoch_avg,train_rel_l2,test_rel_l2\n")
print("Loss metrics log:", loss_log_path)


# Function to extract model architecture information
def get_model_details(model, model_name):
    """Extract model architecture and parameter details."""
    details = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters_all": sum(p.numel() for p in model.parameters()),
    }
    
    # Try to get model config from state_dict or attributes
    if hasattr(model, 'config'):
        details["model_config"] = str(model.config)
    
    # Extract key attributes if available
    model_attrs = {}
    for attr_name in dir(model):
        if not attr_name.startswith('_'):
            try:
                attr_val = getattr(model, attr_name)
                # Only capture simple types
                if isinstance(attr_val, (int, float, str, bool, list, dict)):
                    model_attrs[attr_name] = attr_val
            except:
                pass
    
    if model_attrs:
        details["model_attributes"] = model_attrs
    
    return details


def get_model_architecture_details(model_name, grid_x, grid_t, n_components_encode, n_components_decode):
    """Extract model-specific architecture parameters based on model type."""
    arch_config = {}
    
    if model_name == "GRU":
        arch_config.update({
            "input_dim": n_components_encode, "output_dim": n_components_decode,
            "hidden_dim": 256, "num_layers": 1
        })
    elif model_name == "LSTM":
        arch_config.update({
            "input_dim": n_components_encode, "output_dim": n_components_decode,
            "hidden_dim": 256, "num_layers": 1
        })
    elif model_name == "OSS":
        arch_config.update({
            "input_dim": n_components_encode, "output_dim": n_components_decode,
            "hidden_dim": args.oss_hidden_dim,
            "num_layers": args.oss_num_layers,
            "discretization": args.oss_discretization,
            "dt": args.oss_dt,
            "dt_min": args.oss_dt_min,
            "dt_max": args.oss_dt_max,
            "use_layernorm": bool(args.oss_use_layernorm),
            "residual_weight": args.oss_residual_weight,
            "proj_dropout": args.oss_proj_dropout,
            "robust_dt_init": bool(args.oss_robust_dt_init),
            "use_input_dt": bool(args.oss_use_input_dt),
            "use_d_skip": bool(args.oss_use_d_skip),
            "d_skip_init": args.oss_d_skip_init,
            "use_input_drive_damping": bool(args.oss_use_input_drive_damping),
            "input_drive_scale": args.oss_input_drive_scale,
            "input_damping_scale": args.oss_input_damping_scale,
            "use_osc_gate": bool(args.oss_use_osc_gate),
            "use_causal_prefilter": bool(args.oss_use_causal_prefilter),
            "prefilter_kernel_size": args.oss_prefilter_kernel_size,
            "use_expand_project": bool(args.oss_use_expand_project),
            "expand_factor": args.oss_expand_factor,
            "expand_init_scale": args.oss_expand_init_scale,
            "use_coupled_oscillators": bool(args.oss_use_coupled_oscillators),
            "coupling_rank": args.oss_coupling_rank,
            "coupling_scale": args.oss_coupling_scale,
        })
    elif model_name == "Mamba":
        arch_config.update({
            "d_model": 256, "n_layer": 1, "input_dim": n_components_encode,
            "output_dim": n_components_decode, "rms_norm": True,
            "residual_in_fp32": True, "fused_add_norm": True
        })
    elif model_name in ["MambaScratch", "mamba_scratch"]:
        arch_config.update({
            "input_dim": n_components_encode,
            "output_dim": n_components_decode,
            "hidden_dim": 256,
            "n_layers": 1,
            "d_state": 16,
            "implementation": "pure_torch_scratch",
        })
    elif model_name in ["ST", "GT", "FT"]:
        arch_config.update({
            "dim_in": n_components_encode, "dim_out": n_components_decode,
            "dim_hid": 256, "depth": 1, "heads": 4, "dim_head": 256,
            "attn_type": model_name[0].lower(), "mlp_dim": 256
        })
    elif model_name == "T":
        arch_config.update({
            "ninput": n_components_encode, "noutput": n_components_decode,
            "nhidden": 256, "dim_feedforward": 256, "nhead": 4, "nlayers": 1
        })
    elif model_name == "GNOT":
        arch_config.update({
            "dim_in": n_components_encode, "dim_out": n_components_decode,
            "dim_hid": 256, "depth": 1, "heads": 4, "dim_head": 256, "n_experts": 2
        })
    
    return arch_config


config_file_path = os.path.join(log_dir, "config.json")

print("Generating Data in FOM...")
def generate_data(N, grid_x, grid_t, T):
    x = np.linspace(0, 2 * np.pi, grid_x)
    y = np.linspace(0, 2 * np.pi, grid_x)
    z = np.linspace(0, 2 * np.pi, grid_x)
    t = np.linspace(0, T, grid_t)

    x, y, z, t = np.meshgrid(x, y, z, t, indexing='ij') # grid_x, grid_x, grid_x, grid_t

    x = np.expand_dims(x, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    y = np.expand_dims(y, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    z = np.expand_dims(z, axis=0) # 1, grid_x, grid_x, grid_x, grid_t
    t = np.expand_dims(t, axis=0) # 1, grid_x, grid_x, grid_x, grid_t

    a = np.random.rand(N, 1, 1, 1, 1)
    pm = (np.random.rand(N, 1, 1, 1, 1) > 0.5) + 0.0
    d = np.random.rand(N, 1, 1, 1, 1) + 0.5

    u = -a * (np.exp(a * x) * np.sin(a * y + pm * d * z) + np.exp(a * z) * np.cos(a * x + pm * d * y)) * np.exp(-d*d*t)
    v = -a * (np.exp(a * y) * np.sin(a * z + pm * d * x) + np.exp(a * x) * np.cos(a * y + pm * d * z)) * np.exp(-d*d*t)
    w = -a * (np.exp(a * z) * np.sin(a * x + pm * d * y) + np.exp(a * y) * np.cos(a * z + pm * d * x)) * np.exp(-d*d*t)

    U = np.stack([u, v, w], -1) # N, grid_x, grid_x, grid_x, grid_t, 3

    U_init = U[:, :, :, :, 0, :] # N, grid_x, grid_x, grid_x, 3
    U_init = np.expand_dims(U_init, 1) # N, 1, grid_x, grid_x, grid_x, 3
    U_init = np.tile(U_init, (1, grid_t, 1, 1, 1, 1)) # N, grid_t, grid_x, grid_x, grid_x, 3
    U_init = U_init.reshape(N, grid_t, -1)

    U_x0, U_x1 = U[:, 0, :, :, :, :], U[:, -1, :, :, :, :] # N, grid_x, grid_x, grid_t, 3
    U_x0, U_x1 = np.transpose(U_x0, (0, 3, 1, 2, 4)), np.transpose(U_x1, (0, 3, 1, 2, 4))
    U_x0, U_x1 = U_x0.reshape(N, grid_t, -1), U_x1.reshape(N, grid_t, -1)

    U_y0, U_y1 = U[:, :, 0, :, :, :], U[:, :, -1, :, :, :] # N, grid_x, grid_x, grid_t, 3
    U_y0, U_y1 = np.transpose(U_y0, (0, 3, 1, 2, 4)), np.transpose(U_y1, (0, 3, 1, 2, 4))
    U_y0, U_y1 = U_y0.reshape(N, grid_t, -1), U_y1.reshape(N, grid_t, -1)

    U_z0, U_z1 = U[:, :, :, 0, :, :], U[:, :, :, -1, :, :] # N, grid_x, grid_x, 1, grid_t, 3
    U_z0, U_z1 = np.transpose(U_z0, (0, 3, 1, 2, 4)), np.transpose(U_z1, (0, 3, 1, 2, 4))
    U_z0, U_z1 = U_z0.reshape(N, grid_t, -1), U_z1.reshape(N, grid_t, -1)

    F = np.concatenate([U_init, U_x0, U_x1, U_y0, U_y1, U_z0, U_z1], -1) # N, grid_t, -1


    U = np.transpose(U, (0, 4, 1, 2, 3, 5)) # N, grid_t, grid_x, grid_x, grid_x, 3
    U = U.reshape(N, grid_t, -1)

    return U, F

# u_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T)
# u_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T)

u_train, f_train = [], []
for i in tqdm(range(args.N_train // 10)):
    u_, f_ = generate_data(10, args.grid_x, args.grid_t, args.T)
    u_train.append(u_)
    f_train.append(f_)
u_train = np.concatenate(u_train, 0)
f_train = np.concatenate(f_train, 0)

u_test, f_test = [], []
for i in tqdm(range(args.N_test // 10)):
    u_, f_ = generate_data(10, args.grid_x, args.grid_t, args.T)
    u_test.append(u_)
    f_test.append(f_)
u_test = np.concatenate(u_test, 0)
f_test = np.concatenate(f_test, 0)

print("pca for label...")
u_train_pca = u_train.reshape(args.N_train * args.grid_t, -1)
u_test_pca = u_test.reshape(args.N_test * args.grid_t, -1)
start = time.time()
pca = PCA(n_components=args.n_components_decode).fit(u_train_pca)
end = time.time()
print("PCA Time: ", end - start)
print("# Components:", pca.n_components_)
const = np.sqrt(u_train_pca.shape[-1])
POD_Basis = pca.components_.T * const
POD_Mean = pca.mean_
print("POD shapes: ", POD_Basis.shape, POD_Mean.shape)
f = pca.fit_transform(u_test_pca)
u_test_pred = pca.inverse_transform(f)
print("PCA Error: ", np.linalg.norm(u_test_pca - u_test_pred) / np.linalg.norm(u_test), np.linalg.norm(u_test_pca))


print("pca for input...")
f_train_pca = f_train.reshape(args.N_train * args.grid_t, -1)
f_test_pca = f_test.reshape(args.N_test * args.grid_t, -1)
start = time.time()
pca = PCA(n_components=args.n_components_encode).fit(f_train_pca)
end = time.time()
print("PCA Time: ", end - start)
print("# Components:", pca.n_components_)
const = np.sqrt(f_train_pca.shape[-1])
In_Basis = pca.components_.T * const
In_Mean = pca.mean_
print("InPOD shapes: ", In_Basis.shape, In_Mean.shape)
f = pca.fit_transform(f_test_pca)
f_test_pred = pca.inverse_transform(f)
print("PCA Error: ", np.linalg.norm(f_test_pca - f_test_pred) / np.linalg.norm(f_test), np.linalg.norm(f_test_pca))


# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
print("Data Tensor Shapes: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

POD_Basis = torch.from_numpy(POD_Basis).float().to(device)
POD_Mean = torch.from_numpy(POD_Mean).float().to(device)
In_Basis = torch.from_numpy(In_Basis).float().to(device)
In_Mean = torch.from_numpy(In_Mean).float().to(device)

in_dim = X_train.shape[-1]

def get_model(model):
    if args.model == "GRU":
        model = PODDON_GRU(input_dim=args.n_components_encode, output_dim=args.n_components_decode, hidden_dim=256, num_layers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "LSTM":
        model = PODDON_LSTM(input_dim=args.n_components_encode, output_dim=args.n_components_decode, hidden_dim=256, num_layers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "OSS":
        model = PODDON_OSS_NO(
            input_dim=args.n_components_encode,
            output_dim=args.n_components_decode,
            hidden_dim=args.oss_hidden_dim,
            POD_Basis=POD_Basis,
            POD_Mean=POD_Mean,
            In_Basis=In_Basis,
            In_Mean=In_Mean,
            num_layers=args.oss_num_layers,
            discretization=args.oss_discretization,
            dt=args.oss_dt,
            dt_min=args.oss_dt_min,
            dt_max=args.oss_dt_max,
            use_layernorm=bool(args.oss_use_layernorm),
            residual_weight=args.oss_residual_weight,
            proj_dropout=args.oss_proj_dropout,
            robust_dt_init=bool(args.oss_robust_dt_init),
            use_input_dt=bool(args.oss_use_input_dt),
            use_d_skip=bool(args.oss_use_d_skip),
            d_skip_init=args.oss_d_skip_init,
            use_input_drive_damping=bool(args.oss_use_input_drive_damping),
            input_drive_scale=args.oss_input_drive_scale,
            input_damping_scale=args.oss_input_damping_scale,
            use_osc_gate=bool(args.oss_use_osc_gate),
            use_causal_prefilter=bool(args.oss_use_causal_prefilter),
            prefilter_kernel_size=args.oss_prefilter_kernel_size,
            use_expand_project=bool(args.oss_use_expand_project),
            expand_factor=args.oss_expand_factor,
            expand_init_scale=args.oss_expand_init_scale,
            use_coupled_oscillators=bool(args.oss_use_coupled_oscillators),
            coupling_rank=args.oss_coupling_rank,
            coupling_scale=args.oss_coupling_scale,
        ).to(device)
    elif args.model == "Mamba":
        config = MambaConfig(d_model=256,n_layer=1,vocab_size=0,ssm_cfg=dict(layer="Mamba1"),rms_norm=True,residual_in_fp32=True,fused_add_norm=True)
        model = PODDON_Mamba(256, 1, 0, args.n_components_encode, args.n_components_decode, POD_Basis, POD_Mean, In_Basis, In_Mean, config.ssm_cfg).to(device)
    elif args.model in ["MambaScratch", "mamba_scratch"]:
        model = PODDON_Mamba_Scratch(
            input_dim=args.n_components_encode,
            output_dim=args.n_components_decode,
            hidden_dim=256,
            POD_Basis=POD_Basis,
            POD_Mean=POD_Mean,
            In_Basis=In_Basis,
            In_Mean=In_Mean,
            n_layers=1,
            d_state=16,
        ).to(device)
    elif args.model == "ST":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GT":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "FT":
        model = GalerkinTransformer(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "T":
        model = Transformer(ninput=args.n_components_encode, noutput=args.n_components_decode, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GNOT":
        model = GNOT(dim_in=args.n_components_encode, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)

    return model

model = get_model(args.model)

# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

# ========================================================================
# Save Configuration File
# ========================================================================
model_details = get_model_details(model, args.model)
arch_config = get_model_architecture_details(args.model, args.grid_x, args.grid_t, 
                                             args.n_components_encode, args.n_components_decode)

config_data = {
    "timestamp": datetime.now().isoformat(),
    "run_number": run_number,
    "experiment_config": vars(args),
    "model_details": model_details,
    "model_architecture": arch_config,
    "training_config": {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "seed": args.SEED,
        "dataloader_batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
    },
    "data_config": {
        "grid_x": args.grid_x,
        "grid_t": args.grid_t,
        "T": args.T,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "n_components_encode": args.n_components_encode,
        "n_components_decode": args.n_components_decode,
    }
}

with open(config_file_path, "w") as f_config:
    json.dump(config_data, f_config, indent=2)
print(f"Configuration saved to: {config_file_path}")

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

lr_lambda = lambda epoch: 1-epoch/args.num_epochs
if args.model in ["ST", "GT", "FT", "T", "GNOT"]:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=args.lr,
        total_steps=args.num_epochs,
        div_factor=1e4,
        pct_start=0.2,
        final_div_factor=1e4,
    )
else:
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


# ========================================================================
# Setup Checkpoint Tracking and Best Model Management
# ========================================================================
best_test_error = float('inf')
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")

loss_traj = []

# Training loop
for epoch in tqdm(range(args.num_epochs), file=sys.stdout, desc="Epochs"):
    epoch_loss = 0.0
    num_batches = 0
    timing_enabled = bool(args.profile_timing)
    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0
    
    # Forward pass (no batch-level tqdm, epoch-level tracks progress)
    for batch_idx, (data, targets) in enumerate(train_loader):
        model.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients
        
        t0 = time.perf_counter()
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = criterion(outputs, targets)  # Target is the same as input
        
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        loss.backward()
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        optimizer.step()
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        
        epoch_loss += loss.item()
        num_batches += 1
        if timing_enabled:
            forward_time += (t1 - t0)
            backward_time += (t2 - t1)
            step_time += (t3 - t2)
    
    scheduler.step()
    
    train_loss_epoch = epoch_loss / max(num_batches, 1)
    loss_traj.append(train_loss_epoch)
    print(f"Epoch {epoch+1}, Train Loss (epoch avg): {train_loss_epoch:.3e}")
    if timing_enabled and num_batches > 0:
        print(
            f"Epoch {epoch+1} timing (s/batch): "
            f"forward={forward_time/num_batches:.3f}, "
            f"backward={backward_time/num_batches:.3f}, "
            f"step={step_time/num_batches:.3f}"
        )
    
    with open(loss_log_path, "a") as f_loss:
        f_loss.write(f"{epoch+1},{train_loss_epoch:.8e},nan,nan\n")

    # Print loss and save checkpoints every 100 epochs or at the end
    if (epoch+1)%int(100)==0 or epoch == args.num_epochs - 1: 
        output_train, label_train = [], []

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                output_train.append(outputs.detach().cpu())
                label_train.append(targets.detach().cpu())
        
        output_train = torch.cat(output_train, 0)
        label_train = torch.cat(label_train, 0)

        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm((label_train).reshape(-1))
        

        output_test, label_test = [], []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))

        with open(loss_log_path, "a") as f_loss:
            f_loss.write(
                f"{epoch+1},{train_loss_epoch:.8e},{error_train.item():.8e},{error_test.item():.8e}\n"
            )
        
        # ========================================================================
        # Save Best Model Checkpoint (based on test error)
        # ========================================================================
        test_error_val = error_test.item()
        if test_error_val < best_test_error:
            best_test_error = test_error_val
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_error': best_test_error,
                'train_loss': train_loss_epoch,
            }, best_model_path)
            print(f"  ✓ Best model saved (Test Error: {best_test_error:.3e})")
        
        # ========================================================================
        # Save Last Epoch Checkpoint
        # ========================================================================
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_error': test_error_val,
            'train_loss': train_loss_epoch,
        }, last_model_path)
        print(f"  ✓ Last checkpoint saved")


# ========================================================================
# Save Training Summary
# ========================================================================
training_summary = {
    "run_number": run_number,
    "model_name": args.model,
    "total_epochs": args.num_epochs,
    "best_test_error": best_test_error,
    "best_model_checkpoint": best_model_path,
    "last_checkpoint": last_model_path,
    "loss_log": loss_log_path,
    "config_file": config_file_path,
    "terminal_log": terminal_log_file,
}

summary_path = os.path.join(log_dir, "training_summary.json")
with open(summary_path, "w") as f_summary:
    json.dump(training_summary, f_summary, indent=2)
print(f"\nTraining Summary saved to: {summary_path}")
print(f"Best Test Error achieved: {best_test_error:.3e}")

# Restore streams and close the log file at the end of the script
sys.stdout.flush()
sys.stderr.flush()
sys.stdout = _orig_stdout
sys.stderr = _orig_stderr
_log_handle.close()

print(f"\n{'='*70}")
print(f"Experiment Run #{run_number} completed!")
print(f"Run directory: {log_dir}")
print(f"{'='*70}")

