import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
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
    print("Debugger attached! ")
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
else:
    print("GPU name: CPU-only runtime")
#######

from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2
from FFNO2d import FFNO
# from DON_2d import POD_GRU, POD_LSTM, POD_Mamba
from DON_2d import POD_GRU, POD_LSTM, POD_Mamba, POD_Mamba_Scratch, MambaConfig
from DON_2d import POD_GalerkinTransformer, POD_Transformer, POD_GNOT
from burgers_data import AntideData, AntideAntideData


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




parser = argparse.ArgumentParser(description='DeepONet Training')
parser.add_argument('--SEED', type=int, default=0)

parser.add_argument('--T', type=int, default=1, help="")

parser.add_argument('--grid_x', type=int, default=100, help="x-axis grid size")
parser.add_argument('--grid_t', type=int, default=100, help="t-axis grid size")

parser.add_argument('--N_train', type=int, default=27000)
parser.add_argument('--N_test', type=int, default=3000)

parser.add_argument('--num_epochs', type=int, default=100, help="number of training epochs")
parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")

# parser.add_argument('--model', type=str, default="FNO")
parser.add_argument('--model', type=str, default="OSS")
parser.add_argument('--track_linoss_param', type=str, default="blocks.0.model_t.linoss_A_diag")
parser.add_argument('--track_linoss_index', type=int, default=0)

parser.add_argument('--save_loss', type=int, default=0)
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--amp', type=int, default=1, help="Use automatic mixed precision on CUDA (1/0)")
parser.add_argument('--profile_timing', type=int, default=1, help="Log avg forward/backward/step time per batch (1/0)")
parser.add_argument('--discretization', type=str, default="IMEX", choices=["IM", "IMEX"], help="Discretization scheme for temporal models (IM or IMEX)")

args = parser.parse_args()
print(args)

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

config_file_path = os.path.join(log_dir, "config.json")

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def generate_t(grid_t, T):
    # assert grid_t // 100 == T
    # data
    s0 = [0]
    sensor_in = grid_t
    sensor_out = grid_t
    length_scale = 0.2
    train_num = args.N_train // 4
    test_num = args.N_test // 4

    np.random.seed(args.SEED)
    data = AntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_3_train = data.X_train
    g_2_train = data.y_train
    g_3_test = data.X_test
    g_2_test = data.y_test

    np.random.seed(args.SEED)
    s0 = [0, 0]
    data = AntideAntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_1_train = data.y_train
    g_1_test = data.y_test

    return g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test

g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test = generate_t(args.grid_t, args.T)

def generate_data(grid_x, grid_t, g_1, g_2, g_3):
    g_1 = g_1.T # [1, grid_t]
    g_2 = g_2.T # [1, grid_t]
    g_3 = g_3.T # [1, grid_t]

    x = np.linspace(-5, 5, grid_x)
    t = np.linspace(0.5, 0.5 + args.T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]

    c1 = np.random.rand() * 3
    c2 = np.random.rand() * 6 - 3

    def u1_sol(x, t):
        return x / t - c1 / t - (g_1 - t * g_2) / t
    
    u0 = np.tile(g_2, (grid_x, 1))
    u0_init = u0[:, 0:1] # [grid_x, 1]
    u0_boundary_1 = u0[0:1, :] # [1, grid_t]
    u0_boundary_2 = u0[-1:, :] # [1, grid_t]

    u0_init = np.tile(u0_init, (1, grid_t)) # [grid_x, grid_t]
    u0_boundary_1 = np.tile(u0_boundary_1, (grid_x, 1))
    u0_boundary_2 = np.tile(u0_boundary_2, (grid_x, 1))

    u1 = u1_sol(x, t)
    u1_init = u1[:, 0:1] # [grid_x, 1]
    u1_boundary_1 = u1[0:1, :] # [1, grid_t]
    u1_boundary_2 = u1[-1:, :] # [1, grid_t]

    u1_init = np.tile(u1_init, (1, grid_t)) # [grid_x, grid_t]
    u1_boundary_1 = np.tile(u1_boundary_1, (grid_x, 1))
    u1_boundary_2 = np.tile(u1_boundary_2, (grid_x, 1))
    
    const =  np.sqrt(2 * c1)
    def u2_sol(x, t):
        return - const * np.tanh(0.5 * const * (g_1 - x + c2)) + g_2
    u2 = u2_sol(x, t)
    u2_init = u2[:, 0:1] # [grid_x, 1]
    u2_boundary_1 = u2[0:1, :] # [1, grid_t]
    u2_boundary_2 = u2[-1:, :] # [1, grid_t]

    u2_init = np.tile(u2_init, (1, grid_t)) # [grid_x, grid_t]
    u2_boundary_1 = np.tile(u2_boundary_1, (grid_x, 1))
    u2_boundary_2 = np.tile(u2_boundary_2, (grid_x, 1))

    const =  np.sqrt(2 * c1)
    def u3_sol(x, t):
        return const / t * np.tanh(0.5 * const * ((x - g_1) / t + c2)) + (x - g_1) / t + g_2
    u3 = u3_sol(x, t)
    u3_init = u3[:, 0:1] # [grid_x, 1]
    u3_boundary_1 = u3[0:1, :] # [1, grid_t]
    u3_boundary_2 = u3[-1:, :] # [1, grid_t]

    u3_init = np.tile(u3_init, (1, grid_t)) # [grid_x, grid_t]
    u3_boundary_1 = np.tile(u3_boundary_1, (grid_x, 1))
    u3_boundary_2 = np.tile(u3_boundary_2, (grid_x, 1))

    f = np.tile(g_3, (grid_x, 1))

    f0 = np.stack([f, u0_init, u0_boundary_1, u0_boundary_2], -1) # N_xN_t, 4
    f0 = f0.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f1 = np.stack([f, u1_init, u1_boundary_1, u1_boundary_2], -1) # N_xN_t, 4
    f1 = f1.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f2 = np.stack([f, u2_init, u2_boundary_1, u2_boundary_2], -1) # N_xN_t, 4
    f2 = f2.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f3 = np.stack([f, u3_init, u3_boundary_1, u3_boundary_2], -1) # N_xN_t, 4
    f3 = f3.reshape(grid_x, grid_t, 4) # N_x, N_t, 4

    f = np.stack([f0, f1, f2, f3], 0)
    u = np.stack([u0, u1, u2, u3], 0)
    u = u.reshape(4, -1, 1)

    return f, u

X_train, y_train = [], []
for i in tqdm(range(args.N_train // 4), file=sys.stdout):
    x, y = generate_data(args.grid_x, args.grid_t, g_1_train[i], g_2_train[i], g_3_train[i])
    X_train.append(x)
    y_train.append(y)
X_train = np.concatenate(X_train, 0)
y_train = np.concatenate(y_train, 0)
print(X_train.shape, y_train.shape)

X_test, y_test = [], []
for i in tqdm(range(args.N_test // 4), file=sys.stdout):
    x, y = generate_data(args.grid_x, args.grid_t, g_1_test[i], g_2_test[i], g_3_test[i])
    X_test.append(x)
    y_test.append(y)
X_test = np.concatenate(X_test, 0)
y_test = np.concatenate(y_test, 0)
print(X_test.shape, y_test.shape)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.as_tensor(np.asarray(X_train), dtype=torch.float32, device=device)
y_train = torch.as_tensor(np.asarray(y_train), dtype=torch.float32, device=device)
X_test = torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
y_test = torch.as_tensor(np.asarray(y_test), dtype=torch.float32, device=device)

in_dim = 4

if args.model == "FNO":
    model = FNO2d(modes1=32, modes2=32, width=24, num_layers=2, in_dim=in_dim, out_dim=1).to(device)

elif args.model == "FFNO":
    model = FFNO(modes=32, width=96, input_dim=in_dim, output_dim=1, n_layers=2).to(device)

elif args.model == "FNO_GRU_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU").to(device)
elif args.model == "FNO_GRU_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU").to(device)

elif args.model == "FNO_LSTM_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM").to(device)
elif args.model == "FNO_LSTM_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM").to(device)

elif args.model == "FNO_Mamba_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba").to(device)
elif args.model == "FNO_Mamba_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba").to(device)
elif args.model == "FNO_OSS_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=args.discretization).to(device)
elif args.model == "FNO_OSS_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=args.discretization).to(device)
elif args.model == "OSS":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=args.discretization).to(device)
elif args.model == "FNO_OSS_source_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=args.discretization).to(device)
elif args.model == "FNO_OSS_source_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=args.discretization).to(device)
elif args.model == "OSS_source":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=args.discretization).to(device)

elif args.model == "FNO_linoss_pytorch_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=args.discretization).to(device)
elif args.model == "FNO_linoss_pytorch_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=args.discretization).to(device)
elif args.model == "linoss_pytorch":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=args.discretization).to(device)

elif args.model == "GRU":
    model = POD_GRU(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=256, num_layers=1).to(device)
elif args.model == "LSTM":
    model = POD_LSTM(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=256, num_layers=1).to(device)
elif args.model == "Mamba":
    config = MambaConfig(
            d_model=256,
            n_layer=1,
            vocab_size=0,
            ssm_cfg=dict(layer="Mamba1"),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True
        )
    model = POD_Mamba(256, 1, 0, \
                     in_dim * args.grid_x, args.grid_x, config.ssm_cfg).to(device)
elif args.model in ["MambaScratch", "mamba_scratch"]:
    model = POD_Mamba_Scratch(
        input_dim=in_dim * args.grid_x,
        output_dim=args.grid_x,
        hidden_dim=256,
        num_layers=1,
        d_state=16,
    ).to(device)
elif args.model == "GT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256).to(device)
elif args.model == "ST":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256).to(device)
elif args.model == "FT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256).to(device)
elif args.model == "T":
    model = POD_Transformer(ninput=in_dim * args.grid_x, noutput=args.grid_x, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1).to(device)
elif args.model == "GNOT":
    model = POD_GNOT(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2).to(device)


# ========================================================================
# Conditional JAX Loading (only for OSS_source models)
# ========================================================================
# JAX is only needed for OSS_source temporal models
# Skip for linoss_pytorch and other models
uses_oss_source = args.model in [
    "FNO_OSS_source_1", "FNO_OSS_source_2", "OSS_source"
]

if uses_oss_source:
    try:
        import jax
        print("JAX version:", jax.__version__)
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", jax.devices())
    except Exception as _jax_exc:
        print("JAX backend probe failed:", _jax_exc)
else:
    print(f"Model '{args.model}' does not require JAX (JAX import skipped)")


# ========================================================================
# Conditional Parameter Tracking (only for OSS-based models)
# ========================================================================
# Parameter tracking is useful for OSS_source and linoss_pytorch models
# For other models, we skip tracking setup
tracking_enabled = args.model in [
    "FNO_OSS_source_1", "FNO_OSS_source_2", "OSS_source"
]


# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

# ========================================================================
# Extract Model-Specific Architecture Details
# ========================================================================
def get_model_architecture_details(model_name, in_dim, grid_x, grid_y=None):
    """
    Extract model-specific architecture parameters based on model type.
    These are saved in the config for reproducibility and reference.
    """
    arch_config = {}
    
    if model_name == "FNO":
        arch_config.update({
            "modes1": 32, "modes2": 32, "width": 24, "num_layers": 2,
            "input_dim": in_dim, "output_dim": 1
        })
    elif model_name == "FFNO":
        arch_config.update({
            "modes": 32, "width": 96, "n_layers": 2,
            "input_dim": in_dim, "output_dim": 1
        })
    elif "FNO_GRU" in model_name:
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "GRU", "base_model": model_name
        })
    elif "FNO_LSTM" in model_name:
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "LSTM", "base_model": model_name
        })
    elif "FNO_Mamba" in model_name:
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "Mamba", "base_model": model_name
        })
    elif "FNO_OSS_source" in model_name or "OSS_source" in model_name:
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "OSS_source", "discretization": args.discretization,
            "base_model": model_name
        })
    elif "FNO_OSS" in model_name or model_name == "OSS":
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "OSS", "discretization": args.discretization, "base_model": model_name
        })
    elif "FNO_linoss_pytorch" in model_name or model_name == "linoss_pytorch":
        arch_config.update({
            "input_dim": in_dim, "output_dim": 1, "modes": 32, "width": 128,
            "num_layers": 1, "model_t_type": "linoss_pytorch", "discretization": args.discretization,
            "d_model": 128, "dropout": 0.05, "base_model": model_name
        })
    elif model_name == "GRU":
        arch_config.update({
            "input_dim": in_dim * grid_x, "output_dim": grid_x, "hidden_dim": 256, "num_layers": 1
        })
    elif model_name == "LSTM":
        arch_config.update({
            "input_dim": in_dim * grid_x, "output_dim": grid_x, "hidden_dim": 256, "num_layers": 1
        })
    elif model_name == "Mamba":
        arch_config.update({
            "d_model": 256, "n_layer": 1, "input_dim": in_dim * grid_x, "output_dim": grid_x,
            "ssm_cfg": {"layer": "Mamba1"}, "rms_norm": True, "residual_in_fp32": True, "fused_add_norm": True
        })
    elif model_name in ["MambaScratch", "mamba_scratch"]:
        arch_config.update({
            "input_dim": in_dim * grid_x,
            "output_dim": grid_x,
            "hidden_dim": 256,
            "num_layers": 1,
            "d_state": 16,
        })
    elif model_name == "GT":
        arch_config.update({
            "dim_in": in_dim * grid_x, "dim_out": grid_x, "dim_hid": 256, "depth": 1,
            "heads": 4, "dim_head": 256, "attn_type": "galerkin", "mlp_dim": 256
        })
    elif model_name == "ST":
        arch_config.update({
            "dim_in": in_dim * grid_x, "dim_out": grid_x, "dim_hid": 256, "depth": 1,
            "heads": 4, "dim_head": 256, "attn_type": "standard", "mlp_dim": 256
        })
    elif model_name == "FT":
        arch_config.update({
            "dim_in": in_dim * grid_x, "dim_out": grid_x, "dim_hid": 256, "depth": 1,
            "heads": 4, "dim_head": 256, "attn_type": "fourier", "mlp_dim": 256
        })
    elif model_name == "T":
        arch_config.update({
            "ninput": in_dim * grid_x, "noutput": grid_x, "nhidden": 256,
            "dim_feedforward": 256, "nhead": 4, "nlayers": 1
        })
    elif model_name == "GNOT":
        arch_config.update({
            "dim_in": in_dim * grid_x, "dim_out": grid_x, "dim_hid": 256, "depth": 1,
            "heads": 4, "dim_head": 256, "n_experts": 2
        })
    
    return arch_config

# ========================================================================
# Save Configuration File
# ========================================================================
model_details = get_model_details(model, args.model)
arch_config = get_model_architecture_details(args.model, in_dim, args.grid_x)

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
        "input_channels": in_dim,
    }
}

with open(config_file_path, "w") as f_config:
    json.dump(config_data, f_config, indent=2)
print(f"Configuration saved to: {config_file_path}")

# Loss function
criterion = nn.MSELoss()


def model_has_complex_tensors(module):
    """Return True if the model exposes any complex-valued parameters or buffers."""
    tensors = list(module.parameters()) + list(module.buffers())
    return any(tensor.is_complex() for tensor in tensors)


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# Disable AMP for models that use complex-valued tensors or the LinOSS/JAX bridge.
has_complex_tensors = model_has_complex_tensors(model)
is_linoss_model = "linoss" in args.model.lower() or "oss_source" in args.model.lower()
use_amp = bool(args.amp) and torch.cuda.is_available() and (not has_complex_tensors) and (not is_linoss_model)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
if bool(args.amp) and has_complex_tensors:
    complex_dtypes = sorted({str(t.dtype) for t in list(model.parameters()) + list(model.buffers()) if t.is_complex()})
    print(
        "AMP disabled because the model contains complex-valued tensors "
        f"({', '.join(complex_dtypes)}), which GradScaler does not support"
    )
elif bool(args.amp) and is_linoss_model:
    print("AMP disabled for linoss_pytorch/OSS_source models (use complex dtypes incompatible with GradScaler)")
else:
    print(f"AMP enabled: {use_amp}")

lr_lambda = lambda epoch: 1-epoch/args.num_epochs
if args.model in ['DeepONet_GT', 'DeepONet_FT', 'DeepONet_ST', 'DeepONet_T', 'DeepONet_GNOT', \
                  "GT", "FT", "ST", "T", "GNOT"]:
    print("Using Transformer scheduler")
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

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


def _get_named_param(model, param_name):
    for name, param in model.named_parameters():
        if name == param_name:
            return param
    return None


# ========================================================================
# Setup Parameter Tracking (conditional on model type)
# ========================================================================
tracked_param_at_start = None
linoss_param_logged = False

if tracking_enabled:
    tracked_param_at_start = _get_named_param(model, args.track_linoss_param)
    if tracked_param_at_start is None:
        print("Warning: tracked parameter not found:", args.track_linoss_param)
        # List available parameters that contain "linoss" or temporal model info
        related_param_names = [
            name for name, _ in model.named_parameters() 
            if "linoss" in name.lower() or "model_t" in name.lower()
        ]
        if related_param_names:
            print("Available model_t/LinOSS parameter names:")
            for name in related_param_names:
                print("  -", name)
    else:
        print("Tracking parameter:", args.track_linoss_param)
else:
    print(f"Parameter tracking disabled (not applicable for model '{args.model}')")


loss_traj = []
linoss_param_logged = False

# ========================================================================
# Setup Checkpoint Tracking and Best Model Management
# ========================================================================
best_test_error = float('inf')
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")

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
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(data)
            loss = criterion(outputs, targets)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Conditional parameter tracking (only for OSS-based models)
        tracked_param = None
        before_value = None
        grad_value = None
        
        if tracking_enabled:
            tracked_param = _get_named_param(model, args.track_linoss_param)
            if tracked_param is not None and tracked_param.numel() > args.track_linoss_index:
                before_value = tracked_param.view(-1)[args.track_linoss_index].item()

        scaler.scale(loss).backward()
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        if tracking_enabled and tracked_param is not None:
            if tracked_param.grad is not None and tracked_param.grad.numel() > args.track_linoss_index:
                grad_value = tracked_param.grad.view(-1)[args.track_linoss_index].item()

        scaler.step(optimizer)
        scaler.update()
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        # Log parameter update only once and only if tracking is enabled
        if tracking_enabled and (
            (not linoss_param_logged)
            and (before_value is not None)
            and (grad_value is not None)
        ):
            after_value = tracked_param.view(-1)[args.track_linoss_index].item()
            print(
                f"Parameter update tracked: {args.track_linoss_param}[{args.track_linoss_index}] "
                f"before={before_value:.6e}, grad={grad_value:.6e}, after={after_value:.6e}"
            )
            linoss_param_logged = True

        epoch_loss += loss.item()
        num_batches += 1
        if timing_enabled:
            forward_time += (t1 - t0)
            backward_time += (t2 - t1)
            step_time += (t3 - t2)
        # train_pbar.set_postfix({'loss': loss.item():.4e})

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

    scheduler.step()

    # Print loss every epoch
    if (epoch+1)%int(5)==0 or epoch == args.num_epochs - 1: 
        output_train, label_train = [], []

        model.eval()
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(train_loader):
                outputs = model(data)
                output_train.append(outputs.detach().cpu())
                label_train.append(targets.detach().cpu())
        
        output_train = torch.cat(output_train, 0)
        label_train = torch.cat(label_train, 0)

        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm((label_train).reshape(-1))
        

        output_test, label_test = [], []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                outputs = model(data)
                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)

        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))
        
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.3e}, Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}")
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
        # print(f"Epoch {epoch+1}, Train Loss: {loss.item()}, Train Rel L2: {error_train.item()}, Test Rel L2: {error_test.item()}")

if args.save_loss:
    loss_traj = np.asarray(loss_traj)
    filename = "Data/Burgers_" + \
        "Model=" + str(args.model) + "_" + \
        "Seed=" + str(args.SEED) + \
        ".txt"
    np.savetxt(filename, loss_traj)

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