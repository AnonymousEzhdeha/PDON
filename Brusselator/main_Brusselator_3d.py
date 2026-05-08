import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
import os
import sys

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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_next_run_number(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_nums = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.isdigit():
            existing_nums.append(int(item))
    next_num = max(existing_nums) + 1 if existing_nums else 1
    run_dir = os.path.join(base_dir, str(next_num))
    weights_dir = os.path.join(run_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    return next_num, run_dir, weights_dir


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


base_log_dir = os.path.join(SCRIPT_DIR, "run")
run_number, log_dir, weights_dir = get_next_run_number(base_log_dir)
terminal_log_path = os.path.join(log_dir, "terminal_log.txt")
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_log_handle = open(terminal_log_path, "w", buffering=1)
sys.stdout = TeeStream(_orig_stdout, _log_handle)
sys.stderr = TeeStream(_orig_stderr, _log_handle)
print(f"Experiment Run #{run_number}")
print(f"Log directory: {log_dir}")
print(f"Weights directory: {weights_dir}")
print("Logging to:", terminal_log_path)

def get_args():
    parser = argparse.ArgumentParser(description='DeepONet Training')
    parser.add_argument('--SEED', type=int, default=0)

    parser.add_argument('--grid_x', type=int, default=28, help="x-axis grid size")
    parser.add_argument('--grid_t', type=int, default=201, help="t-axis grid size")

    parser.add_argument('--T', type=int, default=1, help="terminal time")

    parser.add_argument('--num_epochs', type=int, default=1000, help="number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="minibatch size for SGD")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--eval_interval', type=int, default=5, help="epochs between validation/checkpoint logging")

    parser.add_argument('--model', type=str, default="Mamba", choices=["GRU", "LSTM", "Mamba", "MambaScratch", "mamba_scratch", "OSS", "ST", "GT", "FT", "T", "GNOT"])

    parser.add_argument('--n_components_decode', type=int, default=256)
    parser.add_argument('--oss_hidden_dim', type=int, default=256)
    parser.add_argument('--oss_num_layers', type=int, default=1)
    parser.add_argument('--oss_discretization', type=str, default="IMEX", choices=["IM", "IMEX"])
    parser.add_argument('--oss_dt', type=float, default=1.0)
    parser.add_argument('--oss_dt_min', type=float, default=1e-3)
    parser.add_argument('--oss_dt_max', type=float, default=1.0)
    parser.add_argument('--oss_use_layernorm', type=int, default=0)
    parser.add_argument('--oss_residual_weight', type=float, default=0.0)
    parser.add_argument('--oss_proj_dropout', type=float, default=0.0)
    parser.add_argument('--oss_robust_dt_init', type=int, default=1)
    parser.add_argument('--oss_use_input_dt', type=int, default=0)
    parser.add_argument('--oss_use_d_skip', type=int, default=0)
    parser.add_argument('--oss_d_skip_init', type=float, default=1.0)
    parser.add_argument('--oss_use_input_drive_damping', type=int, default=0)
    parser.add_argument('--oss_input_drive_scale', type=float, default=1.0)
    parser.add_argument('--oss_input_damping_scale', type=float, default=1.0)
    parser.add_argument('--oss_use_osc_gate', type=int, default=0)
    parser.add_argument('--oss_use_causal_prefilter', type=int, default=0)
    parser.add_argument('--oss_prefilter_kernel_size', type=int, default=3)

    args = parser.parse_args()
    print(args)

    return args

args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

loss_log_path = os.path.join(log_dir, "loss_values.csv")
with open(loss_log_path, "w") as f_loss:
    f_loss.write("epoch,train_loss_epoch_avg,train_rel_l2,test_rel_l2\n")
print("Loss metrics log:", loss_log_path)


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def get_model_details(model, model_name):
    details = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "total_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
        "total_parameters_all": sum(param.numel() for param in model.parameters()),
    }
    if hasattr(model, "config"):
        details["model_config"] = str(model.config)
    return details


def get_model_architecture_details(model_name):
    if model_name == "GRU":
        return {"input_dim": 1, "output_dim": args.n_components_decode, "hidden_dim": 256, "num_layers": 2}
    if model_name == "LSTM":
        return {"input_dim": 1, "output_dim": args.n_components_decode, "hidden_dim": 256, "num_layers": 2}
    if model_name == "Mamba":
        return {"input_dim": 1, "output_dim": args.n_components_decode, "d_model": 256, "n_layer": 2, "ssm_cfg": {"layer": "Mamba1"}}
    if model_name in ["MambaScratch", "mamba_scratch"]:
        return {"input_dim": 1, "output_dim": args.n_components_decode, "hidden_dim": 256, "n_layers": 1, "d_state": 16}
    if model_name == "OSS":
        return {
            "input_dim": 1,
            "output_dim": args.n_components_decode,
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
        }
    if model_name in ["ST", "GT", "FT"]:
        attn_type = "standard" if model_name == "ST" else ("galerkin" if model_name == "GT" else "fourier")
        return {"input_dim": 1, "output_dim": args.n_components_decode, "dim_hid": 256, "depth": 1, "heads": 4, "dim_head": 256, "attn_type": attn_type, "mlp_dim": 256}
    if model_name == "T":
        return {"input_dim": 1, "output_dim": args.n_components_decode, "nhidden": 256, "dim_feedforward": 256, "nhead": 4, "nlayers": 1}
    if model_name == "GNOT":
        return {"input_dim": 1, "output_dim": args.n_components_decode, "dim_hid": 256, "depth": 1, "heads": 4, "dim_head": 256, "n_experts": 2}
    return {}

f, u = [], []
for path in os.listdir("Data"):
    data = np.load("Data/" + path)
    f_ = data['F'][:, :args.grid_t]
    u_ = data['U'][:, :args.grid_t]
    f.append(f_)
    u.append(u_)
f = np.concatenate(f, 0)
u = np.concatenate(u, 0)


f_train, f_test, u_train, u_test = train_test_split(
    f, u, test_size=0.1, random_state=42
)

args.N_train = f_train.shape[0]
args.N_test = f_test.shape[0]

def process_u(u):
    N, grid_t = u.shape[0], u.shape[1]
    u = u.reshape(N, grid_t, -1)
    return u

u_train = process_u(u_train)
u_test = process_u(u_test)
f_train = process_u(f_train)
f_test = process_u(f_test)


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

In_Basis, In_Mean = np.ones((1, 1)), np.zeros((1,))

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float().to(device), torch.from_numpy(u_train).float().to(device), torch.from_numpy(f_test).float().to(device), torch.from_numpy(u_test).float().to(device)
print("Data Tensor Shapes: ", X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

POD_Basis = torch.from_numpy(POD_Basis).float().to(device)
POD_Mean = torch.from_numpy(POD_Mean).float().to(device)
In_Basis = torch.from_numpy(In_Basis).float().to(device)
In_Mean = torch.from_numpy(In_Mean).float().to(device)

def get_model(model):
    if args.model == "GRU":
        model = PODDON_GRU(input_dim=1, output_dim=args.n_components_decode, hidden_dim=256, num_layers=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "LSTM":
        model = PODDON_LSTM(input_dim=1, output_dim=args.n_components_decode, hidden_dim=256, num_layers=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "Mamba":
        config = MambaConfig(d_model=256,n_layer=2,vocab_size=0,ssm_cfg=dict(layer="Mamba1"),rms_norm=True,residual_in_fp32=True,fused_add_norm=True)
        model = PODDON_Mamba(256, 2, 0, 1, args.n_components_decode, POD_Basis, POD_Mean, In_Basis, In_Mean, config.ssm_cfg).to(device)
    elif args.model in ["MambaScratch", "mamba_scratch"]:
        model = PODDON_Mamba_Scratch(input_dim=1, output_dim=args.n_components_decode, hidden_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean, n_layers=1, d_state=16).to(device)
    elif args.model == "OSS":
        model = PODDON_OSS_NO(
            input_dim=1,
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
        ).to(device)
    elif args.model == "ST":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='standard', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GT":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='galerkin', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "FT":
        model = GalerkinTransformer(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type='fourier', mlp_dim=256, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "T":
        model = Transformer(ninput=1, noutput=args.n_components_decode, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)
    elif args.model == "GNOT":
        model = GNOT(dim_in=1, dim_out=args.n_components_decode, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2, POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean).to(device)

    return model

model = get_model(args.model)

config_file_path = os.path.join(log_dir, "config.json")
config_data = {
    "timestamp": datetime.now().isoformat(),
    "run_number": run_number,
    "script_dir": SCRIPT_DIR,
    "log_dir": log_dir,
    "weights_dir": weights_dir,
    "terminal_log_path": terminal_log_path,
    "loss_log_path": loss_log_path,
    "args": {key: _jsonable(value) for key, value in vars(args).items()},
    "data_shapes": {
        "X_train": list(X_train.shape),
        "Y_train": list(Y_train.shape),
        "X_test": list(X_test.shape),
        "Y_test": list(Y_test.shape),
    },
    "model_details": get_model_details(model, args.model),
    "model_architecture": get_model_architecture_details(args.model),
}
with open(config_file_path, "w") as f_config:
    json.dump(config_data, f_config, indent=2)
print("Config file:", config_file_path)

# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)

lr_lambda = lambda epoch: 1-epoch/args.num_epochs
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

best_test_error = float("inf")
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")
eval_interval = max(1, int(args.eval_interval))

# Training loop
for epoch in tqdm(range(args.num_epochs)):
    model.train()
    epoch_losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    scheduler.step()

    train_loss_epoch_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    if ((epoch + 1) % eval_interval == 0) or (epoch + 1 == args.num_epochs):
        model.eval()
        with torch.no_grad():
            output_train, label_train = [], []
            for batch_idx, (data, targets) in enumerate(train_loader):
                outputs = model(data)
                output_train.append(outputs.detach().cpu())
                label_train.append(targets.detach().cpu())

            output_train = torch.cat(output_train, 0)
            label_train = torch.cat(label_train, 0)
            error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm(label_train.reshape(-1))

            output_test, label_test = [], []
            for batch_idx, (data, targets) in enumerate(test_loader):
                outputs = model(data)
                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

            output_test = torch.cat(output_test, 0)
            label_test = torch.cat(label_test, 0)
            error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm(label_test.reshape(-1))

        with open(loss_log_path, "a") as f_loss:
            f_loss.write(f"{epoch + 1},{train_loss_epoch_avg:.10e},{error_train.item():.10e},{error_test.item():.10e}\n")

        if error_test.item() < best_test_error:
            best_test_error = error_test.item()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss_epoch_avg": train_loss_epoch_avg,
                "train_rel_l2": error_train.item(),
                "test_rel_l2": error_test.item(),
                "args": {key: _jsonable(value) for key, value in vars(args).items()},
            }, best_model_path)

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss_epoch_avg:.3e}, Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}")

torch.save({
    "epoch": args.num_epochs,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_test_error": best_test_error,
    "args": {key: _jsonable(value) for key, value in vars(args).items()},
}, last_model_path)

summary_path = os.path.join(log_dir, "training_summary.json")
summary_data = {
    "timestamp": datetime.now().isoformat(),
    "run_number": run_number,
    "best_test_error": best_test_error,
    "best_model_path": best_model_path,
    "last_model_path": last_model_path,
    "loss_log_path": loss_log_path,
    "config_file_path": config_file_path,
    "terminal_log_path": terminal_log_path,
    "total_parameters": count_parameters(model),
    "model_name": args.model,
}
with open(summary_path, "w") as f_summary:
    json.dump(summary_data, f_summary, indent=2)
print("Training summary:", summary_path)

sys.stdout = _orig_stdout
sys.stderr = _orig_stderr
_log_handle.close()

