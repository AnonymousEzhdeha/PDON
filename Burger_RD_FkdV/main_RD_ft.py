import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DON_2d import (
    MambaConfig,
    POD_GalerkinTransformer,
    POD_GNOT,
    POD_GRU,
    POD_LSTM,
    POD_Mamba,
    POD_Mamba_Scratch,
    POD_Transformer,
)
from FFNO2d_Flexible import FFNO
from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2
from LNO2d import LNO2d
from LNO2d_Jamba import LNO_Jamba_1, LNO_Jamba_2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def _flatten_config(config_data):
    flat = {}
    for key, value in config_data.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat[sub_key] = sub_val
        else:
            flat[key] = value
    return flat


def apply_config_defaults(args, parser):
    if not args.config:
        return args
    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}. Using CLI/default values.")
        return args

    with open(args.config, "r") as f_in:
        config_data = json.load(f_in)

    flat_config = _flatten_config(config_data)
    for key, value in flat_config.items():
        if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)

    return args


def get_parser():
    parser = argparse.ArgumentParser(description="DeepONet Training")

    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config", "config_rd_ft.json"))

    parser.add_argument("--SEED", type=int, default=0)

    parser.add_argument("--N_train", type=int, default=450)
    parser.add_argument("--N_test", type=int, default=50)

    parser.add_argument("--grid_x", type=int, default=20, help="x-axis grid size")
    parser.add_argument("--grid_t", type=int, default=200, help="t-axis grid size")
    parser.add_argument("--T", type=float, default=10, help="terminal time")

    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="minibatch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--eval_interval", type=int, default=10, help="epochs between validation/checkpoint logging")

    parser.add_argument("--model", type=str, default="FNO")

    parser.add_argument("--LSTM_hidden_dim", type=int, default=256)
    parser.add_argument("--LSTM_num_layers", type=int, default=1)

    parser.add_argument("--GRU_hidden_dim", type=int, default=256)
    parser.add_argument("--GRU_num_layers", type=int, default=1)

    parser.add_argument("--MambaLLM_d_model", type=int, default=256)
    parser.add_argument("--MambaLLM_n_layer", type=int, default=1)

    parser.add_argument("--save_loss", type=int, default=0)
    parser.add_argument("--save_model", type=int, default=0)

    parser.add_argument("--oss_num_layers", type=int, default=2)
    parser.add_argument("--discretization", type=str, default="IMEX", choices=["IM", "IMEX"])
    parser.add_argument("--oss_dt", type=float, default=1.0)
    parser.add_argument("--oss_dt_min", type=float, default=1e-3)
    parser.add_argument("--oss_dt_max", type=float, default=1.0)

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_config_defaults(args, parser)
    print(args)
    return args


args = get_args()

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

base_log_dir = os.path.join(SCRIPT_DIR, "run", "run_rd_ft")
run_number, log_dir, weights_dir = get_next_run_number(base_log_dir)
terminal_log_file = os.path.join(log_dir, "terminal_log.txt")
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
_log_handle = open(terminal_log_file, "w", buffering=1)
sys.stdout = TeeStream(_orig_stdout, _log_handle)
sys.stderr = TeeStream(_orig_stderr, _log_handle)

print(f"\n{'=' * 70}")
print(f"Experiment Run #{run_number}")
print(f"Log directory: {log_dir}")
print(f"Weights directory: {weights_dir}")
print(f"{'=' * 70}\n")
print("Logging to:", terminal_log_file)
print("Resolved arguments:", args)

loss_log_path = os.path.join(log_dir, "loss_values.csv")
with open(loss_log_path, "w") as f_loss:
    f_loss.write("epoch,train_loss_epoch_avg,train_rel_l2,test_rel_l2\n")
print("Loss metrics log:", loss_log_path)


def _sample_time_grf(num_samples, grid_t, T, length_scale=0.1):
    t = np.linspace(0.0, T, grid_t, dtype=np.float64)[:, None]
    sqdist = (t - t.T) ** 2
    K = np.exp(-sqdist / (2.0 * (length_scale ** 2)))
    L = np.linalg.cholesky(K + 1e-12 * np.eye(grid_t))
    g = (L @ np.random.randn(grid_t, num_samples)).T
    g = g / (np.std(g, axis=1, keepdims=True) + 1e-6)
    return 0.5 * g


def _solve_rd_1d(forcing, grid_x, grid_t, T, D=0.01, k=0.01):
    # PDE: D*u_xx + k*u^2 - pi^2*u_t = f(x,t), u(x,0)=u(0,t)=u(1,t)=0
    num_samples = forcing.shape[0]
    x = np.linspace(0.0, 1.0, grid_x, dtype=np.float64)
    dx = x[1] - x[0] if grid_x > 1 else 1.0
    dt = T / (grid_t - 1) if grid_t > 1 else T
    coef_t = dt / (np.pi ** 2)

    u = np.zeros((num_samples, grid_x, grid_t), dtype=np.float64)
    for n in range(grid_t - 1):
        u_curr = u[:, :, n]
        lap = np.zeros_like(u_curr)
        if grid_x > 2:
            lap[:, 1:-1] = (u_curr[:, :-2] - 2.0 * u_curr[:, 1:-1] + u_curr[:, 2:]) / (dx ** 2)

        rhs = D * lap + k * (u_curr ** 2) - forcing[:, :, n]
        u_next = u_curr + coef_t * rhs

        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u[:, :, n + 1] = u_next

    return u


def data_gen(num_samples, grid_x, grid_t, T, D=0.01, k=0.01):
    # RD-f(t) case: forcing is time-dependent and constant over x.
    g_t = _sample_time_grf(num_samples, grid_t, T, length_scale=0.1)
    forcing = np.repeat(g_t[:, None, :], grid_x, axis=1)
    u = _solve_rd_1d(forcing, grid_x, grid_t, T, D=D, k=k)

    f = forcing[..., None].astype(np.float32)
    y = u.reshape(num_samples, grid_x * grid_t, 1).astype(np.float32)
    return f, y


f_train, y_train = data_gen(args.N_train, args.grid_x, args.grid_t, args.T)
f_test, y_test = data_gen(args.N_test, args.grid_x, args.grid_t, args.T)


def aug_f(f, num_samples, grid_x, grid_t):
    x = np.linspace(0, 1, grid_x)
    t = np.linspace(0, 1, grid_t)
    x, t = np.meshgrid(x, t, indexing="ij")
    x = x.reshape(1, grid_x, grid_t, 1)
    t = t.reshape(1, grid_x, grid_t, 1)
    x = np.tile(x, (num_samples, 1, 1, 1))
    t = np.tile(t, (num_samples, 1, 1, 1))
    return np.concatenate([f, x, t], -1)


f_train = aug_f(f_train, f_train.shape[0], args.grid_x, args.grid_t)
f_test = aug_f(f_test, f_test.shape[0], args.grid_x, args.grid_t)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.from_numpy(f_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_test = torch.from_numpy(f_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

in_dim = 3

if args.model == "FNO":
    model = FNO2d(modes1=6, modes2=32, width=32, num_layers=2, in_dim=in_dim, out_dim=1).to(device)
elif args.model == "FFNO":
    model = FFNO(modes_x=6, modes_y=32, width=64, input_dim=in_dim, output_dim=1, n_layers=2).to(device)
elif args.model == "FNO_GRU_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="GRU").to(device)
elif args.model == "FNO_LSTM_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="LSTM").to(device)
elif args.model == "FNO_Mamba_1":
    model = FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="Mamba").to(device)
elif args.model == "FNO_GRU_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="GRU").to(device)
elif args.model == "FNO_LSTM_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="LSTM").to(device)
elif args.model == "FNO_Mamba_2":
    model = FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=6, width=128, num_layers=2, model_t_type="Mamba").to(device)
elif args.model == "FNO_OSS_1":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=6,
        width=128,
        num_layers=args.oss_num_layers,
        model_t_type="OSS",
        discretization=args.discretization,
        oss_dt=args.oss_dt,
        oss_dt_min=args.oss_dt_min,
        oss_dt_max=args.oss_dt_max,
    ).to(device)
elif args.model == "FNO_OSS_2":
    model = FNO_Jamba_2(
        input_dim=in_dim,
        output_dim=1,
        modes=6,
        width=128,
        num_layers=args.oss_num_layers,
        model_t_type="OSS",
        discretization=args.discretization,
        oss_dt=args.oss_dt,
        oss_dt_min=args.oss_dt_min,
        oss_dt_max=args.oss_dt_max,
    ).to(device)
elif args.model == "OSS":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=6,
        width=128,
        num_layers=args.oss_num_layers,
        model_t_type="OSS",
        discretization=args.discretization,
        oss_dt=args.oss_dt,
        oss_dt_min=args.oss_dt_min,
        oss_dt_max=args.oss_dt_max,
    ).to(device)
elif args.model == "GRU":
    model = POD_GRU(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=args.GRU_hidden_dim, num_layers=args.GRU_num_layers).to(device)
elif args.model == "LSTM":
    model = POD_LSTM(input_dim=in_dim * args.grid_x, output_dim=args.grid_x, hidden_dim=args.LSTM_hidden_dim, num_layers=args.LSTM_num_layers).to(device)
elif args.model == "Mamba":
    config = MambaConfig(
        d_model=args.MambaLLM_d_model,
        n_layer=args.MambaLLM_n_layer,
        vocab_size=0,
        ssm_cfg=dict(layer="Mamba1"),
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
    )
    model = POD_Mamba(args.MambaLLM_d_model, args.MambaLLM_n_layer, 0, in_dim * args.grid_x, args.grid_x, config.ssm_cfg).to(device)
elif args.model in ["MambaScratch", "mamba_scratch"]:
    model = POD_Mamba_Scratch(
        input_dim=in_dim * args.grid_x,
        output_dim=args.grid_x,
        hidden_dim=args.MambaLLM_d_model,
        num_layers=args.MambaLLM_n_layer,
        d_state=16,
    ).to(device)
elif args.model == "GT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="galerkin", mlp_dim=256).to(device)
elif args.model == "ST":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="standard", mlp_dim=256).to(device)
elif args.model == "FT":
    model = POD_GalerkinTransformer(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="fourier", mlp_dim=256).to(device)
elif args.model == "T":
    model = POD_Transformer(ninput=in_dim * args.grid_x, noutput=args.grid_x, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1).to(device)
elif args.model == "GNOT":
    model = POD_GNOT(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2).to(device)
elif args.model == "LNO":
    t_grid = torch.linspace(0, 1, args.grid_t)
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO2d(input_dim=in_dim, output_dim=1, width=64, modes1=4, modes2=4, T=t_grid, X=x_grid).to(device)
elif args.model == "LNO_LSTM_1":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="LSTM").to(device)
elif args.model == "LNO_LSTM_2":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="LSTM").to(device)
elif args.model == "LNO_GRU_1":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="GRU").to(device)
elif args.model == "LNO_GRU_2":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="GRU").to(device)
elif args.model == "LNO_Mamba_1":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="Mamba").to(device)
elif args.model == "LNO_Mamba_2":
    x_grid = torch.linspace(0, 1, args.grid_x)
    model = LNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=4, width=32, num_layers=4, grid=x_grid, time_model="Mamba").to(device)
else:
    raise ValueError(f"Unsupported model: {args.model}")


def count_parameters(model_obj):
    return sum(p.numel() for p in model_obj.parameters() if p.requires_grad)


print(f"Total number of parameters: {count_parameters(model)}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

lr_lambda = lambda epoch: 1 - epoch / args.num_epochs
if args.model in ["DeepONet_GT", "DeepONet_FT", "DeepONet_ST", "DeepONet_T", "DeepONet_GNOT", "GT", "FT", "ST", "T", "GNOT"]:
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

config_file_path = os.path.join(log_dir, "config.json")
config_data = {
    "timestamp": datetime.now().isoformat(),
    "run_number": run_number,
    "experiment_config": vars(args),
    "model_details": {
        "model_name": args.model,
        "model_class": model.__class__.__name__,
        "total_parameters": count_parameters(model),
        "total_parameters_all": sum(param.numel() for param in model.parameters()),
    },
    "training_config": {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "seed": args.SEED,
        "dataloader_batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "eval_interval": args.eval_interval,
    },
    "data_config": {
        "grid_x": args.grid_x,
        "grid_t": args.grid_t,
        "T": args.T,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "input_channels": in_dim,
    },
}
with open(config_file_path, "w") as f_config:
    json.dump(_jsonable(config_data), f_config, indent=2)
print("Configuration saved to:", config_file_path)

loss_traj = []
best_test_error = float("inf")
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")

for epoch in tqdm(range(args.num_epochs), file=sys.stdout, desc="Epochs"):
    epoch_loss = 0.0
    num_batches = 0

    for data, targets in train_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    train_loss_epoch = epoch_loss / max(num_batches, 1)
    loss_traj.append(train_loss_epoch)
    scheduler.step()

    with open(loss_log_path, "a") as f_loss:
        f_loss.write(f"{epoch + 1},{train_loss_epoch:.8e},nan,nan\\n")

    print(f"Epoch {epoch + 1}, Train Loss (epoch avg): {train_loss_epoch:.3e}")

    if (epoch + 1) % int(max(1, args.eval_interval)) == 0 or epoch == args.num_epochs - 1:
        model.eval()
        output_train, label_train = [], []
        with torch.no_grad():
            for data, targets in train_loader:
                outputs = model(data)
                output_train.append(outputs.detach().cpu())
                label_train.append(targets.detach().cpu())

        output_train = torch.cat(output_train, 0)
        label_train = torch.cat(label_train, 0)
        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm(label_train.reshape(-1))

        output_test, label_test = [], []
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = model(data)
                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)
        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm(label_test.reshape(-1))

        print(
            f"Epoch {epoch + 1}, Train Loss: {loss.item():.3e}, "
            f"Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}"
        )

        with open(loss_log_path, "a") as f_loss:
            f_loss.write(f"{epoch + 1},{train_loss_epoch:.8e},{error_train.item():.8e},{error_test.item():.8e}\\n")

        test_error_val = error_test.item()
        if test_error_val < best_test_error:
            best_test_error = test_error_val
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_error": best_test_error,
                    "train_loss": train_loss_epoch,
                },
                best_model_path,
            )
            print(f"  Best model saved (Test Error: {best_test_error:.3e})")

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_error": test_error_val,
                "train_loss": train_loss_epoch,
            },
            last_model_path,
        )
        print("  Last checkpoint saved")

if args.save_loss:
    loss_traj_arr = np.asarray(loss_traj)
    loss_txt_path = os.path.join(log_dir, f"RD_ft_Model={args.model}_Seed={args.SEED}.txt")
    np.savetxt(loss_txt_path, loss_traj_arr)
    print("Legacy loss trajectory saved:", loss_txt_path)

if args.save_model:
    save_model_path = os.path.join(weights_dir, f"{args.model}_Model.pth")
    torch.save(model.state_dict(), save_model_path)
    print("Legacy model state saved:", save_model_path)

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
    json.dump(_jsonable(training_summary), f_summary, indent=2)
print("Training Summary saved to:", summary_path)
print(f"Best Test Error achieved: {best_test_error:.3e}")

sys.stdout.flush()
sys.stderr.flush()
sys.stdout = _orig_stdout
sys.stderr = _orig_stderr
_log_handle.close()

print(f"\n{'=' * 70}")
print(f"Experiment Run #{run_number} completed!")
print(f"Run directory: {log_dir}")
print(f"{'=' * 70}")
