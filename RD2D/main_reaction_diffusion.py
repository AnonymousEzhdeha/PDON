import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import TensorDataset, DataLoader
import os
import sys
import json
from datetime import datetime

from time_model import GRU, LSTM, Mamba_NO, OSS_NO, MambaScratch_NO

from time_model import GalerkinTransformer, Transformer, GNOT


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
    if "args" in config_data and isinstance(config_data["args"], dict):
        return config_data["args"]
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
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg_flat = _flatten_config(cfg)
    for key, value in cfg_flat.items():
        if hasattr(args, key) and getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
    return args


def get_parser():
    parser = argparse.ArgumentParser(description="DeepONet Training")
    parser.add_argument("--config", type=str, default="", help="optional config json path")
    parser.add_argument("--SEED", type=int, default=0)
    parser.add_argument("--grid_t", type=int, default=100, help="t-axis grid size")
    parser.add_argument("--K", type=int, default=4, help="Fourier modes for synthetic data")
    parser.add_argument("--N_train", type=int, default=10000)
    parser.add_argument("--N_test", type=int, default=1000)

    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="minibatch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--eval_interval", type=int, default=100, help="epochs between validation/checkpoint logging")

    parser.add_argument(
        "--model",
        type=str,
        default="OSS",
        choices=["LSTM", "GRU", "Mamba", "MambaScratch", "mamba_scratch", "OSS", "ST", "GT", "FT", "T", "GNOT"],
    )

    parser.add_argument("--data2cuda", type=int, default=1)

    parser.add_argument("--oss_hidden_dim", type=int, default=256)
    parser.add_argument("--oss_num_layers", type=int, default=1)
    parser.add_argument("--oss_discretization", type=str, default="IMEX", choices=["IM", "IMEX"])
    parser.add_argument("--oss_dt", type=float, default=1.0)
    parser.add_argument("--oss_dt_min", type=float, default=1e-3)
    parser.add_argument("--oss_dt_max", type=float, default=1.0)
    parser.add_argument("--oss_use_layernorm", type=int, default=0)
    parser.add_argument("--oss_residual_weight", type=float, default=0.0)
    parser.add_argument("--oss_proj_dropout", type=float, default=0.0)
    parser.add_argument("--oss_robust_dt_init", type=int, default=0)

    parser.add_argument("--oss_use_input_dt", type=int, default=0)

    parser.add_argument("--oss_use_input_drive_damping", type=int, default=0)
    parser.add_argument("--oss_input_drive_scale", type=float, default=1.0)
    parser.add_argument("--oss_input_damping_scale", type=float, default=1.0)

    parser.add_argument("--oss_use_d_skip", type=int, default=0)
    parser.add_argument("--oss_d_skip_init", type=float, default=1.0)

    parser.add_argument("--oss_use_osc_gate", type=int, default=0)

    parser.add_argument("--oss_use_causal_prefilter", type=int, default=0)
    parser.add_argument("--oss_prefilter_kernel_size", type=int, default=3)

    parser.add_argument("--oss_use_expand_project", type=int, default=0)
    parser.add_argument("--oss_expand_factor", type=int, default=2)
    parser.add_argument("--oss_expand_init_scale", type=float, default=0.02)

    parser.add_argument("--oss_use_coupled_oscillators", type=int, default=0)
    parser.add_argument("--oss_coupling_rank", type=int, default=4)
    parser.add_argument("--oss_coupling_scale", type=float, default=0.05)

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    args = apply_config_defaults(args, parser)
    print(args)
    return args

args = get_args()

args.T = int(args.grid_t / 100)
args.grid_x = args.K * 4 + 1

SEED = args.SEED
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

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

loss_log_path = os.path.join(log_dir, "loss_values.csv")
with open(loss_log_path, "w") as f_loss:
    f_loss.write("epoch,train_loss_epoch_avg,train_rel_l2_fourier,train_rel_l2_grid,test_rel_l2_fourier,test_rel_l2_grid\n")
print("Loss metrics log:", loss_log_path)

def generate_g_x(N, grid_x, K):
    k1 = (np.arange(1, K + 1) + 0.0).reshape(1, K, 1, 1, 1, 1)
    k2 = (np.arange(1, K + 1) + 0.0).reshape(1, 1, K, 1, 1, 1)

    A = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, 1, 1, 1))
    B = np.random.uniform(low = 0.0, high = 1.0, size = (N, K, K, 1, 1, 1))
    C = np.random.rand()

    x = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]
    y = np.linspace(0, 2 * np.pi, grid_x + 1)[:-1]

    x, y = np.meshgrid(x, y, indexing='ij') # grid_x, grid_x
    x = x.reshape(1, 1, 1, grid_x, grid_x, 1) # 1, 1, 1, grid_x, grid_x, 1
    y = y.reshape(1, 1, 1, grid_x, grid_x, 1)

    g = A * np.sin(k1 * x + k2 * y) + \
        B * np.cos(k1 * x + k2 * y) # N, K, K, grid_x, grid_x, 1
    g = g.sum(1) # N, K, grid_x, grid_x, 1
    g = g.sum(1) # N, grid_x, grid_x, 1
    g = g + C

    d2u_dx2 = -(k1 ** 2) * A * np.sin(k1 * x + k2 * y) - \
        (k1 ** 2) * B * np.cos(k1 * x + k2 * y)

    d2u_dy2 = -(k2 ** 2) * A * np.sin(k1 * x + k2 * y) - \
        (k2 ** 2) * B * np.cos(k1 * x + k2 * y)

    Delta_g = d2u_dx2 + d2u_dy2
    Delta_g = Delta_g.sum(1)
    Delta_g = Delta_g.sum(1) # N, grid_x, grid_x, 1

    g = g.reshape(N, grid_x, grid_x, 1, 1)
    Delta_g = Delta_g.reshape(N, grid_x, grid_x, 1, 1)
    
    return g, Delta_g

def generate_h_x(N, grid_t, T):
    from data import AntideData
    s0 = [0]
    length_scale = 0.2
    data = AntideData(T, s0, grid_t, grid_t, length_scale, N, 1)

    h_t, h = data.X_train, data.y_train # N, grid_t, 1

    h_t = h_t.reshape(N, 1, 1, grid_t, 1)
    h = h.reshape(N, 1, 1, grid_t, 1)

    return h_t, h

def generate_data(N, grid_x, grid_t, T, K):
    g, Delta_g = generate_g_x(N, grid_x, K)
    h_t, h = generate_h_x(N, grid_t, T)

    u = g * h # N, grid_x, grid_x, grid_t, 1

    f = Delta_g * h + (g * h) ** 2 - g * h_t

    return u, f

y_train, f_train = generate_data(args.N_train, args.grid_x, args.grid_t, args.T, args.K)
y_test, f_test = generate_data(args.N_test, args.grid_x, args.grid_t, args.T, args.K)

# Setting device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, Y_train, X_test, Y_test = torch.from_numpy(f_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(f_test).float(), torch.from_numpy(y_test).float()

if args.data2cuda:
    X_train, Y_train, X_test, Y_test = X_train.to(device), Y_train.to(device), X_test.to(device), Y_test.to(device)

def Grid2Fourier(X, grid_x):
    # X shape: [N, grid_x, grid_x, grid_t, 1]
    X = X.permute(0, 3, 1, 2, 4).squeeze() # N, grid_t, grid_x, grid_x
    X = torch.fft.rfft2(X) / grid_x ** 2 # N, grid_t, grid_x, grid_x // 2 + 1
    X = X.reshape(X.size(0), X.size(1), -1) # N, grid_t, -1
    X_real = torch.real(X)
    X_imag = torch.imag(X)
    X = torch.cat([X_real, X_imag], -1) # N, grid_t, -1
    return X

def Fourier2Grid(X, grid_x):
    N, grid_t = X.size(0), X.size(1)
    X_real = X[:, :, :X.size(-1)//2]
    X_imag = X[:, :, X.size(-1)//2:]
    X = X_real + X_imag * 1j
    X[:, :, 0] = X_real[:, :, 0]
    X = X * grid_x ** 2
    X = X.reshape(N, grid_t, grid_x, grid_x // 2 + 1)
    X = torch.fft.irfft2(X, s=[grid_x] * 2)
    return X

X_train = Grid2Fourier(X_train, args.grid_x)
y_train = Grid2Fourier(Y_train, args.grid_x)
X_test = Grid2Fourier(X_test, args.grid_x)
y_test = Grid2Fourier(Y_test, args.grid_x)

Y_train = Y_train.squeeze().permute(0, 3, 1, 2)
Y_test = Y_test.squeeze().permute(0, 3, 1, 2)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(Y_train.shape, Y_test.shape)
io_dim = 2 * (args.grid_x // 2 + 1) * args.grid_x

def get_model(args, io_dim, device):
    if args.model == "LSTM":
        return LSTM(input_dim=io_dim, output_dim=io_dim, hidden_dim=256, num_layers=1).to(device)
    if args.model == "GRU":
        return GRU(input_dim=io_dim, output_dim=io_dim, hidden_dim=256, num_layers=1).to(device)
    if args.model == "Mamba":
        return Mamba_NO(d_model=256, n_layer=1, d_intermediate=0, input_dim=io_dim, output_dim=io_dim).to(device)
    if args.model in ["MambaScratch", "mamba_scratch"]:
        return MambaScratch_NO(input_dim=io_dim, output_dim=io_dim, hidden_dim=256, num_layers=1, d_state=16).to(device)
    if args.model == "OSS":
        return OSS_NO(
            d_model=args.oss_hidden_dim,
            input_dim=io_dim,
            output_dim=io_dim,
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
    if args.model == "ST":
        return GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type="standard", mlp_dim=128).to(device)
    if args.model == "GT":
        return GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type="galerkin", mlp_dim=128).to(device)
    if args.model == "FT":
        return GalerkinTransformer(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=2, heads=4, dim_head=128, attn_type="fourier", mlp_dim=128).to(device)
    if args.model == "T":
        return Transformer(ninput=io_dim, noutput=io_dim, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1).to(device)
    if args.model == "GNOT":
        return GNOT(dim_in=io_dim, dim_out=io_dim, dim_hid=128, depth=1, heads=4, dim_head=128, n_experts=2).to(device)
    raise ValueError(f"Unsupported model: {args.model}")


model = get_model(args, io_dim, device)

# Function to count parameters
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {count_parameters(model)}")

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
        "Y_train_fourier": list(y_train.shape),
        "X_test": list(X_test.shape),
        "Y_test_fourier": list(y_test.shape),
        "Y_train_grid": list(Y_train.shape),
        "Y_test_grid": list(Y_test.shape),
    },
    "model_details": {
        "model_name": args.model,
        "model_class": model.__class__.__name__,
        "total_parameters": count_parameters(model),
    },
}
with open(config_file_path, "w") as f_config:
    json.dump(config_data, f_config, indent=2)
print("Config file:", config_file_path)

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

train_dataset = TensorDataset(X_train, y_train, Y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test, Y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

loss_traj = []
best_test_error = float("inf")
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")
eval_interval = max(1, int(args.eval_interval))
# Training loop
for epoch in tqdm(range(args.num_epochs)):
    model.train()
    epoch_losses = []

    for batch_idx, (data, targets, _) in enumerate(train_loader):
        optimizer.zero_grad()
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        loss = torch.linalg.vector_norm(outputs - targets) / torch.linalg.vector_norm(targets)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

        outputs_grid = Fourier2Grid(outputs, args.grid_x)
        targets_grid = Fourier2Grid(targets, args.grid_x)
        loss_saved = criterion(outputs_grid, targets_grid)
        loss_traj.append(loss_saved.item())

    scheduler.step()
    train_loss_epoch_avg = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

    if ((epoch + 1) % eval_interval == 0) or (epoch + 1 == args.num_epochs):
        model.eval()
        output_train, label_train = [], []
        output_train_grid, label_train_grid = [], []

        with torch.no_grad():
            for batch_idx, (data, targets, Targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)

                output_train.append(outputs.detach().cpu())
                label_train.append(targets.detach().cpu())

                outputs_grid = Fourier2Grid(outputs, args.grid_x)
                output_train_grid.append(outputs_grid.detach().cpu())
                label_train_grid.append(Targets.detach().cpu())

            output_train = torch.cat(output_train, 0)
            label_train = torch.cat(label_train, 0)
            error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm(label_train.reshape(-1))

            output_train_grid = torch.cat(output_train_grid, 0)
            label_train_grid = torch.cat(label_train_grid, 0)
            error_train_grid = torch.norm((output_train_grid - label_train_grid).reshape(-1)) / torch.norm(label_train_grid.reshape(-1))

            output_test, label_test = [], []
            output_test_grid, label_test_grid = [], []
            for batch_idx, (data, targets, Targets) in enumerate(test_loader):
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)

                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

                outputs_grid = Fourier2Grid(outputs, args.grid_x)
                output_test_grid.append(outputs_grid.detach().cpu())
                label_test_grid.append(Targets.detach().cpu())

            output_test = torch.cat(output_test, 0)
            label_test = torch.cat(label_test, 0)
            error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm(label_test.reshape(-1))

            output_test_grid = torch.cat(output_test_grid, 0)
            label_test_grid = torch.cat(label_test_grid, 0)
            error_test_grid = torch.norm((output_test_grid - label_test_grid).reshape(-1)) / torch.norm(label_test_grid.reshape(-1))

        with open(loss_log_path, "a") as f_loss:
            f_loss.write(
                f"{epoch + 1},{train_loss_epoch_avg:.10e},{error_train.item():.10e},{error_train_grid.item():.10e},{error_test.item():.10e},{error_test_grid.item():.10e}\n"
            )

        if error_test_grid.item() < best_test_error:
            best_test_error = error_test_grid.item()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_epoch_avg": train_loss_epoch_avg,
                    "train_rel_l2_fourier": error_train.item(),
                    "train_rel_l2_grid": error_train_grid.item(),
                    "test_rel_l2_fourier": error_test.item(),
                    "test_rel_l2_grid": error_test_grid.item(),
                    "args": {key: _jsonable(value) for key, value in vars(args).items()},
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch + 1}, Train: {error_train.item():.3e} {error_train_grid.item():.3e}, "
            f"Test: {error_test.item():.3e} {error_test_grid.item():.3e}"
        )

torch.save(
    {
        "epoch": args.num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_test_error": best_test_error,
        "args": {key: _jsonable(value) for key, value in vars(args).items()},
    },
    last_model_path,
)

loss_traj = np.asarray(loss_traj)
filename = os.path.join(
    log_dir,
    "RD_Fourier_3D_"
    + "Model=" + str(args.model) + "_"
    + "Seed=" + str(args.SEED)
    + ".txt",
)
np.savetxt(filename, loss_traj)

summary_path = os.path.join(log_dir, "training_summary.json")
summary_data = {
    "timestamp": datetime.now().isoformat(),
    "run_number": run_number,
    "best_test_error": best_test_error,
    "best_model_path": best_model_path,
    "last_model_path": last_model_path,
    "loss_log_path": loss_log_path,
    "legacy_loss_txt_path": filename,
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