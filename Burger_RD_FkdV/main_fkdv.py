import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from DON_2d import POD_GalerkinTransformer, POD_GNOT, POD_GRU, POD_LSTM, POD_Mamba, POD_Mamba_Scratch, POD_Transformer, MambaConfig
from FFNO2d import FFNO
from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2


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

    parser.add_argument("--config", type=str, default=os.path.join(SCRIPT_DIR, "config", "config_fkdv.json"))

    parser.add_argument("--SEED", type=int, default=0)

    parser.add_argument("--grid_x", type=int, default=100, help="x-axis grid size")
    parser.add_argument("--grid_t", type=int, default=100, help="t-axis grid size")

    parser.add_argument("--L", type=float, default=5, help="domain size")
    parser.add_argument("--T", type=float, default=5, help="terminal time")

    parser.add_argument("--N_train", type=int, default=27000)
    parser.add_argument("--N_test", type=int, default=3000)

    parser.add_argument("--num_epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="minibatch size for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--eval_interval", type=int, default=10, help="epochs between validation/checkpoint logging")

    parser.add_argument("--model", type=str, default="OSS")

    parser.add_argument("--FNO_hidden_dim", type=int, default=64)
    parser.add_argument("--FNO_num_layers", type=int, default=1)
    parser.add_argument("--FNO_modes_x", type=int, default=32)
    parser.add_argument("--FNO_modes_t", type=int, default=16)

    parser.add_argument("--LSTM_hidden_dim", type=int, default=256)
    parser.add_argument("--LSTM_num_layers", type=int, default=1)

    parser.add_argument("--GRU_hidden_dim", type=int, default=256)
    parser.add_argument("--GRU_num_layers", type=int, default=1)

    parser.add_argument("--MambaLLM_d_model", type=int, default=256)
    parser.add_argument("--MambaLLM_n_layer", type=int, default=1)

    parser.add_argument("--data2cuda", type=int, default=1)
    parser.add_argument("--save_loss", type=int, default=0)
    parser.add_argument("--save_model", type=int, default=0)

    parser.add_argument("--discretization", type=str, default="IMEX", choices=["IM", "IMEX"])
    parser.add_argument("--amp", type=int, default=1, help="Use automatic mixed precision on CUDA (1/0)")
    parser.add_argument("--profile_timing", type=int, default=1, help="Log avg forward/backward/step time per batch (1/0)")
    parser.add_argument("--track_linoss_param", type=str, default="blocks.0.model_t.linoss_A_diag")
    parser.add_argument("--track_linoss_index", type=int, default=0)

    # Included for config parity with burgers/OSS settings.
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--dt_min", type=float, default=1e-3)
    parser.add_argument("--dt_max", type=float, default=1.0)

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

base_log_dir = os.path.join(SCRIPT_DIR, "run_fkdv")
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


def generate_data_1(grid_x=1000, grid_t=100, L=5, T=5):
    def u(k, A, alpha, beta, delta, b1, b0, b, x, t):
        inside_sec = k * (x - delta * k * k * t) - (b * torch.arctan(A * t) + b1 * t + b0)
        return 12 * beta * (k ** 2) / (torch.cosh(inside_sec) ** 2)

    def f(k, A, alpha, beta, delta, b1, b0, b, x, t):
        temp1 = 12 * k * beta / alpha
        temp2 = (k ** 3) * (4 * beta - delta) - b * A / (1 + A ** 2 * t ** 2) - b1
        inside_sec = k * (x - delta * k * k * t) - (b * torch.arctan(A * t) + b1 * t + b0)
        return temp1 * temp2 / (torch.cosh(inside_sec) ** 2)

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    # PDE parameters

    # k = np.random.rand() * 0.5 + 0.001 # [0.5, 1.5]
    # A = np.random.rand() + 2.5 # [2.5, 3.5]
    # alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    # beta = (np.random.rand() * 0.25 + 0.25) / k ** 2 # [0.25, 0.5]
    # b1 = (np.random.rand() - 0.5) * 10 * k # [-3, 3]
    # b0 = (np.random.rand() - 0.5) * 50 * k # [-25, 25]
    # b = (np.random.rand() - 0.5) * 10 * k # [-2.5, 2.5]
    # delta = (np.random.rand() - 0.5) * 8 / k ** 3 # [-4, 4]

    k = np.random.rand() + 0.5 # [0.5, 1.5]
    A = np.random.rand() + 2.5 # [2.5, 3.5]
    alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    b1 = (np.random.rand() - 0.5) * 6 # [-3, 3]
    b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    b = (np.random.rand() - 0.5) * 2 # [-1, 1]
    delta = (np.random.rand() - 0.5) * 8 # [-4, 4]

    x.requires_grad_()
    ff = lambda x, t: f(k, A, alpha, beta, delta, b1, b0, b, x, t)
    out_f = ff(x, t)

    uu = lambda x, t: u(k, A, alpha, beta, delta, b1, b0, b, x, t)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1)
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1) # grid_x, grid_t, 7

    f_x = f_x.detach().to("cpu")
    u_sol = u_sol.detach().to("cpu")

    return f_x, u_sol


def generate_data_2(grid_x=1000, grid_t=100, L=5, T=5):
    def u(k, A, alpha, beta, delta, b2, b1, b0, x, t):
        inside_sec = k * (x - delta * k * k * t) - torch.exp(b2 * t ** 2 + b1 * t + b0)
        return 12 * beta * (k ** 2) / (torch.cosh(inside_sec) ** 2)

    def f(k, A, alpha, beta, delta, b2, b1, b0, x, t):
        temp1 = 12 * k * beta / alpha
        temp2 = (k ** 3) * (4 * beta - delta) - (2 * b2 * t + b1) * torch.exp(b2 * t ** 2 + b1 * t + b0)
        inside_sec = k * (x - delta * k * k * t) - torch.exp(b2 * t ** 2 + b1 * t + b0)
        return temp1 * temp2 / (torch.cosh(inside_sec) ** 2)

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing='ij') # [grid_x, grid_t]
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    # PDE parameters

    k = np.random.rand() + 0.5 # [0.5, 1.5]
    A = np.random.rand() + 2.5 # [2.5, 3.5]
    alpha = np.random.rand() + 1.5 # [1.5, 2.5]
    beta = np.random.rand() * 0.25 + 0.125 # [0.125, 0.375]
    b2 = (np.random.rand() - 2) # [-2, -1]
    b1 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    delta = (np.random.rand() - 0.5) * 8 # [-4, 4]

    # k = np.random.rand() * 0.5 + 0.001 # [0.5, 1.5]
    # A = np.random.rand() + 2.5 # [2.5, 3.5]
    # alpha = (np.random.rand() + 1.5) / k * 24 # [1.5, 2.5]
    # beta = (n, ndom.rand() * 0.25 + 0.25) / k ** 2 # [0.25,, ]
    # b2 = (np.random.rand() - 1) * 2 # [-2, 0]
    # b1 = (np.random.rand() - 0.5) * 6 # [-3, 3]
    # b0 = (np.random.rand() - 0.5) * 2 # [-1, 1]
    # delta = (np.random.rand() - 0.5) * 8 / k ** 3 # [-4, 4]

    x.requires_grad_()
    ff = lambda x, t: f(k, A, alpha, beta, delta, b2, b1, b0, x, t)
    out_f = ff(x, t)
    # f_x = torch.autograd.grad(out_f.sum(), x, create_graph=True)[0] # grid_x \times grid_t

    uu = lambda x, t: u(k, A, alpha, beta, delta, b2, b1, b0, x, t)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1) # grid_x \times grid_t
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1) # grid_x, grid_t, 7

    f_x = f_x.detach().to('cpu')
    u_sol = u_sol.detach().to('cpu')

    return f_x, u_sol


def generate_data_3(grid_x=1000, grid_t=100, L=5, T=5):

    def F(a, b, d, x, t):
        return (x + a) ** 2 + (t + b) ** 2 + d
    
    def H(a, b, d, x, t):
        return (x + a) / torch.sqrt((t + b) ** 2 + d)
    
    def u(a, b, d, beta, gamma, x, t):
        return 12 * beta * gamma / F(a, b, d, x, t)
    
    def f(a, b, d, alpha, beta, gamma, x, t):
        const = 12 * beta * gamma / alpha
        t1 = 6 * beta * (gamma + 1) / (F(a, b, d, x, t) ** 2)
        t2 = 8 * beta * ((t + b) ** 2 + d) / (F(a, b, d, x, t) ** 3)
        t3 = (t + b) * (x + a) / ((t + b) ** 2 + d) / F(a, b, d, x, t)
        t4 = (t + b) / (torch.sqrt((t + b) ** 2 + d) ** 3) * torch.arctan(H(a, b, d, x, t))
        return const * (t1 - t2 - t3 - t4)

    x = np.linspace(-L, L, grid_x)
    t = np.linspace(0, T, grid_t)
    x, t = np.meshgrid(x, t, indexing="ij")
    x, t = x.reshape(-1), t.reshape(-1)
    x, t = torch.from_numpy(x).float(), torch.from_numpy(t).float()
    t0 = torch.zeros(grid_x * grid_t)
    x1 = torch.zeros(grid_x * grid_t) + L
    x2 = torch.zeros(grid_x * grid_t) - L

    a = (np.random.rand() - 0.5) * 6
    alpha = np.random.rand() + 1.5
    beta = np.random.rand() * 0.25 + 0.125
    b = (np.random.rand() - 0.5) * 6 - 1
    d = np.random.rand() * 0.5 + 0.3
    gamma = (np.random.rand() + 0.5) * 1

    x.requires_grad_()
    f_callable = lambda xx, tt: f(a, b, d, alpha, beta, gamma, xx, tt)
    out_f = f_callable(x, t)

    uu = lambda xx, tt: u(a, b, d, beta, gamma, xx, tt)
    u_sol = uu(x, t)
    initial = uu(x, t0)

    boundary_1 = uu(x1, t)

    x2.requires_grad_()
    boundary_2 = uu(x2, t)
    boundary_3 = torch.autograd.grad(boundary_2.sum(), x2, create_graph=True)[0]

    f_x = torch.stack([out_f, initial, boundary_1, boundary_2, boundary_3], -1)
    f_x = f_x.reshape(grid_x, grid_t, 5)
    u_sol = u_sol.reshape(grid_x * grid_t, 1)

    alpha = torch.zeros(grid_x, grid_t, 1) + alpha
    beta = torch.zeros(grid_x, grid_t, 1) + beta

    f_x = torch.cat([f_x, alpha, beta], -1)

    f_x = f_x.detach().to("cpu")
    u_sol = u_sol.detach().to("cpu")
    return f_x, u_sol


X_train, y_train = [], []
for _ in tqdm(range(args.N_train // 3), file=sys.stdout, desc="Build train data"):
    x_data, y_data = generate_data_1(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x_data)
    y_train.append(y_data)
    x_data, y_data = generate_data_2(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x_data)
    y_train.append(y_data)
    x_data, y_data = generate_data_3(args.grid_x, args.grid_t, args.L, args.T)
    X_train.append(x_data)
    y_train.append(y_data)
X_train = torch.stack(X_train, 0)
y_train = torch.stack(y_train, 0)
print(X_train.shape, y_train.shape)
print(X_train.isnan().any(), y_train.isnan().any())

X_test, y_test = [], []
for _ in tqdm(range(args.N_test // 3), file=sys.stdout, desc="Build test data"):
    x_data, y_data = generate_data_1(args.grid_x, args.grid_t, args.L, args.T)
    X_test.append(x_data)
    y_test.append(y_data)
    x_data, y_data = generate_data_2(args.grid_x, args.grid_t, args.L, args.T)
    X_test.append(x_data)
    y_test.append(y_data)
    x_data, y_data = generate_data_3(args.grid_x, args.grid_t, args.L, args.T)
    X_test.append(x_data)
    y_test.append(y_data)
X_test = torch.stack(X_test, 0)
y_test = torch.stack(y_test, 0)
print(X_test.shape, y_test.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.data2cuda:
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

in_dim = 7

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
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS",
        discretization=args.discretization,
    ).to(device)
elif args.model == "FNO_OSS_2":
    model = FNO_Jamba_2(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS",
        discretization=args.discretization,
    ).to(device)
elif args.model == "OSS":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS",
        discretization=args.discretization,
    ).to(device)
elif args.model == "FNO_OSS_source_1":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS_source",
        discretization=args.discretization,
    ).to(device)
elif args.model == "FNO_OSS_source_2":
    model = FNO_Jamba_2(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS_source",
        discretization=args.discretization,
    ).to(device)
elif args.model == "OSS_source":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="OSS_source",
        discretization=args.discretization,
    ).to(device)
elif args.model == "FNO_linoss_pytorch_1":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="linoss_pytorch",
        discretization=args.discretization,
    ).to(device)
elif args.model == "FNO_linoss_pytorch_2":
    model = FNO_Jamba_2(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="linoss_pytorch",
        discretization=args.discretization,
    ).to(device)
elif args.model == "linoss_pytorch":
    model = FNO_Jamba_1(
        input_dim=in_dim,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type="linoss_pytorch",
        discretization=args.discretization,
    ).to(device)
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
        fused_add_norm=True,
    )
    model = POD_Mamba(256, 1, 0, in_dim * args.grid_x, args.grid_x, config.ssm_cfg).to(device)
elif args.model in ["MambaScratch", "mamba_scratch"]:
    model = POD_Mamba_Scratch(
        input_dim=in_dim * args.grid_x,
        output_dim=args.grid_x,
        hidden_dim=256,
        num_layers=1,
        d_state=16,
    ).to(device)
elif args.model == "GT":
    model = POD_GalerkinTransformer(
        dim_in=in_dim * args.grid_x,
        dim_out=args.grid_x,
        dim_hid=256,
        depth=1,
        heads=4,
        dim_head=256,
        attn_type="galerkin",
        mlp_dim=256,
    ).to(device)
elif args.model == "ST":
    model = POD_GalerkinTransformer(
        dim_in=in_dim * args.grid_x,
        dim_out=args.grid_x,
        dim_hid=256,
        depth=1,
        heads=4,
        dim_head=256,
        attn_type="standard",
        mlp_dim=256,
    ).to(device)
elif args.model == "T":
    model = POD_Transformer(
        ninput=in_dim * args.grid_x,
        noutput=args.grid_x,
        nhidden=256,
        dim_feedforward=256,
        nhead=4,
        nlayers=1,
    ).to(device)
elif args.model == "GNOT":
    model = POD_GNOT(dim_in=in_dim * args.grid_x, dim_out=args.grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2).to(device)
else:
    raise ValueError(f"Unsupported model: {args.model}")

uses_oss_source = args.model in ["FNO_OSS_source_1", "FNO_OSS_source_2", "OSS_source"]
if uses_oss_source:
    try:
        import jax

        print("JAX version:", jax.__version__)
        print("JAX backend:", jax.default_backend())
        print("JAX devices:", jax.devices())
    except Exception as jax_exc:
        raise RuntimeError(
            "OSS_source models require JAX, but JAX initialization failed. "
            "Install/activate JAX or choose another model."
        ) from jax_exc


tracking_enabled = args.model in ["FNO_OSS_source_1", "FNO_OSS_source_2", "OSS_source", "FNO_linoss_pytorch_1", "FNO_linoss_pytorch_2", "linoss_pytorch"]


def count_parameters(model_obj):
    return sum(param.numel() for param in model_obj.parameters() if param.requires_grad)


print(f"Total number of parameters: {count_parameters(model)}")


def model_has_complex_tensors(module):
    tensors = list(module.parameters()) + list(module.buffers())
    return any(tensor.is_complex() for tensor in tensors)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

has_complex_tensors = model_has_complex_tensors(model)
is_linoss_model = "linoss" in args.model.lower() or "oss_source" in args.model.lower()
use_amp = bool(args.amp) and torch.cuda.is_available() and (not has_complex_tensors) and (not is_linoss_model)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if bool(args.amp) and has_complex_tensors:
    print("AMP disabled because the model contains complex-valued tensors")
elif bool(args.amp) and is_linoss_model:
    print("AMP disabled for linoss_pytorch/OSS_source models")
else:
    print(f"AMP enabled: {use_amp}")

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
        "L": args.L,
        "T": args.T,
        "N_train": args.N_train,
        "N_test": args.N_test,
        "input_channels": in_dim,
    },
}
with open(config_file_path, "w") as f_config:
    json.dump(_jsonable(config_data), f_config, indent=2)
print("Configuration saved to:", config_file_path)


def _get_named_param(model_obj, param_name):
    for name, param in model_obj.named_parameters():
        if name == param_name:
            return param
    return None


tracked_param_at_start = None
linoss_param_logged = False
if tracking_enabled:
    tracked_param_at_start = _get_named_param(model, args.track_linoss_param)
    if tracked_param_at_start is None:
        print("Warning: tracked parameter not found:", args.track_linoss_param)
    else:
        print("Tracking parameter:", args.track_linoss_param)


loss_traj = []
best_test_error = float("inf")
best_model_path = os.path.join(weights_dir, "best_model.pth")
last_model_path = os.path.join(weights_dir, "last_checkpoint.pth")

for epoch in tqdm(range(args.num_epochs), file=sys.stdout, desc="Epochs"):
    epoch_loss = 0.0
    num_batches = 0
    timing_enabled = bool(args.profile_timing)
    forward_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    for data, targets in train_loader:
        model.train()
        optimizer.zero_grad()

        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(data)
            loss = torch.linalg.vector_norm(outputs - targets) / torch.linalg.vector_norm(targets)
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

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

        if tracking_enabled and tracked_param is not None and tracked_param.grad is not None and tracked_param.grad.numel() > args.track_linoss_index:
            grad_value = tracked_param.grad.view(-1)[args.track_linoss_index].item()

        scaler.step(optimizer)
        scaler.update()
        if timing_enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        if tracking_enabled and (not linoss_param_logged) and (before_value is not None) and (grad_value is not None):
            after_value = tracked_param.view(-1)[args.track_linoss_index].item()
            print(
                f"Parameter update tracked: {args.track_linoss_param}[{args.track_linoss_index}] "
                f"before={before_value:.6e}, grad={grad_value:.6e}, after={after_value:.6e}"
            )
            linoss_param_logged = True

        epoch_loss += loss.item()
        num_batches += 1

        if timing_enabled:
            forward_time += t1 - t0
            backward_time += t2 - t1
            step_time += t3 - t2

    train_loss_epoch = epoch_loss / max(num_batches, 1)
    loss_traj.append(train_loss_epoch)
    scheduler.step()

    with open(loss_log_path, "a") as f_loss:
        f_loss.write(f"{epoch + 1},{train_loss_epoch:.8e},nan,nan\n")

    print(f"Epoch {epoch + 1}, Train Loss (epoch avg): {train_loss_epoch:.3e}")
    if timing_enabled and num_batches > 0:
        print(
            f"Epoch {epoch + 1} timing (s/batch): "
            f"forward={forward_time / num_batches:.3f}, "
            f"backward={backward_time / num_batches:.3f}, "
            f"step={step_time / num_batches:.3f}"
        )

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
        error_train = torch.norm((output_train - label_train).reshape(-1)) / torch.norm((label_train).reshape(-1))

        output_test, label_test = [], []
        with torch.no_grad():
            for data, targets in test_loader:
                outputs = model(data)
                output_test.append(outputs.detach().cpu())
                label_test.append(targets.detach().cpu())

        output_test = torch.cat(output_test, 0)
        label_test = torch.cat(label_test, 0)
        error_test = torch.norm((output_test - label_test).reshape(-1)) / torch.norm((label_test).reshape(-1))

        print(
            f"Epoch {epoch + 1}, Train Loss: {loss.item():.3e}, "
            f"Train Rel L2: {error_train.item():.3e}, Test Rel L2: {error_test.item():.3e}"
        )

        with open(loss_log_path, "a") as f_loss:
            f_loss.write(f"{epoch + 1},{train_loss_epoch:.8e},{error_train.item():.8e},{error_test.item():.8e}\n")

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
    loss_txt_path = os.path.join(log_dir, f"fKdV_Model={args.model}_Seed={args.SEED}.txt")
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
