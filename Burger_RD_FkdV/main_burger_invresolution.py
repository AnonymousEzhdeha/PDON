import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from FNO2d import FNO2d
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2
from FFNO2d import FFNO
from DON_2d import POD_GRU, POD_LSTM, POD_Mamba, POD_Mamba_Scratch, MambaConfig
from DON_2d import POD_GalerkinTransformer, POD_Transformer, POD_GNOT
from burgers_data import AntideData, AntideAntideData


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def save_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def get_parent_run_dir(checkpoint_path):
    weights_dir = os.path.dirname(checkpoint_path)
    return os.path.dirname(weights_dir)


def resolve_training_config_path(checkpoint_path):
    run_dir = get_parent_run_dir(checkpoint_path)
    return os.path.join(run_dir, "config.json")


def is_resolution_locked_model(model_name):
    locked_prefixes = ("GRU", "LSTM", "Mamba", "GT", "ST", "FT", "T", "GNOT")
    return model_name in locked_prefixes or model_name in {"MambaScratch", "mamba_scratch"}


def get_model_details(model, model_name):
    details = {
        "model_name": model_name,
        "model_class": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameters_all": sum(p.numel() for p in model.parameters()),
    }
    if hasattr(model, "config"):
        details["model_config"] = str(model.config)

    model_attrs = {}
    for attr_name in dir(model):
        if attr_name.startswith("_"):
            continue
        try:
            attr_val = getattr(model, attr_name)
        except Exception:
            continue
        if isinstance(attr_val, (int, float, str, bool, list, dict)):
            model_attrs[attr_name] = attr_val

    if model_attrs:
        details["model_attributes"] = model_attrs

    return details


def generate_t(grid_t, T, seed, n_train, n_test):
    s0 = [0]
    sensor_in = grid_t
    sensor_out = grid_t
    length_scale = 0.2
    train_num = n_train // 4
    test_num = n_test // 4

    np.random.seed(seed)
    data = AntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_3_train = data.X_train
    g_2_train = data.y_train
    g_3_test = data.X_test
    g_2_test = data.y_test

    np.random.seed(seed)
    s0 = [0, 0]
    data = AntideAntideData(T, s0, sensor_in, sensor_out, length_scale, train_num, test_num)
    g_1_train = data.y_train
    g_1_test = data.y_test

    return g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test


def generate_data(grid_x, grid_t, T, g_1, g_2, g_3):
    g_1 = g_1.T
    g_2 = g_2.T
    g_3 = g_3.T

    x = np.linspace(-5, 5, grid_x)
    t = np.linspace(0.5, 0.5 + T, grid_t)
    x, t = np.meshgrid(x, t, indexing="ij")

    c1 = np.random.rand() * 3
    c2 = np.random.rand() * 6 - 3

    def u1_sol(xv, tv):
        return xv / tv - c1 / tv - (g_1 - tv * g_2) / tv

    u0 = np.tile(g_2, (grid_x, 1))
    u0_init = u0[:, 0:1]
    u0_boundary_1 = u0[0:1, :]
    u0_boundary_2 = u0[-1:, :]
    u0_init = np.tile(u0_init, (1, grid_t))
    u0_boundary_1 = np.tile(u0_boundary_1, (grid_x, 1))
    u0_boundary_2 = np.tile(u0_boundary_2, (grid_x, 1))

    u1 = u1_sol(x, t)
    u1_init = u1[:, 0:1]
    u1_boundary_1 = u1[0:1, :]
    u1_boundary_2 = u1[-1:, :]
    u1_init = np.tile(u1_init, (1, grid_t))
    u1_boundary_1 = np.tile(u1_boundary_1, (grid_x, 1))
    u1_boundary_2 = np.tile(u1_boundary_2, (grid_x, 1))

    const = np.sqrt(2 * c1)

    def u2_sol(xv, tv):
        return -const * np.tanh(0.5 * const * (g_1 - xv + c2)) + g_2

    u2 = u2_sol(x, t)
    u2_init = u2[:, 0:1]
    u2_boundary_1 = u2[0:1, :]
    u2_boundary_2 = u2[-1:, :]
    u2_init = np.tile(u2_init, (1, grid_t))
    u2_boundary_1 = np.tile(u2_boundary_1, (grid_x, 1))
    u2_boundary_2 = np.tile(u2_boundary_2, (grid_x, 1))

    const = np.sqrt(2 * c1)

    def u3_sol(xv, tv):
        return const / tv * np.tanh(0.5 * const * ((xv - g_1) / tv + c2)) + (xv - g_1) / tv + g_2

    u3 = u3_sol(x, t)
    u3_init = u3[:, 0:1]
    u3_boundary_1 = u3[0:1, :]
    u3_boundary_2 = u3[-1:, :]
    u3_init = np.tile(u3_init, (1, grid_t))
    u3_boundary_1 = np.tile(u3_boundary_1, (grid_x, 1))
    u3_boundary_2 = np.tile(u3_boundary_2, (grid_x, 1))

    f = np.tile(g_3, (grid_x, 1))

    f0 = np.stack([f, u0_init, u0_boundary_1, u0_boundary_2], -1).reshape(grid_x, grid_t, 4)
    f1 = np.stack([f, u1_init, u1_boundary_1, u1_boundary_2], -1).reshape(grid_x, grid_t, 4)
    f2 = np.stack([f, u2_init, u2_boundary_1, u2_boundary_2], -1).reshape(grid_x, grid_t, 4)
    f3 = np.stack([f, u3_init, u3_boundary_1, u3_boundary_2], -1).reshape(grid_x, grid_t, 4)

    f_out = np.stack([f0, f1, f2, f3], 0)
    u_out = np.stack([u0, u1, u2, u3], 0).reshape(4, -1, 1)
    return f_out, u_out


def build_model(model_name, in_dim, grid_x, discretization):
    if model_name == "FNO":
        return FNO2d(modes1=32, modes2=32, width=24, num_layers=2, in_dim=in_dim, out_dim=1)
    if model_name == "FFNO":
        return FFNO(modes=32, width=96, input_dim=in_dim, output_dim=1, n_layers=2)
    if model_name == "FNO_GRU_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU")
    if model_name == "FNO_GRU_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="GRU")
    if model_name == "FNO_LSTM_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM")
    if model_name == "FNO_LSTM_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="LSTM")
    if model_name == "FNO_Mamba_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba")
    if model_name == "FNO_Mamba_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="Mamba")
    if model_name == "FNO_OSS_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=discretization)
    if model_name == "FNO_OSS_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=discretization)
    if model_name == "OSS":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS", discretization=discretization)
    if model_name == "FNO_OSS_source_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=discretization)
    if model_name == "FNO_OSS_source_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=discretization)
    if model_name == "OSS_source":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="OSS_source", discretization=discretization)
    if model_name == "FNO_linoss_pytorch_1":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=discretization)
    if model_name == "FNO_linoss_pytorch_2":
        return FNO_Jamba_2(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=discretization)
    if model_name == "linoss_pytorch":
        return FNO_Jamba_1(input_dim=in_dim, output_dim=1, modes=32, width=128, num_layers=1, model_t_type="linoss_pytorch", discretization=discretization)
    if model_name == "GRU":
        return POD_GRU(input_dim=in_dim * grid_x, output_dim=grid_x, hidden_dim=256, num_layers=1)
    if model_name == "LSTM":
        return POD_LSTM(input_dim=in_dim * grid_x, output_dim=grid_x, hidden_dim=256, num_layers=1)
    if model_name == "Mamba":
        config = MambaConfig(
            d_model=256,
            n_layer=1,
            vocab_size=0,
            ssm_cfg=dict(layer="Mamba1"),
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
        )
        return POD_Mamba(256, 1, 0, in_dim * grid_x, grid_x, config.ssm_cfg)
    if model_name in ["MambaScratch", "mamba_scratch"]:
        return POD_Mamba_Scratch(
            input_dim=in_dim * grid_x,
            output_dim=grid_x,
            hidden_dim=256,
            num_layers=1,
            d_state=16,
        )
    if model_name == "GT":
        return POD_GalerkinTransformer(dim_in=in_dim * grid_x, dim_out=grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="galerkin", mlp_dim=256)
    if model_name == "ST":
        return POD_GalerkinTransformer(dim_in=in_dim * grid_x, dim_out=grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="standard", mlp_dim=256)
    if model_name == "FT":
        return POD_GalerkinTransformer(dim_in=in_dim * grid_x, dim_out=grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, attn_type="fourier", mlp_dim=256)
    if model_name == "T":
        return POD_Transformer(ninput=in_dim * grid_x, noutput=grid_x, nhidden=256, dim_feedforward=256, nhead=4, nlayers=1)
    if model_name == "GNOT":
        return POD_GNOT(dim_in=in_dim * grid_x, dim_out=grid_x, dim_hid=256, depth=1, heads=4, dim_head=256, n_experts=2)
    raise ValueError(f"Unsupported model '{model_name}'")


def infer_resolution_safe(model_name, training_grid_x, target_grid_x):
    if is_resolution_locked_model(model_name) and target_grid_x != training_grid_x:
        raise ValueError(
            f"Model '{model_name}' is resolution-locked in grid_x: trained on {training_grid_x}, requested {target_grid_x}."
        )


def compute_relative_l2(predictions, targets):
    return torch.norm((predictions - targets).reshape(-1)) / torch.norm(targets.reshape(-1))


def parse_args():
    parser = argparse.ArgumentParser(description="DeepOMamba Burgers inverse-resolution evaluation")
    parser.add_argument("--config_res_inv", type=str, default=os.path.join(SCRIPT_DIR, "Example_1+1_Dim/config/config_res_inv.json"))
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--target_grid_x", type=int, default=None)
    parser.add_argument("--target_grid_t", type=int, default=None)
    parser.add_argument("--test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_predictions", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = os.path.dirname(os.path.abspath(args.config_res_inv))
    eval_config = load_json(args.config_res_inv)

    checkpoint_path = args.checkpoint_path or eval_config["checkpoint_path"]
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(os.path.join(config_dir, checkpoint_path))

    target_grid_x = args.target_grid_x if args.target_grid_x is not None else int(eval_config["target_grid_x"])
    target_grid_t = args.target_grid_t if args.target_grid_t is not None else int(eval_config["target_grid_t"])
    test_samples = args.test_samples if args.test_samples is not None else int(eval_config.get("N_test", 3000))

    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = int(eval_config.get("seed", 0))

    training_config_path = resolve_training_config_path(checkpoint_path)
    training_config = load_json(training_config_path)
    model_name = training_config["experiment_config"]["model"]
    training_grid_x = int(training_config["data_config"]["grid_x"])
    training_grid_t = int(training_config["data_config"]["grid_t"])
    T = int(training_config["data_config"]["T"])
    discretization = training_config["experiment_config"].get("discretization", "IMEX")
    input_channels = int(training_config["data_config"]["input_channels"])
    trained_n_train = int(training_config["experiment_config"].get("N_train", 27000))

    infer_resolution_safe(model_name, training_grid_x, target_grid_x)

    run_dir = get_parent_run_dir(checkpoint_path)
    output_dir = args.output_dir or os.path.join(run_dir, "invresolution_eval")
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(SCRIPT_DIR, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    g_1_train, g_1_test, g_2_train, g_2_test, g_3_train, g_3_test = generate_t(
        target_grid_t, T, seed, trained_n_train, test_samples
    )

    X_test, y_test = [], []
    for i in range(test_samples // 4):
        x, y = generate_data(target_grid_x, target_grid_t, T, g_1_test[i], g_2_test[i], g_3_test[i])
        X_test.append(x)
        y_test.append(y)

    X_test = np.concatenate(X_test, 0)
    y_test = np.concatenate(y_test, 0)

    X_test = torch.as_tensor(np.asarray(X_test), dtype=torch.float32, device=device)
    y_test = torch.as_tensor(np.asarray(y_test), dtype=torch.float32, device=device)

    model = build_model(model_name, input_channels, training_grid_x, discretization).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    outputs_all = []
    labels_all = []
    with torch.no_grad():
        for data, targets in test_loader:
            outputs = model(data)
            outputs_all.append(outputs.detach().cpu())
            labels_all.append(targets.detach().cpu())

    outputs_all = torch.cat(outputs_all, 0)
    labels_all = torch.cat(labels_all, 0)
    relative_l2 = compute_relative_l2(outputs_all, labels_all).item()

    results = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "training_config_path": training_config_path,
        "model_name": model_name,
        "training_resolution": {
            "grid_x": training_grid_x,
            "grid_t": training_grid_t,
            "T": T,
        },
        "target_resolution": {
            "grid_x": target_grid_x,
            "grid_t": target_grid_t,
            "T": T,
        },
        "seed": seed,
        "test_samples": test_samples,
        "device": str(device),
        "relative_l2": relative_l2,
        "checkpoint_metadata": {
            "epoch": checkpoint.get("epoch"),
            "train_loss": checkpoint.get("train_loss"),
            "test_error": checkpoint.get("test_error"),
        },
        "model_details": get_model_details(model, model_name),
    }

    save_json(os.path.join(output_dir, "invresolution_results.json"), results)
    save_json(os.path.join(output_dir, "config_res_inv_resolved.json"), {
        **eval_config,
        "checkpoint_path": checkpoint_path,
        "target_grid_x": target_grid_x,
        "target_grid_t": target_grid_t,
        "resolved_training_config_path": training_config_path,
        "resolved_output_dir": output_dir,
    })

    if args.save_predictions:
        torch.save(
            {
                "predictions": outputs_all,
                "targets": labels_all,
                "results": results,
            },
            os.path.join(output_dir, "predictions.pt"),
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()