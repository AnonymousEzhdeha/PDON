import torch

from time_model import OSS_source
from FNO2d_Jamba import FNO_Jamba_1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Temporal boundary shape used by FNO_Jamba temporal modules:
    # (batch * N_x, N_t, width)
    bnx, nt, width = 8, 32, 64
    x_temporal = torch.randn(bnx, nt, width, device=device, requires_grad=True)

    temporal = OSS_source(
        d_model=width,
        input_dim=width,
        output_dim=width,
        discretization="IM",
        seed=0,
    ).to(device)
    temporal_opt = torch.optim.Adam(temporal.parameters(), lr=1e-3)

    temporal_opt.zero_grad()
    y_temporal = temporal(x_temporal)
    print("OSS_source output shape:", tuple(y_temporal.shape))

    loss_temporal = y_temporal.pow(2).mean()
    loss_temporal.backward()
    print("OSS_source backward ok. input grad shape:", tuple(x_temporal.grad.shape))

    # Verify source-backed LinOSS parameters receive gradients and update.
    grad_ok = all(
        p.grad is not None
        for n, p in temporal.named_parameters()
        if n.startswith("linoss_")
    )
    print("OSS_source LinOSS param grads present:", grad_ok)

    before = temporal.linoss_A_diag.detach().clone()
    temporal_opt.step()
    after = temporal.linoss_A_diag.detach()
    updated = not torch.allclose(before, after)
    print("OSS_source LinOSS params updated:", updated)

    if not grad_ok:
        raise RuntimeError("LinOSS source parameters did not receive gradients.")
    if not updated:
        raise RuntimeError("LinOSS source parameters were not updated by optimizer step.")

    # End-to-end FNO_Jamba path with source-backed temporal model.
    model = FNO_Jamba_1(
        input_dim=4,
        output_dim=1,
        modes=8,
        width=32,
        num_layers=1,
        model_t_type="OSS_source",
    ).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(2, 16, 24, 4, device=device, requires_grad=True)
    model_opt.zero_grad()
    y = model(x)
    print("FNO_Jamba_1(OSS_source) output shape:", tuple(y.shape))

    linoss_param_name = "blocks.0.model_t.linoss_A_diag"
    named_params = dict(model.named_parameters())
    before_model = named_params[linoss_param_name].detach().clone()

    loss = y.pow(2).mean()
    loss.backward()
    print("FNO_Jamba_1(OSS_source) backward ok. input grad shape:", tuple(x.grad.shape))

    linoss_grad_ok = named_params[linoss_param_name].grad is not None
    model_opt.step()
    after_model = named_params[linoss_param_name].detach()
    linoss_updated = not torch.allclose(before_model, after_model)
    print("FNO_Jamba_1 LinOSS param grad present:", linoss_grad_ok)
    print("FNO_Jamba_1 LinOSS params updated:", linoss_updated)

    if not linoss_grad_ok:
        raise RuntimeError("FNO_Jamba_1 LinOSS source parameter did not receive gradients.")
    if not linoss_updated:
        raise RuntimeError("FNO_Jamba_1 LinOSS source parameter was not updated by optimizer step.")


if __name__ == "__main__":
    main()
