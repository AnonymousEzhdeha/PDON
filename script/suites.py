"""Per-experiment adapters: rebuild a trained model and draw fresh test data.

Each adapter mirrors exactly what its training script does, because the metric is
only comparable if the architecture, the tensor layout and the relative-L2 form all
match. Where a training script defines the data generators inline, they are
recovered with `common.load_defs` (AST filtering) rather than copied, so this stage
stays in step with the original source and edits nothing.

Two reconstruction hazards are handled explicitly:
  * `modes`/`width` are hardcoded in the training scripts and absent from
    config.json, so they are supplied here per suite (32 for burgers/fkdv, 6 for
    RD-ft/ftx).
  * `dt_min`/`dt_max` and `discretization` are plain Python floats/strings, not
    buffers, so they are absent from the state dict and must be passed correctly or
    inference silently uses the wrong step size.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import common
from common import REPO_ROOT, add_suite_to_path, evaluate, load_defs, rel_l2

TestData = Dict[str, torch.Tensor]


class Suite:
    """Base adapter. Subclasses supply model rebuild + fresh-data generation."""

    name = "?"
    directory = "?"
    script = "?"
    run_subdirs: Tuple[str, ...] = ("run",)
    default_n_test = 100

    @property
    def suite_dir(self) -> str:
        return os.path.join(REPO_ROOT, self.directory)

    @property
    def script_path(self) -> str:
        return os.path.join(self.suite_dir, self.script)

    def available(self) -> Tuple[bool, str]:
        if not os.path.isfile(self.script_path):
            return False, f"missing {self.directory}/{self.script}"
        return True, ""

    def args_of(self, run: "common.Run") -> SimpleNamespace:
        return SimpleNamespace(**run.args)

    def batch_size(self, run: "common.Run") -> int:
        return int(run.args.get("batch_size", 16) or 16)

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        raise NotImplementedError

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        raise NotImplementedError

    def evaluate(self, model: torch.nn.Module, data: TestData, batch_size: int, device: str) -> float:
        return evaluate(model, data["X"], data["Y"], batch_size, device)


# ---------------------------------------------------------------------------
# Burgers and fKdV - flat FNO/Jamba models, in_dim 4 and 7
# ---------------------------------------------------------------------------
class _FlatSuite(Suite):
    """Shared: both use the same model factory, differing only in in_dim/grid."""

    in_dim = 4
    directory = "Burger_RD_FkdV"

    def _factory(self):
        add_suite_to_path(self.suite_dir)
        # main_burger_invresolution.py already carries an args-free build_model plus
        # de-args'd copies of the burgers generators; reuse rather than duplicate.
        helper = os.path.join(self.suite_dir, "main_burger_invresolution.py")
        return load_defs(helper, ["build_model"])["build_model"]

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        a = self.args_of(run)
        grid_x = int(getattr(a, "grid_x", 100))
        disc = str(getattr(a, "discretization", "IMEX"))
        model = self._factory()(run.model, self.in_dim, grid_x, disc)
        return model.to(device)


class Burgers(_FlatSuite):
    name = "burgers"
    script = "main_burgers.py"
    in_dim = 4
    default_n_test = 300  # a full 3000-sample regeneration per test seed is too slow

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        add_suite_to_path(self.suite_dir)
        helper = os.path.join(self.suite_dir, "main_burger_invresolution.py")
        defs = load_defs(helper, ["generate_t", "generate_data"])
        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 100)), int(getattr(a, "grid_t", 100))
        T = float(getattr(a, "T", 1))
        n_test = max(4, (n_test // 4) * 4)
        # Same generator and settings as training; a different seed, so different draws.
        g1_tr, g1_te, g2_tr, g2_te, g3_tr, g3_te = defs["generate_t"](grid_t, T, seed, n_test, n_test)
        X, Y = [], []
        for i in range(n_test // 4):
            x, y = defs["generate_data"](grid_x, grid_t, T, g1_te[i], g2_te[i], g3_te[i])
            X.append(x)
            Y.append(y)
        X = torch.as_tensor(np.concatenate(X, 0), dtype=torch.float32)
        Y = torch.as_tensor(np.concatenate(Y, 0), dtype=torch.float32)
        return {"X": X, "Y": Y}


class FKdV(_FlatSuite):
    name = "fkdv"
    script = "main_fkdv.py"
    in_dim = 7
    run_subdirs = ("run_fkdv", "run")
    default_n_test = 300

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["generate_data_1", "generate_data_2", "generate_data_3"])
        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 100)), int(getattr(a, "grid_t", 100))
        L, T = float(getattr(a, "L", 5)), float(getattr(a, "T", 5))
        np.random.seed(seed)
        X, Y = [], []
        # The three analytic families are drawn in the same 1:1:1 proportion as training.
        for _ in range(max(1, n_test // 3)):
            for gen in ("generate_data_1", "generate_data_2", "generate_data_3"):
                x, y = defs[gen](grid_x, grid_t, L, T)  # calls autograd.grad - keep grad enabled
                X.append(x)
                Y.append(y)
        return {"X": torch.stack(X, 0).float(), "Y": torch.stack(Y, 0).float()}


# ---------------------------------------------------------------------------
# Reaction-diffusion, 1+1D - modes=6 and argparse-driven depth/dt
# ---------------------------------------------------------------------------
class _RD(Suite):
    directory = "Burger_RD_FkdV"
    in_dim = 3
    default_n_test = 50  # the training N_test; generation is cheap

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        add_suite_to_path(self.suite_dir)
        from DON_2d import MambaConfig, POD_Mamba, POD_Mamba_Scratch  # noqa: F401
        from FNO2d_Jamba import FNO_Jamba_1

        a = self.args_of(run)
        grid_x = int(getattr(a, "grid_x", 20))
        if run.model == "OSS":
            model = FNO_Jamba_1(
                input_dim=self.in_dim,
                output_dim=1,
                modes=6,  # hardcoded in the training script, absent from config.json
                width=128,
                num_layers=int(getattr(a, "oss_num_layers", 2)),
                model_t_type="OSS",
                discretization=str(getattr(a, "discretization", "IMEX")),
                oss_dt=float(getattr(a, "oss_dt", 1.0)),
                oss_dt_min=float(getattr(a, "oss_dt_min", 1e-3)),
                oss_dt_max=float(getattr(a, "oss_dt_max", 1.0)),
            )
        elif run.model == "Mamba":
            d_model = int(getattr(a, "MambaLLM_d_model", 256))
            n_layer = int(getattr(a, "MambaLLM_n_layer", 1))
            # rms_norm/fused_add_norm are NOT forwarded by the training script; keeping
            # POD_Mamba's own defaults is what makes the state dict line up.
            model = POD_Mamba(d_model, n_layer, 0, self.in_dim * grid_x, grid_x, dict(layer="Mamba1"))
        elif run.model in ("MambaScratch", "mamba_scratch"):
            model = POD_Mamba_Scratch(
                input_dim=self.in_dim * grid_x,
                output_dim=grid_x,
                hidden_dim=int(getattr(a, "MambaLLM_d_model", 256)),
                num_layers=int(getattr(a, "MambaLLM_n_layer", 1)),
                d_state=16,
            )
        else:
            raise ValueError(f"{self.name}: unsupported model {run.model!r}")
        return model.to(device)

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["data_gen", "aug_f"])
        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 20)), int(getattr(a, "grid_t", 200))
        T = float(getattr(a, "T", 10))
        np.random.seed(seed)
        f, y = defs["data_gen"](n_test, grid_x, grid_t, T)
        f = defs["aug_f"](f, f.shape[0], grid_x, grid_t)
        return {"X": torch.from_numpy(f).float(), "Y": torch.from_numpy(y).float()}


class RDft(_RD):
    name = "rd_ft"
    script = "main_RD_ft.py"
    run_subdirs = ("run/run_rd_ft", "run")


class RDftx(_RD):
    name = "rd_ftx"
    script = "main_RD_ftx.py"
    run_subdirs = ("run/run_rd_ftx", "run")


# ---------------------------------------------------------------------------
# Fourier reaction-diffusion, 2+1D - model works in Fourier space, metric in grid
# ---------------------------------------------------------------------------
class RD2D(Suite):
    name = "rd2d"
    directory = "RD2D"
    script = "main_reaction_diffusion.py"
    default_n_test = 200

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["get_model"])
        a = self.args_of(run)
        grid_x = int(getattr(a, "grid_x", 64))
        io_dim = 2 * (grid_x // 2 + 1) * grid_x
        return defs["get_model"](a, io_dim, device)

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["generate_data", "generate_g_x", "generate_h_x", "Grid2Fourier"])
        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 64)), int(getattr(a, "grid_t", 100))
        T, K = float(getattr(a, "T", 1)), int(getattr(a, "K", 4))
        np.random.seed(seed)
        u, f = defs["generate_data"](n_test, grid_x, grid_t, T, K)
        X = torch.from_numpy(f).float()
        Y = torch.from_numpy(u).float()
        X_f = defs["Grid2Fourier"](X, grid_x)
        Y_f = defs["Grid2Fourier"](Y, grid_x)
        Y_grid = Y.squeeze().permute(0, 3, 1, 2)
        return {"X": X_f, "Y": Y_f, "Y_grid": Y_grid}

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, data: TestData, batch_size: int, device: str) -> float:
        """Selection in training used the GRID-space error, so reproduce that."""
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["Fourier2Grid"])
        grid_x = data["Y_grid"].shape[-1]
        model.eval()
        preds, labels = [], []
        X, Yg = data["X"], data["Y_grid"]
        for i in range(0, X.shape[0], batch_size):
            out = model(X[i : i + batch_size].to(device))
            preds.append(defs["Fourier2Grid"](out, grid_x).detach().cpu())
            labels.append(Yg[i : i + batch_size].detach().cpu())
        return rel_l2(torch.cat(preds, 0), torch.cat(labels, 0))


# ---------------------------------------------------------------------------
# POD suites 
# ---------------------------------------------------------------------------
class Brusselator(Suite):
    name = "brusselator"
    directory = "Brusselator"
    script = "main_Brusselator_3d.py"
    default_n_test = 0  # 0 -> use the whole held-out split from disk

    def available(self) -> Tuple[bool, str]:
        ok, msg = super().available()
        if not ok:
            return ok, msg
        if not os.path.isdir(os.path.join(self.suite_dir, "Data")):
            return False, "Brusselator/Data/ is absent - generate it with generate_data_Brusselator.py first"
        return True, ""

    def _load_split(self, run: "common.Run"):
        from sklearn.model_selection import train_test_split

        a = self.args_of(run)
        grid_t = int(getattr(a, "grid_t", 201))
        data_dir = os.path.join(self.suite_dir, "Data")
        f_all, u_all = [], []
        for path in sorted(os.listdir(data_dir)):  # sorted: os.listdir order is not reproducible
            d = np.load(os.path.join(data_dir, path))
            f_all.append(d["F"][:, :grid_t])
            u_all.append(d["U"][:, :grid_t])
        f = np.concatenate(f_all, 0)
        u = np.concatenate(u_all, 0)
        f_tr, f_te, u_tr, u_te = train_test_split(f, u, test_size=0.1, random_state=42)
        reshape = lambda arr: arr.reshape(arr.shape[0], arr.shape[1], -1)  # noqa: E731
        return reshape(f_tr), reshape(f_te), reshape(u_tr), reshape(u_te)

    def _basis(self, run: "common.Run"):
        from sklearn.decomposition import PCA

        a = self.args_of(run)
        n_dec = int(getattr(a, "n_components_decode", 256))
        key = {"seed": run.seed, "ndec": n_dec, "gridt": int(getattr(a, "grid_t", 201))}
        cached = common.pod_cache_get(self.name, key)
        if cached is not None:
            return cached["POD_Basis"], cached["POD_Mean"]
        _, _, u_tr, _ = self._load_split(run)
        u_tr = u_tr.reshape(u_tr.shape[0] * u_tr.shape[1], -1)
        pca = PCA(n_components=n_dec).fit(u_tr)
        const = np.sqrt(u_tr.shape[-1])
        basis = (pca.components_.T * const).astype(np.float32)  # the sqrt(D) scale is load-bearing
        mean = pca.mean_.astype(np.float32)
        common.pod_cache_put(self.name, key, {"POD_Basis": basis, "POD_Mean": mean})
        return basis, mean

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        add_suite_to_path(self.suite_dir)
        from PODDON_TGV import PODDON_Mamba, PODDON_Mamba_Scratch, PODDON_OSS_NO

        a = self.args_of(run)
        basis, mean = self._basis(run)
        POD_Basis = torch.from_numpy(basis).float().to(device)
        POD_Mean = torch.from_numpy(mean).float().to(device)
        In_Basis = torch.from_numpy(np.ones((1, 1), dtype=np.float32)).to(device)
        In_Mean = torch.from_numpy(np.zeros((1,), dtype=np.float32)).to(device)
        n_dec = int(getattr(a, "n_components_decode", 256))
        if run.model == "OSS":
            model = PODDON_OSS_NO(
                input_dim=1, output_dim=n_dec, hidden_dim=int(getattr(a, "oss_hidden_dim", 256)),
                POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean,
                num_layers=int(getattr(a, "oss_num_layers", 1)),
                discretization=str(getattr(a, "oss_discretization", "IMEX")),
                dt=float(getattr(a, "oss_dt", 1.0)),
                dt_min=float(getattr(a, "oss_dt_min", 1e-3)),
                dt_max=float(getattr(a, "oss_dt_max", 1.0)),
                use_layernorm=bool(getattr(a, "oss_use_layernorm", 1)),
                residual_weight=float(getattr(a, "oss_residual_weight", 0.0)),
                proj_dropout=float(getattr(a, "oss_proj_dropout", 0.0)),
                robust_dt_init=bool(getattr(a, "oss_robust_dt_init", 1)),
                use_input_dt=bool(getattr(a, "oss_use_input_dt", 0)),
                use_d_skip=bool(getattr(a, "oss_use_d_skip", 0)),
                d_skip_init=float(getattr(a, "oss_d_skip_init", 1.0)),
                use_input_drive_damping=bool(getattr(a, "oss_use_input_drive_damping", 0)),
                input_drive_scale=float(getattr(a, "oss_input_drive_scale", 1.0)),
                input_damping_scale=float(getattr(a, "oss_input_damping_scale", 1.0)),
                use_osc_gate=bool(getattr(a, "oss_use_osc_gate", 0)),
                use_causal_prefilter=bool(getattr(a, "oss_use_causal_prefilter", 0)),
                prefilter_kernel_size=int(getattr(a, "oss_prefilter_kernel_size", 3)),
            )
        elif run.model == "Mamba":
            model = PODDON_Mamba(256, 2, 0, 1, n_dec, POD_Basis, POD_Mean, In_Basis, In_Mean, dict(layer="Mamba1"))
        elif run.model in ("MambaScratch", "mamba_scratch"):
            model = PODDON_Mamba_Scratch(
                input_dim=1, output_dim=n_dec, hidden_dim=256,
                POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean,
                n_layers=1, d_state=16,
            )
        else:
            raise ValueError(f"{self.name}: unsupported model {run.model!r}")
        return model.to(device)

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        """Brusselator's data is a fixed file set, so there is no fresh draw to make.

        The best we can do without fabricating physics is a bootstrap resample of the
        held-out split, which varies the evaluation set without reusing training data.
        """
        _, f_te, _, u_te = self._load_split(run)
        rng = np.random.RandomState(seed)
        n = f_te.shape[0] if not n_test else min(n_test, f_te.shape[0])
        idx = rng.choice(f_te.shape[0], size=n, replace=n > f_te.shape[0])
        return {
            "X": torch.from_numpy(f_te[idx]).float(),
            "Y": torch.from_numpy(u_te[idx]).float(),
        }


class Beltrami(Suite):
    name = "beltrami"
    directory = "Beltrami"
    script = "main_beltrami.py"
    default_n_test = 100

    def _basis(self, run: "common.Run"):
        """Refit both PCAs from this seed's training data (nothing is persisted)."""
        from sklearn.decomposition import PCA

        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 17)), int(getattr(a, "grid_t", 100))
        T = float(getattr(a, "T", 1))
        n_train = int(getattr(a, "N_train", 900))
        n_enc = int(getattr(a, "n_components_encode", 128))
        n_dec = int(getattr(a, "n_components_decode", 128))
        key = {"seed": run.seed, "gx": grid_x, "gt": grid_t, "ntr": n_train, "enc": n_enc, "dec": n_dec}
        cached = common.pod_cache_get(self.name, key)
        if cached is not None:
            return cached

        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["generate_data"])
        np.random.seed(run.seed)  # same stream as training, so the same basis
        u_chunks, f_chunks = [], []
        for _ in range(n_train // 10):
            u_, f_ = defs["generate_data"](10, grid_x, grid_t, T)
            u_chunks.append(np.asarray(u_, dtype=np.float32))
            f_chunks.append(np.asarray(f_, dtype=np.float32))
        u_tr = np.concatenate(u_chunks, 0).reshape(-1, u_chunks[0].shape[-1])
        f_tr = np.concatenate(f_chunks, 0).reshape(-1, f_chunks[0].shape[-1])
        del u_chunks, f_chunks

        pca_u = PCA(n_components=n_dec).fit(u_tr)
        arrays = {
            "POD_Basis": (pca_u.components_.T * np.sqrt(u_tr.shape[-1])).astype(np.float32),
            "POD_Mean": pca_u.mean_.astype(np.float32),
        }
        del u_tr, pca_u
        pca_f = PCA(n_components=n_enc).fit(f_tr)
        arrays["In_Basis"] = (pca_f.components_.T * np.sqrt(f_tr.shape[-1])).astype(np.float32)
        arrays["In_Mean"] = pca_f.mean_.astype(np.float32)
        del f_tr, pca_f

        common.pod_cache_put(self.name, key, arrays)
        return arrays

    def build_model(self, run: "common.Run", device: str) -> torch.nn.Module:
        add_suite_to_path(self.suite_dir)
        from PODDON_TGV import PODDON_Mamba, PODDON_Mamba_Scratch, PODDON_OSS_NO

        a = self.args_of(run)
        b = self._basis(run)
        to_t = lambda arr: torch.from_numpy(arr).float().to(device)  # noqa: E731
        POD_Basis, POD_Mean = to_t(b["POD_Basis"]), to_t(b["POD_Mean"])
        In_Basis, In_Mean = to_t(b["In_Basis"]), to_t(b["In_Mean"])
        n_enc = int(getattr(a, "n_components_encode", 128))
        n_dec = int(getattr(a, "n_components_decode", 128))
        if run.model == "OSS":
            model = PODDON_OSS_NO(
                input_dim=n_enc, output_dim=n_dec, hidden_dim=int(getattr(a, "oss_hidden_dim", 256)),
                POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean,
                num_layers=int(getattr(a, "oss_num_layers", 1)),
                discretization=str(getattr(a, "oss_discretization", "IMEX")),
                dt=float(getattr(a, "oss_dt", 1.0)),
                dt_min=float(getattr(a, "oss_dt_min", 1e-3)),
                dt_max=float(getattr(a, "oss_dt_max", 1.0)),
                use_layernorm=bool(getattr(a, "oss_use_layernorm", 1)),
                residual_weight=float(getattr(a, "oss_residual_weight", 0.0)),
                proj_dropout=float(getattr(a, "oss_proj_dropout", 0.0)),
                robust_dt_init=bool(getattr(a, "oss_robust_dt_init", 0)),
                use_input_dt=bool(getattr(a, "oss_use_input_dt", 0)),
                use_d_skip=bool(getattr(a, "oss_use_d_skip", 0)),
                d_skip_init=float(getattr(a, "oss_d_skip_init", 1.0)),
                use_input_drive_damping=bool(getattr(a, "oss_use_input_drive_damping", 0)),
                input_drive_scale=float(getattr(a, "oss_input_drive_scale", 1.0)),
                input_damping_scale=float(getattr(a, "oss_input_damping_scale", 1.0)),
                use_osc_gate=bool(getattr(a, "oss_use_osc_gate", 0)),
                use_causal_prefilter=bool(getattr(a, "oss_use_causal_prefilter", 0)),
                prefilter_kernel_size=int(getattr(a, "oss_prefilter_kernel_size", 3)),
                use_expand_project=bool(getattr(a, "oss_use_expand_project", 0)),
                expand_factor=int(getattr(a, "oss_expand_factor", 2)),
                expand_init_scale=float(getattr(a, "oss_expand_init_scale", 0.02)),
                use_coupled_oscillators=bool(getattr(a, "oss_use_coupled_oscillators", 0)),
                coupling_rank=int(getattr(a, "oss_coupling_rank", 4)),
                coupling_scale=float(getattr(a, "oss_coupling_scale", 0.05)),
            )
        elif run.model == "Mamba":
            model = PODDON_Mamba(256, 1, 0, n_enc, n_dec, POD_Basis, POD_Mean, In_Basis, In_Mean, dict(layer="Mamba1"))
        elif run.model in ("MambaScratch", "mamba_scratch"):
            model = PODDON_Mamba_Scratch(
                input_dim=n_enc, output_dim=n_dec, hidden_dim=256,
                POD_Basis=POD_Basis, POD_Mean=POD_Mean, In_Basis=In_Basis, In_Mean=In_Mean,
                n_layers=1, d_state=16,
            )
        else:
            raise ValueError(f"{self.name}: unsupported model {run.model!r}")
        return model.to(device)

    def make_test_data(self, run: "common.Run", seed: int, n_test: int) -> TestData:
        add_suite_to_path(self.suite_dir)
        defs = load_defs(self.script_path, ["generate_data"])
        a = self.args_of(run)
        grid_x, grid_t = int(getattr(a, "grid_x", 17)), int(getattr(a, "grid_t", 100))
        T = float(getattr(a, "T", 1))
        n_test = max(10, (n_test // 10) * 10)
        np.random.seed(seed)  # fresh draws from the same generator
        u_chunks, f_chunks = [], []
        for _ in range(n_test // 10):
            u_, f_ = defs["generate_data"](10, grid_x, grid_t, T)
            u_chunks.append(u_)
            f_chunks.append(f_)
        U = np.concatenate(u_chunks, 0)
        F = np.concatenate(f_chunks, 0)
        return {"X": torch.from_numpy(F).float(), "Y": torch.from_numpy(U).float()}


REGISTRY: Dict[str, Suite] = {
    s.name: s
    for s in (Burgers(), FKdV(), RDft(), RDftx(), RD2D(), Brusselator(), Beltrami())
}
ORDER = ["burgers", "fkdv", "rd_ft", "rd_ftx", "rd2d", "brusselator", "beltrami"]
