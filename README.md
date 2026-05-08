# PDON

Visual summary (paper-style overview):

- **Main document**: [`sources/Web/main.pdf`](sources/Web/main.pdf)

[![Main overview](sources/Web/main.png)](sources/Web/main.pdf)

## Quick visuals

![2D demo](sources/Web/2d_2.gif)

![3D demo](sources/Web/3d.gif)

## Repository layout (at a glance)

- **`Beltrami/`**: Beltrami-flow experiments. Main entrypoint: `Beltrami/main_beltrami.py`
- **`Brusselator/`**: 3D Brusselator experiments. Main entrypoint: `Brusselator/main_Brusselator_3d.py`
- **`Burger_RD_FkdV/`**: 2D Burgers + related settings. Main entrypoints: `Burger_RD_FkdV/main_burgers.py`, `Burger_RD_FkdV/main_fkdv.py`
- **`RD2D/`**: 2D reaction–diffusion experiments. Main entrypoint: `RD2D/main_reaction_diffusion.py`
- **`sources/Web/`**: pre-rendered visuals used in this README.

## Run experiments (minimal)

One-line runs (examples):

```bash
python Beltrami/main_beltrami.py --model OSS
python Brusselator/main_Brusselator_3d.py --model OSS
python Burger_RD_FkdV/main_burgers.py --model OSS
python RD2D/main_reaction_diffusion.py --model OSS
```

## Key arguments (minimal)

- **`--model`**: selects the temporal backbone. Common options across scripts include `OSS`, `Mamba`, `GRU`, `LSTM` (some scripts also accept `MambaScratch` / `mamba_scratch`).
- **`--num_epochs`**: training epochs.
- **`--batch_size`**: minibatch size.
- **`--lr`**: learning rate.

## Results

- **OSS oscillation figure**: [`sources/Web/oss_oscil.pdf`](sources/Web/oss_oscil.pdf)

[![Results](sources/Web/results.png)](sources/Web/results.png)