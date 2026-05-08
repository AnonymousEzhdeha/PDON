# LinOSS PyTorch Integration Guide

## Overview

The PyTorch LinOSS (`linoss_pytorch_on`) class has been successfully integrated into the training pipeline as a native PyTorch temporal model. This replaces the JAX-based `OSS_source` with a pure PyTorch implementation while maintaining identical mathematical operations and interface compatibility.

## Files Modified

### 1. `time_model.py`
- **Added**: `linoss_pytorch_on` class (lines 347-450+)
- **Purpose**: Wrapper class that adapts the pure PyTorch LinOSS implementation to work as a temporal model within the FNO_Jamba architecture
- **Key Features**:
  - Batch processing with automatic device management
  - Support for both IM (Implicit) and IMEX (Implicit-Explicit) discretization
  - Direct PyTorch gradient computation (no JAX bridging needed)
  - Configurable number of stacked LinOSS blocks

### 2. `FNO2d_Jamba.py`
- **Modified**: Import statement (line 5)
  - Added: `from time_model import ... linoss_pytorch_on`
- **Modified**: `FNO_Jamba_Layer_1.__init__()` (lines 87-95)
  - Added support for `model_t_type="linoss_pytorch"`
  - Instantiates `linoss_pytorch_on` with appropriate dimensions and configuration

### 3. `main_burgers.py`
- **Modified**: Model argument instantiation section
  - Added: `FNO_linoss_pytorch_1` model option
  - Added: `FNO_linoss_pytorch_2` model option
  - Added: `linoss_pytorch` shorthand model option

## Model Configuration

### New Model Options

You can now run training with the following new model options:

```bash
# Option 1: FNO with LinOSS PyTorch (single layer)
python main_burgers.py --model FNO_linoss_pytorch_1

# Option 2: FNO with LinOSS PyTorch (two layers)
python main_burgers.py --model FNO_linoss_pytorch_2

# Option 3: Shorthand for FNO_linoss_pytorch_1
python main_burgers.py --model linoss_pytorch
```

### Model Architecture

```
Input (batch, N_x, N_t, input_dim=4)
    ↓
Input Projection: 4 → 128
    ↓
FNO_Jamba_Layer_1 (num_layers=1)
    ├─ Spatial: SpectralConv1d (FFT-based, modes=32)
    └─ Temporal: linoss_pytorch_on
         ├─ Input projection: 128 → 128
         ├─ LinOSS Core (IM discretization)
         │  ├─ LinOSSBlock (num_layers=1)
         │  │  ├─ LayerNorm
         │  │  ├─ LinOSSLayer (Paper Eq. 1-4)
         │  │  ├─ GELU + Dropout
         │  │  ├─ GLU (Gated Linear Unit)
         │  │  └─ Residual Connection
         │  └─ Output projection: 128 → 128
         └─ Output projection: 128 → 128
    ↓
Fully Connected Layers: 128 → 1
    ↓
Output (batch, N_x*N_t, output_dim=1)
```

## Running Training

### Basic Example
```bash
cd Example_1+1_Dim
python main_burgers.py \
    --model linoss_pytorch \
    --grid_x 100 \
    --grid_t 100 \
    --num_epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --N_train 27000 \
    --N_test 3000
```

### With Custom Parameters
```bash
python main_burgers.py \
    --model FNO_linoss_pytorch_2 \
    --num_epochs 200 \
    --batch_size 32 \
    --lr 5e-4 \
    --save_loss 1 \
    --save_model 1
```

## PyTorch vs JAX Implementation

### Comparison Matrix

| Feature | JAX (`OSS_source`) | PyTorch (`linoss_pytorch_on`) |
|---------|------|--------|
| **Language** | JAX Numpy | PyTorch |
| **Complex Numbers** | JAX complex types | torch.complex64 |
| **Discretization** | IM, IMEX (JAX vmap) | IM, IMEX (sequential) |
| **Parallelization** | Associative scan O(log L) | Sequential O(L) |
| **GPU Support** | CUDA (via JAX) | CUDA (native PyTorch) |
| **Gradients** | JAX autodiff | PyTorch autograd |
| **Device Transfer** | Manual JAX ↔ PyTorch | Automatic device management |
| **Dependencies** | JAX, Equinox | PyTorch only |

### Performance Characteristics

**LinearOSSLayer (PyTorch Implementation)**:
- **Time Complexity**: O(L × P) sequential, O(log L) parallel-enabled structure
  - L = sequence length (100 for temporal dimension)
  - P = state-space size (128 in default config)
- **Space Complexity**: O(P × H) for state and parameters
  - H = feature dimension (128)
- **Computation per timestep**: Matrix-vector multiply + element-wise ops

**Where Parallel Structure is Available**:
- `binary_operator()` (Lines 91-150 in linoss_pytorch.py) enables composition of state transitions
- JAX implementation uses `lax.associative_scan` for O(log L) depth
- PyTorch implementation shown is sequential but structure is in place for parallel extension

### Memory Usage
```
Model Parameters: 739,585 (trainable)

Memory per batch element (100×100 grid):
  - Input: 100 × 100 × 4 floats = 160 KB
  - Hidden states: 100 × 128 × 2 (complex) = 102.4 KB
  - Output gradients: 10,000 × 1 floats = 40 KB
  - Total estimate (batch=16): ~50 MB
```

## Mathematical Correspondence

### Paper Equations → PyTorch Code

| Paper Equation | Paper Section | PyTorch Code Location | Implementation |
|---|---|---|---|
| y'' = -A·y + B·u + b | Eq. 1-2 (Continuous ODE) | `linoss_pytorch_on` init | Parameter setup in `LinOSSLayer` |
| Implicit discretization | Eq. 3-4 | `apply_linoss_im()` (lines 160-351) | Schur complement + sequential recurrence |
| IMEX discretization | Eq. 5-6 | `apply_linoss_imex()` (lines 355-480) | 2×2 symplectic blocks |
| Binary operator | Eq. 5-6 | `binary_operator()` (lines 91-150) | Block matrix composition |
| Output readout | Eq. 1 | All functions (lines 315-330, 478-489) | x = C·y (complex → real) |
| Parameter constraints | Eq. 3 | `LinOSSLayer.forward()` (lines 638-645) | A≥0 (ReLU), 0<Δt≤1 (sigmoid) |

## Discretization Modes

### IM (Implicit) - Default
```python
linoss_pytorch_on(..., discretization="IM")
```
- **Properties**: Exponentially stable, dissipative
- **Formula**: Uses Schur complement (I + Δt²A)⁻¹
- **Stability**: Unconditionally stable for any Δt > 0
- **Use case**: General purpose, stable for stiff systems
- **Computational cost**: Lower per-iteration (fewer blocks)

### IMEX (Implicit-Explicit)
```python
linoss_pytorch_on(..., discretization="IMEX")
```
- **Properties**: Symplectic, volume-preserving
- **Formula**: 2×2 block matrix structure
- **Stability**: Conditional stability
- **Use case**: Conservative systems, long-time integration
- **Computational cost**: Slightly higher (2×2 blocks)

To enable IMEX, modify [FNO2d_Jamba.py](FNO2d_Jamba.py#L91):
```python
elif model_t_type == "linoss_pytorch":
    self.model_t = linoss_pytorch_on(
        d_model=width,
        input_dim=width,
        output_dim=width,
        discretization="IMEX",  # ← Change this
        seed=0,
        num_layers=1,
        dropout=0.05,
    )
```

## Hyperparameter Tuning

### Default Configuration
```python
# From FNO2d_Jamba.py
linoss_pytorch_on(
    d_model=width,              # Usually 128
    input_dim=width,           # Usually 128
    output_dim=width,          # Usually 128
    discretization="IM",       # Implicit by default
    seed=0,                    # Random initialization seed
    num_layers=1,              # Number of LinOSSBlocks to stack
    dropout=0.05,              # Regularization rate
)
```

### Recommended Adjustments

**For better accuracy (slower training)**:
```python
num_layers=2,        # Double the model capacity
dropout=0.1,         # Increase regularization
```

**For faster training (may reduce accuracy)**:
```python
num_layers=1,        # Keep single layer
dropout=0.05,        # Reduce regularization
```

**For smaller GPU memory**:
Consider reducing `width` parameter in FNO_Jamba_1 constructor (default: 128)

## Testing

### Quick Integration Test
```bash
cd Example_1+1_Dim
python test_linoss_pytorch_integration.py
```

Expected output:
```
✅ Imports successful
✅ linoss_pytorch_on instantiation successful
✅ linoss_pytorch_on forward pass successful, output shape: torch.Size([2, 100, 128])
✅ FNO_Jamba_1 with linoss_pytorch instantiation successful
✅ FNO_Jamba_1 forward pass successful, output shape: torch.Size([2, 10000, 1])

✅ All tests passed!
```

### Training with Logging
```bash
# Model will log to run/expname/linoss_output.txt
python main_burgers.py --model linoss_pytorch --num_epochs 10

# View logs
tail -f run/expname/linoss_output.txt
```

## Troubleshooting

### Issue: Out of memory during training
**Solution**: Reduce batch size or model width
```bash
python main_burgers.py --model linoss_pytorch --batch_size 8 --width 64
```

### Issue: Slow training on CPU
**Solution**: Use GPU (if available)
```bash
# The script automatically detects and uses CUDA if available
# Check your GPU setup in connect_gpu.txt
```

### Issue: Model not improving
**Suggestions**:
- Increase `num_layers` for more model capacity
- Reduce learning rate (try `--lr 5e-4`)
- Train longer (`--num_epochs 200`)
- Try IMEX discretization for better stability

## Parameter Tracking

The training script tracks LinOSS parameters by default:

```bash
python main_burgers.py --model linoss_pytorch \
    --track_linoss_param "blocks.0.model_t.linoss_core.blocks.0.ssm.A_diag" \
    --track_linoss_index 0
```

This logs parameter updates like:
```
LinOSS tracked param: blocks.0.model_t.linoss_core.blocks.0.ssm.A_diag[0] 
  before=0.123456, grad=0.000123, after=0.123445
```

## Advanced: Comparing IM vs IMEX

To compare discretization schemes, run:

```bash
# Train with IM (implicit)
python main_burgers.py --model linoss_pytorch --SEED 42

# Train with IMEX (implicit-explicit)
# First, modify FNO2d_Jamba.py line 91 to use discretization="IMEX"
python main_burgers.py --model linoss_pytorch --SEED 42
```

Then compare the loss curves:
```bash
# View both runs
tail -20 run/expname/loss_train.txt
```

## Key Differences from OSS_source

1. **No JAX dependency**: Pure PyTorch implementation
2. **Simpler gradients**: Uses native PyTorch autograd (no custom Function needed)
3. **Device handling**: Automatic device detection and transfer
4. **Same interface**: Identical forward signature and behavior
5. **Identical mathematics**: Uses same discretization schemes (IM/IMEX)

## Next Steps

1. **Run a test training**: 
   ```bash
   python main_burgers.py --model linoss_pytorch --num_epochs 10
   ```

2. **Compare with other models**:
   ```bash
   # Compare different temporal models
   for model in FNO FNO_GRU_1 FNO_Mamba_1 linoss_pytorch; do
       echo "Training $model..."
       python main_burgers.py --model $model --num_epochs 10
   done
   ```

3. **Fine-tune hyperparameters**: Adjust `num_layers`, `dropout`, `discretization`

4. **Integrate into other test cases**: Replace `model_t_type="OSS_source"` with `"linoss_pytorch"` in other problem scripts

## References

- **LinOSS Paper**: "Linear Oscillatory State-Space Models: Stable and Efficient Sequence Learning"
- **PyTorch Implementation**: [linoss_pytorch/linoss_pytorch.py](../linoss_pytorch/linoss_pytorch.py)
- **Equation Mapping**: [linoss_pytorch/EQUATIONS_TO_CODE_MAPPING.md](../linoss_pytorch/EQUATIONS_TO_CODE_MAPPING.md)
- **Test Suite**: [linoss_pytorch/test_linoss_pytorch.py](../linoss_pytorch/test_linoss_pytorch.py)

## Support

For issues or questions:
1. Check the test suite: `python test_linoss_pytorch_integration.py`
2. Review equation mapping for mathematical details
3. Check logging output: `tail run/expname/linoss_output.txt`
