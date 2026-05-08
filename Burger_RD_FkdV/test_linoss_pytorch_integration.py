#!/usr/bin/env python
"""
Quick test to verify linoss_pytorch integration with FNO_Jamba models.
"""
import torch
import sys
sys.path.insert(0, '..')

# Test basic imports
from time_model import linoss_pytorch_on
from FNO2d_Jamba import FNO_Jamba_1, FNO_Jamba_2

print('✅ Imports successful')

# Test linoss_pytorch_on instantiation
try:
    model = linoss_pytorch_on(
        d_model=128,
        input_dim=128,
        output_dim=128,
        discretization='IM',
        seed=0,
        num_layers=1
    )
    print('✅ linoss_pytorch_on instantiation successful')
except Exception as e:
    print(f'❌ linoss_pytorch_on instantiation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test forward pass with batch input (matching FNO_Jamba usage)
try:
    x = torch.randn(2, 100, 128)  # (batch, seq_len, d_model)
    output = model(x)
    assert output.shape == (2, 100, 128), f'Unexpected output shape: {output.shape}'
    print(f'✅ linoss_pytorch_on forward pass successful, output shape: {output.shape}')
except Exception as e:
    print(f'❌ linoss_pytorch_on forward pass failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FNO_Jamba_1 with linoss_pytorch
try:
    model = FNO_Jamba_1(
        input_dim=4,
        output_dim=1,
        modes=32,
        width=128,
        num_layers=1,
        model_t_type='linoss_pytorch'
    )
    print('✅ FNO_Jamba_1 with linoss_pytorch instantiation successful')
except Exception as e:
    print(f'❌ FNO_Jamba_1 instantiation failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FNO_Jamba_1 forward pass
try:
    x = torch.randn(2, 100, 100, 4)  # (batch, N_x, N_t, in_dim)
    output = model(x)
    print(f'✅ FNO_Jamba_1 forward pass successful, output shape: {output.shape}')
except Exception as e:
    print(f'❌ FNO_Jamba_1 forward pass failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

print('\n✅ All tests passed!')
