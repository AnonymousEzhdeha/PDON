"""
PyTorch implementation of a linear oscillatory state-space model.

This implementation focuses on the Implicit (IM) discretization scheme (Paper Equations 3-4).

Module structure mirrors the JAX implementation in linoss/models/LinOSS.py but uses PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np


# ============================================================================
# PAPER EQUATION REFERENCE
# ============================================================================
"""
Paper Equation 1-2: Continuous Second-Order ODE System
    y''(t) = -A·y(t) + B·u(t) + b
    x(t) = C·y(t) + D·u(t)
    
    With auxiliary state z(t) = y'(t):
    z'(t) = -A·y(t) + B·u(t)
    y'(t) = z(t)

Paper Equation 3-4: Implicit (IM) Discretization
    
    Discretized recurrence:
    z_n = z_{n-1} + Δt·(-A·y_n + B·u_n)
    y_n = y_{n-1} + Δt·z_n
    
    Matrix form: M·x_n = x_{n-1} + F_n
    where x_n = [z_n, y_n]^T
    
    M = [[I,        Δt·A    ]
         [-Δt·I,    I       ]]
    
    F_n = [[Δt·B·u_n],
           [0        ]]
    
    Using Schur complement (since A is diagonal):
    S = (I + Δt²·A)⁻¹  (diagonal, O(m) to invert)
    
    M⁻¹ = [[S - Δt²·A·S,  -Δt·A·S],
           [Δt·S,         S      ]]
    
    Final recurrence (Paper Eq. 4):
    x_n = M_IM · x_{n-1} + F_IM_n
"""
# ============================================================================


def init_uniform(shape: Tuple, std: float = 1.0, device: str = 'cpu') -> torch.Tensor:
    """
    Initialize uniform random tensor in [-std, std].
    
    Args:
        shape: Output shape
        std: Standard deviation (half-width of uniform distribution)
        device: Device to create tensor on
        
    Returns:
        Initialized tensor with shape `shape`
    """
    return torch.empty(shape, device=device).uniform_(-std, std)


class LinearAttention(nn.Module):
    """
    Simple gated linear unit activation: y = x·sigmoid(Wx + b)
    Used in LinOSSBlock for feature gating.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.fc2 = nn.Linear(input_dim, output_dim, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            Gated output (..., output_dim)
        """
        return self.fc1(x) * torch.sigmoid(self.fc2(x))


# ============================================================================
# PARALLEL SCAN OPERATIONS (Paper Eq. 5-6)
# ============================================================================
# Binary operator for associative scan of linear recurrence
# Structure: Compose two state transition steps using matrix multiplication
# This enables O(log L) parallel computation in JAX; PyTorch implementation
# shown here demonstrates the parallel structure

def binary_operator(
    q_i: Tuple[torch.Tensor, torch.Tensor],
    q_j: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary operator for parallel associative scan (Paper Eq. 5-6).
    
    Composes two sequential state transitions into a single transition.
    Enables O(log L) depth parallel computation via associative scan.
    
    Paper Eq. 5-6: Composition of transition matrices for IMEX scheme
    
    Args:
        q_i: Tuple (A_i, b_i) where A_i contains [A, B, C, D] blocks [4P]
                              and b_i contains [b1, b2] forcing terms [2P]
        q_j: Tuple (A_j, b_j) same structure
        
    Returns:
        Composed transition: (A_new, b_new)
        
    Mathematical operation (for 2x2 blocks):
    $$A_{new} = A_j \\cdot A_i$$
    $$b_{new} = A_j \\cdot b_i + b_j$$
    
    This allows combining two steps into one via matrix multiplication.
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    
    device = A_i.device
    P = A_i.shape[0] // 4  # Extract state dimension from block size
    
    # Extract 2x2 block matrices from concatenated representation
    # Storage: [A_block || B_block || C_block || D_block], each of size P
    iA = A_i[0 * P: 1 * P]       # Top-left: A
    iB = A_i[1 * P: 2 * P]       # Top-right: B
    iC = A_i[2 * P: 3 * P]       # Bottom-left: C
    iD = A_i[3 * P: 4 * P]       # Bottom-right: D
    
    jA = A_j[0 * P: 1 * P]       # Top-left: A
    jB = A_j[1 * P: 2 * P]       # Top-right: B
    jC = A_j[2 * P: 3 * P]       # Bottom-left: C
    jD = A_j[3 * P: 4 * P]       # Bottom-right: D
    
    # Compose matrices: M_new = M_j * M_i (for 2x2 blocks)
    # [[A_new, B_new],    [[jA*iA + jB*iC,  jA*iB + jB*iD],
    #  [C_new, D_new]] =   [jC*iA + jD*iC,  jC*iB + jD*iD]]
    
    A_new = jA * iA + jB * iC  # Paper Eq. 5-6: Composed top-left block
    B_new = jA * iB + jB * iD  # Paper Eq. 5-6: Composed top-right block
    C_new = jC * iA + jD * iC  # Paper Eq. 5-6: Composed bottom-left block
    D_new = jC * iB + jD * iD  # Paper Eq. 5-6: Composed bottom-right block
    
    # Reconstruct concatenated block matrix
    A_new_full = torch.cat([A_new, B_new, C_new, D_new], dim=0)
    
    # Compose forcing terms: b_new = M_j * b_i + b_j
    b_i1 = b_i[0:P]   # First P elements
    b_i2 = b_i[P:2*P]  # Last P elements
    
    new_b1 = jA * b_i1 + jB * b_i2  # Paper Eq. 5-6: Composed forcing (part 1)
    new_b2 = jC * b_i1 + jD * b_i2  # Paper Eq. 5-6: Composed forcing (part 2)
    
    b_new_full = torch.cat([new_b1, new_b2], dim=0)
    
    # Return composed step
    return A_new_full, b_new_full + b_j


def apply_linoss_im(
    A_diag: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    input_sequence: torch.Tensor,
    step: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Apply LinOSS-IM (Implicit discretization) to an input sequence.
    
    Implements Paper Equations 3-4: Implicit time integration with Schur complement inversion.
    
    Args:
        A_diag (torch.Tensor): Diagonal state matrix A ∈ R^P (shape: [P])
                              Paper Eq. 1: A is diagonal for efficiency
                              
        B (torch.Tensor): Complex-valued input matrix B ∈ C^(P×H) 
                         Stored as real+imag in shape [P, H, 2]
                         Paper Eq. 1: Input-to-state coupling
                         
        C (torch.Tensor): Complex-valued output matrix C ∈ C^(H×P)
                         Stored as real+imag in shape [H, P, 2]
                         Paper Eq. 1: State-to-output projection
                         
        input_sequence (torch.Tensor): Input sequence U = [u_0, u_1, ..., u_{L-1}]
                                      Shape: [L, H] where L=sequence length, H=feature dim
                                      Paper Eq. 2: u(t) drives the system
                                      
        step (torch.Tensor): Discretization timestep Δt ∈ R^P (shape: [P])
                            Paper Eq. 3-4: Step size for numerical integration
                            Constraint: 0 < Δt ≤ 1 for stability
                            
        initial_state (tuple, optional): Initial state (z_0, y_0)
                                        If None, initialize to zero
                                        
    Returns:
        outputs (torch.Tensor): Model outputs X = [x_0, x_1, ..., x_{L-1}]
                               Shape: [L, H]
                               Paper Eq. 1: x(t) = C·y(t) + D·u(t) (D term added in Layer)
    
    Implementation Details:
    ----------------------
    1. Compute B·u for each timestep (Paper Eq. 2 term)
    2. Compute Schur complement S = (I + Δt²·A)⁻¹ (Paper Eq. 3, O(m) diagonal inversion)
    3. Build transition matrix M_IM⁻¹ blocks (Paper Eq. 3)
    4. Build forcing vector F_IM (Paper Eq. 3-4)
    5. Apply sequential recurrence: x_n = M_IM·x_{n-1} + F_IM·u_n
       Note: Sequential version for clarity; JAX uses associative_scan for parallelization
    6. Extract hidden state y = x[:, P:2P] (Paper Eq. 2: y is second half of x=[z,y])
    7. Apply readout C·y (Paper Eq. 1)
    """
    
    device = A_diag.device
    squeeze_output = False
    if input_sequence.dim() == 2:
        input_sequence = input_sequence.unsqueeze(0)  # [1, L, H]
        squeeze_output = True
    elif input_sequence.dim() != 3:
        raise ValueError(f"input_sequence must have shape [L, H] or [B, L, H], got {tuple(input_sequence.shape)}")

    Bsz, L, H = input_sequence.shape  # Batch, sequence length, feature dimension
    P = A_diag.shape[0]          # State dimension (ssm_size)
    
    # ========================================================================
    # STEP 1: Compute B·u_n for each timestep n (Paper Eq. 2: Bu term)
    # ========================================================================
    # Convert B from real storage [P, H, 2] to complex [P, H]
    B_complex = torch.complex(B[..., 0], B[..., 1])  # Shape: [P, H]
    
    # Convert input to complex (cast to match B's dtype for matmul)
    input_complex = input_sequence.to(dtype=B_complex.dtype)  # Shape: [B, L, H]
    
    # Compute Bu for each input: Bu_n = B @ u_n (batched matrix-vector product)
    # Result shape: [L, P] (L timesteps, P state dimensions)
    Bu_elements = torch.einsum("blh,ph->blp", input_complex, B_complex)  # [B, L, P]
    
    # ========================================================================
    # STEP 2: Compute Schur complement (Paper Eq. 3)
    # ========================================================================
    # S = (I + Δt²·A)⁻¹  (diagonal matrix, O(m) inversion)
    # Since A is diagonal, we can invert element-wise
    
    # Convert to complex dtype for consistency with B, C, and Bu computations
    A_diag_complex = A_diag.to(dtype=B_complex.dtype)
    step_complex = step.to(dtype=B_complex.dtype)
    
    # schur_comp represents S (shape: [P])
    # Numerically: S = 1 / (1 + Δt²·A)
    schur_comp = 1.0 / (1.0 + step_complex ** 2 * A_diag_complex)  # Paper Eq. 3: S = (I + Δt²A)⁻¹
    
    # ========================================================================
    # STEP 3: Build M_IM⁻¹ blocks (Paper Eq. 3)
    # ========================================================================
    # M⁻¹_IM = [[S - Δt²·A·S,  -Δt·A·S],
    #           [Δt·S,         S      ]]
    #
    # We represent this as 4 diagonal blocks [M11, M12, M21, M22]:
    
    # Top-left block: M11 = S - Δt²·A·S = S(I - Δt²A)
    M_IM_11 = 1.0 - step_complex ** 2 * A_diag_complex * schur_comp  # Shape: [P]
    
    # Top-right block: M12 = -Δt·A·S
    M_IM_12 = -1.0 * step_complex * A_diag_complex * schur_comp     # Shape: [P]
    
    # Bottom-left block: M21 = Δt·S
    M_IM_21 = step_complex * schur_comp                      # Shape: [P]
    
    # Bottom-right block: M22 = S
    M_IM_22 = schur_comp                             # Shape: [P]
    
    # ========================================================================
    # STEP 4: Initialize state variables
    # ========================================================================
    # State is represented as x = [z, y] where:
    #   z ∈ C^P: auxiliary velocity state (complex, z = dy/dt)
    #   y ∈ C^P: hidden state (complex)
    # Total state dimension: 2P (in complex)
    
    if initial_state is None:
        # Initialize to zero if not provided
        z_n = torch.zeros(Bsz, P, device=device, dtype=B_complex.dtype)
        y_n = torch.zeros(Bsz, P, device=device, dtype=B_complex.dtype)
    else:
        z_n, y_n = initial_state
    
    # ========================================================================
    # STEP 5: Sequential recurrence (Paper Eq. 4)
    # ========================================================================
    # x_n = M_IM · x_{n-1} + F_IM_n
    # where x_n = [z_n, y_n]^T
    #
    # Note: This is sequential (O(L) depth). JAX version uses associative_scan
    # for parallel O(log L) depth, but sequential is clearer for PyTorch.
    
    outputs = []
    
    for n in range(L):
        # Get current input u_n (shape: [H])
        u_n = input_sequence[:, n, :]
        
        # Compute forcing term F_IM_n (Paper Eq. 4)
        # F_IM = [[Δt·B·u_n], [0]]  →  after applying M_IM⁻¹:
        # F1_n = M11·(Δt·B·u_n) + M12·0 = (S - Δt²AS)·Δt·B·u_n
        # F2_n = M21·(Δt·B·u_n) + M22·0 = ΔtS·Δt·B·u_n
        
        Bu_n = Bu_elements[:, n, :]  # Shape: [B, P]
        
        # Forcing terms (Paper Eq. 4)
        F1_n = M_IM_11 * (step_complex * Bu_n)  # Paper Eq. 4: M11·(ΔtBu)
        F2_n = M_IM_21 * (step_complex * Bu_n)  # Paper Eq. 4: M21·(ΔtBu)
        
        # ====================================================================
        # Apply state transition (Paper Eq. 4): x_n = M_IM · x_{n-1} + F_n
        # ====================================================================
        # Matrix multiplication broken down by blocks:
        # [z_n']   [M11  M12] [z_n-1]   [F1_n]
        # [y_n'] = [M21  M22] [y_n-1] + [F2_n]
        
        # z_n = M11·z_{n-1} + M12·y_{n-1} + F1_n
        z_n_new = M_IM_11 * z_n + M_IM_12 * y_n + F1_n
        
        # y_n = M21·z_{n-1} + M22·y_{n-1} + F2_n
        y_n_new = M_IM_21 * z_n + M_IM_22 * y_n + F2_n
        
        # Update state for next iteration
        z_n = z_n_new
        y_n = y_n_new
        
        # Store output for this timestep
        outputs.append(y_n.unsqueeze(1))  # Shape: [B, 1, P]
    
    # Concatenate all output timesteps (shape: [L, P])
    ys = torch.cat(outputs, dim=1)
    
    # ========================================================================
    # STEP 6: Apply output projection (Paper Eq. 1): x(t) = C·y(t)
    # ========================================================================
    # Convert C from real storage [H, P, 2] to complex [H, P]
    C_complex = torch.complex(C[..., 0], C[..., 1])  # Shape: [H, P]
    
    # Apply readout: x = C·y (shape: [L, P] @ [P, H]^T = [L, H])
    # Note: C @ y.T gives [H, P] @ [P, L] = [H, L], so we need to transpose
    outputs_final = torch.einsum("blp,hp->blh", ys, C_complex)  # [B, L, P] x [H, P] -> [B, L, H]
    
    # Take real part (discard imaginary part from complex computation)
    outputs_final = outputs_final.real
    
    if squeeze_output:
        outputs_final = outputs_final.squeeze(0)
        z_n = z_n.squeeze(0)
        y_n = y_n.squeeze(0)

    return outputs_final, (z_n, y_n)  # Return outputs and final state


def apply_linoss_imex(
    A_diag: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    input_sequence: torch.Tensor,
    step: torch.Tensor,
    initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> torch.Tensor:
    """
    Apply LinOSS-IMEX (Implicit-Explicit discretization) to an input sequence.
    
    Implements Paper Equations 5-6: Implicit-Explicit time integration with symplectic structure.
    
    IMEX scheme provides:
    - Symplectic integration (volume-preserving, good for conservative systems)
    - Better handling of stiff systems (large B matrix)
    - Slightly larger truncation error but better long-term stability
    
    Args:
        A_diag (torch.Tensor): Diagonal state matrix A ∈ R^P (shape: [P])
                              Paper Eq. 1: A is diagonal for efficiency
                              
        B (torch.Tensor): Complex-valued input matrix B ∈ C^(P×H) 
                         Stored as real+imag in shape [P, H, 2]
                         Paper Eq. 1: Input-to-state coupling
                         
        C (torch.Tensor): Complex-valued output matrix C ∈ C^(H×P)
                         Stored as real+imag in shape [H, P, 2]
                         Paper Eq. 1: State-to-output projection
                         
        input_sequence (torch.Tensor): Input sequence U = [u_0, u_1, ..., u_{L-1}]
                                      Shape: [L, H] where L=sequence length, H=feature dim
                                      Paper Eq. 2: u(t) drives the system
                                      
        step (torch.Tensor): Discretization timestep Δt ∈ R^P (shape: [P])
                            Paper Eq. 5-6: Step size for numerical integration
                            Constraint: 0 < Δt ≤ 1 for stability
                            
        initial_state (tuple, optional): Initial state (z_0, y_0)
                                        If None, initialize to zero
                                        
    Returns:
        outputs (torch.Tensor): Model outputs X = [x_0, x_1, ..., x_{L-1}]
                               Shape: [L, H]
                               Paper Eq. 1: x(t) = C·y(t) + D·u(t) (D term added in Layer)
    
    Implementation Details:
    ----------------------
    The IMEX scheme uses a 2x2 block matrix:
    
    Paper Equation 5-6:
    $$M_{IMEX} = \\begin{bmatrix} I & -\\Delta t A \\\\ \\Delta t I & I - \\Delta t^2 A \\end{bmatrix}$$
    
    Blocks:
    - A_block = I (identity)
    - B_block = -Δt·A
    - C_block = Δt·I (identity scaled by timestep)
    - D_block = I - Δt²·A
    
    These blocks are composed efficiently via the binary_operator for parallel computation.
    """
    
    device = A_diag.device
    squeeze_output = False
    if input_sequence.dim() == 2:
        input_sequence = input_sequence.unsqueeze(0)  # [1, L, H]
        squeeze_output = True
    elif input_sequence.dim() != 3:
        raise ValueError(f"input_sequence must have shape [L, H] or [B, L, H], got {tuple(input_sequence.shape)}")

    Bsz, L, H = input_sequence.shape  # Batch, sequence length, feature dimension
    P = A_diag.shape[0]          # State dimension (ssm_size)
    
    # ========================================================================
    # STEP 1: Compute B·u_n for each timestep n (Paper Eq. 2: Bu term)
    # ========================================================================
    # Convert B from real storage [P, H, 2] to complex [P, H]
    B_complex = torch.complex(B[..., 0], B[..., 1])  # Shape: [P, H]
    
    # Convert input to complex (cast to match B's dtype for matmul)
    input_complex = input_sequence.to(dtype=B_complex.dtype)  # Shape: [B, L, H]
    
    # Compute Bu for each input: Bu_n = B @ u_n (batched matrix-vector product)
    # Result shape: [L, P] (L timesteps, P state dimensions)
    Bu_elements = torch.einsum("blh,ph->blp", input_complex, B_complex)  # [B, L, P]
    
    # ========================================================================
    # STEP 2: Build IMEX transition matrix blocks (Paper Eq. 5-6)
    # ========================================================================
    # Paper Eq. 5-6: 2x2 block matrix structure
    # M_IMEX = [[I,              -Δt·A    ],
    #           [Δt·I,           I - Δt²·A]]
    #
    # Stored as [A_block || B_block || C_block || D_block] for each timestep
    
    # Convert to complex dtype for consistency
    A_diag_complex = A_diag.to(dtype=B_complex.dtype)
    step_complex = step.to(dtype=B_complex.dtype)
    
    # Block matrices (Paper Eq. 5-6):
    A_block = torch.ones(P, device=device, dtype=B_complex.dtype)           # Top-left: I
    B_block = -1.0 * step_complex * A_diag_complex                          # Top-right: -Δt·A
    C_block = step_complex                                                   # Bottom-left: Δt·I
    D_block = 1.0 - (step_complex ** 2) * A_diag_complex                    # Bottom-right: I - Δt²·A
    
    # Concatenate blocks (Paper Eq. 5-6 structure)
    M_IMEX = torch.cat([A_block, B_block, C_block, D_block], dim=0)
    
    # Replicate across L timesteps
    M_IMEX_elements = M_IMEX.unsqueeze(0).expand(L, -1)
    
    # ========================================================================
    # STEP 3: Build forcing vectors (Paper Eq. 5-6)
    # ========================================================================
    # Forcing structure: [Δt·Bu, Δt²·Bu]
    # F1_n = Δt·B·u_n (for z update)
    # F2_n = Δt²·B·u_n (for y update)
    
    F1 = step_complex.unsqueeze(0).unsqueeze(0) * Bu_elements        # Paper Eq. 5-6: Δt·Bu [B, L, P]
    F2 = (step_complex.unsqueeze(0).unsqueeze(0) ** 2) * Bu_elements  # Paper Eq. 5-6: Δt²·Bu [B, L, P]
    
    # Concatenate forcing terms
    F = torch.cat([F1, F2], dim=-1)  # Shape: [B, L, 2P]
    
    # ========================================================================
    # STEP 4: Sequential recurrence with IMEX blocks
    # ========================================================================
    # Paper Eq. 5-6: Recurrence with 2x2 block structure
    # [z_n]       [A_block  B_block] [z_{n-1}]       [F1_n]
    # [y_n]   =   [C_block  D_block] [y_{n-1}]   +   [F2_n]
    #
    # Note: JAX version uses associative_scan(binary_operator, ...) for O(log L) depth
    #       PyTorch version shown is sequential O(L) for clarity
    
    if initial_state is None:
        z_n = torch.zeros(Bsz, P, device=device, dtype=B_complex.dtype)
        y_n = torch.zeros(Bsz, P, device=device, dtype=B_complex.dtype)
    else:
        z_n, y_n = initial_state
    
    outputs = []
    
    for n in range(L):
        # Extract current IMEX blocks and forcing
        A_b = A_block              # Top-left: I
        B_b = B_block              # Top-right: -Δt·A
        C_b = C_block              # Bottom-left: Δt·I
        D_b = D_block              # Bottom-right: I - Δt²·A
        
        F1_n = F1[:, n, :]  # Δt·Bu_n
        F2_n = F2[:, n, :]  # Δt²·Bu_n
        
        # Apply IMEX blocks (Paper Eq. 5-6)
        # z_n = A_block·z_{n-1} + B_block·y_{n-1} + F1_n
        z_n_new = A_b * z_n + B_b * y_n + F1_n
        
        # y_n = C_block·z_{n-1} + D_block·y_{n-1} + F2_n
        y_n_new = C_b * z_n + D_b * y_n + F2_n
        
        z_n = z_n_new
        y_n = y_n_new
        
        outputs.append(y_n.unsqueeze(1))
    
    # Concatenate all output timesteps
    ys = torch.cat(outputs, dim=1)
    
    # ========================================================================
    # STEP 5: Apply output projection (Paper Eq. 1): x(t) = C·y(t)
    # ========================================================================
    C_complex = torch.complex(C[..., 0], C[..., 1])  # Shape: [H, P]
    
    # Apply readout observation: x = C·y
    outputs_final = torch.einsum("blp,hp->blh", ys, C_complex)  # [B, L, P] x [H, P] -> [B, L, H]
    
    # Take real part (observation is real-valued)
    outputs_final = outputs_final.real
    
    if squeeze_output:
        outputs_final = outputs_final.squeeze(0)
        z_n = z_n.squeeze(0)
        y_n = y_n.squeeze(0)

    return outputs_final, (z_n, y_n)


class LinOSSLayer(nn.Module):
    """
    LinOSS State-Space Layer implementing Paper Equations 1-4.
    
    This layer learns a linear state-space model with the form:
    $$y''(t) = -A·y(t) + B·u(t) + b$$
    $$x(t) = C·y(t) + D·u(t)$$
    
    Key design choices:
    - A is diagonal (O(m) complexity instead of O(m³))
    - B, C use complex-valued matrices (oscillatory dynamics)
    - Δt (step) is learned per dimension for flexibility
    - Two discretization schemes: IM (implicit) or IMEX (implicit-explicit)
    
    Currently implements: Implicit (IM) discretization (Paper Eq. 3-4)
    """
    
    def __init__(
        self,
        ssm_size: int,
        feature_dim: int,
        discretization: str = 'IM',
        device: str = 'cpu'
    ):
        """
        Initialize a LinOSS layer.
        
        Args:
            ssm_size (int): Size of the state-space model (P in paper)
                          Controls the number of oscillatory modes
                          Larger → more expressive, slower computation
                          
            feature_dim (int): Feature dimension (H in paper)
                             Input/output feature dimension
                             Must match sequence feature dimension
                             
            discretization (str): Either 'IM' (implicit) or 'IMEX' (implicit-explicit)
                                Currently only 'IM' is implemented
                                'IM': exponentially stable, dissipative
                                'IMEX': symplectic, volume-preserving
                                
            device (str): Device to place parameters on ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.ssm_size = ssm_size      # P: state dimension
        self.feature_dim = feature_dim  # H: feature dimension
        self.discretization = discretization
        self.device = device
        
        # ====================================================================
        # LEARNABLE PARAMETERS (Paper Eq. 1-2)
        # ====================================================================
        
        # A_diag: Diagonal state coupling matrix A ∈ R^P (Paper Eq. 1)
        # Constraint: A ≥ 0 (enforced via ReLU) for stability
        # Initialized uniformly in [0, 1]
        self.register_parameter('A_diag', nn.Parameter(
            init_uniform((ssm_size,), std=1.0, device=device)
        ))
        
        # B: Complex-valued input matrix B ∈ C^(P×H) (Paper Eq. 1)
        # Stored as real + imaginary parts in shape [P, H, 2]
        # Initialized uniformly with scaling 1/sqrt(H)
        self.register_parameter('B', nn.Parameter(
            init_uniform((ssm_size, feature_dim, 2), std=1.0/math.sqrt(feature_dim), device=device)
        ))
        
        # C: Complex-valued output matrix C ∈ C^(H×P) (Paper Eq. 1)
        # Stored as real + imaginary parts in shape [H, P, 2]
        # Initialized uniformly with scaling 1/sqrt(ssm_size)
        self.register_parameter('C', nn.Parameter(
            init_uniform((feature_dim, ssm_size, 2), std=1.0/math.sqrt(ssm_size), device=device)
        ))
        
        # D: Direct feedthrough matrix D ∈ R^H (Paper Eq. 1)
        # Allows input to directly affect output: x += D·u
        # Initialized from normal distribution with std=1.0
        self.register_parameter('D', nn.Parameter(
            torch.randn(feature_dim, device=device)
        ))
        
        # steps: Discretization timestep Δt ∈ R^P (Paper Eq. 3-4)
        # One timestep per state dimension for flexibility
        # Constraint: 0 < Δt ≤ 1 (enforced via sigmoid) for stability
        # Initialized uniformly in [0, 1]
        self.register_parameter('steps', nn.Parameter(
            init_uniform((ssm_size,), std=1.0, device=device)
        ))
    
    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply LinOSS layer to an input sequence.
        
        Args:
            input_sequence (torch.Tensor): Input sequence of shape [L, H]
                                         L = sequence length
                                         H = feature dimension (must match self.feature_dim)
                                         
        Returns:
            outputs (torch.Tensor): Output sequence of shape [L, H]
                                   Result of forward pass through the ODE system
                                   
        Forward pass computation:
        1. Enforce parameter constraints (A ≥ 0, 0 < Δt ≤ 1)
        2. Convert real storage to complex matrices (B, C)
        3. Apply discretized ODE solver (apply_linoss_im)
        4. Add direct feedthrough (D·u term)
        """
        
        # ====================================================================
        # Enforce parameter constraints (Paper Eq. 1, 3-4 requirements)
        # ====================================================================
        
        # Constraint: A ≥ 0 for Schur complement well-definedness (Paper Eq. 3)
        # Apply ReLU: max(0, A)
        A_diag_constrained = F.relu(self.A_diag)
        
        # Convert B from real storage [P, H, 2] to complex [P, H]
        # Real + i·Imag representation captures oscillatory eigenmodes
        B_complex = torch.complex(self.B[..., 0], self.B[..., 1])
        
        # Convert C from real storage [H, P, 2] to complex [H, P]
        C_complex = torch.complex(self.C[..., 0], self.C[..., 1])
        
        # Constraint: 0 < Δt ≤ 1 for stability (Paper Eq. 3-4)
        # Apply sigmoid to map R → (0, 1): Δt = sigmoid(learned_steps)
        steps_constrained = torch.sigmoid(self.steps)
        
        # ====================================================================
        # Apply selected discretization scheme (Paper Eq. 3-4 or 5-6)
        # ====================================================================
        
        if self.discretization == 'IM':
            # Apply Implicit discretization (Paper Eq. 3-4)
            # Exponentially stable, dissipative dynamics
            ys, _ = apply_linoss_im(
                A_diag=A_diag_constrained,
                B=self.B,  # Pass real storage; apply_linoss_im converts to complex
                C=self.C,  # Pass real storage; apply_linoss_im converts to complex
                input_sequence=input_sequence,
                step=steps_constrained
            )
        elif self.discretization == 'IMEX':
            # Apply Implicit-Explicit discretization (Paper Eq. 5-6)
            # Symplectic integration, volume-preserving
            ys, _ = apply_linoss_imex(
                A_diag=A_diag_constrained,
                B=self.B,  # Pass real storage; apply_linoss_imex converts to complex
                C=self.C,  # Pass real storage; apply_linoss_imex converts to complex
                input_sequence=input_sequence,
                step=steps_constrained
            )
        else:
            raise NotImplementedError(f"Discretization '{self.discretization}' not implemented. "
                                    f"Currently supported: 'IM', 'IMEX' (Paper Eq. 3-6)")
        
        # ====================================================================
        # Add direct feedthrough (Paper Eq. 1): x(t) = C·y(t) + D·u(t)
        # ====================================================================
        # D ∈ R^H is a diagonal matrix (stored as vector)
        # D·u_n = D * u_n (element-wise multiplication)
        Du = self.D.unsqueeze(0) * input_sequence  # Broadcast and multiply
        
        # Final output: y_output + D·u
        outputs = ys + Du
        
        return outputs


class LinOSSBlock(nn.Module):
    """
    LinOSS Building Block combining SSM, normalization, and nonlinearity.
    
    Architecture:
    Input → BatchNorm → LinOSSLayer → Activation → GLU → Dropout → Output
                                                    ↓
                                        Residual Connection
    
    This follows modern deep learning conventions with:
    - Batch normalization for stability
    - Gated linear units for feature selection
    - Residual connections for gradient flow
    - Dropout for regularization
    """
    
    def __init__(
        self,
        ssm_size: int,
        feature_dim: int,
        discretization: str = 'IM',
        drop_rate: float = 0.05,
        device: str = 'cpu'
    ):
        """
        Initialize a LinOSS block.
        
        Args:
            ssm_size (int): Size of the state-space model
            feature_dim (int): Feature dimension
            discretization (str): 'IM' or 'IMEX'
            drop_rate (float): Dropout probability (default: 0.05)
            device (str): Device to place parameters on
        """
        super().__init__()
        
        # Batch normalization (normalize across batch dimension)
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        
        # LinOSS state-space model layer
        self.ssm = LinOSSLayer(
            ssm_size=ssm_size,
            feature_dim=feature_dim,
            discretization=discretization,
            device=device
        )
        
        # Gated Linear Unit (GLU) for learned feature gating
        # GLU: y = x * sigmoid(W·x + b) (learnable gating mechanism)
        self.glu = LinearAttention(feature_dim, feature_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=drop_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LinOSS block with residual connection.
        
        Args:
            x (torch.Tensor): Input sequence [L, H]
            
        Returns:
            output (torch.Tensor): Output sequence [L, H]
        """
        # Save input for residual connection
        skip = x
        
        # Layer normalization (normalizes last dimension features)
        x = self.norm(x)
        
        # Apply LinOSS state-space model (Paper Eq. 1-4)
        x = self.ssm(x)
        
        # Activation + dropout
        x = self.dropout(F.gelu(x))
        
        # Gated linear unit (conditional feature weighting)
        x = self.glu(x)
        
        # Dropout after gating
        x = self.dropout(x)
        
        # Residual connection (ensures gradient flow)
        x = skip + x
        
        return x


class LinOSS(nn.Module):
    """
    Full LinOSS model stacking multiple LinOSSBlocks.
    
    Architecture:
    Input Projection → [LinOSSBlock × N] → Output Projection → Tanh
    
    Implements the complete state-space model with:
    - Input embedding/projection
    - Multiple stacked blocks for expressiveness
    - Output projection to target dimension
    - Final nonlinear activation (tanh)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ssm_size: int,
        feature_dim: int,
        num_blocks: int = 2,
        discretization: str = 'IM',
        dropout: float = 0.05,
        device: str = 'cpu'
    ):
        """
        Initialize a full LinOSS model.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            ssm_size (int): State-space model size (P)
            feature_dim (int): Internal feature dimension (H)
            num_blocks (int): Number of stacked LinOSSBlocks (default: 2)
            discretization (str): 'IM' or 'IMEX' (default: 'IM')
            dropout (float): Dropout rate (default: 0.05)
            device (str): Device to place parameters on (default: 'cpu')
        """
        super().__init__()
        
        # Input projection: input_dim → feature_dim
        self.input_proj = nn.Linear(input_dim, feature_dim)
        
        # Stack of LinOSS blocks
        self.blocks = nn.ModuleList([
            LinOSSBlock(
                ssm_size=ssm_size,
                feature_dim=feature_dim,
                discretization=discretization,
                drop_rate=dropout,
                device=device
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection: feature_dim → output_dim
        self.output_proj = nn.Linear(feature_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full LinOSS model.
        
        Args:
            x (torch.Tensor): Input sequence [L, input_dim]
            
        Returns:
            output (torch.Tensor): Output sequence [L, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply stacked LinOSS blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Final activation (tanh)
        x = torch.tanh(x)
        
        return x
