"""Unit tests for CPCModule."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from sorrel.models.pytorch.cpc_module import CPCModule


def test_cpc_module_initialization():
    """Test CPCModule initialization."""
    latent_dim = 256
    cpc_horizon = 30
    projection_dim = 128
    temperature = 0.07
    
    module = CPCModule(
        latent_dim=latent_dim,
        cpc_horizon=cpc_horizon,
        projection_dim=projection_dim,
        temperature=temperature,
    )
    
    assert module.latent_dim == latent_dim
    assert module.cpc_horizon == cpc_horizon
    assert module.projection_dim == projection_dim
    assert module.temperature == temperature
    assert isinstance(module.cpc_proj, nn.Linear)
    assert module.cpc_proj.in_features == latent_dim
    assert module.cpc_proj.out_features == projection_dim
    assert isinstance(module.latent_proj, nn.Linear)
    assert module.latent_proj.in_features == latent_dim
    assert module.latent_proj.out_features == projection_dim


def test_cpc_module_initialization_default_projection():
    """Test CPCModule initialization with default projection_dim."""
    latent_dim = 256
    module = CPCModule(latent_dim=latent_dim)
    
    assert module.projection_dim == latent_dim
    assert isinstance(module.latent_proj, nn.Identity)


def test_cpc_module_forward():
    """Test CPCModule forward pass."""
    latent_dim = 256
    projection_dim = 128
    batch_size = 2
    seq_length = 10
    
    module = CPCModule(latent_dim=latent_dim, projection_dim=projection_dim)
    
    # Test with batch dimension
    c_seq = torch.randn(batch_size, seq_length, latent_dim)
    output = module.forward(c_seq)
    
    assert output.shape == (batch_size, seq_length, projection_dim)
    
    # Test without batch dimension
    c_seq_2d = torch.randn(seq_length, latent_dim)
    output_2d = module.forward(c_seq_2d)
    
    assert output_2d.shape == (seq_length, projection_dim)


def test_cpc_module_compute_loss_basic():
    """Test CPCModule compute_loss with basic inputs."""
    latent_dim = 256
    cpc_horizon = 5
    seq_length = 20
    batch_size = 1
    
    module = CPCModule(latent_dim=latent_dim, cpc_horizon=cpc_horizon)
    device = torch.device("cpu")
    module.to(device)
    
    # Create test sequences
    z_seq = torch.randn(batch_size, seq_length, latent_dim, device=device, requires_grad=True)
    c_seq = torch.randn(batch_size, seq_length, latent_dim, device=device, requires_grad=True)
    
    # Compute loss
    loss = module.compute_loss(z_seq, c_seq)
    
    assert loss.requires_grad
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_cpc_module_compute_loss_with_mask():
    """Test CPCModule compute_loss with masking."""
    latent_dim = 256
    cpc_horizon = 5
    seq_length = 20
    batch_size = 1
    
    module = CPCModule(latent_dim=latent_dim, cpc_horizon=cpc_horizon)
    device = torch.device("cpu")
    module.to(device)
    
    # Create test sequences
    z_seq = torch.randn(batch_size, seq_length, latent_dim, device=device, requires_grad=True)
    c_seq = torch.randn(batch_size, seq_length, latent_dim, device=device, requires_grad=True)
    
    # Create mask (all valid)
    mask = torch.ones(batch_size, seq_length, dtype=torch.bool, device=device)
    
    # Compute loss
    loss = module.compute_loss(z_seq, c_seq, mask)
    
    assert loss.requires_grad
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_cpc_module_compute_loss_with_episode_boundary():
    """Test CPCModule compute_loss with episode boundary (done flag)."""
    latent_dim = 256
    cpc_horizon = 5
    seq_length = 20
    
    module = CPCModule(latent_dim=latent_dim, cpc_horizon=cpc_horizon)
    device = torch.device("cpu")
    module.to(device)
    
    # Create test sequences
    z_seq = torch.randn(1, seq_length, latent_dim, device=device, requires_grad=True)
    c_seq = torch.randn(1, seq_length, latent_dim, device=device, requires_grad=True)
    
    # Create done flags (episode ends at timestep 10)
    dones = torch.zeros(seq_length, device=device)
    dones[10] = 1.0
    
    # Create mask from dones
    mask = module.create_mask_from_dones(dones, seq_length)
    
    # Compute loss
    loss = module.compute_loss(z_seq, c_seq, mask)
    
    assert loss.requires_grad
    assert loss.item() >= 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_cpc_module_create_mask_from_dones():
    """Test CPCModule create_mask_from_dones."""
    seq_length = 20
    module = CPCModule(latent_dim=256)
    device = torch.device("cpu")
    
    # No done flags
    dones = torch.zeros(seq_length, device=device)
    mask = module.create_mask_from_dones(dones, seq_length)
    
    assert mask.shape == (1, seq_length)
    assert mask.all()  # All should be True
    
    # Episode ends at timestep 10
    dones[10] = 1.0
    mask = module.create_mask_from_dones(dones, seq_length)
    
    assert mask.shape == (1, seq_length)
    # After timestep 10, mask should be False
    assert mask[0, 10].item() == True  # Timestep 10 itself is valid
    assert mask[0, 11:].all() == False  # After timestep 10 should be False


def test_cpc_module_gradient_flow():
    """Test that gradients flow through CPCModule."""
    latent_dim = 256
    projection_dim = 128  # Use different projection_dim to test Linear layer
    cpc_horizon = 5
    seq_length = 20
    
    module = CPCModule(latent_dim=latent_dim, cpc_horizon=cpc_horizon, projection_dim=projection_dim)
    device = torch.device("cpu")
    module.to(device)
    
    # Create test sequences with requires_grad
    z_seq = torch.randn(1, seq_length, latent_dim, device=device, requires_grad=True)
    c_seq = torch.randn(1, seq_length, latent_dim, device=device, requires_grad=True)
    
    # Compute loss
    loss = module.compute_loss(z_seq, c_seq)
    
    # Backward
    loss.backward()
    
    # Check gradients
    assert z_seq.grad is not None
    assert c_seq.grad is not None
    assert module.cpc_proj.weight.grad is not None
    # latent_proj is Linear when projection_dim != latent_dim
    assert isinstance(module.latent_proj, nn.Linear)
    assert module.latent_proj.weight.grad is not None


if __name__ == "__main__":
    test_cpc_module_initialization()
    test_cpc_module_initialization_default_projection()
    test_cpc_module_forward()
    test_cpc_module_compute_loss_basic()
    test_cpc_module_compute_loss_with_mask()
    test_cpc_module_compute_loss_with_episode_boundary()
    test_cpc_module_create_mask_from_dones()
    test_cpc_module_gradient_flow()
    print("All CPC module tests passed!")

