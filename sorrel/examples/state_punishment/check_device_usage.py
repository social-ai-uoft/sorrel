"""Script to verify that MPS device is actually being used during training."""

import torch
import time
import numpy as np
from sorrel.examples.state_punishment.environment_setup import setup_environments
from sorrel.examples.state_punishment.config import create_config


def check_model_device(model, device_name):
    """Check if model parameters are on the specified device."""
    device = torch.device(device_name)
    all_on_device = True
    for name, param in model.named_parameters():
        if param.device != device:
            print(f"  WARNING: {name} is on {param.device}, not {device}")
            all_on_device = False
    return all_on_device


def test_device_performance(device_name="cpu", num_epochs=10):
    """Test training performance on different devices."""
    print(f"\n{'='*60}")
    print(f"Testing device: {device_name}")
    print(f"{'='*60}")
    
    # Validate device
    if device_name == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, skipping test")
        return None
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return None
    
    device = torch.device(device_name)
    
    # Create config
    config = create_config(
        num_agents=3,
        epochs=num_epochs,
        device=device_name,
    )
    
    # Setup environment
    multi_env, _, _ = setup_environments(config, False, 0.2, False)
    
    # Check if models are on correct device
    print(f"\nChecking model device placement:")
    for i, env in enumerate(multi_env.individual_envs):
        agent = env.agents[0]
        model = agent.model
        print(f"Agent {i}:")
        print(f"  Model device attribute: {model.device}")
        print(f"  qnetwork_local on device: {next(model.qnetwork_local.parameters()).device}")
        print(f"  qnetwork_target on device: {next(model.qnetwork_target.parameters()).device}")
        
        # Verify all parameters are on device
        if check_model_device(model.qnetwork_local, device_name):
            print(f"  ✓ All local network parameters on {device_name}")
        if check_model_device(model.qnetwork_target, device_name):
            print(f"  ✓ All target network parameters on {device_name}")
    
    # Test training speed
    print(f"\nTesting training speed (running {num_epochs} epochs):")
    start_time = time.time()
    
    # Run a few epochs
    for epoch in range(num_epochs):
        multi_env.reset()
        for env in multi_env.individual_envs:
            for agent in env.agents:
                agent.model.start_epoch_action(epoch=epoch)
        
        # Run a few steps
        for turn in range(10):
            for env in multi_env.individual_envs:
                for agent in env.agents:
                    state = env.get_observation(agent)
                    action = agent.take_action(state)
                    env.step(agent, action)
                    # Train if memory is full enough
                    if len(agent.model.memory) > agent.model.batch_size:
                        agent.model.train_step()
    
    elapsed_time = time.time() - start_time
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Time per epoch: {elapsed_time/num_epochs:.3f} seconds")
    
    return elapsed_time


def main():
    """Run device performance tests."""
    print("Device Usage Verification")
    print("=" * 60)
    
    # Check available devices
    print("\nAvailable devices:")
    print(f"  CPU: Always available")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Devices: {torch.cuda.device_count()}")
    print(f"  MPS: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    
    # Test CPU
    # cpu_time = test_device_performance("cpu", num_epochs=5)
    
    # Test MPS if available
    if torch.backends.mps.is_available():
        mps_time = test_device_performance("mps", num_epochs=5)
        if cpu_time and mps_time:
            speedup = cpu_time / mps_time
            print(f"\n{'='*60}")
            print(f"Performance Comparison:")
            print(f"  CPU: {cpu_time:.2f}s")
            print(f"  MPS: {mps_time:.2f}s")
            if speedup > 1:
                print(f"  MPS is {speedup:.2f}x faster")
            else:
                print(f"  MPS is {1/speedup:.2f}x slower (this can happen with small models)")
            print(f"{'='*60}")
    
    print("\nNote: MPS performance can vary significantly:")
    print("  - Small models/batches may be slower due to overhead")
    print("  - First few operations are slower (kernel compilation)")
    print("  - Large batches typically show better speedup")


if __name__ == "__main__":
    main()


