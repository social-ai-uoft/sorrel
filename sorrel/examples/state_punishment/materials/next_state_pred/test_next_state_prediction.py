"""
Unit tests for Next-State Prediction Module.

Tests cover:
1. NextStatePredictionModule basic functionality
2. Image and vector observation types
3. IQN adapter data extraction
4. PPO adapter data extraction
5. Loss computation
6. Gradient flow
7. Edge cases and error handling
"""

import unittest
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# Import from materials package (sorrel.models.pytorch.auxiliary not yet integrated)
from sorrel.examples.state_punishment.materials.next_state_pred.next_state_prediction import (
    NextStatePredictionModule,
    NextStatePredictionAdapter,
    IQNNextStatePredictionAdapter,
    PPONextStatePredictionAdapter,
    create_next_state_predictor,
)


class TestNextStatePredictionModule(unittest.TestCase):
    """Test the core NextStatePredictionModule."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.hidden_size = 128
        self.action_space = 4
        torch.manual_seed(42)
        np.random.seed(42)
    
    def test_image_observation_creation(self):
        """Test module creation with image observations."""
        obs_shape = (3, 84, 84)  # RGB image
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        self.assertEqual(module.obs_type, "image")
        self.assertTrue(module.use_deconv)
        self.assertEqual(module.channels, 3)
        self.assertEqual(module.height, 84)
        self.assertEqual(module.width, 84)
        
        # Check that deconv layers exist
        self.assertIsNotNone(module.deconv1)
        self.assertIsNotNone(module.deconv2)
        self.assertIsNotNone(module.deconv3)
    
    def test_vector_observation_creation(self):
        """Test module creation with vector observations."""
        obs_shape = (512,)  # Flattened feature vector
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        self.assertEqual(module.obs_type, "vector")
        self.assertFalse(module.use_deconv)
        self.assertEqual(module.obs_dim, 512)
        
        # Check that FC layers exist
        self.assertIsNotNone(module.fc_hidden)
        self.assertIsNotNone(module.fc_output)
    
    def test_predict_next_state_vector(self):
        """Test next-state prediction with vector observations."""
        obs_shape = (512,)
        batch_size = 16
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        # Create dummy inputs
        hidden_state = torch.randn(batch_size, self.hidden_size)
        action = torch.randint(0, self.action_space, (batch_size,))
        
        # Predict next state
        predicted = module.predict_next_state(hidden_state, action)
        
        # Check output shape
        self.assertEqual(predicted.shape, (batch_size, 512))
        self.assertFalse(torch.isnan(predicted).any())
        self.assertFalse(torch.isinf(predicted).any())
    
    def test_predict_next_state_image(self):
        """Test next-state prediction with image observations."""
        obs_shape = (3, 84, 84)
        batch_size = 8
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
            use_deconv=True,
        )
        
        # Create dummy inputs
        hidden_state = torch.randn(batch_size, self.hidden_size)
        action = torch.randint(0, self.action_space, (batch_size,))
        
        # Predict next state
        predicted = module.predict_next_state(hidden_state, action)
        
        # Check output shape
        self.assertEqual(predicted.shape, (batch_size, 3, 84, 84))
        self.assertFalse(torch.isnan(predicted).any())
        self.assertFalse(torch.isinf(predicted).any())
    
    def test_compute_loss_vector(self):
        """Test loss computation with vector observations."""
        obs_shape = (512,)
        batch_size = 16
        seq_len = 32
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        # Create dummy data (flattened format)
        hidden_states = torch.randn(batch_size * seq_len, self.hidden_size)
        actions = torch.randint(0, self.action_space, (batch_size * seq_len,))
        next_states = torch.randn(batch_size * seq_len, 512)
        
        # Compute loss
        loss = module.compute_loss(hidden_states, actions, next_states)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)  # MAE is always positive
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_compute_loss_sequence_format(self):
        """Test loss computation with sequence-first format."""
        obs_shape = (512,)
        batch_size = 16
        seq_len = 32
        
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        # Create dummy data (sequence-first format)
        hidden_states = torch.randn(seq_len, batch_size, self.hidden_size)
        actions = torch.randint(0, self.action_space, (seq_len, batch_size))
        next_states = torch.randn(seq_len, batch_size, 512)
        
        # Compute loss
        loss = module.compute_loss(hidden_states, actions, next_states)
        
        # Should work regardless of format
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the module correctly."""
        obs_shape = (512,)
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        # Create dummy data with requires_grad
        hidden_states = torch.randn(16, self.hidden_size, requires_grad=True)
        actions = torch.randint(0, self.action_space, (16,))
        next_states = torch.randn(16, 512)
        
        # Compute loss and backward
        loss = module.compute_loss(hidden_states, actions, next_states)
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(torch.isnan(hidden_states.grad).any())
        
        # Check that module parameters have gradients
        for param in module.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_forward_method(self):
        """Test that forward() is an alias for predict_next_state()."""
        obs_shape = (512,)
        module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=obs_shape,
            device=self.device,
        )
        
        hidden_state = torch.randn(8, self.hidden_size)
        action = torch.randint(0, self.action_space, (8,))
        
        # Both methods should give same result
        pred1 = module.predict_next_state(hidden_state, action)
        pred2 = module.forward(hidden_state, action)
        
        self.assertTrue(torch.allclose(pred1, pred2))
    
    def test_invalid_obs_shape(self):
        """Test that invalid observation shapes raise errors."""
        with self.assertRaises(ValueError):
            # 2D shape is not supported (only 1D vectors or 3D images)
            NextStatePredictionModule(
                hidden_size=self.hidden_size,
                action_space=self.action_space,
                obs_shape=(28, 28),  # 2D not supported
                device=self.device,
            )
    
    def test_different_activations(self):
        """Test module with different activation functions."""
        obs_shape = (512,)
        
        for activation in ["relu", "tanh", "leaky_relu"]:
            module = NextStatePredictionModule(
                hidden_size=self.hidden_size,
                action_space=self.action_space,
                obs_shape=obs_shape,
                device=self.device,
                activation=activation,
            )
            
            hidden_state = torch.randn(8, self.hidden_size)
            action = torch.randint(0, self.action_space, (8,))
            predicted = module.predict_next_state(hidden_state, action)
            
            self.assertEqual(predicted.shape, (8, 512))
            self.assertFalse(torch.isnan(predicted).any())


class TestIQNAdapter(unittest.TestCase):
    """Test IQNNextStatePredictionAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.hidden_size = 128
        self.action_space = 4
        self.obs_dim = 512
        self.batch_size = 8
        self.unroll_len = 20
        
        # Create prediction module
        self.module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=(self.obs_dim,),
            device=self.device,
        )
        
        # Create adapter
        self.adapter = IQNNextStatePredictionAdapter(self.module)
        
        torch.manual_seed(42)
    
    def test_extract_training_data(self):
        """Test data extraction from IQN format."""
        # Create IQN-format data (matching RecurrentIQNModelCPC._train_step)
        states_unroll = torch.randn(self.batch_size, self.unroll_len + 1, self.obs_dim)
        lstm_out = torch.randn(self.unroll_len + 1, self.batch_size, self.hidden_size)
        actions_unroll = torch.randint(0, self.action_space, (self.batch_size, self.unroll_len))
        
        # Extract data
        h_flat, actions_flat, next_flat = self.adapter.extract_training_data(
            states_unroll=states_unroll,
            lstm_out=lstm_out,
            actions_unroll=actions_unroll,
        )
        
        # Check shapes
        expected_n = self.batch_size * self.unroll_len
        self.assertEqual(h_flat.shape, (expected_n, self.hidden_size))
        self.assertEqual(actions_flat.shape, (expected_n,))
        self.assertEqual(next_flat.shape, (expected_n, self.obs_dim))
    
    def test_compute_auxiliary_loss(self):
        """Test auxiliary loss computation with IQN data."""
        # Create IQN-format data
        states_unroll = torch.randn(self.batch_size, self.unroll_len + 1, self.obs_dim)
        lstm_out = torch.randn(self.unroll_len + 1, self.batch_size, self.hidden_size)
        actions_unroll = torch.randint(0, self.action_space, (self.batch_size, self.unroll_len))
        
        # Compute loss
        loss = self.adapter.compute_auxiliary_loss(
            states_unroll=states_unroll,
            lstm_out=lstm_out,
            actions_unroll=actions_unroll,
        )
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_gradient_flow_through_adapter(self):
        """Test that gradients flow through adapter correctly."""
        # Create IQN-format data with requires_grad
        states_unroll = torch.randn(self.batch_size, self.unroll_len + 1, self.obs_dim)
        lstm_out = torch.randn(
            self.unroll_len + 1, self.batch_size, self.hidden_size,
            requires_grad=True
        )
        actions_unroll = torch.randint(0, self.action_space, (self.batch_size, self.unroll_len))
        
        # Compute loss and backward
        loss = self.adapter.compute_auxiliary_loss(
            states_unroll=states_unroll,
            lstm_out=lstm_out,
            actions_unroll=actions_unroll,
        )
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(lstm_out.grad)
        self.assertFalse(torch.isnan(lstm_out.grad).any())


class TestPPOAdapter(unittest.TestCase):
    """Test PPONextStatePredictionAdapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.hidden_size = 128
        self.action_space = 4
        self.obs_shape = (3, 84, 84)  # Image observations
        self.seq_len = 100
        
        # Create prediction module
        self.module = NextStatePredictionModule(
            hidden_size=self.hidden_size,
            action_space=self.action_space,
            obs_shape=self.obs_shape,
            device=self.device,
        )
        
        # Create adapter
        self.adapter = PPONextStatePredictionAdapter(self.module)
        
        torch.manual_seed(42)
    
    def test_extract_training_data(self):
        """Test data extraction from PPO format."""
        # Create PPO-format data (matching RecurrentPPOLSTMCPC.train)
        states = torch.randn(self.seq_len, *self.obs_shape)
        features_all = torch.randn(self.seq_len, self.hidden_size)
        actions = torch.randint(0, self.action_space, (self.seq_len,))
        dones = torch.zeros(self.seq_len)
        
        # Extract data
        hidden_states, actions_seq, next_states = self.adapter.extract_training_data(
            states=states,
            features_all=features_all,
            actions=actions,
            dones=dones,
        )
        
        # Check shapes (should exclude last timestep)
        expected_len = self.seq_len - 1
        self.assertEqual(hidden_states.shape, (expected_len, self.hidden_size))
        self.assertEqual(actions_seq.shape, (expected_len,))
        self.assertEqual(next_states.shape, (expected_len, *self.obs_shape))
    
    def test_compute_auxiliary_loss(self):
        """Test auxiliary loss computation with PPO data."""
        # Create PPO-format data
        states = torch.randn(self.seq_len, *self.obs_shape)
        features_all = torch.randn(self.seq_len, self.hidden_size)
        actions = torch.randint(0, self.action_space, (self.seq_len,))
        
        # Compute loss
        loss = self.adapter.compute_auxiliary_loss(
            states=states,
            features_all=features_all,
            actions=actions,
        )
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_dimension_mismatch_error(self):
        """Test that dimension mismatches are caught."""
        # Create mismatched data
        states = torch.randn(self.seq_len, *self.obs_shape)
        features_all = torch.randn(self.seq_len - 5, self.hidden_size)  # Wrong length!
        actions = torch.randint(0, self.action_space, (self.seq_len,))
        
        # Should raise assertion error
        with self.assertRaises(AssertionError):
            self.adapter.extract_training_data(
                states=states,
                features_all=features_all,
                actions=actions,
            )
    
    def test_gradient_flow_through_adapter(self):
        """Test that gradients flow through adapter correctly."""
        # Create PPO-format data with requires_grad
        states = torch.randn(self.seq_len, *self.obs_shape)
        features_all = torch.randn(self.seq_len, self.hidden_size, requires_grad=True)
        actions = torch.randint(0, self.action_space, (self.seq_len,))
        
        # Compute loss and backward
        loss = self.adapter.compute_auxiliary_loss(
            states=states,
            features_all=features_all,
            actions=actions,
        )
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(features_all.grad)
        self.assertFalse(torch.isnan(features_all.grad).any())


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_create_next_state_predictor_iqn(self):
        """Test creating predictor + adapter for IQN."""
        predictor, adapter = create_next_state_predictor(
            hidden_size=128,
            action_space=4,
            obs_shape=(512,),
            device="cpu",
            model_type="iqn",
        )
        
        self.assertIsInstance(predictor, NextStatePredictionModule)
        self.assertIsInstance(adapter, IQNNextStatePredictionAdapter)
        self.assertIs(adapter.prediction_module, predictor)
    
    def test_create_next_state_predictor_ppo(self):
        """Test creating predictor + adapter for PPO."""
        predictor, adapter = create_next_state_predictor(
            hidden_size=256,
            action_space=6,
            obs_shape=(3, 84, 84),
            device="cpu",
            model_type="ppo",
        )
        
        self.assertIsInstance(predictor, NextStatePredictionModule)
        self.assertIsInstance(adapter, PPONextStatePredictionAdapter)
        self.assertIs(adapter.prediction_module, predictor)
    
    def test_create_next_state_predictor_invalid_type(self):
        """Test that invalid model_type raises error."""
        with self.assertRaises(ValueError):
            create_next_state_predictor(
                hidden_size=128,
                action_space=4,
                obs_shape=(512,),
                device="cpu",
                model_type="invalid",
            )


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end integration scenarios."""
    
    def test_iqn_training_loop_simulation(self):
        """Simulate IQN training loop with auxiliary loss."""
        # Setup
        device = torch.device("cpu")
        hidden_size = 128
        action_space = 4
        obs_dim = 512
        batch_size = 8
        unroll_len = 20
        
        # Create predictor and adapter
        predictor, adapter = create_next_state_predictor(
            hidden_size=hidden_size,
            action_space=action_space,
            obs_shape=(obs_dim,),
            device=device,
            model_type="iqn",
        )
        
        # Simulate IQN training step
        states_unroll = torch.randn(batch_size, unroll_len + 1, obs_dim)
        lstm_out = torch.randn(unroll_len + 1, batch_size, hidden_size, requires_grad=True)
        actions_unroll = torch.randint(0, action_space, (batch_size, unroll_len))
        
        # Compute auxiliary loss
        aux_loss = adapter.compute_auxiliary_loss(
            states_unroll=states_unroll,
            lstm_out=lstm_out,
            actions_unroll=actions_unroll,
        )
        
        # Simulate total loss (dummy loss is non-leaf after **2; retain_grad to check grad)
        iqn_loss = torch.randn(1, requires_grad=True) ** 2  # Dummy IQN loss
        iqn_loss.retain_grad()
        total_loss = iqn_loss + 0.5 * aux_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist everywhere
        self.assertIsNotNone(lstm_out.grad)
        self.assertIsNotNone(iqn_loss.grad)
        for param in predictor.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_ppo_training_loop_simulation(self):
        """Simulate PPO training loop with auxiliary loss."""
        # Setup
        device = torch.device("cpu")
        hidden_size = 256
        action_space = 6
        obs_shape = (3, 84, 84)
        seq_len = 100
        
        # Create predictor and adapter
        predictor, adapter = create_next_state_predictor(
            hidden_size=hidden_size,
            action_space=action_space,
            obs_shape=obs_shape,
            device=device,
            model_type="ppo",
        )
        
        # Simulate PPO training step
        states = torch.randn(seq_len, *obs_shape)
        features_all = torch.randn(seq_len, hidden_size, requires_grad=True)
        actions = torch.randint(0, action_space, (seq_len,))
        
        # Compute auxiliary loss
        aux_loss = adapter.compute_auxiliary_loss(
            states=states,
            features_all=features_all,
            actions=actions,
        )
        
        # Simulate total loss (dummy loss is non-leaf after **2; retain_grad to check grad)
        ppo_loss = torch.randn(1, requires_grad=True) ** 2  # Dummy PPO loss
        ppo_loss.retain_grad()
        total_loss = ppo_loss + 1.0 * aux_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist everywhere
        self.assertIsNotNone(features_all.grad)
        self.assertIsNotNone(ppo_loss.grad)
        for param in predictor.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
    
    def test_parameter_sharing(self):
        """Test that module parameters can be added to model optimizer."""
        device = torch.device("cpu")
        predictor, adapter = create_next_state_predictor(
            hidden_size=128,
            action_space=4,
            obs_shape=(512,),
            device=device,
            model_type="iqn",
        )
        
        # Get all parameters
        params = list(predictor.parameters())
        
        # Should have multiple parameters
        self.assertGreater(len(params), 0)
        
        # All should be tensors with requires_grad=True
        for param in params:
            self.assertIsInstance(param, torch.Tensor)
            self.assertTrue(param.requires_grad)
        
        # Should be able to create optimizer
        optimizer = torch.optim.Adam(params, lr=1e-4)
        self.assertIsNotNone(optimizer)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
