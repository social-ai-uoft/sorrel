"""Basic test for asynchronous training functionality."""

import time
import unittest
import numpy as np

from sorrel.models.base_model import BaseModel
from sorrel.training import AsyncTrainer


class SimpleTestModel(BaseModel):
    """Simple model for testing async training."""
    
    def __init__(self):
        super().__init__(input_size=10, action_space=4, memory_size=100, epsilon=0.1)
        self.train_count = 0
        
    def take_action(self, state):
        return np.random.randint(0, self.action_space)
        
    def train_step(self):
        """Simulate a training step."""
        self.train_count += 1
        time.sleep(0.01)  # Simulate some work
        return np.array(1.0)


class TestAsyncTraining(unittest.TestCase):
    
    def test_async_trainer_starts_and_stops(self):
        """Test that async trainer starts and stops cleanly."""
        model = SimpleTestModel()
        trainer = AsyncTrainer(model, train_interval=0.01)
        
        # Start the trainer
        trainer.start()
        self.assertTrue(trainer.running)
        
        # Let it run for a bit
        time.sleep(0.1)
        
        # Stop the trainer
        trainer.stop()
        self.assertFalse(trainer.running)
        
    def test_async_trainer_performs_training(self):
        """Test that async trainer actually performs training steps."""
        model = SimpleTestModel()
        trainer = AsyncTrainer(model, train_interval=0.01)
        
        initial_count = model.train_count
        
        # Start the trainer
        trainer.start()
        
        # Wait for some training steps
        time.sleep(0.15)
        
        # Stop the trainer
        trainer.stop()
        
        # Verify training happened
        self.assertGreater(model.train_count, initial_count)
        self.assertGreater(trainer.total_steps, 0)
        
    def test_async_trainer_stats(self):
        """Test that async trainer tracks statistics correctly."""
        model = SimpleTestModel()
        trainer = AsyncTrainer(model, train_interval=0.01)
        
        # Start the trainer
        trainer.start()
        
        # Wait for some training
        time.sleep(0.1)
        
        # Get stats
        stats = trainer.get_stats()
        
        # Stop the trainer
        trainer.stop()
        
        # Verify stats
        self.assertGreater(stats['total_steps'], 0)
        self.assertGreater(stats['total_loss'], 0)
        self.assertGreater(stats['avg_loss'], 0)
        
    def test_concurrent_training_and_acting(self):
        """Test that training can happen while 'acting' in main thread."""
        model = SimpleTestModel()
        trainer = AsyncTrainer(model, train_interval=0.001)
        
        # Start async training
        trainer.start()
        
        # Simulate acting in the main thread
        for _ in range(10):
            state = np.random.rand(10)
            action = model.take_action(state)
            time.sleep(0.01)  # Simulate environment step
        
        # Stop trainer
        trainer.stop()
        
        # Verify that training happened while we were "acting"
        stats = trainer.get_stats()
        self.assertGreater(stats['total_steps'], 0)


if __name__ == '__main__':
    unittest.main()
