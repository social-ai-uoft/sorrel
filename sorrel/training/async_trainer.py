"""Asynchronous training infrastructure for Sorrel.

This module provides thread-safe asynchronous training capabilities, allowing
agents to collect experiences while training happens in parallel.
"""

import threading
import time
from typing import Optional

from sorrel.models.base_model import BaseModel


class AsyncTrainer:
    """Asynchronous trainer that runs model training in a background thread.
    
    This allows agents to continuously collect experiences while training
    happens in parallel, maximizing throughput.
    
    Attributes:
        model: The model to train asynchronously
        train_interval: Minimum seconds between train steps (0 = as fast as possible)
        max_steps_per_sec: Optional cap on training steps per second
        running: Whether the trainer is currently running
        total_steps: Total number of training steps completed
        total_loss: Cumulative loss across all training steps
    """
    
    def __init__(
        self,
        model: BaseModel,
        train_interval: float = 0.0,
        max_steps_per_sec: Optional[int] = None
    ):
        """Initialize the asynchronous trainer.
        
        Args:
            model: The model to train asynchronously
            train_interval: Minimum seconds between train steps (0 = as fast as possible)
            max_steps_per_sec: Optional cap on training steps per second
        """
        self.model = model
        self.train_interval = train_interval
        self.max_steps_per_sec = max_steps_per_sec
        
        self.running = False
        self.thread = None
        self.lock = threading.Lock()  # For thread-safe operations
        
        self.total_steps = 0
        self.total_loss = 0.0
        self._last_step_time = 0.0
        
    def start(self) -> None:
        """Start the background training thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._training_loop, daemon=True)
        self.thread.start()
        
    def stop(self) -> None:
        """Stop the background training thread."""
        if not self.running:
            return
            
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
            
    def _training_loop(self) -> None:
        """Background thread training loop."""
        while self.running:
            try:
                # Throttle if max_steps_per_sec is set
                if self.max_steps_per_sec is not None:
                    min_interval = 1.0 / self.max_steps_per_sec
                    elapsed = time.time() - self._last_step_time
                    if elapsed < min_interval:
                        time.sleep(min_interval - elapsed)
                
                # Wait for minimum interval
                if self.train_interval > 0:
                    time.sleep(self.train_interval)
                
                # Check if model has enough samples in memory
                # PyTorch models typically need batch_size samples
                skip_training = False
                if hasattr(self.model, 'memory') and hasattr(self.model, 'batch_size'):
                    if len(self.model.memory) < self.model.batch_size:
                        skip_training = True
                        time.sleep(0.01)  # Short sleep if not enough samples yet
                
                if not skip_training:
                    # Use model's lock if available, otherwise use AsyncTrainer's lock
                    model_lock = getattr(self.model, '_lock', self.lock)
                    
                    # Perform training step with lock
                    with model_lock:
                        loss = self.model.train_step()
                    
                    # Update statistics (outside lock to minimize lock time)
                    self.total_steps += 1
                    self.total_loss += float(loss)
                    self._last_step_time = time.time()
                
            except Exception as e:
                # Log error but don't crash the thread
                print(f"AsyncTrainer error: {e}")
                time.sleep(0.1)
                
    def get_stats(self) -> dict:
        """Get training statistics.
        
        Returns:
            dict with keys: 'total_steps', 'total_loss', 'avg_loss'
        """
        with self.lock:
            avg_loss = self.total_loss / max(1, self.total_steps)
            return {
                'total_steps': self.total_steps,
                'total_loss': self.total_loss,
                'avg_loss': avg_loss
            }
            
    def reset_stats(self) -> None:
        """Reset training statistics."""
        with self.lock:
            self.total_steps = 0
            self.total_loss = 0.0
