from sorrel.models.pytorch.iqn import iRainbowModel
from sorrel.models.threadsafe_base_model import ThreadsafeBaseModel
from sorrel.threadsafe.buffers import ThreadsafeBuffer


class ThreadsafePyTorchIQN(iRainbowModel, ThreadsafeBaseModel):
    """Threadsafe opt-in IQN model.

    This class keeps default IQN untouched and provides an alternative for concurrent
    actor/learner workflows.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = ThreadsafeBuffer(
            capacity=self.memory.capacity,
            obs_shape=self.memory.obs_shape,
            n_frames=self.n_frames,
        )
