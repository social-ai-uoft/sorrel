import numpy as np

from sorrel.agents import Agent
from sorrel.environment import Environment
from sorrel.worlds import Gridworld


class ThreadsafeEnvironment[W: Gridworld](Environment[W]):
    """Opt-in environment variant to route model calls through threadsafe wrappers."""

    def _model_start_epoch_action(self, agent: Agent, epoch: int) -> None:
        start_action = getattr(agent.model, "threadsafe_start_epoch_action", None)
        if callable(start_action):
            start_action(epoch=epoch)
            return
        agent.model.start_epoch_action(epoch=epoch)

    def _model_end_epoch_action(self, agent: Agent, epoch: int) -> None:
        end_action = getattr(agent.model, "threadsafe_end_epoch_action", None)
        if callable(end_action):
            end_action(epoch=epoch)
            return
        agent.model.end_epoch_action(epoch=epoch)

    def _model_train_step(
        self, agent: Agent
    ) -> np.ndarray:  # pyright: ignore[reportIncompatibleMethodOverride]
        train_step = getattr(agent.model, "threadsafe_train_step", None)
        if callable(train_step):
            return train_step()  # pyright: ignore[reportReturnType]
        return agent.model.train_step()
