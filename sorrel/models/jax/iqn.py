# JAX Imports for automatic differentiation and numerical operations
import jax
import jax.numpy as jnp
import jax.random

# Optimization library specific to JAX
import optax

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

# Sorrel imports
from sorrel.buffers import Buffer
from sorrel.models import SorrelModel
from sorrel.models.jax.jax_base import IQNetwork, compute_quantile_td_target_from_state, quantile_bellman_residual_loss

class IQNAgent(SorrelModel):
    """
    doubleDQN: An implementation of the Double Deep Q-Network (DDQN) algorithm.

    This class encapsulates the functionality of the Double DQN algorithm, an enhancement over the standard DQN.
    It addresses the overestimation of Q-values by decoupling the selection of actions from their evaluation.

    Attributes:
    - local_model: A QNetwork instance used for selecting actions and training.
    - target_model: A QNetwork instance used for evaluating the actions chosen by the local model.
    - replay_buffer: A buffer for storing experiences. Can be either standard or prioritized (PER).
    - per: A boolean indicating whether to use Prioritized Experience Replay (PER).
    - optimizer: An optimizer for updating the local model's parameters.
    - local_model_params: The parameters of the local model.
    - target_model_params: The parameters of the target model.
    - optimizer_state: The state of the optimizer.

    Initialization:
    - The constructor initializes two QNetwork instances: one as the local model and the other as the target model.
    - It also initializes a replay buffer, which can be either a standard buffer or a prioritized buffer based on the `per` flag.
    - The models are initialized with dummy data to set up their parameters. The shape of the dummy data should match the expected input shape.
    - An optimizer (Adam in this case) is initialized with the parameters of the local model.

    Usage:
    - The local model is used to select actions and is regularly trained with experiences from the replay buffer.
    - The target model's parameters are periodically updated with those of the local model,
      reducing the overestimation bias inherent in standard DQN.
    - If `per` is set to True, the replay buffer prioritizes experiences based on their TD error,
      potentially leading to more efficient learning.

    Notes:
    - The separation of action selection (local model) and action evaluation (target model) is key to the Double DQN approach.
    - The use of PER is optional and can be beneficial in complex environments where some experiences are significantly more important.
    - The models and optimizer should be appropriately configured to match the specifics of the environment and task.
    - Regular synchronization of the local and target models is crucial for the algorithm's stability and performance.

    Example:
    ```
    model = doubleDQN(per=True)
    # Use model.local_model and agent.target_model for training and action selection
    ```
    """

    def __init__(
        self,
        input_size=1134,
        action_space=4,
        lr=0.001,
        gamma=0.99,
        per=False,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0006,
        memory_size=5000,
        seed=0
    ):
        self.input_size = input_size
        self.action_space = action_space
        self.lr = lr
        self.per = per
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory_size = memory_size

        self.local_model = IQNetwork(action_space=action_space)
        self.target_model = IQNetwork(action_space=action_space)

        # Initialize RNG key for the model
        self.rng_key = jax.random.PRNGKey(seed)

        self.memory = Buffer(capacity=memory_size, obs_shape=(input_size, ))

        # Initialize models with dummy data
        dummy_input = jnp.zeros((1, input_size))  # Should be part of the constructor

        # Initialize the parameters of the models
        self.local_model_params = self.local_model.init(
            jax.random.PRNGKey(seed), dummy_input, self.rng_key
        )["params"]
        self.target_model_params = self.target_model.init(
            jax.random.PRNGKey(seed+1), dummy_input, self.rng_key
        )["params"]

        # Initialize the optimizer with the parameters of the local model
        self.optimizer = optax.adam(lr)

        self.train_state: TrainState = TrainState.create(
            apply_fn=self.local_model.apply,
            params=self.local_model_params,
            tx=self.optimizer,
        )
        self.target_train_state: TrainState = TrainState.create(
            apply_fn=self.target_model.apply,
            params=self.target_model_params,
            tx=optax.set_to_zero,
        )


    def take_action(self, state, epsilon):
        """
        Selects an action based on the current state, using an epsilon-greedy strategy.

        This method decides between exploration and exploitation using the epsilon value.
        With probability epsilon, it chooses a random action (exploration),
        and with probability 1 - epsilon, it chooses the best action based on the model's predictions (exploitation).

        Parameters:
        - state: The current state of the environment.
        - epsilon: The probability of choosing a random action.
        - rng: The JAX random number generator key.

        Things to Check:
        - Ensure that the RNG key is split or advanced correctly instead of being reseeded each time.
        In JAX, you typically use `jax.random.split` to get a new subkey.
        - Validate that the state is correctly reshaped and can be processed by the model during exploitation.
        - Confirm that the action space defined by `action_space` aligns with the environment's action space.
        - Check that the `jax.random.uniform` and `jax.random.randint` functions are used correctly to 
          generate random numbers and actions.

        Potential Improvements:
        - Modify the RNG handling to use `jax.random.split` for generating new keys.
        This will maintain the stochasticity without the need to reset the seed each time.
        - If the state requires specific preprocessing before being fed into the model,
        ensure that it's handled appropriately within this method or prior to calling it.
        - Consider parameterizing the number of actions if it varies across different environments or parts of your application.

        Returns:
        - The selected action, either random or the best according to the model.
        """

        def action_fn(state, model: TrainState, key, action_space, epsilon):
            # Split the RNG key
            rng_key, rng_key_action, rng_key_taus = jax.random.split(key, 3)

            # Generate a random number using JAX's random number generator
            random_number = jax.random.uniform(rng_key_action, shape=())

            if random_number < epsilon:
                # Exploration: choose a random action
                rng_key, rng_key_random = jax.random.split(rng_key)
                action = jax.random.randint(
                    rng_key_random, shape=(), minval=0, maxval=action_space
                )
            else:
                # Exploitation: choose the best action based on model prediction
                q_values = model.apply_fn(model.params, state, rng_key_taus)
                action = jnp.argmax(jnp.mean(q_values, axis=-1), axis=-1)[0]

            return action, rng_key
        
        action, self.rng_key = jax.jit(action_fn, static_argnums=(3))(
            state, self.train_state, self.rng_key, self.action_space, epsilon)
        return action


    def train_step(self, batch_size, soft_update=True):
        """
        Perform a training step, with control over batch size, discount factor, and update type of the target model.

        Parameters:
        - batch_size: Determines the number of samples to be used in each training step. A larger batch size
        generally leads to more stable gradient estimates, but it requires more memory and computational power.
        - gamma: The discount factor used in the calculation of the return. It determines the importance of
        future rewards. A value of 0 makes the agent short-sighted by only considering current rewards, while
        a value close to 1 makes it strive for long-term high rewards.
        - soft_update: A boolean flag that controls the type of update performed on the target model's parameters.
        If True, a soft update is performed, where the target model parameters are gradually blended with the
        local model parameters. If False, a hard update is performed, directly copying the local model parameters
        to the target model.

        The function conducts a single step of training, which involves sampling a batch of experiences,
        computing the loss, updating the model parameters based on the computed gradients, and then updating
        the target model parameters.

        The choice between soft and hard updates for the target model is crucial for the stability of the training process.
        Soft updates provide more stable but slower convergence, while hard updates can lead to faster convergence
        but might cause instability in training dynamics.
        """

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones, valid = batch

        def update_fn(states, actions, rewards, next_states, dones, valid, model: TrainState, target_model: TrainState, keys):
            next_key, key = jax.random.split(keys)
            dones = dones.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)

            def loss_fn(params):
                quantile_target = compute_quantile_td_target_from_state(
                    next_states, rewards, dones, target_model, self.gamma, self.rng_key
                )
                loss, loss_dict = quantile_bellman_residual_loss(
                    states, quantile_target, model, valid, self.rng_key

                )

                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, gradients = grad_fn(model.params)
            model = model.apply_gradients(grads=gradients)

            # Soft update

            tau = 0.01
            target_model.params = jax.tree_map(
                lambda t, l: tau * l + (1 - tau) * t,
                target_model.params,
                model.params,
            )

            return loss, next_key, model, target_model
        
        loss, self.rng_key, self.train_state, self.target_train_state = jax.jit(jax.vmap(update_fn, in_axes=(
            0, 0, 0, 0, 0, 0, None, None, None)))(
            states, actions, rewards, next_states, dones, valid, self.train_state, self.target_train_state, self.rng_key)
        return loss


    def hard_update(self):
        """
        Perform a hard update on the target model parameters.

        This function directly copies the parameters from the local model to the
        target model. Unlike a soft update, which gradually blends the parameters
        of the local and target models, a hard update sets the target model
        parameters to be exactly equal to the local model parameters. This is
        typically done less frequently than soft updates and can be used to
        periodically reset the target model with the latest local model parameters.

        Usage:
            This function should be called at specific intervals (e.g., every 100
            epochs) during the training process, depending on the specific needs
            and dynamics of the training regimen.

        Note:
            Frequent hard updates can lead to more volatile training dynamics,
            whereas infrequent updates might cause the target model to lag too far
            behind the local model. The frequency of hard updates should be chosen
            carefully based on empirical results and the specific characteristics
            of the model and training data.
        """
        self.target_model_params = jax.tree_map(
            lambda l: l,
            self.local_model_params,
        )

