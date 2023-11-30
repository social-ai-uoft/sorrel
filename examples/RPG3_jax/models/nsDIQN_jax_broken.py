# JAX Imports for automatic differentiation and numerical operations
import jax
import jax.numpy as jnp
import jax.random
from flax import linen as nn

# Standard Python library imports for data structures and randomness
import heapq
from collections import deque
import random

# Numerical and array operations
import numpy as np

# Optimization library specific to JAX
import optax

# Utility for time measurements
import time


"""
Usage:
    This example demonstrates how to initialize and use the doubleDQN model for reinforcement learning tasks. The model is created with specified parameters for the input size, number of actions, learning rate, discount factor (gamma), and settings for Prioritized Experience Replay (PER).

    Parameters:
    - input_size: The size of the input layer, set to 1134.
    - number_of_actions: The number of possible actions the agent can take, set to 4.
    - lr: Learning rate for the optimizer, set to 0.001.
    - gamma: Discount factor for future rewards, set to 0.97 (default is 0.99).
    - per: Boolean flag to enable or disable Prioritized Experience Replay, enabled here.
    - alpha: Determines the degree of prioritization in PER, set to 0.6 (default is also 0.6).
    - beta: Controls the bias compensation in importance sampling weights for PER, set to 0.05 (default is 0.4).
    - beta_increment: The rate at which beta is increased, set to 0.0006 (default is 0.0006).
    - capacity: The capacity of the replay buffer, set to 5000 (default is 50000).

Example:
    model = doubleDQN(
        input_size=1134,
        number_of_actions=4,
        lr=0.001,
        gamma=0.97,
        per=True,
        alpha=0.6,
        beta=0.05,
        beta_increment=0.0006,
        capacity=5000,
    )
"""


class ReplayBuffer:
    """
    ReplayBuffer for Reinforcement Learning
    """

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sampled_experiences = random.sample(self.buffer, batch_size)

        # Convert tensors to NumPy arrays and ensure consistent shapes
        states = np.array([exp[0].numpy() for exp in sampled_experiences])
        actions = np.array([exp[1] for exp in sampled_experiences])
        rewards = np.array([exp[2] for exp in sampled_experiences])
        next_states = np.array([exp[3].numpy() for exp in sampled_experiences])
        dones = np.array([exp[4] for exp in sampled_experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, default_priority=1.0):
        self.buffer = []
        self.priority_queue = []  # Min-heap for efficient priority management
        self.capacity = capacity
        self.default_priority = default_priority
        self.size = 0
        self.total_priority = 0  # Initialize total_priority

    def add(self, state, action, reward, next_state, done):
        priority = self.default_priority
        experience = (state, action, reward, next_state, done, priority)
        if self.size < self.capacity:
            heapq.heappush(self.priority_queue, (priority, self.size))
            self.buffer.append(experience)
            self.size += 1
        else:
            _, min_priority_index = heapq.heappop(self.priority_queue)
            self.total_priority -= self.buffer[min_priority_index][5]
            self.buffer[min_priority_index] = experience
            heapq.heappush(self.priority_queue, (priority, min_priority_index))
        self.total_priority += priority  # Update total_priority when adding

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        priorities = [exp[5] for exp in self.buffer]
        scaled_priorities = np.array(priorities) ** alpha
        probabilities = scaled_priorities / np.sum(scaled_priorities)

        sampled_indices = np.random.choice(self.size, batch_size, p=probabilities)
        sampled_experiences = [self.buffer[idx] for idx in sampled_indices]

        # Calculate importance-sampling weights
        weights = [
            (1.0 / (self.size * probabilities[idx])) ** beta for idx in sampled_indices
        ]
        max_weight = max(weights)
        normalized_weights = [w / max_weight for w in weights]

        return sampled_experiences, normalized_weights, sampled_indices

    def update_priorities(self, indices, priorities):
        for i, priority in zip(indices, priorities):
            # Update the experience with the new priority
            self.buffer[i] = self.buffer[i][:5] + (priority,)

    def __len__(self):
        return len(self.buffer)


class NoisyDense(nn.Module):
    features: int  # Number of output features
    sigma_init: float = 0.017  # Standard deviation for noise
    bias: bool = True

    """
    Custom Noisy Dense Layer

    Args:
        x (jax.numpy.ndarray): The input tensor.
        rng_key (jax.random.PRNGKey): A random number generator key.

    Returns:
        jax.numpy.ndarray: The output tensor after applying the noisy dense layer.
    """

    @nn.compact
    def __call__(self, x, rng_key):
        input_features = x.shape[-1]

        # Initialize weights and biases
        weight = self.param(
            "weight", nn.initializers.lecun_normal(), (self.features, input_features)
        )
        bias = (
            self.param("bias", nn.initializers.zeros, (self.features,))
            if self.bias
            else None
        )

        # Initialize noise parameters
        sigma_weight = self.param(
            "sigma_weight",
            nn.initializers.constant(self.sigma_init),
            (self.features, input_features),
        )
        sigma_bias = (
            self.param(
                "sigma_bias",
                nn.initializers.constant(self.sigma_init),
                (self.features,),
            )
            if self.bias
            else None
        )

        # Generate noise

        seed = int(time.time() * 1000)
        rng_key = jax.random.PRNGKey(seed)

        epsilon_weight = jax.random.normal(rng_key, (self.features, input_features))

        seed = int(time.time() * 1500)
        rng_key = jax.random.PRNGKey(seed)

        epsilon_bias = (
            jax.random.normal(rng_key, (self.features,)) if self.bias else None
        )

        # Apply noise
        weight += sigma_weight * epsilon_weight
        bias = bias + sigma_bias * epsilon_bias if self.bias else bias

        # Linear transformation
        return jnp.dot(x, weight.T) + bias if self.bias else jnp.dot(x, weight.T)

class QNetwork(nn.Module):
    """
    QNetwork: A neural network architecture for Q-learning.

    This class defines a Q-network commonly used in Deep Q-Networks (DQN) for reinforcement learning.
    The network estimates the Q-values for each action in a given state. It consists of multiple layers
    and offers an option to include a noisy dense layer for exploration.

    Structure:
    - The network has several fully connected dense layers with ReLU activations.
    - The input `x` is first passed through two dense layers of 512 units each.
    - Optionally, a noisy dense layer can be included to add stochasticity to the action selection,
      which can be useful for exploration in reinforcement learning.
      The standard deviation for the noise can be adjusted via the `std_init` parameter.
    - If the noisy layer is not used, an additional regular dense layer of 512 units is applied.
    - The final part of the network splits into two streams: one for the value function and one for the advantage function.
    - The value stream reduces to a single output, representing the value of the state.
    - The advantage stream outputs a value for each possible action.
    - The final Q-values are computed by combining the value and the advantage streams,
      with the mean advantage subtracted for numerical stability.

    Parameters:
    - `noisy` (bool): Whether to include a noisy dense layer in the network.
      When set to `True`, it adds stochasticity to the action selection process, aiding exploration.

    Usage:
    - The Q-network is used to estimate the Q-values for each action in a given state.
    - The `noisy` parameter can be toggled based on whether exploration is desired in the action selection.
    - This architecture is typical for DQN-based reinforcement learning algorithms.

    Example:
    ```
    q_network = QNetwork()
    state = jnp.array([...])  # Example state
    q_values = q_network(state, noisy=True)
    ```

    Notes:
    - The effectiveness of a noisy layer versus a regular dense layer can depend on the specific learning environment and task.
    - Adjusting the `std_init` parameter in the `NoisyDense` layer can significantly impact exploration efficiency.
    - The separation of value and advantage streams is a technique from the Dueling DQN architecture,
      which can lead to more stable and efficient learning.
    """

    input_size: int
    number_of_actions: int
    noisy: bool = True
    num_quantiles: int = 32  

    @staticmethod
    def quantile_embedding(quantiles, embedding_dim=64):
        # Embed quantiles into a higher-dimensional space
        seq = jnp.arange(embedding_dim, dtype=jnp.float32)
        embedding = jnp.cos(jnp.pi * quantiles[:, None] * seq[None, :])
        return embedding

    @nn.compact
    def __call__(self, x, rng_key=None):
        if self.noisy:
            noisy_rng_key = jax.random.PRNGKey(int(time.time() * 1000))
            # _, noisy_rng_key = jax.random.split(rng_key)
            x = NoisyDense(features=512)(x, noisy_rng_key)
            x = nn.relu(x)
            noisy_rng_key = jax.random.PRNGKey(int(time.time() * 1500))
            # _, noisy_rng_key = jax.random.split(rng_key)
            x = NoisyDense(features=512)(x, noisy_rng_key)
            x = nn.relu(x)
            noisy_rng_key = jax.random.PRNGKey(int(time.time() * 2500))
            # _, noisy_rng_key = jax.random.split(rng_key)
            x = NoisyDense(features=256)(x, noisy_rng_key)
            x = nn.relu(x)
        else:
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            x = nn.Dense(512)(x)
            x = nn.relu(x)
            x = nn.Dense(256)(x)
            x = nn.relu(x)

        # Next step prediction
        predicted_next_state = nn.Dense(self.input_size)(x)

        # Sample quantiles

        seed = int(time.time() * 1000)
        rng_key = jax.random.PRNGKey(seed)
        quantiles = jax.random.uniform(rng_key, shape=(self.num_quantiles,))

        quantile_embeddings = self.quantile_embedding(quantiles)

        # Expand quantile embeddings to match (batch_size, num_quantiles, embedding_dim)
        batch_size = x.shape[0]
        quantile_embeddings_expanded = jnp.repeat(quantile_embeddings[None, :, :], batch_size, axis=0)

        # Concatenate along the last dimension
        # x: (batch_size, num_quantiles, state_dim)
        # quantile_embeddings_expanded: (batch_size, num_quantiles, embedding_dim)
        state_quantile_combined = jnp.concatenate([x[:, None, :].repeat(self.num_quantiles, axis=1), quantile_embeddings_expanded], axis=2)



        value = nn.Dense(1)(state_quantile_combined)  # Shape: (batch_size, num_quantiles, 1)
        advantage = nn.Dense(self.number_of_actions)(state_quantile_combined)  # Shape: (batch_size, num_quantiles, number_of_actions)

        # Combine value and advantage to get final quantile values
        # Advantage mean is subtracted for stability
        q_values = value + advantage - jnp.mean(advantage, axis=2, keepdims=True)


        return q_values, quantiles, predicted_next_state

class doubleDQN:
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
        number_of_actions=4,
        lr=0.001,
        gamma=0.99,
        per=False,
        alpha=0.6,
        beta=0.4,
        beta_increment=0.0006,
        capacity=5000,
        noisy=True,
        next_state_prediction = True
    ):
        self.input_size = input_size
        self.number_of_actions = number_of_actions
        self.lr = lr
        self.per = per
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.capacity = capacity


        self.local_model = QNetwork(input_size=input_size, number_of_actions=number_of_actions, noisy=noisy
        )
        self.target_model = QNetwork(input_size=input_size, number_of_actions=number_of_actions, noisy=noisy
        )

        # Initialize RNG key for the model
        self.rng_key = jax.random.PRNGKey(0)

        if per:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=capacity)
        else:
            self.replay_buffer = ReplayBuffer(capacity=capacity)

        # Initialize models with dummy data
        dummy_input = jnp.zeros((1, input_size))  # Should be part of the constructor

        # Initialize the parameters of the models
        self.local_model_params = self.local_model.init(
            jax.random.PRNGKey(0), dummy_input
        )["params"]
        self.target_model_params = self.target_model.init(
            jax.random.PRNGKey(1), dummy_input
        )["params"]

        # Initialize the optimizer with the parameters of the local model
        self.optimizer = optax.adam(lr)
        self.optimizer_state = self.optimizer.init(self.local_model_params)

    def take_action(self, state, epsilon, rng):
        """
        Selects an action based on the current state, using an epsilon-greedy strategy.

        This method decides between exploration and exploitation using the epsilon value.
        With probability epsilon, it chooses a random action (exploration),
        and with probability 1 - epsilon, it chooses the best action based on the model's predictions (exploitation).

        Concern:
        - The method currently resets the RNG key using a new seed derived from the current time in milliseconds.
        This approach might not be ideal as it can lead to predictable sequences in the long run and does not
        utilize JAX's PRNG system effectively.

        Parameters:
        - state: The current state of the environment.
        - epsilon: The probability of choosing a random action.
        - rng: The JAX random number generator key.

        Things to Check:
        - Ensure that the RNG key is split or advanced correctly instead of being reseeded each time.
        In JAX, you typically use `jax.random.split` to get a new subkey.
        - Validate that the state is correctly reshaped and can be processed by the model during exploitation.
        - Confirm that the action space defined by `number_of_actions` aligns with the environment's action space.
        - Check that the `jax.random.uniform` and `jax.random.randint` functions are used correctly to generate random numbers and actions.

        Potential Improvements:
        - Modify the RNG handling to use `jax.random.split` for generating new keys.
        This will maintain the stochasticity without the need to reset the seed each time.
        - If the state requires specific preprocessing before being fed into the model,
        ensure that it's handled appropriately within this method or prior to calling it.
        - Consider parameterizing the number of actions if it varies across different environments or parts of your application.

        Returns:
        - The selected action, either random or the best according to the model.
        """

        seed = int(time.time() * 1000)
        rng_key = jax.random.PRNGKey(seed)

        seed = int(time.time() * 1500)
        rng_key_action = jax.random.PRNGKey(seed)

        seed = int(time.time() * 1500)
        rng_key_quantiles = jax.random.PRNGKey(seed)

        # Split the RNG key
        #self.rng_key, rng_key_action, rng_key_quantiles = jax.random.split(self.rng_key, 3)

        # Flatten the state if necessary
        state = state.reshape(-1)


        # Generate a random number using JAX's random number generator
        random_number = jax.random.uniform(rng, minval=0.0, maxval=1.0)

        if random_number < epsilon:
            # Exploration: choose a random action
            action = jax.random.randint(
                rng_key_action, shape=(), minval=0, maxval=self.number_of_actions
            )
        else:
            # Exploitation: choose the best action based on the distribution of model predictions
            q_values_distribution, quantiles, _ = self.local_model.apply(
                {"params": self.local_model_params}, jnp.array([state]), rng_key_quantiles
            )
            expected_q_values = jnp.mean(q_values_distribution, axis=1)
            action = jnp.argmax(expected_q_values, axis=1)[0]

        return action



    def compute_td_errors(self, params, states, actions, rewards, next_states, dones):
        q_values, _ , _= self.local_model.apply({"params": params}, states)
        next_q_values, _ ,_ = self.target_model.apply({"params": params}, next_states)

        # Expand and reshape rewards and dones for broadcasting
        rewards_expanded = rewards[:, None, None]  # Shape: (batch_size, 1, 1)
        dones_expanded = dones[:, None, None]  # Shape: (batch_size, 1, 1)

        max_next_q_values = jnp.max(next_q_values, axis=-1, keepdims=True)
        target_q_values = rewards_expanded + (self.gamma * max_next_q_values * (1 - dones_expanded))

        # Expand and reshape actions for broadcasting
        actions_one_hot = jax.nn.one_hot(actions, self.number_of_actions)
        actions_one_hot_expanded = actions_one_hot[:, None, :]  # Shape: (batch_size, 1, number_of_actions)

        # Select the predicted Q-values for the taken actions
        predicted_q_values = jnp.sum(q_values * actions_one_hot_expanded, axis=-1)

        # Compute TD errors across all quantiles
        td_errors = jnp.abs(predicted_q_values - target_q_values)

        return td_errors

    
    def state_prediction_loss(self, predicted_next_states, actual_next_states):
        return jnp.mean(jnp.square(predicted_next_states - actual_next_states))

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

        if self.per:
            """
            Perform a training step using Prioritized Experience Replay (PER).

            This function is intended to facilitate the training of a model using a PER buffer. It samples experiences based on their priority, calculates the loss using the Temporal Difference (TD) error, updates the model parameters, and adjusts the priorities of experiences based on the new TD errors.

            Steps:
            1. Increment beta towards 1.
            2. Sample a batch of experiences based on their priorities.
            3. Extract relevant data (states, actions, rewards, next_states, dones) from the sampled experiences.
            4. Compute the loss function, which includes calculating TD errors and a weighted loss using importance sampling weights.
            5. Use gradient descent to update the model parameters.
            6. Perform a soft update of the target model parameters.
            7. Update the priorities in the replay buffer based on the new TD errors.

            Troubleshooting and Checks:
            - Ensure that the sampled batch has the correct format and that each component (states, actions, etc.) is correctly extracted.
            - Verify that the loss function returns a scalar value. If the function returns a non-scalar value (like an array), it will cause errors during gradient calculation.
            - Check that the calculated TD errors and importance sampling weights are used correctly in the loss function.
            - Confirm that the gradient calculation and model parameter updates are performed correctly. If there's an issue with the gradient calculation, it might be due to incorrect input to the loss function or an error within the loss function itself.
            - Ensure that the replay buffer's `update_priorities` method is called with the correct arguments and that it's functioning as intended.

            Potential Enhancements:
            - Consider implementing a more sophisticated method for calculating TD errors or importance sampling weights to improve learning efficiency.
            - Explore different strategies for updating priorities in the replay buffer, which might lead to better sampling of experiences.
            - Review and optimize the update rate of the target model parameters (tau value) for better stability and performance.
            """

            # Increment beta towards 1
            self.beta = min(self.beta + self.beta_increment, 1)

            sampled_batch, weights, indices = self.replay_buffer.sample(
                batch_size, self.alpha, self.beta
            )

            # Extract data from the sampled batch
            states = np.array([exp[0].numpy() for exp in sampled_batch])
            actions = np.array([exp[1] for exp in sampled_batch])
            rewards = np.array([exp[2] for exp in sampled_batch])
            next_states = np.array([exp[3].numpy() for exp in sampled_batch])
            dones = np.array([exp[4] for exp in sampled_batch])
            weights = jnp.array(weights)

            def loss_fn(params):
                q_values, quantiles, predicted_next_states = self.local_model.apply({"params": params}, states)
                next_q_values, _, _ = self.target_model.apply({"params": params}, next_states)

                # Expand and reshape rewards and dones for broadcasting
                rewards_expanded = rewards[:, None, None]  # Shape: (batch_size, 1, 1)
                dones_expanded = dones[:, None, None]  # Shape: (batch_size, 1, 1)

                # Maximize over action dimension of next_q_values for each quantile
                max_next_q_values = jnp.max(next_q_values, axis=-1, keepdims=True)  # Shape: (batch_size, num_quantiles, 1)

                # Calculate target Q values
                target_q_values = rewards_expanded + (self.gamma * max_next_q_values * (1 - dones_expanded))  # Shape: (batch_size, num_quantiles, 1)

                # Expand and reshape actions for selecting the appropriate q_values
                actions_one_hot = jax.nn.one_hot(actions, self.number_of_actions)
                actions_one_hot_expanded = actions_one_hot[:, None, :]  # Shape: (batch_size, 1, number_of_actions)

                # Select the Q-values for the taken actions, resulting in shape (batch_size, num_quantiles, 1)
                selected_q_values = jnp.sum(q_values * actions_one_hot_expanded, axis=-1, keepdims=True)

                # Compute TD errors across all quantiles
                td_errors = target_q_values - selected_q_values  # Shape should now be compatible

                quantile_loss = jax.lax.stop_gradient(quantiles)[None, :, None] - (td_errors < 0).astype(jnp.float32)
                quantile_loss = jnp.abs(quantile_loss) * jnp.square(td_errors)
                weighted_loss = jnp.mean(td_errors * jnp.array(weights))
                scalar_weighted_loss = jnp.reshape(weighted_loss, ())  # Reshape to scalar

                # State prediction loss
                state_pred_loss = self.state_prediction_loss(predicted_next_states, next_states)

                loss = scalar_weighted_loss + 0.1 * state_pred_loss

                return loss


            grad_fn = jax.value_and_grad(loss_fn)
            loss, gradients = grad_fn(self.local_model_params)

            updates, self.optimizer_state = self.optimizer.update(
                gradients, self.optimizer_state
            )
            self.local_model_params = optax.apply_updates(self.local_model_params, updates)

            if soft_update:
                tau = 0.01
                self.target_model_params = jax.tree_map(
                    lambda t, l: tau * l + (1 - tau) * t,
                    self.target_model_params,
                    self.local_model_params,
                )

            td_errors = self.compute_td_errors(
                self.local_model_params, states, actions, rewards, next_states, dones
            )

            not_broken = False
            if not_broken:
                new_priorities = np.abs(td_errors) + 1e-6  # Ensure no zero priorities
                self.replay_buffer.update_priorities(indices, new_priorities.tolist())

            return loss

        else:
            batch = self.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = batch

            def loss_fn(params): # update this for IQN
                q_values, predicted_next_step = self.local_model.apply({"params": params}, states)
                next_q_values, _ = self.target_model.apply({"params": params}, next_states)
                max_next_q_values = jnp.max(next_q_values, axis=1)
                target_q_values = rewards + (
                    self.gamma * max_next_q_values * (1 - dones)
                )
                actions_one_hot = jax.nn.one_hot(actions, self.number_of_actions)
                predicted_q_values = jnp.sum(q_values * actions_one_hot, axis=1)
                loss = jnp.mean((predicted_q_values - target_q_values) ** 2)
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, gradients = grad_fn(self.local_model_params)
            updates, self.optimizer_state = self.optimizer.update(
                gradients, self.optimizer_state
            )
            self.local_model_params = optax.apply_updates(
                self.local_model_params, updates
            )

            # Soft update

            tau = 0.01
            self.target_model_params = jax.tree_map(
                lambda t, l: tau * l + (1 - tau) * t,
                self.target_model_params,
                self.local_model_params,
            )

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
