import jax  # Importing JAX for automatic differentiation and GPU/TPU acceleration
import jax.numpy as jnp  # JAX NumPy for efficient numerical computations
import flax.linen as nn  # Flax Linen for defining neural network models
from jax import (
    random,
    grad,
    jit,
    value_and_grad,
)  # Importing specific JAX functionalities
from jax.nn import softmax  # Softmax function from JAX for probability distribution
from flax.core.frozen_dict import (
    FrozenDict,
)  # FrozenDict for immutable dictionary structures
from typing import Tuple, Any  # Importing typing for type annotations
import time  # Importing time for time-based operations (like seeding random number generators)


"""
This script implements the Proximal Policy Optimization (PPO) algorithm using JAX and Flax.
PPO is an advanced policy gradient method that trains a policy by maximizing an objective function.
It includes an actor-critic architecture with separate networks for policy (actor) and value estimation (critic).

The script defines classes for the actor and critic networks, as well as the main PPO agent class,
which includes methods for policy execution, computing returns, generalized advantage estimation (GAE),
and learning (updating the actor and critic networks).

Note: This script requires an external environment for the agent to interact with.

# Example Usage

def train(environment, num_episodes, batch_size):
    Trains the PPO model on a given environment.

    Args:
        environment: The environment in which the agent operates.
        num_episodes: Total number of episodes for training.
        batch_size: Number of steps to take in the environment per episode.

    This function runs the PPO training loop. For each episode, it collects states, actions, rewards,
    and old log probabilities for a number of steps equal to the batch size. It then computes returns
    and updates the PPO model using the learn function.

    # Initialize the PPO model
    num_actions = environment.action_space.n
    model = PPO(num_actions=num_actions)

    for episode in range(num_episodes):
        states, actions, rewards, dones, old_log_probs = [], [], [], [], []
        state = environment.reset()

        for _ in range(batch_size):
            action = model.policy(state)
            next_state, reward, done, _ = environment.step(action)
            log_prob = model.log_prob_of_action(model.actor_params, state, action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            old_log_probs.append(log_prob)

            state = next_state
            if done:
                state = environment.reset()

        # Compute returns using either compute_returns or compute_gae based on the use case
        returns = model.compute_returns(rewards, dones)
        # Normalize returns if necessary
        normalized_returns = model.normalize_returns(returns)
        model.learn(states, actions, jnp.array(old_log_probs), normalized_returns)

# Example call to the training function
# train(your_environment, num_episodes=100, batch_size=1000)

"""


class ActorNetwork(nn.Module):
    """
    ActorNetwork defines a neural network model for the 'actor' in reinforcement learning.
    It outputs action probabilities based on the state input.

    Attributes:
        num_actions (int): Number of possible actions the actor can take.

    Methods:
        __call__(x): Forward pass through the network. It takes the state as input and
        returns the softmax probabilities of actions.
    """

    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.tanh(x)
        x = nn.Dense(512)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.num_actions)(x)
        return softmax(x)


class CriticNetwork(nn.Module):
    """
    CriticNetwork defines a neural network model for the 'critic' in reinforcement learning.
    It estimates the value of the state input.

    Methods:
        __call__(x): Forward pass through the network. It takes the state as input and
        returns the estimated value of that state.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(512)(x)
        x = nn.tanh(x)
        x = nn.Dense(512)(x)
        x = nn.tanh(x)
        x = nn.Dense(256)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


class PPO:
    """
    PPO class implements the Proximal Policy Optimization algorithm.

    Attributes:
        num_actions (int): Number of actions in the action space.
        gamma (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        clip_ratio (float): Clipping parameter for PPO.
        update_epochs (int): Number of epochs for updating the policy.
        entropy_coef (float): Coefficient for the entropy bonus in the loss function.
        actor_network (ActorNetwork): The actor network model.
        critic_network (CriticNetwork): The critic network model.
        actor_params (FrozenDict): Parameters of the actor network.
        critic_params (FrozenDict): Parameters of the critic network.

    Methods:
        policy(state): Chooses an action based on the current state.
        compute_returns(rewards, dones): Computes returns for each time step.
        compute_gae(...): Computes Generalized Advantage Estimation (GAE).
        normalize_returns(returns): Normalizes the returns.
        learn(states, actions, old_log_probs, returns): Updates the actor and critic networks.
        compute_actor_loss(...): Computes loss for the actor network.
        compute_critic_loss(...): Computes loss for the critic network.
        log_prob_of_action(actor_params, state, action): Computes the log probability of a given action.
        entropy(actor_params, state): Computes the entropy of the action distribution.
    """

    def __init__(
        self,
        num_actions,
        gamma=0.8,
        learning_rate=1e-3,
        clip_ratio=0.2,
        update_epochs=10,  # 10
        entropy_coef=0.01,  # 0.01
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        self.key = random.PRNGKey(0)

        # Initialize actor and critic networks
        self.actor_network = ActorNetwork(num_actions=num_actions)
        self.critic_network = CriticNetwork()

        dummy_input = jnp.ones((1134,))  # Example input
        seed = int(
            time.time() * 1000
        )  # need to figure out the jax random number generator key
        rng = jax.random.PRNGKey(seed)

        self.actor_params = self.actor_network.init(rng, dummy_input)
        seed = int(
            time.time() * 2000
        )  # need to figure out the jax random number generator key
        rng = jax.random.PRNGKey(seed)

        self.critic_params = self.critic_network.init(rng, dummy_input)

    def policy(self, state):
        """
        Chooses an action based on the current state using the actor network.

        Args:
            state: The current state of the environment.

        Returns:
            An action selected based on the probability distribution from the actor network.

        This function first computes the action probabilities using the actor network. It uses
        the current parameters of the actor network and the input state to generate a probability
        distribution over actions. Then, it utilizes a time-based seed to initialize a random number
        generator (RNG) and selects an action according to the probability distribution.

        Note:
        - Using time-based seeds for RNG can lead to non-reproducible results. For consistency and
        reproducibility in experiments, it's recommended to manage RNG keys in a controlled manner,
        potentially by passing and splitting keys using JAX's PRNGKey system.
        - The policy is stochastic, meaning it randomly selects an action based on the computed probabilities.
        In some applications, a deterministic policy (selecting the highest probability action) might be preferable.
        """

        action_probs = self.actor_network.apply(self.actor_params, state)
        seed = int(
            time.time() * 1000
        )  # need to figure out the jax random number generator key
        rng = jax.random.PRNGKey(seed)
        return random.choice(rng, len(action_probs), p=action_probs)

    def compute_returns(self, rewards, dones):
        """
        Computes the returns for each timestep in an episode.

        Args:
            rewards: A list of rewards received at each timestep.
            dones: A list of boolean values indicating whether each timestep is the end of an episode.

        Returns:
            A JAX numpy array of the computed returns for each timestep.

        This function calculates the returns (total discounted future rewards) for each timestep in a sequence.
        It iterates through each reward and corresponding 'done' signal in reverse order (starting from the end of the episode).
        If a 'done' signal is True, it signifies the end of an episode, and the cumulative reward (Gt) is reset to 0.
        Otherwise, the cumulative reward is updated as the sum of the current reward and the discounted cumulative reward from the next timestep.

        Note:
        - The rewards are processed in reverse order, ensuring that the return at each timestep is calculated based on the future rewards.
        - The 'gamma' attribute of the class is used as the discount factor, influencing the importance given to future rewards.
        """
        returns = []
        Gt = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                Gt = 0  # Reset the cumulative reward for a new episode
            Gt = reward + self.gamma * Gt
            returns.insert(0, Gt)
        return jnp.array(returns)

    def compute_gae(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        """
        Computes Generalized Advantage Estimation (GAE) for each timestep.

        Args:
            rewards: A list of rewards received at each timestep.
            values: A list of value function estimates at each timestep.
            dones: A list of boolean values indicating whether each timestep is the end of an episode.
            gamma (float): The discount factor for future rewards (default is 0.99).
            lambda_ (float): The GAE parameter for balancing bias and variance (default is 0.95).

        Returns:
            A JAX numpy array of the computed GAE values for each timestep.

        This function calculates GAE, which is a method for estimating the advantages of actions in reinforcement learning.
        GAE is used to reduce the variance of the advantage estimation while keeping a reasonable level of bias. The function
        iterates in reverse through the rewards, values, and dones to calculate the GAE for each timestep.

        Note:
        - This function is currently not working as expected. The issue may lie in the calculations or the inputs provided to the function.
        - The calculation of delta (the temporal difference error) and the subsequent GAE update are key parts of this function.
        - The values list is expected to have one more element than the rewards and dones lists, as it includes the value estimate at the next timestep.

        TODO: Investigate and resolve the issues causing this function to not work as intended.
        """
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * lambda_ * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        return jnp.array(returns)

    def normalize_returns(self, returns):
        """
        Normalizes the returns using mean and standard deviation.

        Args:
            returns: A JAX numpy array of returns for each timestep.

        Returns:
            A JAX numpy array of normalized returns.

        This function normalizes the input returns by subtracting the mean and dividing by the standard deviation.
        Normalization helps in stabilizing the training process by ensuring that the scale of the returns does not
        adversely affect the gradients. A small constant (1e-8) is added to the standard deviation to prevent division
        by zero in cases where the standard deviation is extremely small.

        Note:
        - The choice of the small constant (1e-8) is arbitrary and can be adjusted based on the scale of the returns.
        - Normalization is especially important when the range of returns is large or varies significantly between episodes.
        """
        return (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-8)

    def learn(self, states, actions, old_log_probs, returns):
        """
        Updates the policy (actor) and value (critic) networks based on provided training data.

        Args:
            states: A list or array of states encountered in the environment.
            actions: A list or array of actions taken in those states.
            old_log_probs: A list or array of log probabilities of the taken actions under the old policy.
            returns: A list or array of returns (cumulative discounted rewards) for each state.

        Returns:
            actor_loss: The loss computed for the actor network after the update.
            critic_loss: The loss computed for the critic network after the update.

        This function updates the actor and critic networks using the Proximal Policy Optimization (PPO) algorithm.
        It first ensures that the inputs are properly batched and then performs several epochs of updates to both networks.
        For the actor network, it computes the loss using the specified actor loss function and applies gradient descent.
        Similarly, it updates the critic network by computing its loss and adjusting its parameters.

        Note:
        - The dimensions of the input arrays are checked to ensure they are properly batched.
        - The learning rate defined in the class is used for updating the network parameters.
        - The function performs a specified number of update epochs (self.update_epochs) on both networks.
        """
        # Ensure inputs are properly batched
        states = jnp.array(states)
        actions = jnp.array(actions)
        old_log_probs = jnp.array(old_log_probs)
        returns = jnp.array(returns)

        assert states.ndim == 2, "States must be batched (batch_size, state_dim)"
        assert actions.ndim == 1, "Actions must be batched (batch_size,)"
        assert (
            old_log_probs.ndim == 1
        ), "Old log probabilities must be batched (batch_size,)"

        # Update the policy and value networks
        for _ in range(self.update_epochs):
            # Update actor
            actor_loss, actor_grads = value_and_grad(self.compute_actor_loss)(
                self.actor_params, states, actions, old_log_probs, returns
            )

            self.actor_params = jax.tree_map(
                lambda p, g: p - self.learning_rate * g, self.actor_params, actor_grads
            )

            # Update critic
            critic_loss, critic_grads = value_and_grad(self.compute_critic_loss)(
                self.critic_params, states, returns
            )

            self.critic_params = jax.tree_map(
                lambda p, g: p - self.learning_rate * g,
                self.critic_params,
                critic_grads,
            )

        return actor_loss, critic_loss

    def compute_actor_loss(self, actor_params, states, actions, old_log_probs, returns):
        """
        Computes the loss for the actor network using the PPO algorithm.

        Args:
            actor_params: Current parameters of the actor network.
            states: A batch of states from the environment.
            actions: A batch of actions taken in those states.
            old_log_probs: Log probabilities of the actions under the old policy.
            returns: Returns (cumulative discounted rewards) for each state.

        Returns:
            The computed loss for the actor network.

        This function computes the actor loss based on the Proximal Policy Optimization (PPO) algorithm.
        It calculates the new log probabilities of the actions under the current policy and uses them to
        compute the ratio of new and old probabilities. The advantage is calculated as the difference between
        the returns and the value estimates from the critic network. The surrogate losses (surr1 and surr2) are
        then calculated and clipped based on the PPO clipping technique. An entropy bonus is also added to encourage
        exploration by the policy. The final loss is a combination of the clipped surrogate loss and the entropy bonus.

        Note:
        - The use of `jax.vmap` allows for efficient vectorized computation over batches.
        - The entropy bonus is controlled by the `entropy_coef` attribute, which can be tuned.
        """
        new_log_probs = jax.vmap(self.log_prob_of_action, in_axes=(None, 0, 0))(
            actor_params, states, actions
        )
        ratio = jnp.exp(new_log_probs - old_log_probs)
        advantage = (
            returns
            - jax.vmap(self.critic_network.apply, in_axes=(None, 0))(
                self.critic_params, states
            ).squeeze()
        )
        surr1 = ratio * advantage
        surr2 = jnp.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        entropy_bonus = (
            -self.entropy_coef
            * jax.vmap(self.entropy, in_axes=(None, 0))(actor_params, states).mean()
        )
        return -jnp.minimum(surr1, surr2).mean() + entropy_bonus

    def compute_critic_loss(self, critic_params, states, returns):
        """
        Computes the loss for the critic network.

        Args:
            critic_params: Current parameters of the critic network.
            states: A batch of states from the environment.
            returns: Returns (cumulative discounted rewards) for each state.

        Returns:
            The computed loss for the critic network.

        This function calculates the loss for the critic network, which is responsible for estimating the value
        of states. It first uses the critic network to predict the value of each state in the batch. The loss is
        then computed as the mean squared error between the predicted values and the actual returns. This loss
        function helps in training the critic network to make more accurate value predictions.

        Note:
        - The `jax.vmap` function is used to efficiently apply the critic network to each state in the batch.
        - The loss is calculated using mean squared error, which is a common choice for regression problems like
        value estimation.
        """
        values = jax.vmap(self.critic_network.apply, in_axes=(None, 0))(
            critic_params, states
        ).squeeze()
        return jnp.mean((returns - values) ** 2)

    def log_prob_of_action(self, actor_params, state, action):
        """
        Computes the log probability of a specific action given a state, under the current policy.

        Args:
            actor_params: Current parameters of the actor network.
            state: The state of the environment for which the action was taken.
            action: The action taken in the given state.

        Returns:
            The log probability of the specified action given the state under the current policy.

        This function calculates the log probability of a specific action given a state. It first obtains the
        probability distribution over actions from the actor network using the current state. Then, it selects
        the probability corresponding to the given action and computes its logarithm. A small constant (1e-10) is
        added to the probability before taking the logarithm to avoid numerical issues with log(0).

        Note:
        - The small constant (1e-10) added to the probability is to ensure numerical stability, as taking the log
        of zero leads to negative infinity.
        - The actor network's apply method is used to compute the probability distribution of actions for the given state.
        """
        action_probs = self.actor_network.apply(actor_params, state)
        return jnp.log(action_probs[action] + 1e-10)

    def entropy(self, actor_params, state):
        """
        Computes the entropy of the action probability distribution for a given state.

        Args:
            actor_params: Current parameters of the actor network.
            state: The state of the environment.

        Returns:
            The entropy of the action probability distribution for the given state.

        This function calculates the entropy of the action probability distribution for a given state.
        High entropy indicates a more exploratory policy, while low entropy suggests a more deterministic policy.
        The function first computes the action probability distribution for the given state using the actor network.
        It then calculates the entropy of this distribution, which is the sum of the product of probabilities and
        their logarithms, multiplied by -1. A small constant (1e-10) is added to the final result to avoid numerical
        instability, especially in cases where probabilities are very low.

        Note:
        - Entropy is an important measure in reinforcement learning as it encourages exploration by the policy.
        - The small constant (1e-10) added to the entropy is to ensure numerical stability.
        """
        action_probs = self.actor_network.apply(actor_params, state)
        return -jnp.sum(action_probs * jnp.log(action_probs)) + 1e-10


# model = PPO(num_actions=4)


# def train(environment, num_episodes, batch_size):
#    for episode in range(num_episodes):
#        states, actions, rewards, old_log_probs = [], [], [], []
#        state = environment.reset()
#
#        for _ in range(batch_size):
#            action = PPO.policy(state)
#            next_state, reward, done, _ = environment.step(action)
#            log_prob = PPO.log_prob_of_action(PPO.actor_params, state, action)
#
#            states.append(state)
#            actions.append(action)
#            rewards.append(reward)
#            old_log_probs.append(log_prob)

#            state = next_state
#            if done:
#                state = environment.reset()

#        returns = PPO.compute_returns(rewards)
#        PPO.learn(states, actions, jnp.array(old_log_probs), returns)
