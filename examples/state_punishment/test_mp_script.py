import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import shared_memory, Lock, Value
import numpy as np
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import shared_memory, Lock, Value
import numpy as np
import time

# Define the agent network
class AgentNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(AgentNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def get_flat_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_flat_weights(self, flat_tensor):
        pointer = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(flat_tensor[pointer:pointer + numel].view_as(param))
            pointer += numel

# Shared dummy environment
class SharedEnv:
    def reset(self):
        return [np.random.randn(4).astype(np.float32) for _ in range(6)]

    def step(self, actions):
        next_obs = [np.random.randn(4).astype(np.float32) for _ in range(6)]
        rewards = [np.random.rand() for _ in range(6)]
        dones = [np.random.rand() > 0.95 for _ in range(6)]
        return next_obs, rewards, dones, {}

# Learner process that executes once per epoch
def learner_process(agent_id, shared_name, shape, lock, shared_epoch, done_epoch):
    shm = shared_memory.SharedMemory(name=shared_name)
    weight_np = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    model = AgentNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dummy_input = torch.randn((32, 4))
    dummy_target = torch.randint(0, 2, (32,))
    last_epoch = 0

    while True:
        current_epoch = shared_epoch.value
        if current_epoch > last_epoch:
            output = model(dummy_input)
            loss = nn.CrossEntropyLoss()(output, dummy_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with lock:
                weight_np[:] = model.get_flat_weights().numpy()
            print(f"[Learner {agent_id}] Trained for epoch {current_epoch}")

            done_epoch[agent_id] = current_epoch
            last_epoch = current_epoch

        time.sleep(0.05)

# Actor loop
def actor_loop(num_agents, shared_names, shapes, locks, shared_epoch, done_epoch, max_epochs=5):
    models = [AgentNet() for _ in range(num_agents)]
    shms = [shared_memory.SharedMemory(name=shared_names[i]) for i in range(num_agents)]
    weight_nps = [np.ndarray(shapes[i], dtype=np.float32, buffer=shms[i].buf) for i in range(num_agents)]

    env = SharedEnv()
    obs_list = env.reset()
    epoch = 0
    steps_per_epoch = 10
    step_count = 0

    while True:
        actions = []
        for i in range(num_agents):
            with locks[i]:
                weights_tensor = torch.tensor(weight_nps[i].copy(), dtype=torch.float32)
            models[i].load_flat_weights(weights_tensor)
            obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32)
            with torch.no_grad():
                action = models[i](obs_tensor).argmax().item()
            actions.append(action)

        next_obs, rewards, dones, _ = env.step(actions)
        obs_list = next_obs
        step_count += 1

        for i in range(num_agents):
            print(f"[Agent {i}] Action: {actions[i]} Reward: {rewards[i]:.2f} Done: {dones[i]}")

        if step_count >= steps_per_epoch:
            step_count = 0
            epoch += 1
            shared_epoch.value = epoch
            print(f"[Actor] Epoch {epoch} completed. Waiting for learners...")

            while not all(done_epoch[i] == epoch for i in range(num_agents)):
                time.sleep(0.1)

            print(f"[Actor] All learners completed training for epoch {epoch}")

            if epoch >= max_epochs:
                print("[Actor] Max epochs reached. Exiting actor loop.")
                break

        time.sleep(0.1)

    # Cleanup shared memory (only actor owns these references)
    for shm in shms:
        shm.close()
        shm.unlink()

# Setup and entry point
def main():
    num_agents = 6
    shared_names = []
    shapes = []
    locks = []
    learners = []
    shms = []  # Keep SharedMemory objects alive
    shared_epoch = Value('i', 0)
    done_epoch = mp.Array('i', [0] * num_agents)

    for i in range(num_agents):
        dummy_model = AgentNet()
        flat_weights = dummy_model.get_flat_weights()
        shape = flat_weights.shape
        shm = shared_memory.SharedMemory(create=True, size=flat_weights.numel() * 4)
        weight_np = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        weight_np[:] = flat_weights.numpy()

        lock = Lock()
        learner = mp.Process(target=learner_process, args=(i, shm.name, shape, lock, shared_epoch, done_epoch))
        learner.start()

        shared_names.append(shm.name)
        shapes.append(shape)
        locks.append(lock)
        learners.append(learner)
        shms.append(shm)  # Keep reference to avoid unlinking by GC

    actor_loop(num_agents, shared_names, shapes, locks, shared_epoch, done_epoch)

    for learner in learners:
        learner.terminate()
        learner.join()

if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
    print("Main process completed.")
