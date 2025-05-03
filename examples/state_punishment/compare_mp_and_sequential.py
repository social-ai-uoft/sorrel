import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import shared_memory, Lock
import numpy as np
import time
import psutil
import os

class AgentNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=512, output_size=2):
        super(AgentNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

    def get_flat_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def load_flat_weights(self, flat_tensor):
        pointer = 0
        for param in self.parameters():
            numel = param.numel()
            param.data.copy_(flat_tensor[pointer:pointer + numel].view_as(param))
            pointer += numel

class SharedEnv:
    def __init__(self, num_agents=4, input_size=256):
        self.num_agents = num_agents
        self.input_size = input_size

    def reset(self):
        return [np.random.randn(self.input_size).astype(np.float32) for _ in range(self.num_agents)]

    def step(self, actions):
        next_obs = [np.random.randn(self.input_size).astype(np.float32) for _ in range(self.num_agents)]
        rewards = [np.random.rand() for _ in range(self.num_agents)]
        dones = [np.random.rand() > 0.95 for _ in range(self.num_agents)]
        return next_obs, rewards, dones, {}

def train_one_epoch(model, optimizer, input_size=256):
    model.train()
    for _ in range(50):  # Increase if needed
        input_tensor = torch.randn(4096, input_size)
        target_tensor = torch.randint(0, 2, (4096,), dtype=torch.long)
        output = model(input_tensor)
        loss = nn.CrossEntropyLoss()(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.get_flat_weights()

def learner_process(agent_id, shared_name, shape, lock, train_event, done_event, input_size):
    # p = psutil.Process(os.getpid())
    # core_id = agent_id % os.cpu_count()
    # p.cpu_affinity([core_id])  # Optional pinning

    shm = shared_memory.SharedMemory(name=shared_name)
    weight_np = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
    model = AgentNet(input_size=input_size)
    model.load_flat_weights(torch.tensor(weight_np.copy()))  # 🔁 Load initial weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    while True:
        train_event.wait()
        train_event.clear()

        weights = train_one_epoch(model, optimizer, input_size)
        with lock:
            weight_np[:] = weights.numpy()
        done_event.set()

def actor_loop(num_agents, shared_names, shapes, locks, train_events, done_events, max_epochs=5, input_size=256):
    models = [AgentNet(input_size=input_size) for _ in range(num_agents)]
    shms = [shared_memory.SharedMemory(name=shared_names[i]) for i in range(num_agents)]
    weight_nps = [np.ndarray(shapes[i], dtype=np.float32, buffer=shms[i].buf) for i in range(num_agents)]

    env = SharedEnv(num_agents=num_agents, input_size=input_size)
    obs_list = env.reset()
    epoch = 0
    steps_per_epoch = 10
    step_count = 0
    start = None

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
            print(f"[Actor] Epoch {epoch} completed. Signaling learners...")

            for ev in train_events:
                ev.set()

            for ev in done_events:
                ev.wait()
                ev.clear()

            print(f"[Actor] All learners completed training for epoch {epoch}")
            if epoch == 1:
                start = time.time()
            if epoch >= max_epochs:
                end = time.time()
                print(f"Multiprocessing Mode Time (post-epoch-1): {end - start:.2f} seconds")
                print("[Actor] Max epochs reached. Exiting actor loop.")
                break

    for shm in shms:
        shm.close()
        shm.unlink()

def run_multiprocessing_mode(num_agents, max_epochs=10, input_size=256):
    shared_names = []
    shapes = []
    locks = []
    learners = []
    shms = []
    train_events = []
    done_events = []

    dummy_model = AgentNet(input_size=input_size)
    flat_weights = dummy_model.get_flat_weights()
    shape = flat_weights.shape

    for i in range(num_agents):
        shm = shared_memory.SharedMemory(create=True, size=flat_weights.numel() * 4)
        weight_np = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)
        weight_np[:] = flat_weights.numpy()

        lock = Lock()
        train_event = mp.Event()
        done_event = mp.Event()
        learner = mp.Process(target=learner_process, args=(i, shm.name, shape, lock, train_event, done_event, input_size))
        learner.start()

        shared_names.append(shm.name)
        shapes.append(shape)
        locks.append(lock)
        learners.append(learner)
        shms.append(shm)
        train_events.append(train_event)
        done_events.append(done_event)

    actor_loop(num_agents, shared_names, shapes, locks, train_events, done_events, max_epochs, input_size)

    for learner in learners:
        learner.terminate()
        learner.join()

def run_sequential_mode(num_agents, max_epochs=10, input_size=256):
    models = [AgentNet(input_size=input_size) for _ in range(num_agents)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in models]
    weight_nps = [model.get_flat_weights().numpy().copy() for model in models]
    locks = [Lock() for _ in range(num_agents)]

    env = SharedEnv(num_agents=num_agents, input_size=input_size)
    obs_list = env.reset()
    epoch = 0
    steps_per_epoch = 10
    step_count = 0
    start = None

    while epoch < max_epochs:
        actions = []
        for i in range(num_agents):
            models[i].load_flat_weights(torch.tensor(weight_nps[i]))
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
            print(f"[Sequential Actor] Epoch {epoch} completed. Training agents...")
            if epoch == 1:
                start = time.time()
            for i in range(num_agents):
                weights = train_one_epoch(models[i], optimizers[i], input_size)
                weight_nps[i][:] = weights.numpy()
            print(f"[Sequential Actor] All agents trained for epoch {epoch}")

    end = time.time()
    print(f"Sequential Mode Time (post-epoch-1): {end - start:.2f} seconds")

def main():
    mode = "multiprocessing"  # or "sequential"
    num_agents = 6
    max_epochs = 10
    input_size = 4

    if mode == "multiprocessing":
        mp.set_start_method("spawn", force=True)
        run_multiprocessing_mode(num_agents, max_epochs, input_size)
    else:
        run_sequential_mode(num_agents, max_epochs, input_size)

    print("Main process completed.")

if __name__ == "__main__":
    main()
    print("Script executed successfully.")