# MAIN
shared_model = Model()
shared_model.share_memory()              # shared params
start_process(learner, args=(shared_model,))
for i in range(num_actors):
    start_process(actor, args=(shared_model,))

# LEARNER PROCESS
def learner(shared_model):
    loop:
        data = receive_from_actors()     # or from memory buffer
        loss = compute_loss(shared_model, data)
        update(shared_model, loss)       # optimizer.step(shared_model)

# ACTOR PROCESS
def actor(shared_model):
    local_model = copy(shared_model)     # initial sync
    loop:
        exp = run_env_with(local_model) # collect rollout
        send_to_learner(exp)            # or to the memory buffer
        if time_to_sync:
            local_model = copy(shared_model)  # refresh weights
