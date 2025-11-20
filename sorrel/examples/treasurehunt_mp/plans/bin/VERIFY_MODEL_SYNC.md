# Verifying Local Model Sync from Shared Model

## Hypothesis
The local model in the actor might not be successfully syncing from the shared model, causing the actor to use stale weights even though the learner is updating the shared model.

## Verification Methods

### 1. Add Sync Event Logging

**Location**: `mp_actor.py` around line 141-144

**Add logging when sync happens:**
```python
if self.sync_counter % self.config.sync_interval == 0:
    for agent_id in range(len(self.agents)):
        # Log BEFORE sync
        local_epsilon_before = self.local_models[agent_id].epsilon
        shared_epsilon_before = self.shared_models[agent_id].epsilon
        
        # Read from shared model (atomic read via load_state_dict, no lock needed)
        copy_model_state_dict(self.shared_models[agent_id], self.local_models[agent_id])
        
        # Log AFTER sync
        local_epsilon_after = self.local_models[agent_id].epsilon
        shared_epsilon_after = self.shared_models[agent_id].epsilon
        
        print(f"[Actor Sync] Agent {agent_id}, Turn {self.env.turn}, Epoch {epoch}")
        print(f"  Before: local_epsilon={local_epsilon_before:.4f}, shared_epsilon={shared_epsilon_before:.4f}")
        print(f"  After:  local_epsilon={local_epsilon_after:.4f}, shared_epsilon={shared_epsilon_after:.4f}")
        print(f"  Changed: {local_epsilon_after != local_epsilon_before}")
```

**What to look for:**
- Does sync actually trigger? (Check if print statements appear)
- Does `local_epsilon_after` match `shared_epsilon_before`?
- Does sync happen frequently enough? (Every 50 turns by default)

---

### 2. Compare Model Weights Before/After Sync

**Add weight comparison:**
```python
if self.sync_counter % self.config.sync_interval == 0:
    for agent_id in range(len(self.agents)):
        # Get first layer weights as a sample
        local_weight_before = self.local_models[agent_id].qnetwork_local.head1.weight.data.clone()
        shared_weight_before = self.shared_models[agent_id].qnetwork_local.head1.weight.data.clone()
        
        # Check if they're different (proving local is stale)
        weight_diff_before = torch.abs(local_weight_before - shared_weight_before).mean().item()
        
        # Sync
        copy_model_state_dict(self.shared_models[agent_id], self.local_models[agent_id])
        
        # Check if they're now the same
        local_weight_after = self.local_models[agent_id].qnetwork_local.head1.weight.data.clone()
        shared_weight_after = self.shared_models[agent_id].qnetwork_local.head1.weight.data.clone()
        weight_diff_after = torch.abs(local_weight_after - shared_weight_after).mean().item()
        
        print(f"[Actor Sync] Agent {agent_id}, Weight diff before: {weight_diff_before:.6f}, after: {weight_diff_after:.6f}")
        
        if weight_diff_before > 0.001:  # Significant difference
            print(f"  ⚠️  WARNING: Local model was stale before sync!")
        if weight_diff_after > 0.001:  # Should be ~0 after sync
            print(f"  ❌ ERROR: Sync failed! Weights still different after sync!")
```

**What to look for:**
- `weight_diff_before > 0`: Local model is stale (expected if learner updated)
- `weight_diff_after ≈ 0`: Sync worked correctly
- `weight_diff_after > 0`: Sync failed (CRITICAL BUG)

---

### 3. Track Model Versions

**Add version tracking to shared models:**
```python
# In mp_system.py, add to shared_state:
shared_state['model_versions'] = [mp.Value('i', 0) for _ in range(num_agents)]

# In mp_learner.py, increment when publishing:
if device.type in ('cuda', 'mps') and training_step % config.publish_interval == 0:
    # ... publish weights ...
    shared_state['model_versions'][agent_id].value += 1  # Increment version

# In mp_actor.py, check version when syncing:
if self.sync_counter % self.config.sync_interval == 0:
    for agent_id in range(len(self.agents)):
        old_version = self.shared_state['model_versions'][agent_id].value
        copy_model_state_dict(self.shared_models[agent_id], self.local_models[agent_id])
        new_version = self.shared_state['model_versions'][agent_id].value
        
        if new_version > old_version:
            print(f"[Actor Sync] Agent {agent_id} synced to version {new_version} (was {old_version})")
        else:
            print(f"[Actor Sync] Agent {agent_id} synced but version unchanged ({old_version})")
```

**What to look for:**
- Does version increment when learner publishes?
- Does actor sync to latest version?
- How many versions behind is the actor?

---

### 4. Verify Sync Counter Logic

**Check if sync actually triggers:**
```python
# In mp_actor.py, add debug logging:
self.sync_counter += 1
if epoch % 10 == 0:  # Log every 10 epochs
    print(f"[Actor] Epoch {epoch}, Turn {self.env.turn}, Sync counter: {self.sync_counter}, Sync interval: {self.config.sync_interval}")
    print(f"  Will sync: {self.sync_counter % self.config.sync_interval == 0}")

if self.sync_counter % self.config.sync_interval == 0:
    print(f"[Actor] SYNC TRIGGERED at counter={self.sync_counter}, interval={self.config.sync_interval}")
    # ... sync code ...
```

**What to look for:**
- Does `sync_counter` increment correctly?
- Does sync trigger when `sync_counter % sync_interval == 0`?
- Is `sync_interval` too large? (Default: 50 turns)

---

### 5. Compare Epsilon Values Continuously

**Epsilon is easier to track than weights:**
```python
# In mp_actor.py, add periodic epsilon comparison:
if epoch % 10 == 0 and self.env.turn % 10 == 0:  # Every 10 epochs, every 10 turns
    for agent_id in range(len(self.agents)):
        local_epsilon = self.local_models[agent_id].epsilon
        shared_epsilon = self.shared_models[agent_id].epsilon
        diff = abs(local_epsilon - shared_epsilon)
        
        if diff > 0.001:  # Significant difference
            print(f"[Actor] ⚠️  Epsilon mismatch! Agent {agent_id}, Epoch {epoch}, Turn {self.env.turn}")
            print(f"  Local epsilon: {local_epsilon:.6f}, Shared epsilon: {shared_epsilon:.6f}, Diff: {diff:.6f}")
            print(f"  Sync counter: {self.sync_counter}, Next sync in: {self.config.sync_interval - (self.sync_counter % self.config.sync_interval)} turns")
```

**What to look for:**
- Are epsilon values different between local and shared?
- How long do they stay different?
- Do they match after sync?

---

### 6. Test copy_model_state_dict() Function

**Verify the sync function actually works:**
```python
# Add test in mp_actor.py __init__ or after first sync:
# After initial sync from shared model
test_local = self.local_models[0]
test_shared = self.shared_models[0]

# Manually change shared model epsilon
test_shared.epsilon = 0.999

# Sync
copy_model_state_dict(test_shared, test_local)

# Check if it worked
if abs(test_local.epsilon - 0.999) < 0.001:
    print("[Actor] ✅ copy_model_state_dict() works for epsilon")
else:
    print(f"[Actor] ❌ copy_model_state_dict() FAILED! Expected 0.999, got {test_local.epsilon}")

# Test with weights
test_shared.qnetwork_local.head1.weight.data.fill_(1.0)
copy_model_state_dict(test_shared, test_local)
if torch.allclose(test_local.qnetwork_local.head1.weight.data, torch.ones_like(test_local.qnetwork_local.head1.weight.data)):
    print("[Actor] ✅ copy_model_state_dict() works for weights")
else:
    print("[Actor] ❌ copy_model_state_dict() FAILED for weights!")
```

---

### 7. Check if Sync Happens Before Action Selection

**Critical timing check:**
```python
# In step_environment(), before getting action:
# Check if we should sync NOW (before action selection)
if (self.sync_counter + 1) % self.config.sync_interval == 0:
    # We're about to sync next turn, but maybe we should sync now?
    print(f"[Actor] ⚠️  Will sync next turn, but using potentially stale model now")

# Get action using local model copy
action = agent.get_action(state)

# After action, check if we should have synced first
if self.sync_counter % self.config.sync_interval == 0:
    print(f"[Actor] ⚠️  Just used model, now syncing - might have used stale model!")
```

**What to look for:**
- Is sync happening at the right time?
- Are we using stale models for action selection?

---

### 8. Monitor Learner Publishing Frequency

**Check if learner is actually updating shared model:**
```python
# In mp_learner.py, when publishing:
if device.type in ('cuda', 'mps') and training_step % config.publish_interval == 0:
    print(f"[Learner {agent_id}] Publishing weights at step {training_step}, version {version}")
    print(f"  Epsilon before publish: {train_model_cpu.epsilon:.6f}")
    
    # Publish
    shared_model.load_state_dict(train_model_cpu.state_dict())
    shared_model.epsilon = train_model_cpu.epsilon
    
    print(f"  Shared model epsilon after publish: {shared_model.epsilon:.6f}")
    print(f"  ✅ Published successfully")
```

**What to look for:**
- Does learner publish frequently enough?
- Does shared model actually get updated?
- What's the time gap between learner publishing and actor syncing?

---

## Quick Diagnostic Script

Add this to `mp_actor.py` at the end of each epoch:

```python
# At end of epoch, after epsilon decay:
if epoch % 10 == 0:  # Every 10 epochs
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch} - MODEL SYNC DIAGNOSTICS")
    print(f"{'='*60}")
    
    for agent_id in range(len(self.agents)):
        local = self.local_models[agent_id]
        shared = self.shared_models[agent_id]
        
        # Compare epsilon
        epsilon_match = abs(local.epsilon - shared.epsilon) < 0.001
        print(f"Agent {agent_id}:")
        print(f"  Epsilon: local={local.epsilon:.6f}, shared={shared.epsilon:.6f}, match={epsilon_match}")
        
        # Compare first layer weights
        local_w = local.qnetwork_local.head1.weight.data
        shared_w = shared.qnetwork_local.head1.weight.data
        weight_diff = torch.abs(local_w - shared_w).mean().item()
        weight_match = weight_diff < 0.001
        print(f"  Weights: diff={weight_diff:.6f}, match={weight_match}")
        
        # Sync status
        turns_until_sync = self.config.sync_interval - (self.sync_counter % self.config.sync_interval)
        print(f"  Sync counter: {self.sync_counter}, Next sync in: {turns_until_sync} turns")
        
        if not epsilon_match or not weight_match:
            print(f"  ⚠️  WARNING: Models are out of sync!")
    print(f"{'='*60}\n")
```

---

## Expected Results if Hypothesis is TRUE

If local model is NOT syncing:
1. ✅ Epsilon in logger shows decay (shared model updates)
2. ❌ Local model epsilon doesn't match shared model epsilon
3. ❌ Weight differences persist after sync
4. ❌ `copy_model_state_dict()` might be failing silently
5. ❌ Sync might not be triggering (sync_counter issue)
6. ❌ Sync might be happening but not working (function bug)

---

## Expected Results if Hypothesis is FALSE

If local model IS syncing correctly:
1. ✅ Epsilon values match between local and shared
2. ✅ Weight differences are ~0 after sync
3. ✅ Sync triggers regularly
4. ✅ `copy_model_state_dict()` works correctly
5. ❌ But learning still fails → problem is elsewhere

---

## Implementation Priority

1. **Start with Method 1** (Sync Event Logging) - Easiest, shows if sync happens
2. **Then Method 5** (Epsilon Comparison) - Simple, shows if values match
3. **Then Method 2** (Weight Comparison) - More complex, but definitive
4. **Then Method 3** (Version Tracking) - Shows timing issues

These methods will quickly reveal if the sync is working or not.

