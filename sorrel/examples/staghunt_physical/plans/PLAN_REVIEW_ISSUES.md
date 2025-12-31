# Plan Review Issues

This document lists all issues identified during the review of `agent_identity_system_plan.md`.

## Critical Issues

### 1. Duplicate Code Blocks in Identity Map Generation (Lines 318-355)

**Location:** `Phase 2.2` - `StagHuntObservation.__init__()` identity_map generation

**Issue:** There are duplicate `elif` blocks for `"unique_and_group"` and `"custom"` modes.

**Current Code:**
```python
if mode == "unique_onehot" or mode == "unique_and_group":
    # ... code for both modes ...
    self.identity_map[(agent_id, agent_kind, orientation)] = identity_code
elif mode == "custom":
    # ... code ...
    self.identity_map = {}

elif mode == "unique_and_group":  # ❌ DUPLICATE - already handled above
    # ... duplicate code ...
    self.identity_map[(agent_id, agent_kind, orientation)] = identity_code

elif mode == "custom":  # ❌ DUPLICATE - already handled above
    # ... duplicate code ...
    self.identity_map = {}
```

**Fix:** Remove lines 337-355 (duplicate blocks). The first `if` statement already handles both `"unique_onehot"` and `"unique_and_group"` modes.

---

### 2. Missing Variable Definition in Simplified Code Example (Line 418)

**Location:** `Phase 3` - Simplified `observe()` method example (around line 418)

**Issue:** `obs_y` and `obs_x` are used in the loop but never defined.

**Current Code:**
```python
# Create identity channels
identity_channels = np.zeros((identity_size, height, width), dtype=np.float32)

# Iterate through visual field cells to add identity codes
for y in range(height):
    for x in range(width):
        # Calculate world coordinate
        world_y = obs_y - vision_radius + y  # ❌ obs_y not defined!
        world_x = obs_x - vision_radius + x  # ❌ obs_x not defined!
```

**Fix:** Add before the loop:
```python
# Get observer's world coordinates
obs_y, obs_x = location[0:2]
```

---

### 3. Inconsistent Indentation in Step 3 Code (Line 1611)

**Location:** `Step 3: Visual Field Processing` - Complete code example (line 1611)

**Issue:** Code after line 1610 has inconsistent indentation - some lines are not properly indented.

**Current Code:**
```python
       # Step 3.9: Get the agent at observation location to extract inventory and ready state
    agent = None  # ❌ Wrong indentation (should be 7 spaces, not 4)
    if hasattr(world, "agents"):
        for a in world.agents:
            if a.location == location:
                agent = a
                break
    
    # Step 3.8: Extract extra features (existing code)  # ❌ Wrong indentation
```

**Fix:** Ensure all code after line 1610 is properly indented with 7 spaces (matching the function body).

---

### 4. Step Numbering Inconsistency (Line 1618)

**Location:** `Step 3: Visual Field Processing` - Complete code example (line 1618)

**Issue:** Comment says "Step 3.8" but should be "Step 3.10" based on the sequence.

**Current Code:**
```python
    # Step 3.8: Extract extra features (existing code)  # ❌ Should be 3.10
```

**Sequence:**
- Step 3.1: Get base visual field
- Step 3.2: Calculate dimensions
- Step 3.3: Reshape base visual field
- Step 3.4: Get observer's world coordinates
- Step 3.5: Iterate through visual field cells
- Step 3.6: Concatenate entity and identity channels
- Step 3.7: Flatten visual field
- Step 3.8: Handle padding
- Step 3.9: Get the agent at observation location
- Step 3.10: Extract extra features ← **This is the correct number**
- Step 3.11: Generate positional embedding
- Step 3.12: Concatenate final observation

**Fix:** Change "Step 3.8" to "Step 3.10" on line 1618.

---

## Moderate Issues

### 5. Outdated Code Examples - Agent Detection Method

**Location:** Multiple locations in the plan

**Issue:** Some code examples still reference the old `isinstance(entity, StagHuntAgent)` approach instead of the new `hasattr(entity, 'identity_code')` approach.

**Locations:**
- Line 1735: "Agent Detection" section still shows `isinstance` example
- Line 1740-1746: "Identity Extraction" section shows old approach

**Current Code:**
```python
2. **Agent Detection:**
   ```python
   if isinstance(entity, StagHuntAgent):  # ❌ Outdated - should use hasattr
   ```
```

**Fix:** Update to:
```python
2. **Agent Detection:**
   ```python
   if hasattr(entity, 'identity_code') and entity.identity_code is not None:
   ```
```

---

### 6. Missing Import Statement in Code Examples

**Location:** `Step 3: Visual Field Processing` - Complete code example (line 1632)

**Issue:** Code uses `embedding.positional_embedding()` but doesn't show the import statement.

**Current Code:**
```python
       # Step 3.12: Generate positional embedding (existing code)
       pos_code = embedding.positional_embedding(
           location, world, (self.embedding_size, self.embedding_size)
       )
```

**Fix:** Add at the top of the code example:
```python
from sorrel.observation import embedding
```

Or clarify that this import is assumed to exist (since it's in the existing codebase).

---

## Minor Issues

### 7. Inconsistent Section Numbering

**Location:** `Phase 5` section (line 625)

**Issue:** Section is labeled "Phase 5" but the subsection is "5.1", and then "3.2" appears (should be "5.2").

**Current:**
```markdown
### Phase 5: Integration with Environment

#### 5.1 Pass Identity Config to Observation Spec
...
#### 3.2 Handle Edge Cases  # ❌ Should be 5.2
```

**Fix:** Change "3.2" to "5.2" on line 663.

---

### 8. Missing Step 3.4 in Complete Code Example

**Location:** `Step 3: Visual Field Processing` - Complete code example

**Issue:** The code jumps from Step 3.3 to Step 3.5, but Step 3.4 is mentioned in comments. The actual Step 3.4 code is present but the comment numbering might be confusing.

**Current:** The code has Step 3.4 defined correctly, but the flow could be clearer.

**Fix:** Verify all step numbers are sequential and match the actual code blocks.

---

## Summary

**Critical Issues (Must Fix):**
1. ✅ Duplicate code blocks (lines 337-355)
2. ✅ Missing obs_y, obs_x definition (line 418)
3. ✅ Inconsistent indentation (line 1611)
4. ✅ Step numbering inconsistency (line 1618)

**Moderate Issues (Should Fix):**
5. ⚠️ Outdated code examples (multiple locations)
6. ⚠️ Missing import statement (line 1632)

**Minor Issues (Nice to Fix):**
7. ⚠️ Inconsistent section numbering (line 663)
8. ⚠️ Missing step verification (throughout)

---

## Recommended Fix Order

1. Fix duplicate code blocks (Issue #1) - **CRITICAL**
2. Fix missing variable definitions (Issue #2) - **CRITICAL**
3. Fix indentation (Issue #3) - **CRITICAL**
4. Fix step numbering (Issue #4) - **CRITICAL**
5. Update outdated examples (Issue #5) - **MODERATE**
6. Add import statement (Issue #6) - **MODERATE**
7. Fix section numbering (Issue #7) - **MINOR**
8. Verify all steps (Issue #8) - **MINOR**

