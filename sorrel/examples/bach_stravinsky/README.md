# Bach-Stravinsky (Battle of the Sexes)

A gridworld game simulating the classic Bach-Stravinsky/"Battle of the Sexes" coordination game. Two agents must coordinate to attend the same concert, but they have different preferences.

## Game Logic

- **Concerts**: Concert objects spawn randomly in the environment.
- **Coordination**: Agents must "zap" a concert simultaneously to attend.
- **Beams**: Agents can choose between two types of zaps:
    - **BachBeam**: Represents choosing the Bach option.
    - **StravinskyBeam**: Represents choosing the Stravinsky option.
- **Rules**:
    - A concert must be hit twice to be consumed.
    - If both hits are `BachBeam`, it resolves as a Bach concert.
    - If both hits are `StravinskyBeam`, it resolves as a Stravinsky concert.
    - Mixed hits result in 0 reward (coordination failure).
- **Preferences**:
    - **Agent 0** prefers Bach (+5) over Stravinsky (+1).
    - **Agent 1** prefers Stravinsky (+5) over Bach (+1).

## Running the example

```bash
python sorrel/examples/bach_stravinsky/main.py
```

The script runs the experiment, logging metrics to TensorBoard and saving data to `data/`.

## Project structure

```
bach_stravinsky/
├─ assets/                    # Sprite images
├─ data/                      # Generated logs and outputs
│   ├─ checkpoints/
│   ├─ gifs/
│   └─ logs/
├─ agents.py                  # BachStravinskyAgent implementation
├─ entities.py                # Concert, Beam, and other entity definitions
├─ env.py                     # Environment wrapper and setup
├─ main.py                    # Experiment entry point and configuration
├─ metrics_collector.py       # Custom metrics tracking (beam usage, rewards)
└─ world.py                   # World definition and parameters
```
