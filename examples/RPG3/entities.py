from collections import deque
from ast import literal_eval as make_tuple

# TODO: 
# get rid of dead agent and just make it a function of an agent
# what's the deal with these entities having vision of 1 --- I assume its for it to interact with agents when they move into that space?
# come back here to see if any entities require transiton

class EmptyObject:
    def __init__(self):
        self.appearance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.vision = 1
        self.value = 0  
        self.passable = 1  # whether the object blocks movement
        self.action_type = "empty"

class Wall:
    def __init__(self):
        self.appearance = [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.vision = 0  
        self.value = -0.1  
        self.passable = 0  
        self.action_type = "static"  # rays disappear after one turn

class Gem():
    def __init__(self, cfg, type):
        self.health = 1  # for the gen, whether it has been mined or not
        self.vision = 1  # gems can see one radius around them
        self.passable = 1  # whether the object blocks movement
        self.action_type = "static"
        if type == 1:
            self.appearance = make_tuple(cfg.entity.Gem1.appearance)
            self.value = cfg.entity.Gem1.value
        elif type == 2:
            self.appearance = make_tuple(cfg.entity.Gem2.appearance)
            self.value = cfg.entity.Gem2.value
        elif type == 3:
            self.appearance = make_tuple(cfg.entity.Gem3.appearance)
            self.value = cfg.entity.Gem3.value