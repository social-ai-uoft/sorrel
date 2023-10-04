from collections import deque
from ast import literal_eval as make_tuple

# TODO: 
# get rid of dead agent and just make it a function of an agent
# what's the deal with these entities having vision of 1 --- I assume its for it to interact with agents when they move into that space?
# come back here to see if any entities require transiton

class EmptyObject:
    def __init__(self):
        self.apperance = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.appearance = [255, 255, 255]  # empty is white
        # self.sprite = 'examples/RPG/assets/white.png'
        self.vision = 1  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.passable = 1  # whether the object blocks movement
        self.action_type = "empty"

class Wall:
    def __init__(self):
        self.appearance = [0.0, 255.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.appearance = [153.0, 51.0, 102.0]  # walls are purple
        # self.sprite = 'examples/RPG/assets/pink.png'
        self.vision = 0  # wall stuff is basically empty
        self.value = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.action_type = "static"  # rays disappear after one turn

class Gem():
    def __init__(self, cfg, type):
        self.health = 1  # for the gen, whether it has been mined or not
        self.sprite = 'examples/RPG/assets/gem.png'
        self.vision = 1  # gems can see one radius around them
        self.static = 1  # whether the object gets to take actions or not
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