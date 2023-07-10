from collections import deque

# TODO: 
# get rid of dead agent and just make it a function of an agent
# what's the deal with these entities having vision of 1 --- I assume its for it to interact with agents when they move into that space?
# come back here to see if any entities require transiton

class EmptyObject:
    def __init__(self):
        self.appearance = [255, 255, 255]  # empty is white
        self.sprite = 'examples/RPG/assets/white.png'
        self.vision = 1  # empty stuff is basically empty
        self.value = 0  # empty stuff is basically empty
        self.passable = 1  # whether the object blocks movement
        self.has_transitions = False
        self.action_type = "empty"

class Collectable:
    pass

class Wall:
    def __init__(self):
        self.appearance = [153.0, 51.0, 102.0]  # walls are purple
        self.sprite = 'examples/RPG/assets/pink.png'
        self.vision = 0  # wall stuff is basically empty
        self.value = -0.1  # wall stuff is basically empty
        self.static = 1  # wall stuff is basically empty
        self.passable = 0  # you can't walk through a wall
        self.has_transitions = False
        self.action_type = "static"  # rays disappear after one turn

class Gem(Collectable):
    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearance = color  # gems are green
        self.sprite = 'examples/RPG/assets/gem.png'
        self.vision = 1  # gems can see one radius around them
        self.value = value  # the value of this gem
        self.static = 1  # whether the object gets to take actions or not
        self.passable = 1  # whether the object blocks movement
        self.has_transitions = False
        self.action_type = "static"

class Coin(Collectable):
    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearance = color  # gems are green
        self.sprite = 'examples/RPG/assets/coin.png'
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.passable = 1  # whether the object blocks movement
        self.has_transitions = False
        self.action_type = "static"

class Food(Collectable):
    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearance = color  # gems are green
        self.sprite = 'examples/RPG/assets/food.png'
        self.vision = 1  # gems can see one radius around them
        self.value = value  # the value of this gem
        self.passable = 1  # whether the object blocks movement
        self.has_transitions = False
        self.action_type = "static"

class Bone(Collectable):
    def __init__(self, value, color):
        super().__init__()
        self.health = 1  # for the gen, whether it has been mined or not
        self.appearance = color  # gems are green
        self.sprite = 'examples/RPG/assets/bone.png'
        self.vision = 1  # gems can see one radius around them
        self.policy = "NA"  # gems do not do anything
        self.value = value  # the value of this gem
        self.passable = 1  # whether the object blocks movement
        self.has_transitions = False
        self.action_type = "static"