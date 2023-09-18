from gem.primitives import Object
import random

# ----------------------------------------------------- #
# region: Environment object classes for Baker ToM task #
# ----------------------------------------------------- #

class EmptyObject(Object):
    '''
    Base empty object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.passable = True # EmptyObjects can be traversed
        self.sprite = f'{cfg.root}/examples/RPG/assets/white.png'

    def transition(self, env):
        '''
        Transition function for EmptyObjects in the RPG environment. 
        EmptyObjects can randomly spawn into Gems, Coins, etc. according
        to the item spawn probabilities dictated in the environmnet.
        '''
        
        if random.random() < env.item_spawn_prob * 0.02: # NOTE: If this rate is too high, the environment gets overrun
            env.spawn(self.location)

class Wall(Object):
    '''
    Base wall object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.value = -1 # Walls penalize contact
        self.sprite = f'{cfg.root}/examples/RPG/assets/pink.png'

class Gem(Object):
    '''
    Base gem object.

    Parameters:
        color: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.cfg = cfg
        self.value = cfg.entity.Gem.value
        self.passable = True
        self.sprite = f'{cfg.root}/examples/RPG/assets/gem.png'

class Coin(Object):
    '''
    Base gem object.

    Parameters:
        color: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.cfg = cfg
        self.value = cfg.entity.Coin.value
        self.passable = True
        self.sprite = f'{cfg.root}/examples/RPG/assets/coin.png'

class Food(Object):
    '''
    Base gem object.

    Parameters:
        color: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.cfg = cfg
        self.value = cfg.entity.Food.value
        self.passable = True
        self.sprite = f'{cfg.root}/examples/RPG/assets/food.png'

class Bone(Object):
    '''
    Base gem object.

    Parameters:
        color: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, color, cfg):
        super().__init__(color)
        self.cfg = cfg
        self.value = cfg.entity.Bone.value
        self.passable = True
        self.sprite = f'{cfg.root}/examples/RPG/assets/bone.png'

# ----------------------------------------------------- #
# endregion                                             #
# ----------------------------------------------------- #