from agentarium.primitives import Entity
import random

# ----------------------------------------------------- #
# region: Environment object classes for Baker ToM task #
# ----------------------------------------------------- #

class EmptyObject(Entity):
    '''
    Base empty object.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.passable = True # EmptyObjects can be traversed
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/white.png'
        # self.appearance = cfg.entity.EmptyObject.appearance
        self.respawn_rate = cfg.env.prob.respawn_rate
        self.type = 'emptyobject'

    def transition(self, env):
        '''
        Transition function for EmptyObjects in the state_punishment environment. 
        EmptyObjects can randomly spawn into Gems, Coins, etc. according
        to the item spawn probabilities dictated in the environmnet.
        '''
        
        if random.random() < env.item_spawn_prob * self.respawn_rate: # NOTE: If this rate is too high, the environment gets overrun
            env.spawn(self.location)

class Wall(Entity):
    '''
    Base wall object.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.value = -1 # Walls penalize contact
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/pink.png'
        self.type = 'wall'
        # self.appearance = cfg.entity.Wall.appearance

class Gem(Entity):
    '''
    Base gem object.

    Parameters:
        appearance: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.cfg = cfg
        self.value = cfg.entity.Gem.value
        self.passable = True
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/gem.png'
        self.social_harm = cfg.entity.Gem.social_harm
        self.type = 'gem'
        # self.appearance = cfg.entity.Gem.appearance

# class Coin(Entity):
#     '''
#     Base gem object.

#     Parameters:
#         appearance: The appearance of the gem. \n
#         cfg: The configuration object.
#     '''
#     def __init__(self, appearance, cfg):
#         super().__init__(appearance)
#         self.cfg = cfg
#         self.value = cfg.entity.Coin.value
#         self.passable = True
#         self.sprite = f'{cfg.root}/examples/state_punishment/assets/coin.png'

# class Food(Entity):
#     '''
#     Base gem object.

#     Parameters:
#         appearance: The appearance of the gem. \n
#         cfg: The configuration object.
#     '''
#     def __init__(self, appearance, cfg):
#         super().__init__(appearance)
#         self.cfg = cfg
#         self.value = cfg.entity.Food.value
#         self.passable = True
#         self.sprite = f'{cfg.root}/examples/state_punishment/assets/food.png'

class Bone(Entity):
    '''
    Base gem object.

    Parameters:
        appearance: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.cfg = cfg
        self.value = cfg.entity.Bone.value
        self.passable = True
        self.social_harm = cfg.entity.Bone.social_harm
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/bone.png'
        self.type = 'bone'

# ----------------------------------------------------- #
# new entities for state_punishment                     #
# ----------------------------------------------------- # 

class Coin(Entity):
    '''
    Base gem object.

    Parameters:
        appearance: The appearance of the gem. \n
        cfg: The configuration object.
    '''
    def __init__(self, appearance, cfg):
        super().__init__(appearance)
        self.cfg = cfg
        self.value = cfg.entity.Coin.value
        self.passable = True
        self.social_harm = cfg.entity.Coin.social_harm
        self.sprite = f'{cfg.root}/examples/state_punishment/assets/coin.png'
        # self.appearance = cfg.entity.Coin.appearance
        self.type = 'coin'

# ----------------------------------------------------- #
# endregion                                             #
# ----------------------------------------------------- #