import numpy as np
import random 
from copy import deepcopy
from scipy.stats import entropy
from scipy.special import softmax

class partner_pool:
    def __init__(self, agents):
        self.partner_to_select = []
        self.appearance_of_others = None
        self.pool = agents
        self.time = 0
    
    def agents_sampling(self, focal_agent=None, default=False, cfg=None, epoch=None):
        """
        Sample two agents as potential partner choices and one agent as the focal agent.
        """
        if not default:
            # sample all needed agents
            if focal_agent:
                if cfg.study == 1:
                    focal_agent_ixs = focal_agent.ixs
                    qualified_pool = [agent for agent in self.pool if agent.ixs != focal_agent_ixs]
                    sampled_agents = [agent for agent in self.pool if agent.ixs in [1, 2]]
                    if epoch % 1 == 0:
                        random.shuffle(sampled_agents)
                    partner_choices = sampled_agents
            else:
                sampled_agents = random.sample(self.pool, 1)
                focal_agent = sampled_agents[0]
                focal_agent_ixs = focal_agent.ixs
                qualified_pool = [agent for agent in self.pool if agent.ixs != focal_agent_ixs]
                probs = softmax(focal_agent.sampling_weight)
                if np.count_nonzero(probs) == 1:
                    first_idx = np.where(probs == 1.0)[0][0]
                    first_choice = qualified_pool[first_idx]
                    # Pick second randomly from the remaining items
                    remaining_pool = np.delete(qualified_pool, first_idx)
                    second_choice = np.random.choice(remaining_pool)
                    partner_choices = [first_choice, second_choice]
                else:
                    partner_choices = np.random.choice(qualified_pool, p=probs, size=2, replace=False)
              
            partner_ixs = [a.ixs for a in partner_choices]
            self.partner_to_select = deepcopy(partner_choices)
            self.focal_ixs = focal_agent.ixs

            if cfg.only_show_available_options:
                self.appearance_of_others = np.concatenate([partner.appearance for partner in partner_choices])
            else:
                self.appearance_of_others = np.concatenate([agent.appearance for agent in self.pool])
            
            self.appearance_of_current_options = np.concatenate([partner.appearance for partner in partner_choices])
            
        # update time
        self.time += 1

        return focal_agent, partner_choices, partner_ixs
    
    def state(self, focal_agent, agent_list, cfg, partner=None): #TODO: block num + stage number + step number + self icon + partner icon + other icons  
        
        # add time marker
        state = np.array([])
        state = np.concatenate([state, np.array([self.block])]) 
        state = np.concatenate([state, np.array([self.stage])])
        state = np.concatenate([state, np.array([self.step])])

        

        # add partner icon
        if partner is None:
            state = np.concatenate([state, np.zeros(focal_agent.generate_icon().shape)])
        else:
            state = np.concatenate([state, partner.generate_icon()])
       
        # add agent icon
        state = np.concatenate([state, focal_agent.generate_icon()])
        # add all agents' icons
        for agent in agent_list:
            state = np.concatenate([state, agent.generate_icon()])

        # add partner preferences & appearances
        if cfg.random_selection:
            selected_partner = random.choices(self.partner_to_select, k=1)[0]
            selected_partner_ixs = selected_partner.ixs

        # # add marker of being selected
        # state = np.concatenate([state, np.array([1.*focal_agent.selected_in_last_turn])])

        if not cfg.random_selection:
            return state
        else:
            return state, selected_partner_ixs
    
 