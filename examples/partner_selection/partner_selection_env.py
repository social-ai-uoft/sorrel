import numpy as np
import random 
from copy import deepcopy
from scipy.stats import entropy

class partner_pool:
    def __init__(self, agents):
        self.partner_to_select = []
        self.partner_to_select_appearance = None
        self.pool = agents
        self.time = 0
    
    def agents_sampling(self, focal_agent=None, default=True):
        """
        Sample two agents as potential partner choices and one agent as the focal agent.
        """
        if not default:
            # sample all needed agents
            if focal_agent:
                focal_agent_ixs = focal_agent.ixs
                qualified_pool = [agent for agent in self.pool if agent.ixs != focal_agent_ixs]
                sampled_agents = random.sample(qualified_pool, 2)
                partner_choices = sampled_agents
            else:
                sampled_agents = random.sample(self.pool, 3)
                sampled_agents_indices = [agent.ixs for agent in sampled_agents]
                # pick the focal agent
                focal_agent_ixs = random.sample([i for i in range(len(sampled_agents_indices))], 1)[0]
                focal_agent = sampled_agents[focal_agent_ixs]
                partner_choices = [sampled_agents[i] for i in range(len(sampled_agents)) 
                                if i != focal_agent_ixs]
            partner_ixs = [a.ixs for a in partner_choices]
            self.partner_to_select = deepcopy(partner_choices)
            self.focal_ixs = focal_agent.ixs
            self.partner_to_select_appearance = np.concat([partner.appearance for partner in partner_choices])
        # else:
        #     focal_agent = [a for a in self.pool if a.ixs == 0][0]
        #     qualified_pool = [a for a in self.pool if a.ixs != 0]
        #     sampled_agents = random.sample(qualified_pool, 2)
        #     partner_ixs = [a.ixs for a in partner_choices]
        #     self.focal_ixs = 0 
        #     self.partner_to_select = deepcopy(partner_choices)
        #     self.partner_to_select_appearance = np.concat([partner.appearance for partner in partner_choices])
        # update time
        self.time += 1

        return focal_agent, partner_choices, partner_ixs
    
    def state(self, agent, cfg):
        assert len(self.partner_to_select) == 2, 'number of partners to select larger than 2'

        # add stage marker
        if agent.ixs == self.focal_ixs:
            if cfg.with_partner_to_select_appearance:
                state = np.concatenate([self.partner_to_select_appearance, np.array([1])])
            else:
                state = np.concatenate([self.partner_to_select_appearance*0, np.array([1])])
        else:
            if cfg.with_partner_to_select_appearance:
                state = np.concatenate([self.partner_to_select_appearance, np.array([0])])
            else:
                state = np.concatenate([self.partner_to_select_appearance*0, np.array([0])])
        
        if not cfg.experiment.is_SB_task:
            # add variability
            if cfg.with_self_variability:
                state = np.concatenate([state, np.array ([agent.variability])])
            else:
                state = np.concatenate([state, np.array([0])])
            # add partner variability
            for partner in self.partner_to_select:
                if cfg.with_partner_variability:
                    state = np.concatenate([state, np.array([partner.variability])])
                else:
                    state = np.concatenate([state, np.array([0])])
                if cfg.with_partner_to_select_appearance:
                    state = np.concatenate([state, np.array(partner.appearance)])
                else:
                    state = np.concatenate([state, np.array(partner.appearance)*0])
        else:
            # add preferences
            if cfg.with_self_preferences:
                state = np.concatenate([state, np.array(agent.preferences)])
            else:
                state = np.concatenate([state, np.array([0,0])])
            # add partner preferences
            for partner in self.partner_to_select:
                if cfg.with_partner_preferences:
                    state = np.concatenate([state, np.array(partner.preferences)])
                else:
                    state = np.concatenate([state, np.array([0,0])])
                if cfg.with_partner_to_select_appearance:
                    state = np.concatenate([state, np.array(partner.appearance)])
                else:
                    state = np.concatenate([state, np.array(partner.appearance)*0])


        # add time
        state = np.concatenate([state, np.array([self.time])])
      
        return state
    
    def get_max_variability_partner_ixs(self):
        """
        Get the ixs of the most variable partner among all options.
        """
        variability = [partner.variability for partner in self.partner_to_select]
        
        # Sort indices based on variability values in descending order
        sorted_ixs = sorted(range(len(variability)), key=lambda x: variability[x], reverse=True)
        
        # Get sorted variability values
        sorted_vals = [variability[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[0]].ixs
        
        return agent_ixs
    
    def get_max_entropic_partner_ixs(self):
        """
        Get the ixs of the most entropic partner among all options.
        """
        entropy_val = [entropy(partner.preferences) for partner in self.partner_to_select]

        # Sort indices based on entropy values in descending order
        sorted_ixs = sorted(range(len(entropy_val)), key=lambda x: entropy_val[x], reverse=True)

        # Get sorted entropy values
        sorted_vals = [entropy_val[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[0]].ixs

        agent = self.partner_to_select[sorted_ixs[0]]

        return agent, agent_ixs
    
    def get_min_variability_partner_ixs(self):
        """
        Get the ixs of the least variable partner among all options.
        """
        variability = [partner.variability for partner in self.partner_to_select]
        
        # Sort indices based on variability values in descending order
        sorted_ixs = sorted(range(len(variability)), key=lambda x: variability[x], reverse=True)
        
        # Get sorted variability values
        sorted_vals = [variability[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[-1]].ixs

        agent = self.partner_to_select[sorted_ixs[-1]]
        
        return agent, agent_ixs

    
    