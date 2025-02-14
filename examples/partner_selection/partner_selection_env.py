import numpy as np
import random 
from copy import deepcopy

class partner_pool:
    def __init__(self, agents):
        self.partner_to_select = []
        self.partner_to_select_appearance = None
        self.pool = agents
    
    def agents_sampling(self, focal_agent=None):
        """
        Sample two agents as potential partner choices and one agent as the focal agent.
        """
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
        self.partner_to_select = partner_choices
        self.focal_ixs = focal_agent.ixs
        self.partner_to_select_appearance = np.concat([partner.appearance for partner in partner_choices])

        return focal_agent, partner_choices, partner_ixs
    
    def state(self, agent):
        assert len(self.partner_to_select) == 2, 'number of partners to select larger than 2'
        if agent.ixs == self.focal_ixs:
            state = np.concat([self.partner_to_select_appearance, np.array([1])])
            # add variability
            state = np.concat([state, np.array ([agent.variability])])
            
            for partner in self.partner_to_select:
                state = np.concat([state, np.array([partner.variability])])
                state = np.concat([state, np.array(partner.appearance)])
        else:
            state = np.concat([self.partner_to_select_appearance, np.array([0])])
            # add variability
            state = np.concat([state, np.array ([agent.variability])])
            # print('len, partner_to_select', len(self.partner_to_select))
            for partner in self.partner_to_select:
                state = np.concat([state, np.array([partner.variability])])
                state = np.concat([state, np.array(partner.appearance)])
        return state
    
    def get_max_variability_partner_ixs(self):
        """
        Get the ixs of the partner among all options.
        """
        variability = [partner.variability for partner in self.partner_to_select]
        
        # Sort indices based on variability values in descending order
        sorted_ixs = sorted(range(len(variability)), key=lambda x: variability[x], reverse=True)
        
        # Get sorted variability values
        sorted_vals = [variability[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[0]].ixs
        
        return agent_ixs
    
    def get_min_variability_partner_ixs(self):
        """
        Get the ixs of the partner among all options.
        """
        variability = [partner.variability for partner in self.partner_to_select]
        
        # Sort indices based on variability values in descending order
        sorted_ixs = sorted(range(len(variability)), key=lambda x: variability[x], reverse=True)
        
        # Get sorted variability values
        sorted_vals = [variability[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[-1]].ixs

        agent = self.partner_to_select[sorted_ixs[-1]]
        
        return agent, agent_ixs

    
    