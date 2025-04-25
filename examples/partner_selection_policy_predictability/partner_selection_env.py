import numpy as np
import random 
from copy import deepcopy
from scipy.stats import entropy
from scipy.special import softmax

class partner_pool:
    def __init__(self, agents):
        self.partner_to_select = []
        self.partner_to_select_appearance = None
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
                    # sampled_agents = random.sample(qualified_pool, 2)
                    partner_choices = sampled_agents
            else:
                # sampled_agents = random.sample(self.pool, 3)
                # sampled_agents_indices = [agent.ixs for agent in sampled_agents]
                # # pick the focal agent
                # focal_agent_ixs = random.sample([i for i in range(len(sampled_agents_indices))], 1)[0]
                # focal_agent = sampled_agents[focal_agent_ixs]
                # partner_choices = [sampled_agents[i] for i in range(len(sampled_agents)) 
                #                 if i != focal_agent_ixs]
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
                # except:
                #     print(f'{len(qualified_pool)}, {len(probs)}, {len(self.pool)}, {len(probs)}, {probs}')

            partner_ixs = [a.ixs for a in partner_choices]
            self.partner_to_select = deepcopy(partner_choices)
            self.focal_ixs = focal_agent.ixs
            self.partner_to_select_appearance = np.concatenate([partner.appearance for partner in partner_choices])
        # else:
        #     focal_agent = [a for a in self.pool if a.ixs == 0][0]
        #     qualified_pool = [a for a in self.pool if a.ixs != 0]
        #     sampled_agents = random.sample(qualified_pool, 2)
        #     partner_ixs = [a.ixs for a in partner_choices]
        #     self.focal_ixs = 0 
        #     self.partner_to_select = deepcopy(partner_choices)
        #     self.partner_to_select_appearance = np.concatenate([partner.appearance for partner in partner_choices])
        # update time
        self.time += 1

        return focal_agent, partner_choices, partner_ixs
    
    def state(self, agent, cfg):
        assert len(self.partner_to_select) == 2, 'number of partners to select larger than 2'
        # print('agent ixs', [a.ixs for a in self.partner_to_select])
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
                state = np.concatenate([state, 10*np.array(agent.preferences)])
            else:
                state = np.concatenate([state, np.array([0,0])])
            # add partner preferences & appearances
            if cfg.random_selection:
                selected_partner = random.choices(self.partner_to_select, k=1)[0]
                selected_partner_ixs = selected_partner.ixs

            for partner in self.partner_to_select:

                if cfg.random_selection:
                    partner = selected_partner

                if cfg.with_partner_preferences:
                    state = np.concatenate([state, np.array(partner.preferences)])
                else:
                    state = np.concatenate([state, np.array([0,0])])
                
                if cfg.with_partner_entropy:
                    state = np.concatenate([state, np.array([partner.entropy*100])])
                else:
                    state = np.concatenate([state, np.array([0])])

                if cfg.with_partner_to_select_appearance:
                    state = np.concatenate([state, np.array(partner.appearance)])
                else:
                    state = np.concatenate([state, np.array(partner.appearance)*0])

        # add marker of being selected
        state = np.concatenate([state, np.array([1.*agent.selected_in_last_turn])])
        # add time
        if cfg.with_time:
            state = np.concatenate([state, np.array([self.time])])
        else:
            state = np.concatenate([state, np.array([0])])
        # add internal state
        state = np.concatenate([state, np.array([agent.internal_state])])

        if not cfg.random_selection:
            return state
        else:
            return state, selected_partner_ixs
    
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
        entropy_val = [partner.entropy for partner in self.partner_to_select]

        # Sort indices based on entropy values in descending order
        sorted_ixs = sorted(range(len(entropy_val)), key=lambda x: entropy_val[x], reverse=True)

        # Get sorted entropy values
        sorted_vals = [entropy_val[j] for j in sorted_ixs]

        agent_ixs = self.partner_to_select[sorted_ixs[0]].ixs

        agent = self.partner_to_select[sorted_ixs[0]]

        entropy_val_diff_among_partner_choices = abs(
            self.partner_to_select[0].entropy - self.partner_to_select[1].entropy
            )
        # print(agent_ixs, entropy_val_diff_among_partner_choices, [(partner.ixs, partner.entropy) for partner in self.partner_to_select])
        return agent, agent_ixs, entropy_val_diff_among_partner_choices
    
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

    
    