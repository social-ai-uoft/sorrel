import pandas as pd
import random 
import numpy as np
from examples.state_punishment.utils import inspect_the_env, add_extra_info

class state_sys():
    def __init__(self, init_prob, prob_list, magnitude, taboo, change_per_vote) -> None:
        self.prob = init_prob
        self.prob_list = prob_list
        self.init_prob = init_prob 
        self.magnitude = magnitude
        self.taboo = taboo
        self.change_per_vote = change_per_vote
        self.prob_record = []
    
    def reset_prob_record(self):
        self.prob_record = []

    def update_prob_record(self):
        self.prob_record.append(self.prob)


class Monitor():
    def __init__(
            self, 
            max_duration_between_checks,
            resource_to_monitor,
            env_size
            ):
        self.record = pd.DataFrame(columns=[
            'type',
            'location',
            'over',
            'being_checked',
            'timepoint',
            'check_index', 
        ])
        self.max_duration_between_checks = max_duration_between_checks # duration between two checks
        self.time = 0 # clock to record the time 
        self.check_time = 0 # ?
        self.wait_for_check = False # waiting for check
        self.resource_to_monitor = resource_to_monitor # the type of resource to monitor 
        self.wait_time = 0 # 
        self.check_index = 0 # the number of checks undergone 
        self.size_of_env = env_size # size of the environment
        self.check_time_record = []
        self.time_record = []
        self.signal_detection_record = pd.DataFrame(columns=[
            'location',
            'punished_agent',
            'current_time',
            'last_check_time',
        ])
    

    def clear_mem(self, time):
        self.record = self.record[self.record.timepoint >= (time-2*self.max_duration_between_checks)]
    
    def get_taxicab_distance_points(self, a, b, distance, size_of_env):
        """
        Generate region of suspect. 
        This version does not consider what was observed in last check. Therefore, 
        any agents that have a distance of smaller than X are considered as guilty. 

        if last observation is considered, we need to calculate the possible current 
        locations of that agent considering the last location.
        """
        points = []

        # Iterate over all possible dx values within the distance range [0, X]
        for dx in range(-distance, distance + 1):
            for dy in range(-distance + abs(dx), distance - abs(dx) + 1):
                # Calculate new position
                x, y = a + dx, b + dy
                # Clip the coordinates within the range [0, l]
                x = max(0, min(x, size_of_env-1))
                y = max(0, min(y, size_of_env-1))
                
                points.append((x, y))

        # Remove duplicates (in case some points have been added more than once)
        points = list(set(points))
        return points
    

    def time_of_next_check(self):
        """
        Figure out the time of the next check
        """
        if not self.wait_for_check:
            time = random.randint(1, self.max_duration_between_checks)
            self.check_time = self.time + time 
            self.wait_time = time
            self.wait_for_check = True   
    
    
    def collect_new_state_info(self, env, types, turn, done):
        """
        Get the info of the new state.
        """
        df = inspect_the_env(env, types)
        df = add_extra_info(df, turn, done)
        return df


    def update(self, new_rows):
        """
        Add a new line of record to the dataframe 
        """
        if (
            (self.time == self.check_time) 
            or 
            (self.time == 0)
        ):
            new_rows['being_checked'] = True # 'being_checked' become True
            new_rows['check_index'] = self.check_index
        else:
            new_rows['being_checked'] = False
            # new_rows['check_index']
        # self.record[len(self.record)] = new_rows
        self.record = pd.concat([self.record, new_rows], ignore_index=True)
        self.check_time_record.append(self.check_time)
        self.time_record.append(self.wait_time)

    def regular_check(self, timepoint, resource_type, state_sys, agents):
        """
        At time T, check through all agents for a specific type of transgression. Punish
        the transgressors.
        """
        if (self.time == self.check_time):
            filtered_record = self.record[
                (self.record.over == False) &
                (self.record.being_checked == True) &
                (self.record.timepoint == self.time)
            ]
            # print(len(self.record[(self.record.over == False) &
            #     (self.record.being_checked == True) &
            #     (self.record.timepoint == self.time)]))
            # print(self.time, self.record['timepoint'].unique())
            # ll
            # print(timepoint, self.record['timepoint'].unique())
            
            if len(filtered_record) > 0:
                
                index_of_last_check = self.check_index - 1
                last_resource_state = self.record[
                    (self.record.over == False) &
                    (self.record.being_checked == True) &
                    (self.record.type == resource_type) & 
                    (self.record.check_index == index_of_last_check)   
                ]
                # print('this', self.check_time, self.time, self.wait_time, index_of_last_check, resource_type, last_resource_state, filtered_record)
                
                for loc in list(set(last_resource_state['location'])):
                    # print('here', filtered_record[filtered_record.location == loc])
                    # ll
                    if (
                        resource_type
                        !=
                        (filtered_record[filtered_record.location == loc]['type'].iloc[0])
                    ):
                        # print('punish')
                        # ll
                        region_of_suspect = self.get_taxicab_distance_points(
                            loc[0], 
                            loc[1], 
                            self.wait_time, 
                            self.size_of_env
                            )

                        suspects = []
                        for agent_ind, agent in enumerate(agents): ##TODO if there are multiple agents in the region, who should be punished? Currently the program punishes all
                            if (agent.location[0], agent.location[1]) in region_of_suspect:
                                if state_sys.prob * 1. > random.random(): # the prob of the agent made the transgression * the prob of punishing it given the transgression
                                    agent.to_be_punished[resource_type] += 1 
                                    new_row = [loc, agent.ixs, self.time, self.time-self.check_time]
                                    self.signal_detection_record.iloc[len(self.signal_detection_record)] = new_row
                        
                        # randomly decide the transgressor #TODO if one agent is in multiple regions of suspect
                        # judge = random.choice(suspects)
                        # agent_of_suspect = agents[judge]
                        # if state_sys.prob * 1. > random.random():
                        #     agent_of_suspect.to_be_punished[resource_type] += 1

                                    
            self.wait_for_check = False
            self.check_index += 1 #TODO check whether this is correct

        if self.time == 0:
            self.check_index += 1
        

        

    def regular_check_all_resources(self, timepoint, state_sys, agents):
        """
        At time T, check through all resources types for any transgressions. 
        """
        
        for resource_type in self.resource_to_monitor:
            self.regular_check(timepoint, resource_type, state_sys, agents)
        
    
                        
## before agent loop: collect_new_state_info(), update(), time_next_check()
## after the agent loop: regular_check_all_resources(), 


## unit tests:
## game rules: 1. during regular checks, when it is found that a taboo disappears, the system draws a region around the original location (1), punish agents
## within the region (2), also need to inspect the dataframe (3) 