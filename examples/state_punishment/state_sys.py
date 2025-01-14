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


class monitor():
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
            'check_index'
        ])
        self.max_duration_between_checks = max_duration_between_checks # duration between two checks
        self.time = 0 # clock to record the time 
        self.check_time = 0 # ?
        self.wait_for_check = False # waiting for check
        self.resource_to_monitor = resource_to_monitor # the type of resource to monitor 
        self.wait_time = 0 # 
        self.check_index = 0 # the number of checks undergone 
        self.size_of_env = env_size # size of the environment
    
    def get_taxicab_distance_points(self, a, b, distance, size_of_env):
        """
        get the taxicab distance between two locations in the environment. 
        """
        points = []

        # Iterate over all possible dx values within the distance range [0, X]
        for dx in range(distance + 1):
            # Corresponding dy based on the taxicab distance formula
            dy = distance - dx

            # Generate points for all combinations of +dx/-dx and +dy/-dy
            for sign_x in [1, -1]:
                for sign_y in [1, -1]:
                    x = a + sign_x * dx
                    y = b + sign_y * dy
                    
                    # Clip the coordinates within the range [0, l]
                    x = max(0, min(x, size_of_env))
                    y = max(0, min(y, size_of_env))
                    
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
        if self.time == self.check_time:
            new_rows['being_checked'] = True # 'being_checked' become True
            new_rows['check_index'] = self.check_index
        else:
            new_rows['being_checked'] = False
            # new_rows['check_index']
        # self.record[len(self.record)] = new_rows
        self.record = pd.concat([self.record, new_rows], ignore_index=True)
    

    def regular_check(self, timepoint, resource_type, state_sys, agents):
        """
        At time T, check through all agents for a specific type of transgression. Punish
        the transgressors.
        """
        if (self.time == self.check_time) and (self.time != 0) :
            
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
                print('this', resource_type, last_resource_state, filtered_record)
                ll
                for loc in last_resource_state['location']:
                    if (
                        resource_type
                        !=
                        (filtered_record[filtered_record.location == loc]['type'].iloc[0])
                    ):
                        print('punish')
                        ll
                        region_of_suspect = self.get_taxicab_distance_points(
                            loc[0], 
                            loc[1], 
                            self.wait_time, 
                            self.size_of_env
                            )
                        for agent in agents: ##TODO if there are multiple agents in the region, who should be punished? Currently the program punishes all
                            print(agent.location, region_of_suspect)
                            ll
                            if agent.location in region_of_suspect:
                                if state_sys.prob * 1. + 1. > random.random(): # the prob of the agent made the transgression * the prob of punishing it given the transgression
                                    agent.to_be_punished[resource_type] += 1 #TODO: to_be_punished should be a dict {a: 0, b:0, ...}
            self.wait_for_check = False
        self.check_index += 1 #TODO check whether this is correct
        

        

    def regular_check_all_resources(self, timepoint, state_sys, agents):
        """
        At time T, check through all resources types for any transgressions. 
        """
        for resource_type in self.resource_to_monitor:
            self.regular_check(timepoint, resource_type, state_sys, agents)
        self.time += 1
    
                        
## before agent loop: collect_new_state_info(), update(), time_next_check()
## after the agent loop: regular_check_all_resources(), 


## unit tests:
## game rules: 1. during regular checks, when it is found that a taboo disappears, the system draws a region around the original location (1), punish agents
## within the region (2), also need to inspect the dataframe (3) 