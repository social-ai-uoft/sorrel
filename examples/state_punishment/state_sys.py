import pandas as pd
import random 

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


def monitor():
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
            'is_existing',
            'timepoint',
            'check_index'
        ])
        self.max_duration_between_checks = max_duration_between_checks
        self.time = 0 
        self.check_time = 0
        self.wait_for_check = False
        self.resource_to_monitor = resource_to_monitor
        self.wait_time = 0
        self.check_index = 0
        self.size_of_env = env_size
    
    def get_taxicab_distance_points(a, b, distance, size_of_env):
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
        if not self.wait_for_check:
            time = random.randint(1, self.max_duration_between_checks)
            self.check_time = self.time + time 
            self.wait_time = time
            self.time += 1
            self.wait_for_check = True

    def update(self, new_row):
        if self.time == self.check_time:
            new_row[3] = True
            self.check_index += 1
        self.record[len(self.record)] = new_row
    
    def regular_check(self, timepoint, resource_type, state_sys, agents):
        filtered_record = self.record[
            (self.record.over == False) &
            (self.record.being_checked == True) &
            (self.record.type == resource_type) & 
            (self.record.timepoint == timepoint)
        ]
        if len(filtered_record) > 0:
            index_of_last_check = self.check_index - 1
            last_resource_state = self.record[
                (self.record.over == False) &
                (self.record.being_checked == True) &
                (self.record.type == resource_type) & 
                (self.record.check_index == index_of_last_check)   
            ]
            for loc in filtered_record['location']:
                if last_resource_state['is_existing'] and (not filtered_record['is_existing']):
                    region_of_suspect = get_taxicab_distance_points(
                        loc[0], 
                        loc[1], 
                        self.wait_time, 
                        self.size_of_env
                        )
                    for agent in agents: ##TODO if there are multiple agents in the region, who should be punished?
                        if agent.location in region_of_suspect:
                            if state_sys.criterion * 1. > random.random(): # the prob of the agent made the transgression * the prob of punishing it given the transgression
                                agent.to_be_punished[resource_type] = True #TODO: to_be_punished should be a dict {a: 0, b:0, ...}
    
    def regular_check_all_resources(self, timepoint, state_sys, agents):
        for resource_type in self.resource_to_monitor:
            regular_check(self, timepoint, resource_type, state_sys, agents)

    
                        
## before agent loop: update(), time_next_check()
## after the agent loop: regular_check_all_resources(), 