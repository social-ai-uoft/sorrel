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