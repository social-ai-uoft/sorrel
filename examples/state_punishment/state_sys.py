class state_sys():
    def __init__(self, init_prob, magnitude, taboo, change_per_vote) -> None:
        self.prob = init_prob
        self.init_prob = init_prob 
        self.magnitude = magnitude
        self.taboo = taboo
        self.change_per_vote = change_per_vote
        self.time = None
        self.prob_record = []
    
    def reset_prob_record(self):
        self.prob_record = []

    def update_prob_record(self):
        self.prob_record.append(self.prob)