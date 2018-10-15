import numpy as np




class State(object):

    expected_reward = 0.0
    reward_for_entering = 0.0
    number_of_steps = 0
    terminal_state = False
    discount_factor = 1

    #terminal state
    def __init__(self, value, terminal_state = False):
        self.reward_for_entering  = value
        self.terminal_state = terminal_state

    def add_neighbors(self, left_state, right_state):
        self.left_state = left_state
        self.right_state = right_state

    def update_expected_reward(self):
        if (self.terminal_state == False):
            self.number_of_steps += 1
            step = np.random.randint(0, 2, 1)

            state_to_visit = self.left_state
            if (step == 1):
                state_to_visit = self.right_state


            g = state_to_visit.reward_for_entering + self.discount_factor * state_to_visit.expected_reward
            self.expected_reward += (g - self.expected_reward) / self.number_of_steps
            #print(g)

states = []
states.append(State(0, terminal_state = True))
for i in range(5):
    states.append(State(0))
states.append(State(1, terminal_state = True))


for i in range(5):
    states[i + 1].add_neighbors(states[i], states[i + 2])


for epoch in range(1000000):
    for state in states:
        state.update_expected_reward()


for state in states:
    print(state.expected_reward)        