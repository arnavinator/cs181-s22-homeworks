# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
# from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        # there are 2 possible actions, 7 possible x-values, 9 possible y-values
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))   # 2x7x9 matrix

        # to have decaying alpha for Q-learning to converge to the optimal solution,
        # we choose to have a_t(s,a) =  1/N(s,a), where N(s,a) is number of times (s,a) visited
        self.counts = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))
        self.gamma = 0.8    # discount factor
        self.epsilon = 0.001  # epsilon-greedy exploration vs optimal policy

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    # instead of having continuously evolving value of horiz and vert distance,
    # we perform floor division such that horiz and vert dist change more slowly
    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    # as seen in SwingyMonkey.py, action_callback state arg is a dictionary describing game
    def action_callback(self, state):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        # TODO (currently monkey just jumps around randomly)
        # 1. Discretize 'state' to get your transformed 'current state' features.
        # 2. Perform the Q-Learning update using 'current state' and the 'last state'.
        # 3. Choose the next action using an epsilon-greedy policy.

        # if this the first state, initialize last state as current state, random last_action, and 0 last_reward
        if self.last_state is None:
            self.last_state = state
            self.last_action = int(npr.rand() < 0.5)
            self.last_reward = 0

        # discretize current state (s', a') and last state (s, a)
        c_state = self.discretize_state(state)
        l_state = self.discretize_state(self.last_state)

        # update counts
        self.counts[self.last_action, l_state[0], l_state[1]] += 1

        # calculate temporal difference error: r + gamma*max_{a'}[Q(s',a')] - Q(s,a)
        qmax = np.max(self.Q[:, c_state[0], c_state[1]])
        td = self.last_reward + self.gamma*qmax - self.Q[self.last_action, l_state[0], l_state[1]]

        # iteration for Q-Learning
        # Q(s,a) = Q(s,a) + alpha*td
        alpha = 1/self.counts[self.last_action, l_state[0], l_state[1]]
        self.Q[self.last_action, l_state[0], l_state[1]] +=  alpha * td

        # epsilon-greedy choice of current action
        if npr.rand() < self.epsilon:
            current_action = int(npr.rand() < 0.5)  # choose 0 or 1 with prob 50-50
        else:
            current_action = np.argmax(self.Q[:, c_state[0], c_state[1]])


        # iterate action/state for next callback
        self.last_action = current_action
        self.last_state = state

        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        # for every iter, while game is running, keep sending action and reward to game (class SwingyMonkey)
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)
        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    # run_games(agent, hist, 100, 100)
    run_games(agent, hist, 100, 0)
    print(hist)
    print("Average score from epochs 50-100: ", np.average(np.array(hist)[49:]))
    print("Max score over all epochs: ", np.max(np.array(hist)))

    # Save history.
    np.save('hist', np.array(hist))
