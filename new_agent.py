"""
File to complete. Contains the agents
"""
import numpy as np
import math

class Agent(object):
    """Agent base class. DO NOT MODIFY THIS CLASS
    """

    def __init__(self, mdp):
        super(Agent, self).__init__()
        # Init with a random policy
        self.mdp = mdp
        self.policy = np.zeros((4, mdp.env.observation_space.n)) + 0.25
        self.discount = 0.9

        # Initialize V or Q depending on your agent
        # self.V = np.zeros(self.mdp.env.observation_space.n)
        # self.Q = np.zeros((4, self.mdp.env.observation_space.n))

    def update(self, observation, action, reward):
        # DO NOT MODIFY. This is an example
        pass

    def action(self, observation):
        # DO NOT MODIFY. This is an example
        return self.mdp.env.action_space.sample()


class QLearning(Agent):
    def __init__(self, mdp):
        super(QLearning, self).__init__(mdp)
        self.Q = np.zeros((4, self.mdp.env.observation_space.n))
        self.lr = 0.1
        self.epsilon = 1

    def update(self, observation, action, reward):
        self.Q[action, observation] = (1-self.lr) * self.Q[action, observation] + \
        self.lr * (reward + self.discount * np.max(self.Q[:, self.mdp.observe()]))

    def action(self, observation):
        if np.random.random() > self.epsilon:
            choice = np.argmax(self.Q[:, observation])
        else:
            choice = np.random.randint(0, 4)
        self.epsilon *= 0.95
        return choice


class SARSA(Agent):
    def __init__(self, mdp):
        super(SARSA, self).__init__(mdp)
        self.Q = np.zeros((4, self.mdp.env.observation_space.n))
        self.lr = 0.1
        self.epsilon = 1

    def update(self, observation, action, reward):
        self.Q[action, observation] = (1-self.lr) * self.Q[action, observation] + \
        self.lr * (reward + self.discount * self.Q[self.action(self.mdp.observe()), self.mdp.observe()])

    def action(self, observation):
        if np.random.random() > self.epsilon:
            choice = np.argmax(self.Q[:, observation])
        else:
            choice = np.random.randint(0, 4)
        self.epsilon *= 0.99
        return choice


class ValueIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9

    def optimal_value_function(self):
        """1 step of value iteration algorithm
            Return: State Value V
        """
        # Initialize random V
        V = np.zeros(self.mdp.env.nS)

        nS = len(V)
        nA = self.mdp.env.nA

        nb_iter = 100
        for _ in range(nb_iter):
            for i in range(nS):
                new_states = []
                values = []
                for j in range(nA):
                    new_states.append(self.mdp.env.P[i][j][0][1])
                    values.append(V[new_states[-1]])
                V[i] = self.mdp.env.P[i][0][0][2] + self.gamma * np.max(values)
        print(V)
        return V

    def optimal_policy_extraction(self, V):
        """2 step of policy iteration algorithm
            Return: the extracted policy
        """
        nS = self.mdp.env.nS
        nA = self.mdp.env.nA
        policy = np.zeros(nS)
        for i in range(nS):
            new_states = []
            values = []
            for j in range(nA):
                new_states.append(self.mdp.env.P[i][j][0][1])
                values.append(V[new_states[-1]])
            policy[i] = np.argmax(values)
        return policy

    def value_iteration(self):
        """This is the main function of value iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        V = self.optimal_value_function()
        policy = self.optimal_policy_extraction(V)

        return policy, V


class PolicyIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9

    def policy_evaluation(self, policy, V):
        """1 step of policy iteration algorithm
            Return: State Value V
        """
        nS = self.mdp.env.nS
        for i in range(nS):
            new_state = self.mdp.env.P[i][policy[i]][0][1]
            value = V[new_state]
            V[i] = self.mdp.env.P[i][0][0][2] + self.gamma * value
        return V

    def policy_improvement(self, V, policy):
        """2 step of policy iteration algorithm
            Return: the improved policy
        """
        nA = self.mdp.env.nA
        nS = self.mdp.env.nS
        for i in range(nS):
            new_states = []
            values = []
            for j in range(nA):
                new_states.append(self.mdp.env.P[i][j][0][1])
                values.append(V[new_states[-1]])
            policy[i] = np.argmax(values)
        return policy


    def policy_iteration(self):
        """This is the main function of policy iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        # Start with a random policy
        nS = self.mdp.env.nS
        policy = np.ones(nS)
        V = np.zeros(self.mdp.env.nS)

        loop = True
        count = 0
        while loop:
            count += 1
            V = self.policy_evaluation(policy, V)
            policy2 = self.policy_improvement(V, np.copy(policy))
            loop = (np.linalg.norm(policy2 - policy) != 0)
            policy = policy2
        
        return policy, V
