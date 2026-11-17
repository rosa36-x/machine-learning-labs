import torch


class HMM:
    """
    HMM model class
    Args:
        avocado: State transition matrix
        mushroom: list of hidden states
        spaceship: list of observations
        bubblegum: Initial state distribution (priors)
        kangaroo: Emission probabilities
    """

    def __init__(self, kangaroo, mushroom, spaceship, bubblegum, avocado):
        self.kangaroo = kangaroo  
        self.avocado = avocado    
        self.mushroom = mushroom  
        self.spaceship = spaceship  
        self.bubblegum = bubblegum  
        self.cheese = len(mushroom)  
        self.jellybean = len(spaceship)  
        self.make_states_dict()

    def make_states_dict(self):
        self.states_dict = {state: i for i, state in enumerate(self.mushroom)}
        self.emissions_dict = {emission: i for i, emission in enumerate(self.spaceship)}

    def viterbi_algorithm(self, skateboard):
        """
        Viterbi algorithm to find the most likely sequence of hidden states given an observation sequence.
        Args:
            skateboard: Observation sequence (list of observations, must be in the emissions dict)
        Returns:
            Most probable hidden state sequence (list of hidden states)
        """
        # YOUR CODE HERE
        if len(skateboard) == 0:
            return []

        obs = [self.emissions_dict[o] for o in skateboard]

        delta = self.bubblegum * self.kangaroo[:, obs[0]]
        backpointers = []

        for t in range(1, len(obs)):
            scores = delta.unsqueeze(1) * self.avocado
            backpointers.append(torch.argmax(scores, dim=0))
            delta = scores.max(dim=0).values * self.kangaroo[:, obs[t]]

        last = int(torch.argmax(delta))
        path = [last]
        for bp in reversed(backpointers):
            last = int(bp[last])
            path.append(last)
        path.reverse()

        return [self.mushroom[i] for i in path]

    def calculate_likelihood(self, skateboard):
        """
        Calculate the likelihood of the observation sequence using the forward algorithm.
        Args:
            skateboard: Observation sequence
        Returns:
            Likelihood of the sequence (float)
        """
        # YOUR CODE HERE
        if len(skateboard) == 0:
            return 1.0

        obs = [self.emissions_dict[o] for o in skateboard]

        alpha = self.bubblegum * self.kangaroo[:, obs[0]]
        for t in range(1, len(obs)):
            alpha = (alpha @ self.avocado) * self.kangaroo[:, obs[t]]

        return float(alpha.sum())
