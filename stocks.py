import numpy as np
import levy
import random


class StockMarkov:
    def __init__(self, name: str, price: float, ini, q):
        """ Basic Stock class.
            Stock movement modeled as a Markov Chain.
            Movement intensity is modeled in subclasses listed below."""
        self.name = name
        self.price = price
        self.state = ini
        self.upstate, self.downstate = np.c_[[1, 0]], np.c_[[0, 1]]
        self.q = q
        self.n = 0

    def fixed_point(self):
        """Returns average fixed point."""
        # insert 'not stochastic condition' here

        x = self.q[0, 1]/(self.q[0, 1] + self.q[1, 0])
        y = self.q[1, 0]/(self.q[0, 1] + self.q[1, 0])
        fix = np.c_[x, y]
        return fix

    def update(self):
        """Updates state of chain through the transition matrix q. Updates time by 1 unit."""
        next_chance = self.q.dot(self.state)
        next_state = np.array(random.choices(population=[self.upstate, self.downstate],
                                             weights=[next_chance[0, 0], next_chance[1, 0]]))
        self.state = next_state
        self.n += 1


class StockGauss(StockMarkov):
    """ Stock movement modeled as a Markov Chain.
        Movement intensity modeled through Gaussian pdf."""
    def __init__(self, name, price, ini, q, mu_up, sigma_up, mu_down, sigma_down):
        super().__init__(name, price, ini, q)
        self.mu_up = mu_up
        self.sigma_up = sigma_up
        self.mu_down = mu_down
        self.sigma_down = sigma_down
        self.movement = 0  # in time dependent model, this will not be 0

    def move(self, n=1):
        """ Updates Markovian state.
            n = number of times stock moves.
            Moves stock price in given direction by % dominated by Gaussian pdf."""
        for bins in range(n):
            if self.price > 0:
                self.update()
                if self.state[0, 0] == 1:
                    self.movement = random.gauss(self.mu_up, self.sigma_up)
                    if self.movement < 0:
                        self.state = self.downstate
                else:
                    self.movement = random.gauss(self.mu_down, self.sigma_down)
                    if self.movement > 0:
                        self.state = self.upstate
                self.price = self.price*(1 + self.movement)


class StockUniform(StockMarkov):
    """ Stock Movement modeled as a Markkov Chain.
        Movement intensity modeled through Uniform pdf."""
    def __init__(self, name, price, ini, q, a_up, b_up, a_down, b_down):
        super().__init__(name, price, ini, q)
        self.a_up = a_up
        self.b_up = b_up
        self.a_down = a_down
        self.b_down = b_down
        self.movement = 0  # in time dependent model, this will not be 0

    def move(self, n=1):
        """ Updates Markovian state.
            n = number of times stock moves.
            Moves stock price in given direction by % dominated by Uniform pdf."""
        for bins in range(n):
            if self.price > 0:
                self.update()
                if self.state[0, 0] == 1:
                    self.movement = random.uniform(self.a_up, self.b_up)
                else:
                    self.movement = random.uniform(self.a_down, self.b_down)
                self.price = self.price*(1 + self.movement)


class StockLevyStable(StockMarkov):
    """ Stock Movement modeled as a Markov Chain.
        Movement intensity modeled through Levy-Stable pdf."""
    def __init__(self, name, price, ini, q, alpha, beta, delta, gamma):
        super().__init__(name, price, ini, q)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.movement = 0  # in time dependent model, this will not be 0

    def move(self, n=1):
        """ Updates Markovian state.
            n = number of times stock moves.
            Moves stock price n times in given direction by % dominated by Levy-Stable pdf."""
        for bins in range(n):
            if self.price > 0:
                self.update()
                temp_movement = 0
                if self.state[0, 0] == 1:
                    temp_movement = -0.001
                    while temp_movement < 0:
                        temp_movement = levy.random(self.alpha, self.beta, self.delta, self.gamma)
                if self.state[0, 0] == 0:
                    temp_movement = 0.001
                    while temp_movement > 0 or temp_movement < -1:
                        temp_movement = levy.random(self.alpha, self.beta, self.delta, self.gamma)

                self.movement = temp_movement
                self.price = self.price * (1 + self.movement)


if __name__ == '__main__':
    initial_price = 50
    upstate = np.c_[[1, 0]]

    M = np.array([[0.50183452, 0.53757696],
                  [0.49816548, 0.46242304]])
    alpha1 = 1.5744042025830018
    beta1 = -0.13434961296351516
    mu1 = 0.0008798043941681043
    sigma1 = 0.009221584194093038
    stock = StockLevyStable('ABC', initial_price, upstate, M, alpha1, beta1, mu1, sigma1)

    days = 2500
    stock.move(days)
    print(stock.price)
