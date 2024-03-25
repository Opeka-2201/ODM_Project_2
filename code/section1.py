# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 1: Implementation of the domain

## IMPORTS
import numpy as np
import random
random.seed(0)

## CONSTANTS
N = 10
U = [-4, 4]
M = 1
G = 9.81
TIME_STEP = 0.1
INTEGRATION_STEP = 0.001
DISCOUNT_FACTOR = 0.95
TERMINAL_P = 1
TERMINAL_S = 3

## DERIVED CONSTANTS
DYNAMIC_STEP = int(TIME_STEP / INTEGRATION_STEP)

## FUNCTIONS
def generate_trajectory(domain, agent, N):
    p_prev = agent.init_p
    s_prev = agent.init_s

    print("Generating trajectory...")
    for t in range(N+1):
        u = agent.policy()
        p_next, s_next = domain.dynamics(p_prev, s_prev, u)
        r = domain.reward(p_next, s_next, u)
        print(f"(x_{t} = ({p_prev:.4f}, {s_prev:.4f}), u_{t} = {u}, r_{t} = {r}, x_{t+1} = ({p_next:.3f}, {s_next:.3f}))")
        p_prev = p_next
        s_prev = s_next

## CLASSES
class Domain:
    def __init__(self):
        pass

    def hill(self, p):
        if p < 0:
            return p**2 + p
        else:
            return p / np.sqrt(1 + 5*p**2)
    
    def hill_prime(self, p):
        if p < 0:
            return 2*p + 1
        else:
            return 1 / (1 + 5*p**2)**(3/2)
        
    def hill_prime_prime(self, p):
        if p < 0:
            return 2
        else:
            return -15*p / (1 + 5*p**2)**(5/2)
        
    def dynamics(self, p, s, u):
        if np.abs(p) > TERMINAL_P or np.abs(s) > TERMINAL_S:
            return p, s
        
        p_next = p
        s_next = s

        for t in range(DYNAMIC_STEP):
            p_prime = s
            s_prime = (u / M) - (G * self.hill_prime(p)) - (s**2 * self.hill_prime(p) * self.hill_prime_prime(p))
            s_prime /= (1 + (self.hill_prime(p))**2)
            p_next += INTEGRATION_STEP * p_prime
            s_next += INTEGRATION_STEP * s_prime
        
        return p_next, s_next
    
    def reward(self, p, s, u):
        p_next, s_next = self.dynamics(p, s, u)
        if p_next < -TERMINAL_P or np.abs(s_next) > TERMINAL_S:
            return -1
        elif p_next > TERMINAL_P and np.abs(s_next) <= TERMINAL_S:
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self):
        self.init_p = random.uniform(-0.1, 0.1)
        self.init_s = 0

    def policy(self):
        return U[1]
    
## MAIN
def main():
    domain = Domain()
    agent = Agent()
    generate_trajectory(domain, agent, N)

if __name__ == "__main__":
    main()