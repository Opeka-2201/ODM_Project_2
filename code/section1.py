# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 1: Implementation of the domain

## IMPORTS
import numpy as np
np.random.seed(0)

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
    """
        This function generates a trajectory of N steps in the domain following the policy of the agent.
        
        Parameters
        ----------
        domain : Domain
            The domain in which the agent evolves
        agent : Agent
            The agent that evolves in the domain
        N : int
            The number of steps of the trajectory
        
        Returns
        ----------
        None
    """

    p_prev = agent.init_p
    s_prev = agent.init_s

    print("Generating trajectory...")
    for n in range(N+1):
        u = agent.policy()
        r = domain.reward(p_prev, s_prev, u)
        p_next, s_next = domain.dynamics(p_prev, s_prev, u)
        print(f"(x_{n} = ({p_prev:.4f}, {s_prev:.4f}), u_{n} = {u}, r_{n} = {r}, x_{n+1} = ({p_next:.4f}, {s_next:.4f}))")
        p_prev = p_next
        s_prev = s_next

## CLASSES
class Domain:
    """
        This class represents the domain in which the agent evolves.

        Attributes
        ----------
        None
    """

    def __init__(self):
        pass

    def hill(self, p):
        """
            This function computes the height of the hill at a given position.
            
            Parameters
            ----------
            p : float
                The position of the agent
            
            Returns
            -------
            float
                The height of the hill at the given position
        """
        if p < 0:
            return (p**2) + p
        else:
            return (p / (np.sqrt(1 + 5*p**2)))
    
    def hill_prime(self, p):
        """
            This function computes the derivative of the height of the hill at a given position.
            
            Parameters
            ----------
            p : float
                The position of the agent
                
            Returns
            -------
            float
                The derivative of the height of the hill at the given position
        """
        if p < 0:
            return (2*p) + 1
        else:
            return ((1) / ((1 + 5*p**2)**(3/2)))
        
    def hill_prime_prime(self, p):
        """
            This function computes the second derivative of the height of the hill at a given position.

            Parameters
            ----------
            p : float
                The position of the agent
            
            Returns
            -------
            float
                The second derivative of the height of the hill at the given position
        """
        if p < 0:
            return 2
        else:
            return ((-15*p) / ((1 + 5*p**2)**(5/2)))
        
    def dynamics(self, p, s, u):
        """
            This function computes the next position and speed of the agent given his current position, speed and acceleration.
            
            Parameters
            ----------
            p : float
                The position of the agent
            s : float
                The speed of the agent
            u : int
                The acceleration of the agent (-4 or 4)
            
            Returns
            -------
            p_next : float
                The next position of the agent
            s_next : float
                The next speed of the agent
        """
        if np.abs(p) > TERMINAL_P or np.abs(s) > TERMINAL_S:
            return p, s
        
        p_next = p
        s_next = s

        for t in range(DYNAMIC_STEP):
            p_prime = s
            s_prime = ((u) / (M*(1 + (self.hill_prime(p))**2))) - ((G * self.hill_prime(p))*(1 + (self.hill_prime(p))**2)) - ((s**2 * self.hill_prime(p) * self.hill_prime_prime(p))/((1 + (self.hill_prime(p))**2)))
            p_next += INTEGRATION_STEP * p_prime
            s_next += INTEGRATION_STEP * s_prime
        
        return p_next, s_next
    
    def reward(self, p, s, u):
        """
            This function computes the reward of the agent given his current position, speed and acceleration.
            
            Parameters
            ----------
            p : float
                The position of the agent    
            s : float
                The speed of the agent
            u : int
                The acceleration of the agent (-4 or 4)
                
            Returns
            -------
            int
                The reward of the agent (-1, 0 or 1)
        """
        p_next, s_next = self.dynamics(p, s, u)
        if p_next < -TERMINAL_P or np.abs(s_next) > TERMINAL_S:
            return -1
        elif p_next > TERMINAL_P and np.abs(s_next) <= TERMINAL_S:
            return 1
        else:
            return 0
        
class Agent:
    """
        This class represents an agent evolving in a continuous domain.
        
        Attributes
        ----------
        init_p : float
            The initial position of the agent
        init_s : float
            The initial speed of the agent
        randomized : bool
            Whether the policy is randomized or fixed in acceleration
    """
    def __init__(self, randomized=False):
        self.init_p = np.random.uniform(-0.1, 0.1)
        self.init_s = 0
        self.randomized = randomized

    def policy(self):
        """
            This function returns the acceleration of the agent using his behavior.
            
            Returns
            -------
            int
                The acceleration of the agent (-4 or 4)
        """
        if self.randomized:
            return np.random.choice(U)
        else:
            return U[1]
    
## MAIN
def main():
    domain = Domain()
    agent = Agent()
    generate_trajectory(domain, agent, N)

if __name__ == "__main__":
    main()