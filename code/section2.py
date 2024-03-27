# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 2: Expected Return of a Policy in a Continuous Domain

## IMPORTS
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

## CONSTANTS
N = 500
NB_INITIAL_STATES = 50
RANDOMIZED = False # Change this to True to use a randomized policy instead of the always accelerate policy

## IMPORTS FROM OTHER SECTIONS
## IMPORT CONSTANTS
from section1 import DISCOUNT_FACTOR

## IMPORT CLASSES
from section1 import Domain, Agent

## FUNCTIONS
def expected_return_continuous(Domain, Agent, N):
    """
    This function computes the expected return of a policy over N steps in the continuous domain.

    Parameters
    ----------
    Domain : Domain
        The domain in which the agent evolves
    Agent : Agent
        The agent that evolves in the domain
    N : int
        The number of steps of the trajectory
    """
    p = np.random.uniform(-0.1, 0.1)
    s = 0
    J = 0
    J_N = np.zeros(N)

    for n in range(N-1):
        u = Agent.policy()
        r = Domain.reward(p, s, u)
        p_next, s_next = Domain.dynamics(p, s, u)
        
        if p != p_next or s != s_next:
            # If we are not in a terminal state, we continue accumulating 
            # the expected return of the policy inside J
            J += (DISCOUNT_FACTOR ** n) * r

        p = p_next
        s = s_next
        J_N[n+1] = J
    
    return J_N

def monte_carlo_simulations_continuous(Domain, Agent, nb_initial_states, N):
    """
    This function computes the expected return of a policy over N steps in the continuous domain by averaging over several initial states.

    Parameters
    ----------
    Domain : Domain
        The domain in which the agent evolves
    Agent : Agent
        The agent that evolves in the domain
    nb_initial_states : int
        The number of initial states to average the expected return over
    N : int
        The number of steps of the trajectory
    """
    J_N = np.zeros(N)
    for _ in range(nb_initial_states):
        J_N += expected_return_continuous(Domain, Agent, N)
    J_N = J_N / nb_initial_states
    return J_N

## MAIN
def main():
    domain = Domain()
    agent = Agent(randomized=RANDOMIZED)
    J_N = monte_carlo_simulations_continuous(domain, agent, NB_INITIAL_STATES, N)
    print("Final expected return of the policy:", J_N[-1])

    plt.plot(J_N)
    plt.xlabel("Time step")
    plt.ylabel(f"Expected return over {NB_INITIAL_STATES} simulations")
    plt.title(r"Evolution of $J^{\mu}_{" + str(N) + "}$ over " + str(NB_INITIAL_STATES) + " simulations for a " + ("random" if agent.randomized else "always accelerate") + " policy")
    plt.savefig("figures/expected_return_" + str(NB_INITIAL_STATES) + "_states_over_" + str(N) + "_steps_" + ("random" if agent.randomized else "accelerate") + ".png")

if __name__ == "__main__":
    main()