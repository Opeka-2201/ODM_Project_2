# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 3: Visualization

## IMPORTS
import imageio.v3 as imageiov3
import imageio
from tqdm import tqdm

## IMPORTS FROM OTHER SECTIONS
## IMPORT CONSTANTS
from section1 import TERMINAL_P, TERMINAL_S
from section2 import N, RANDOMIZED

## IMPORT FUNCTIONS
from display_caronthehill import save_caronthehill_image

## IMPORT CLASSES
from section1 import Domain, Agent

## FUNCTIONS
def generate_gif(Domain, Agent, N):
    """
        This function generates a gif of the car on the hill for N steps following the policy of the agent.
        
        Parameters
        ----------
        Domain : Domain
            The domain in which the agent evolves
        Agent : Agent
            The agent that evolves in the domain
        N : int
            The number of steps of the trajectory
        
        Returns
        ----------
        None
    """
    filename = f"figures/car_visualization_{N}_{RANDOMIZED}.gif"
    images = []

    p = Agent.init_p
    s = Agent.init_s

    for n in tqdm(range(N)):
        subfile = f"figures/gif/{RANDOMIZED}/{n+1}_{N}.jpg"

        p = min(max(-TERMINAL_P, p), TERMINAL_P)
        s = min(max(-TERMINAL_S, s), TERMINAL_S)

        save_caronthehill_image(p, s, subfile)
        images.append(imageiov3.imread(subfile))

        u = Agent.policy()
        p, s = Domain.dynamics(p, s, u)

    imageio.mimsave(filename, images)

## MAIN
def main():
    domain = Domain()
    agent = Agent(randomized=RANDOMIZED)
    agent.init_p = 0 # for this exercise we will fix the position of the car to 0
    agent.init_s = 0 # for this exercise we will fix the speed of the car to 0
    generate_gif(domain, agent, N)

if __name__ == "__main__":
    main()