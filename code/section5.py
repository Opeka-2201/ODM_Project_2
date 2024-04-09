# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 5: Parametric Q-Learning

## IMPORTS
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
import imageio.v3 as imageiov3
import imageio
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
import csv

## CONSTANTS
OSST_SIZES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
MAX_LENGTH = 60
N = 500

## IMPORTS FROM OTHER SECTIONS
## IMPORT CONSTANTS
from section1 import TERMINAL_P, TERMINAL_S, DISCOUNT_FACTOR, U
from section2 import NB_INITIAL_STATES

## IMPORT CLASSES
from section1 import Domain, Agent

## IMPORT FUNCTIONS
from display_caronthehill import save_caronthehill_image

## FUNCTIONS
def generate_osst(agent: Agent, domain: Domain, osst_size: int) -> list:
    """
    Generate an OSST of a given size with a given agent and domain.
    
    Parameters:
    ------------
    agent: Agent
        The agent used to generate the OSST
    domain: Domain
        The domain in which the agent evolves
    osst_size: int
        The size of the OSST
        
    Returns:
    ------------
    osst: list
        The OSST generated
    """
    osst = []

    while len(osst) < osst_size:
        p = np.random.uniform(-1, 1)
        s = 0

        for _ in range(MAX_LENGTH):
            if np.abs(p) > TERMINAL_P or np.abs(s) > TERMINAL_S or len(osst) >= osst_size:
                break
            
            u = agent.policy()
            p_next, s_next = domain.dynamics(p, s, u)
            r = domain.reward(p, s, u)
            osst.append(((p, s), u, r, (p_next, s_next)))
            p = p_next
            s = s_next
        
    return osst

def fitted_q_iteration(osst: list) -> MLPRegressor:
    """
    Perform the Fitted Q-Iteration algorithm on the given OSST.
    
    Parameters:
    ------------
    osst: list
        The OSST used to perform the Fitted Q-Iteration
    
    Returns:
    ------------
    model: MLPRegressor
        The model learned by the Fitted Q-Iteration
    """
    pass


def parametric_q_learning(osst: list) -> MLPRegressor:
    """
    Perform the Parametric Q-Learning algorithm on the given OSST.
    
    Parameters:
    ------------
    osst: list
        The OSST used to perform the Parametric Q-Learning
    
    Returns:
    ------------
    model: MLPRegressor
        The model learned by the Parametric Q-Learning
    """
    pass

def expected_return_continuous(domain: Domain, N: int, model: MLPRegressor) -> np.array:
    """
    Computes the expected return of a policy over N steps in the continuous domain.

    Parameters:
    ------------
    domain: Domain
        The domain in which the agent evolves
    N: int
        The horizon of the simulations
    model: MLPRegressor
        The model used to predict the optimal action
    
    Returns:
    ------------
    J_N: np.array
        The expected return of the policy over N steps
    """
    p = np.random.uniform(-0.1, 0.1)
    s = 0
    J = 0
    J_N = np.zeros(N)

    for n in range(N-1):
        u = max(U, key=lambda u: model.predict(np.array([[p, s, u]]))[0])
        r = domain.reward(p, s, u)
        p_next, s_next = domain.dynamics(p, s, u)

        if p != p_next or s != s_next:
            J += (DISCOUNT_FACTOR ** n) * r

        p = p_next
        s = s_next
        J_N[n+1] = J

    return J_N

def monte_carlo_simulations_continuous(domain: Domain, nb_initial_states: int, N: int, model: MLPRegressor) -> np.array:
    """
    Perform Monte Carlo simulations in the continuous domain to approximate the expected return.
    
    Parameters:
    ------------
    domain: Domain
        The domain in which the agent evolves
    nb_initial_states: int
        The number of initial states
    N: int
        The horizon of the simulations
    model: MLPRegressor
        The model used to predict the optimal action
    
    Returns:
    ------------
    J_N: np.array
        The expected return of the policy over N steps averaged over nb_initial_states initial states
    """
    J_N = np.zeros(N)
    for _ in range(nb_initial_states):
        J_N += expected_return_continuous(domain, N, model)
    J_N /= nb_initial_states
    return J_N

def generate_gif(domain: Domain, model: MLPRegressor, osst_size: int, N: int) -> int:
    """
    Generate a GIF of the car evolving in the domain with the given model.
    
    Parameters:
    ------------
    domain: Domain
        The domain in which the agent evolves
    model: MLPRegressor
        The model used to predict the optimal action
    osst_size: int
        The number of samples in the OSST
    N: int
        The number of frames in the GIF
    
    Returns:
    ------------
    n + 1: int
        The number of frames in the GIF
    """
    filename = f"figures/section5/gifs/parametric_OSST_size_{osst_size}_over_{N}_frames.gif"
    images = []

    p = 0
    s = 0

    for n in tqdm(range(N), desc=f"Generating GIF for OSST size {osst_size}"):
        subfile = f"figures/gif/section5/{n}_{N}.jpg"

        save_caronthehill_image(p, s , subfile)
        images.append(imageiov3.imread(subfile))

        if np.abs(p) > TERMINAL_P or np.abs(s) > TERMINAL_S:
            break

        u = max(U, key=lambda u: model.predict(np.array([[p, s, u]]))[0])
        p, s = domain.dynamics(p, s, u)

    imageio.mimsave(filename, images)   

    return n + 1

def generate_result_plots(model: MLPRegressor, osst_size: int) -> None:
    """
    Generate the plots for the results of the parametric Q-learning.
    
    Parameters:
    ------------
    model: MLPRegressor
        The model used to predict the optimal action
    osst_size: int
        The number of samples in the OSST
    """
    p_range = np.arange(-TERMINAL_P, TERMINAL_P, 0.01)
    s_range = np.arange(-TERMINAL_S, TERMINAL_S, 0.01)
    Q = np.zeros((len(p_range), len(s_range), 2))
    x, y = np.meshgrid(p_range, s_range)

    for i, p in enumerate(p_range):
        print(f"Computing Q: step {i} over {len(p_range)}\r", end="")
        for j, s in enumerate(s_range):
            for k, u in enumerate(U):
                Q[i, j, k] = model.predict(np.array([[p, s, u]]))

    map_0 = Q[:, :, 0].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_0, levels=20, cmap="RdBu", vmax=np.max(map_0), vmin=np.min(map_0))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(r"$\hat{Q}_N$ for u = " + str(U[0]) + " with parametric Q-Learning\n and OSST size = " + str(osst_size))
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section5/Q_N/parametric_OSST_size_{osst_size}_u_{U[0]}.png")
    plt.close()

    map_1 = Q[:, :, 1].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_1, levels=20, cmap="RdBu", vmax=np.max(map_1), vmin=np.min(map_1))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(r"$\hat{Q}_N$ for u = " + str(U[1]) + " with parametric Q-Learning\n and OSST size = " + str(osst_size))
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section5/Q_N/parametric_OSST_size_{osst_size}_u_{U[1]}.png")
    plt.close()

    optimal = np.argmax(Q, axis=2)
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, optimal.T, levels=2, cmap="RdBu", vmax=1, vmin=0)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='u = ' + str(U[0]), markerfacecolor='b', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='u = ' + str(U[1]), markerfacecolor='r', markersize=10)]
    ax.legend(handles=legend_elements)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(r"Optimal policy with parametric Q-Learning\n and OSST size = " + str(osst_size))
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section5/optimal/parametric_OSST_size_{osst_size}.png")
    plt.close()

def dump_results(results: list) -> None:
    """
    Dump the results in a CSV file.
    
    Parameters:
    ------------
    results: list
        The list of results to dump
    """
    with open("results/section5/results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["OSST size", "Number of steps", "Expected return"])
        writer.writerows(results)

## MAIN
def main() -> None:
    """
    Main function for the section 5.
    """
    agent = Agent(randomized=True)
    domain = Domain()
    return_fitted = np.zeros(len(OSST_SIZES))
    return_parametric = np.zeros(len(OSST_SIZES))
    results = []

    for i, osst_size in enumerate(OSST_SIZES):
        osst = generate_osst(agent, domain, osst_size)
        model_fitted = fitted_q_iteration(osst)
        model_parametric = parametric_q_learning(osst)

        return_fitted[i] = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_fitted)
        return_parametric[i] = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_parametric)

        steps = generate_gif(domain, model_parametric, osst_size, N)
        generate_result_plots(model_parametric, osst_size)

        results.append((osst_size, steps, return_parametric[i][-1]))

    plt.plot(OSST_SIZES, return_fitted, label="Fitted Q-Iteration")
    plt.plot(OSST_SIZES, return_parametric, label="Parametric Q-Learning")
    plt.xlabel("OSST size")
    plt.ylabel("Expected Return")
    plt.xscale("log")
    plt.legend()
    plt.savefig("figures/section5/returns/fitted_vs_parametric.png")
    plt.close()

    j_parametric = return_parametric[i][-1]
    print(f"Expected return with parametric Q-learning: {j_parametric}")

    dump_results(results)

if __name__ == "__main__":
    main()