# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 2: Reinforcement Learning in a Continuous Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 4: Fitted-Q Iteration

## IMPORTS
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import imageio.v3 as imageiov3
import imageio
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm

## CONSTANTS
RANDOM_STATE = 0
EPOCHS = 50
NB_SAMPLES = 100
TECHNIQUE = "full"
BOUND = 0.01
GRID_SIZE = 200
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
def generate_model(model_type, inputs, outputs):
    """
        Generate a model based on the type of model given as input and fit it with the inputs and outputs.

        Parameters:
        ------------
        model_type : str
            The type of model to generate
        inputs : np.array
            The inputs to fit the model
        outputs : np.array
            The outputs to fit the model
            
        Returns:
        ------------
        model : model
            The model generated and fitted
    """
    if model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=10, random_state=RANDOM_STATE)
    elif model_type == "neural_network":
        model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError("Model type not recognized")

    inputs = inputs.reshape(-1, 3)
    outputs = outputs.ravel()

    model.fit(inputs, outputs)
    return model

def generate_osst(agent, domain, epochs, nb_samples, technique="reduced"):
    """
        Generate the observed state, state, action, reward, next state tuples.
        
        Parameters:
        ------------
        agent : Agent
            The agent that evolves in the domain
        domain : Domain
            The domain in which the agent evolves
        epochs : int
            The number of epochs to generate the data
        nb_samples : int
            The number of samples to generate at each epoch
        technique : str
            The technique to generate the initial position (reduced space or full space)
        
        Returns:
        ------------
        ostt : list
            The observed state, state, action, reward, next state tuples
    """
    ostt = []
    for _ in range(epochs):
        p = np.random.uniform(-1, 1) if technique == "full" else np.random.uniform(-0.1, 0.1)
        s = 0
        for _ in range(nb_samples):
            if np.abs(p) > TERMINAL_P or np.abs(s) > TERMINAL_S:
                break
            u = agent.policy()
            p_next, s_next = domain.dynamics(p, s, u)
            r = domain.reward(p, s, u)
            ostt.append(((p, s), u, r, (p_next, s_next)))
            p = p_next
            s = s_next

    return ostt

def stop_criterion(model, model_prev, N, bound, osst, mode):
    """
        Check if the stop criterion is met.
        
        Parameters:
        ------------
        model : model
            The current model
        model_prev : model
            The previous model    
        N : int
            The current iteration
        bound : float
            The bound to check the stop criterion
        osst : list
            The observed state, state, action, reward, next state tuples
        mode : int
            The mode of the stop criterion
        
        Returns:
        ------------
        bool
            Whether the stop criterion is met
    """
    if mode == 1:
        if N == 0 or N == 1:
            return False

        predictions_model = []
        predictions_prev = []

        for (p, s), u, _, _ in osst:
            predictions_model.append(model.predict(np.array([[p, s, u]]))[0])
            predictions_prev.append(model_prev.predict(np.array([[p, s, u]]))[0])

        predictions_model = np.array(predictions_model)
        predictions_prev = np.array(predictions_prev)

        return np.linalg.norm(predictions_model - predictions_prev) < bound
    elif mode == 2:
        Br = 1
        upper = (2 * (DISCOUNT_FACTOR**N) * Br) / (1 - DISCOUNT_FACTOR)**2
        return upper < bound
    else:
        raise ValueError("Mode not recognized")
    
def fitted_q_iteration(model_type, osst, stop_criterion_mode, bound):
    """
        Perform the Fitted Q-Iteration algorithm to generate a model.
    
        Parameters:
        ------------
        model_type : str
            The type of model to generate
        osst : list
            The observed state, state, action, reward, next state tuples
        stop_criterion_mode : int
            The mode of the stop criterion
        bound : float
            The bound to check the stop criterion
        
        Returns:
        ------------
        model : model
            The model generated
    """
    N = 0
    model = 0
    model_prev = 0

    print(f"Generating model with {model_type}, stop criterion mode {stop_criterion_mode} and bound {bound}...")

    while not stop_criterion(model, model_prev, N, bound, osst, stop_criterion_mode):
        N += 1
        print(f"Iteration {N}\r", end="")
        inputs = []
        outputs = []

        for (p, s), u, r, (p_next, s_next) in osst:
            inputs.append([p, s, u])
        
            if N != 1:
                fetch_max = max(model.predict(np.array([[p_next, s_next, U[0]]])), \
                                model.predict(np.array([[p_next, s_next, U[1]]])))
            else:
                fetch_max = 0

            outputs.append(r + DISCOUNT_FACTOR * fetch_max)
        
        model_prev = model
        model = generate_model(model_type, np.array(inputs), np.array(outputs))
    
    print(f"Model generated after {N} iterations")
    return model

def expected_return_continuous(domain, N, model):
    """
        This function computes the expected return of a policy over N steps in the continuous domain.
    
        Parameters:
        ------------
        domain : Domain
            The domain in which the agent evolves
        N : int
            The number of steps of the trajectory
        model : model
            The model to use to predict the best action
            
        Returns:
        ------------
        J_N : np.array
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
            # If we are not in a terminal state, we continue accumulating 
            # the expected return of the policy inside J
            J += (DISCOUNT_FACTOR ** n) * r

        p = p_next
        s = s_next
        J_N[n+1] = J
    
    return J_N

def monte_carlo_simulations_continuous(domain, nb_initial_states, N, model):
    """
        This function computes the expected return of a policy over N steps in the continuous domain by averaging over several initial states.
    
        Parameters:
        ------------
        domain : Domain
            The domain in which the agent evolves
        nb_initial_states : int
            The number of initial states to average the expected return over
        N : int
            The number of steps of the trajectory
        model : model
            The model to use to predict the best action
        
        Returns:
        ------------
        J_N : np.array
            The expected return of the policy over N steps averaged over nb_initial_states initial states
    """
    J_N = np.zeros(N)
    for _ in range(nb_initial_states):
        J_N += expected_return_continuous(domain, N, model)
    J_N = J_N / nb_initial_states
    return J_N

def generate_gif(domain, model, N, model_type):
    """
        Generate a gif of the car evolving in the domain with the model.
        
        Parameters:
        ------------
        domain : Domain
            The domain in which the agent evolves
        model : model
            The model to use to predict the best action
        N : int
            The number of steps of the trajectory
        model_type : str
            The type of model to generate
            
        Returns:
        ------------
        None
    """
    filename = f"figures/car_visualization_{N}_{model_type}.gif"
    images = []

    p = np.random.uniform(-0.1, 0.1)
    s = 0

    for n in tqdm(range(N)):
        subfile = f"figures/gif/{model_type}/{n+1}_{N}.jpg"

        p = min(max(-TERMINAL_P, p), TERMINAL_P)
        s = min(max(-TERMINAL_S, s), TERMINAL_S)

        save_caronthehill_image(p, s, subfile)
        images.append(imageiov3.imread(subfile))

        u = max(U, key=lambda u: model.predict(np.array([[p, s, u]]))[0])
        p, s = domain.dynamics(p, s, u)

    imageio.mimsave(filename, images)

def generate_result_plots(model, model_type):
    """
        Generate the result plots for the model.
    
        Parameters:
        ------------
        model : model
            The model to use to predict the best action
        model_type : str
            The type of model to generate
        
        Returns:
        ------------
        None
    """
    p_range = np.linspace(-TERMINAL_P, TERMINAL_P, GRID_SIZE)
    s_range = np.linspace(-TERMINAL_S, TERMINAL_S, GRID_SIZE)
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(U)))
    x, y = np.meshgrid(p_range, s_range)

    for i, p in enumerate(p_range):
        for j, s in enumerate(s_range):
            for k, u in enumerate(U):
                Q[i, j, k] = model.predict(np.array([[p, s, u]]))[0]

    map_0 = Q[:, :, 0].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_0, levels=20, cmap="RdBu", vmax=np.max(map_0), vmin=np.min(map_0))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(f"Value function for u = {str(U[0])}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/value_function_{model_type}_u_{str(U[0])}.png")
    plt.close()

    map_1 = Q[:, :, 1].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_1, levels=20, cmap="RdBu", vmax=np.max(map_1), vmin=np.min(map_1))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(f"Value function for u = {str(U[1])}")
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/value_function_{model_type}_u_{str(U[1])}.png")
    plt.close()

    optimal = np.argmax(Q, axis=2)
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, optimal.T, levels=20, cmap="RdBu", vmax=1, vmin=0)
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title("Optimal policy")
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/optimal_policy_{model_type}.png")
    plt.close()

## MAIN
def main():
    domain = Domain()
    agent = Agent(randomized=True)
    osst = generate_osst(agent, domain, EPOCHS, NB_SAMPLES, technique=TECHNIQUE)

    print("### LINEAR REGRESSION ###")
    model_linear = fitted_q_iteration("linear_regression", osst, 2, BOUND)
    generate_result_plots(model_linear, "linear_regression")
    print(f"Expected return with linear regression: {monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_linear)}")
    generate_gif(domain, model_linear, N, "linear_regression")

    model_trees = fitted_q_iteration("extra_trees", osst, 2, BOUND)
    generate_result_plots(model_trees, "extra_trees")
    print(f"Expected return with extra trees: {monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_trees)}")
    generate_gif(domain, model_trees, N, "extra_trees")

    model_nn = fitted_q_iteration("neural_network", osst, 2, BOUND)
    generate_result_plots(model_nn, "neural_network")
    print(f"Expected return with neural network: {monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_nn)}")
    generate_gif(domain, model_nn, N, "neural_network")

if __name__ == "__main__":
    main()