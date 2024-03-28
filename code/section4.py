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
import csv

## CONSTANTS
RANDOM_STATE = 0
EPOCHS = 60
NB_SAMPLES = 100
BOUND = 0.01
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
def generate_model(model, model_type, inputs, outputs):
    """
        Generate a model based on the type of model given as input

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
            The model generated
    """
    if model is None:
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "extra_trees":
            model = ExtraTreesRegressor(n_estimators=20, random_state=RANDOM_STATE)
        elif model_type == "neural_network":
            model = MLPRegressor(hidden_layer_sizes=(10, 15, 20, 15, 10), max_iter=1000, random_state=RANDOM_STATE, activation="tanh")
        else:
            raise ValueError("Model type not recognized")
        
    if inputs is not None and outputs is not None:
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

def stop_criterion(model, model_prev, N, osst, mode):
    """
        Computes the upper bound of the stop criterion.
        
        Parameters:
        ------------
        model : model
            The current model
        model_prev : model
            The previous model    
        N : int
            The current iteration
        osst : list
            The observed state, state, action, reward, next state tuples
        mode : int
            The mode of the stop criterion
        
        Returns:
        ------------
        float : 
            The upper bound of the stop criterion
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

        return np.linalg.norm(predictions_model - predictions_prev)
    
    elif mode == 2:
        Br = 1
        upper = (2 * (DISCOUNT_FACTOR**N) * Br) / (1 - DISCOUNT_FACTOR)**2
        return upper
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
    model = generate_model(None, model_type, None, None)
    model_prev = 0

    print(f"Generating model with {model_type}, stop criterion mode {stop_criterion_mode} and bound {bound}...")

    criterion = bound + 1
    while not criterion < bound:
        N += 1
        print(f"Iteration {N}: {criterion} < {bound}\r", end="")
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
        model = generate_model(model, model_type, np.array(inputs).reshape(-1, 3), np.array(outputs).ravel())

        if N > 2:
            criterion = stop_criterion(model, model_prev, N, osst, stop_criterion_mode)
    
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

def generate_gif(domain, model, N, model_type, stop_criterion_mode, generation_mode):
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
        stop_criterion_mode : int
            The mode of the stop criterion
        generation_mode : int
            The mode of the generation

        Returns:
        ------------
        None
    """
    filename = f"figures/section4/gifs/car_{N}_{model_type}_{stop_criterion_mode}_{generation_mode}.gif"
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

def generate_result_plots(model, model_type, stop_criterion_mode, generation_mode):
    """
        Generate the result plots for the model.
    
        Parameters:
        ------------
        model : model
            The model to use to predict the best action
        model_type : str
            The type of model to generate
        stop_criterion_mode : int
            The mode of the stop criterion
        generation_mode : int
            The mode of the generation
        
        Returns:
        ------------
        None
    """
    p_range = np.arange(-TERMINAL_P, TERMINAL_P, 0.01)
    s_range = np.arange(-TERMINAL_S, TERMINAL_S, 0.01)
    Q = np.zeros((len(p_range), len(s_range), 2))
    x, y = np.meshgrid(p_range, s_range)

    for i, p in enumerate(p_range):
        print(f"Computing Q: step {i} over {len(p_range)}\r", end="")
        for j, s in enumerate(s_range):
            for k, u in enumerate(U):
                Q[i, j, k] = model.predict(np.array([[p, s, u]]))[0]

    map_0 = Q[:, :, 0].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_0, levels=20, cmap="RdBu", vmax=np.max(map_0), vmin=np.min(map_0))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(r"$\hat{Q}_N$ for u = " + str(U[0]) + " with " + model_type + " model\nstop criterion mode: " + str(stop_criterion_mode) + " and generation mode: " + generation_mode)
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section4/Q_N/{model_type}_u_{str(U[0])}_stop_{stop_criterion_mode}_generation_{generation_mode}.png")
    plt.close()

    map_1 = Q[:, :, 1].T
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, map_1, levels=20, cmap="RdBu", vmax=np.max(map_1), vmin=np.min(map_1))
    fig.colorbar(c)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title(r"$\hat{Q}_N$ for u = " + str(U[1]) + " with " + model_type + " model\nstop criterion mode: " + str(stop_criterion_mode) + " and generation mode: " + generation_mode)
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section4/Q_N/{model_type}_u_{str(U[1])}_stop_{stop_criterion_mode}_generation_{generation_mode}.png")
    plt.close()

    optimal = np.argmax(Q, axis=2)
    fig, ax = plt.subplots()
    c = ax.contourf(x, y, optimal.T, levels=2, cmap="RdBu", vmax=1, vmin=0)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='u = 4',
                              markerfacecolor='b', markersize=10),
                       plt.Line2D([0], [0], marker='o', color='w', label='u = -4',
                              markerfacecolor='r', markersize=10)]
    ax.legend(handles=legend_elements)
    ax.set_xlim([-TERMINAL_P, TERMINAL_P])
    ax.set_ylim([-TERMINAL_S, TERMINAL_S])
    ax.set_title("Optimal policy with " + model_type + " model\nstop criterion mode: " + str(stop_criterion_mode) + " and generation mode: " + generation_mode)
    ax.set_xlabel("Position")
    ax.set_ylabel("Speed")
    plt.savefig(f"figures/section4/optimal/{model_type}_stop_{stop_criterion_mode}_generation_{generation_mode}.png")
    plt.close()

def dump_expected_returns(expected_returns, model_type):
    """
        Dump the expected returns in a csv file.
    
        Parameters:
        ------------
        expected_returns : list
            The list of expected returns
        model_type : str
            The type of model to generate
            
        Returns:
        ------------
        None
    """
    with open(f'results/section4/{model_type}/expected_returns.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'generation_mode', 'stop_criterion', 'expected_return'])
        for model, generation_mode, stop_criterion, expected_return in expected_returns:
            writer.writerow([model, generation_mode, stop_criterion, expected_return])

## MAIN
def main():
    domain = Domain()
    agent = Agent(randomized=True)
    osst_full = generate_osst(agent, domain, EPOCHS, NB_SAMPLES, technique="full")
    osst_reduced = generate_osst(agent, domain, EPOCHS, NB_SAMPLES, technique="reduced")

    print("\n### LINEAR REGRESSION STOP CRITERION 1 AND REDUCED SPACE ###")
    model_linear_1_reduced = fitted_q_iteration("linear_regression", osst_reduced, 1, BOUND)
    generate_result_plots(model_linear_1_reduced, "linear_regression", 1, "reduced")
    J_linear_1_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_linear_1_reduced)
    print(f"Expected return with linear regression: {J_linear_1_reduced[-1]}")
    generate_gif(domain, model_linear_1_reduced, N, "linear_regression", 1, "reduced")

    print("\n### LINEAR REGRESSION STOP CRITERION 1 AND FULL SPACE ###")
    model_linear_1_full = fitted_q_iteration("linear_regression", osst_full, 1, BOUND)
    generate_result_plots(model_linear_1_full, "linear_regression", 1, "full")
    J_linear_1_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_linear_1_full)
    print(f"Expected return with linear regression: {J_linear_1_full[-1]}")
    generate_gif(domain, model_linear_1_full, N, "linear_regression", 1, "full")

    print("\n### LINEAR REGRESSION STOP CRITERION 2 AND REDUCED SPACE ###")
    model_linear_2_reduced = fitted_q_iteration("linear_regression", osst_reduced, 2, BOUND)
    generate_result_plots(model_linear_2_reduced, "linear_regression", 2, "reduced")
    J_linear_2_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_linear_2_reduced)
    print(f"Expected return with linear regression: {J_linear_2_reduced[-1]}")
    generate_gif(domain, model_linear_2_reduced, N, "linear_regression", 2, "reduced")

    print("\n### LINEAR REGRESSION STOP CRITERION 2 AND FULL SPACE ###")
    model_linear_2_full = fitted_q_iteration("linear_regression", osst_full, 2, BOUND)
    generate_result_plots(model_linear_2_full, "linear_regression", 2, "full")
    J_linear_2_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_linear_2_full)
    print(f"Expected return with linear regression: {J_linear_2_full[-1]}")
    generate_gif(domain, model_linear_2_full, N, "linear_regression", 2, "full")

    expected_returns_linear = [
        ('linear_regression', 'reduced', 1, J_linear_1_reduced[-1]),
        ('linear_regression', 'full', 1, J_linear_1_full[-1]),
        ('linear_regression', 'reduced', 2, J_linear_2_reduced[-1]),
        ('linear_regression', 'full', 2, J_linear_2_full[-1]),
    ]

    dump_expected_returns(expected_returns_linear, "linear_regression")

    print("\n### EXTRA TREES STOP CRITERION 1 AND REDUCED SPACE ###")
    model_extra_trees_1_reduced = fitted_q_iteration("extra_trees", osst_reduced, 1, BOUND)
    generate_result_plots(model_extra_trees_1_reduced, "extra_trees", 1, "reduced")
    J_extra_trees_1_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_extra_trees_1_reduced)
    print(f"Expected return with extra trees: {J_extra_trees_1_reduced[-1]}")
    generate_gif(domain, model_extra_trees_1_reduced, N, "extra_trees", 1, "reduced")
    
    print("\n### EXTRA TREES STOP CRITERION 1 AND FULL SPACE ###")
    model_extra_trees_1_full = fitted_q_iteration("extra_trees", osst_full, 1, BOUND)
    generate_result_plots(model_extra_trees_1_full, "extra_trees", 1, "full")
    J_extra_trees_1_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_extra_trees_1_full)
    print(f"Expected return with extra trees: {J_extra_trees_1_full[-1]}")
    generate_gif(domain, model_extra_trees_1_full, N, "extra_trees", 1, "full")
    
    print("\n### EXTRA TREES STOP CRITERION 2 AND REDUCED SPACE ###")
    model_extra_trees_2_reduced = fitted_q_iteration("extra_trees", osst_reduced, 2, BOUND)
    generate_result_plots(model_extra_trees_2_reduced, "extra_trees", 2, "reduced")
    J_extra_trees_2_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_extra_trees_2_reduced)
    print(f"Expected return with extra trees: {J_extra_trees_2_reduced[-1]}")
    generate_gif(domain, model_extra_trees_2_reduced, N, "extra_trees", 2, "reduced")
    
    print("\n### EXTRA TREES STOP CRITERION 2 AND FULL SPACE ###")
    model_extra_trees_2_full = fitted_q_iteration("extra_trees", osst_full, 2, BOUND)
    generate_result_plots(model_extra_trees_2_full, "extra_trees", 2, "full")
    J_extra_trees_2_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_extra_trees_2_full)
    print(f"Expected return with extra trees: {J_extra_trees_2_full[-1]}")
    generate_gif(domain, model_extra_trees_2_full, N, "extra_trees", 2, "full")

    expected_returns_extra_trees = [
        ('extra_trees', 'reduced', 1, J_extra_trees_1_reduced[-1]),
        ('extra_trees', 'full', 1, J_extra_trees_1_full[-1]),
        ('extra_trees', 'reduced', 2, J_extra_trees_2_reduced[-1]),
        ('extra_trees', 'full', 2, J_extra_trees_2_full[-1]),
    ]

    dump_expected_returns(expected_returns_extra_trees, "extra_trees")

    print("\n### NEURAL NETWORK STOP CRITERION 1 AND REDUCED SPACE ###")
    model_neural_network_1_reduced = fitted_q_iteration("neural_network", osst_reduced, 1, BOUND)
    generate_result_plots(model_neural_network_1_reduced, "neural_network", 1, "reduced")
    J_neural_network_1_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_neural_network_1_reduced)
    print(f"Expected return with neural network: {J_neural_network_1_reduced[-1]}")
    generate_gif(domain, model_neural_network_1_reduced, N, "neural_network", 1, "reduced")
    
    print("\n### NEURAL NETWORK STOP CRITERION 1 AND FULL SPACE ###")
    model_neural_network_1_full = fitted_q_iteration("neural_network", osst_full, 1, BOUND)
    generate_result_plots(model_neural_network_1_full, "neural_network", 1, "full")
    J_neural_network_1_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_neural_network_1_full)
    print(f"Expected return with neural network: {J_neural_network_1_full[-1]}")
    generate_gif(domain, model_neural_network_1_full, N, "neural_network", 1, "full")
    
    print("\n### NEURAL NETWORK STOP CRITERION 2 AND REDUCED SPACE ###")
    model_neural_network_2_reduced = fitted_q_iteration("neural_network", osst_reduced, 2, BOUND)
    generate_result_plots(model_neural_network_2_reduced, "neural_network", 2, "reduced")
    J_neural_network_2_reduced = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_neural_network_2_reduced)
    print(f"Expected return with neural network: {J_neural_network_2_reduced[-1]}")
    generate_gif(domain, model_neural_network_2_reduced, N, "neural_network", 2, "reduced")
    
    print("\n### NEURAL NETWORK STOP CRITERION 2 AND FULL SPACE ###")
    model_neural_network_2_full = fitted_q_iteration("neural_network", osst_full, 2, BOUND)
    generate_result_plots(model_neural_network_2_full, "neural_network", 2, "full")
    J_neural_network_2_full = monte_carlo_simulations_continuous(domain, NB_INITIAL_STATES, N, model_neural_network_2_full)
    print(f"Expected return with neural network: {J_neural_network_2_full[-1]}")
    generate_gif(domain, model_neural_network_2_full, N, "neural_network", 2, "full")

    expected_returns_neural_network = [
        ('neural_network', 'reduced', 1, J_neural_network_1_reduced[-1]),
        ('neural_network', 'full', 1, J_neural_network_1_full[-1]),
        ('neural_network', 'reduced', 2, J_neural_network_2_reduced[-1]),
        ('neural_network', 'full', 2, J_neural_network_2_full[-1]),
    ]

    dump_expected_returns(expected_returns_neural_network, "neural_network")

    pass

if __name__ == "__main__":
    main()