import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SimulationResult:
    prey_captured: bool = False
    steps_until_capture: int = 0
    state: np.ndarray = None
    params: dict = None

@njit
def positive_distance(state):
    distance = state[:, 0, 0, :] - state[:, 1, 0, :]
    return norm_list(distance)

@njit
def norm(v):
    return np.sqrt(np.sum(v ** 2))

@njit
def norm_list(v):
    out = np.empty(v.shape[0])
    for i in range(v.shape[0]):
        out[i] = np.sqrt(v[i, 0] ** 2 + v[i, 1] ** 2)
    return out

@njit
def normalize(v):
    length = norm(v)
    if length == 0:
        return np.zeros_like(v)
    return v / length

@njit
def rotate_right_90(v):
    """Rotate a 2D vector 90 degrees to the right (clockwise)."""
    return np.array([v[1], -v[0]])

@njit
def get_trajectory(deg: float, n_steps: int, prey_speed_max: float):
    """
    Initialize a trajectory state array for a 2D predator-prey simulation.

    The state array has shape (n_steps, 2 agents, 3 features, 2 dimensions), where:
    - time: t in {0, ..., n_steps - 1}
    - agents: 0 = predator, 1 = prey
    - features: 0 = position, 1 = velocity, 2 = acceleration
    - dimensions: 2D (x, y)

    Predator starts at origin with zero velocity and acceleration.
    Prey starts at position (1, 0) and moves in a direction specified by `deg` at max speed.
    Acceleration is initialized to zero for both agents.

    Args:
        deg (float): Direction angle (in degrees) of prey's initial velocity.
        params (dict): Dictionary containing simulation parameters including 'n_steps' and '11_max' (prey's max speed).

    Returns:
        np.ndarray: Initialized state array.
    """
    
    state = np.zeros((n_steps, 2, 3, 2), dtype=float)

    # initial predator pos, vel, acc
    state[0, 0, 0] = [0, 0] 
    state[0, 0, 1] = [0, 0] 
    state[0, 0, 2] = [0, 0] 

    rad = np.deg2rad(deg) 

    # initial prey pos, vel, acc
    state[0, 1, 0] = [1, 0]
    state[0, 1, 1] = [np.sin(rad) * prey_speed_max, np.cos(rad) * prey_speed_max]
    state[0, 1, 2] = [0, 0] 
    
    return state

@njit
def step(state, t, dt, pred_speed_max, prey_speed_max):
    """
    Update position and velocity for timestep t based on previous state.
    Applies max speed limits separately for predator (agent 0) and prey (agent 1).
    """
    # Position update
    state[t, :, 0, :] = state[t-1, :, 0, :] + state[t-1, :, 1, :] * dt

    # Velocity update
    state[t, :, 1, :] = state[t-1, :, 1, :] + state[t-1, :, 2, :] * dt

    # Limit predator speed
    pred_speed = norm(state[t, 0, 1, :])
    if pred_speed > pred_speed_max:
        state[t, 0, 1, :] = state[t, 0, 1, :] / pred_speed * pred_speed_max

    # Limit prey speed
    prey_speed = norm(state[t, 1, 1, :])
    if prey_speed > prey_speed_max:
        state[t, 1, 1, :] = state[t, 1, 1, :] / prey_speed * prey_speed_max

    return state

def simulate(state, params):
    return simul(state, params["dt"], params["R_kill"], params["R_react"], params["pred_acc"], params["prey_acc"], params["pred_speed_max"], params["prey_speed_max"])

@njit
def simul(state, dt, r_kill, r_react, pred_acc, prey_acc, pred_speed_max, prey_speed_max):
    """
    Run the simulation and return the final state tensor.
    state: (n_steps, 2 agents, 3 features, 2 dims)
    params: dictionary containing simulation parameters
    """
    n_steps = state.shape[0]

    for t in range(1, n_steps):
        direction = state[t-1, 1, 0] - state[t-1, 0, 0]
        distance = norm(direction)

        if distance < r_kill:
            return state, t            
        elif distance < r_react:
            # Prey reacts: rotate predator-prey vector 90° right
            state[t-1, 1, 2, :] = normalize(rotate_right_90(direction)) * prey_acc

        # Predator always steers toward prey
        state[t-1, 0, 2, :] = normalize(direction) * pred_acc

        # Step positions and velocities
        step(state, t, dt, pred_speed_max, prey_speed_max)

    return state, n_steps    


def visualize_evolution(prey_fitness_history, pred_fitness_history):
    """
    Visualize the evolutionary progress of prey and predator populations
    """
    plt.figure(figsize=(10, 6))
    
    generations = range(1, len(prey_fitness_history) + 1)
    
    plt.plot(generations, prey_fitness_history, 'b-', label='Best Prey Fitness')
    plt.plot(generations, pred_fitness_history, 'r-', label='Best Predator Fitness')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolutionary Progress')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_best_simulation(best_predator, best_prey, simulation_params):
    """
    Visualize a simulation using the best predator and prey
    """
    sim_result = run_simulation(best_predator, best_prey, simulation_params)
    
    # Plotting setup
    plt.figure(figsize=(10, 10))
    
    # Extract positions
    predator_positions = sim_result.state[:sim_result.steps_until_capture, 0, 0, :]
    prey_positions = sim_result.state[:sim_result.steps_until_capture, 1, 0, :]
    
    # Plot trajectories
    plt.plot(predator_positions[:, 0], predator_positions[:, 1], 'r-', label='Predator')
    plt.plot(prey_positions[:, 0], prey_positions[:, 1], 'b-', label='Prey')
    
    # Mark start positions
    plt.plot(predator_positions[0, 0], predator_positions[0, 1], 'ro', markersize=10, label='Predator Start')
    plt.plot(prey_positions[0, 0], prey_positions[0, 1], 'bo', markersize=10, label='Prey Start')
    
    # Mark end positions if prey was captured
    if sim_result.prey_captured:
        plt.plot(predator_positions[-1, 0], predator_positions[-1, 1], 'rx', markersize=10, label='Predator End')
        plt.plot(prey_positions[-1, 0], prey_positions[-1, 1], 'bx', markersize=10, label='Prey End')
    
    plt.title(f'Predator-Prey Simulation\nCapture: {sim_result.prey_captured}, Steps: {sim_result.steps_until_capture}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()

def run_simulation(predator_genome, prey_genome, simulation_params):
    """
    Run a single simulation with given predator and prey genomes
    Returns a SimulationResult object with the outcome
    """
    # Set up simulation parameters from genomes
    params = simulation_params.copy()
    params['pred_speed_max'] = predator_genome.max_speed
    params['prey_speed_max'] = prey_genome.max_speed
    params['pred_acc'] = predator_genome.max_acceleration
    params['prey_acc'] = prey_genome.max_acceleration
    params['R_react'] = prey_genome.react_radius
    
    initial_state = get_trajectory(
        deg=prey_genome.evasion_angle * 180 / np.pi,  # Convert to degrees
        n_steps=params['n_steps'],
        prey_speed_max=params['prey_speed_max']
    )
    
    final_state, steps_until_capture = simulate(initial_state, params)
    
    # Create and return simulation result
    result = SimulationResult()
    result.prey_captured = steps_until_capture < params['n_steps']
    result.steps_until_capture = steps_until_capture
    result.state = final_state
    result.params = params
    
    return result