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
def cross_product_2d(v1, v2):
    """Compute the z-component of the cross product of two 2D vectors"""
    return v1[0] * v2[1] - v1[1] * v2[0]

@njit
def rotate_right_90(v):
    """Rotate a 2D vector 90 degrees to the right (clockwise)."""
    return np.array([v[1], -v[0]])

@njit
def get_trajectory(deg: float, n_steps: int, prey_speed_max: float):
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
    return simul(state, params["dt"], params["R_kill"], params["R_react"], params["pred_acc"], params["prey_acc"], params["pred_speed_max"], params["prey_speed_max"], params["pred_acc_max"], params["prey_acc_max"], params['navigation_gain'])

@njit
def simul(state, dt, r_kill, r_react, pred_acc, prey_acc, pred_speed_max, prey_speed_max, pred_acc_max, prey_acc_max, navigation_gain):
    """
    Run the simulation and return the final state tensor.
    state: (n_steps, 2 agents, 3 features, 2 dims)
    params: dictionary containing simulation parameters
    """
    n_steps = state.shape[0]

    for t in range(1, n_steps):
        pred_pos = state[t-1, 0, 0, :]  # Predator position
        pred_vel = state[t-1, 0, 1, :]  # Predator velocity
        prey_pos = state[t-1, 1, 0, :]  # Prey position
        prey_vel = state[t-1, 1, 1, :]  # Prey velocity

        r_vec = prey_pos - pred_pos # line of sight (LOS)
        v_rel = prey_vel - pred_vel # relative velocity

        distance = norm(r_vec)

        if distance < r_kill:
            return state, t
        
        pred_acc_vec = proportional_navigation_acceleration(r_vec, v_rel, navigation_gain)

        pred_acc_norm = norm(pred_acc_vec)
        if pred_acc_norm > pred_acc_max:
            pred_acc_vec = pred_acc_vec * (pred_acc_max / pred_acc_norm)

        prey_acc_vec = np.zeros(2)
        if distance < r_react:
            # Normalize the line-of-sight vector
            if distance > 1e-10:
                r_unit = r_vec / distance
                # Calculate perpendicular direction (90 degrees clockwise)
                escape_dir = np.array([-r_unit[1], r_unit[0]])
                prey_acc_vec = escape_dir * prey_acc_max

        # Store accelerations
        state[t-1, 0, 2, :] = pred_acc_vec
        state[t-1, 1, 2, :] = prey_acc_vec

        # Update step
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

@njit
def proportional_navigation_acceleration(r_vec, v_rel, navigation_gain):
    # Calculate distance and normalized LOS vector
    distance = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    
    # Avoid division by zero
    if distance < 1e-10:
        return np.zeros(2)
    
    # LOS unit vector
    r_unit_x = r_vec[0] / distance
    r_unit_y = r_vec[1] / distance
    
    # Calculate closing velocity (projection of v_rel along LOS)
    v_closing = r_unit_x * v_rel[0] + r_unit_y * v_rel[1]
    
    # Calculate LOS rotation rate (omega)
    cross_product = r_vec[0] * v_rel[1] - r_vec[1] * v_rel[0]
    omega = cross_product / (distance * distance)
    
    # Calculate perpendicular unit vector to LOS
    perp_unit_x = -r_unit_y
    perp_unit_y = r_unit_x
    
    # Calculate acceleration magnitude
    acc_magnitude = navigation_gain * v_closing * omega
    
    # Calculate acceleration vector
    acc_x = perp_unit_x * acc_magnitude
    acc_y = perp_unit_y * acc_magnitude
    
    return np.array([acc_x, acc_y])