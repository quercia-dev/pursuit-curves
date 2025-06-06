import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib import gridspec
from numba import njit

@njit
def restack_state_noacc(traj_flat):
    """
    Convert a flat RK4 trajectory array of shape (n_steps, 8)
    into the “state tensor” shape (n_steps, 2 agents, 3 features, 2 dims),
    where:
      feature 0 = position (x, y)
      feature 1 = velocity (vx, vy)
      feature 2 = acceleration = (0, 0)

    traj_flat[i] = [p_x, p_y, v_x, v_y, q_x, q_y, u_x, u_y]
      predator pos = (p_x, p_y)
      predator vel = (v_x, v_y)
      prey    pos = (q_x, q_y)
      prey    vel = (u_x, u_y)

    Returns a numpy array of shape (n_steps, 2, 3, 2), with zeros for accelerations.
    """
    n_steps = traj_flat.shape[0]
    state = np.zeros((n_steps, 2, 3, 2), dtype=np.float64)

    for t in range(n_steps):
        # Unpack flat trajectory at time t
        p_x = traj_flat[t, 0]
        p_y = traj_flat[t, 1]
        v_x = traj_flat[t, 2]
        v_y = traj_flat[t, 3]
        q_x = traj_flat[t, 4]
        q_y = traj_flat[t, 5]
        u_x = traj_flat[t, 6]
        u_y = traj_flat[t, 7]

        # --- Fill predator position & velocity ---
        state[t, 0, 0, 0] = p_x   # predator x
        state[t, 0, 0, 1] = p_y   # predator y
        state[t, 0, 1, 0] = v_x   # predator vx
        state[t, 0, 1, 1] = v_y   # predator vy

        # --- Fill prey position & velocity ---
        state[t, 1, 0, 0] = q_x   # prey x
        state[t, 1, 0, 1] = q_y   # prey y
        state[t, 1, 1, 0] = u_x   # prey vx
        state[t, 1, 1, 1] = u_y   # prey vy

        # --- Feature 2 (acceleration) remains zero ---
        # state[t, 0, 2, :] and state[t, 1, 2, :] are already zeros

    return state

    
def plot_trajectory(state, params: dict, bound: int = 20, title='', ax_bounded=None, ax_autoscaled=None):
    """
    Plot predator and prey trajectories in two views: bounded and auto-scaled.
    
    Parameters:
    -----------
    state : numpy.ndarray
        System state array with shape (time_steps, num_agents, features, dimensions)
    params : dict
        Dictionary of simulation parameters
    bound : int, optional
        Boundary for the bounded view plot
    title : str, optional
        Title for the plots
    ax_bounded : matplotlib.axes.Axes, optional
        Axes for the bounded view plot. If None, new axes will be created.
    ax_autoscaled : matplotlib.axes.Axes, optional
        Axes for the auto-scaled view plot. If None, new axes will be created.
        
    Returns:
    --------
    tuple
        (ax_bounded, ax_autoscaled) - The axes objects used for plotting
    """
    
    state = state.copy()

    # Extract data for trajectory plots
    predator_pos = state[:, 0, 0, :]
    prey_pos = state[:, 1, 0, :]
    
    # Prepare connecting line indices
    n_steps = state.shape[0]
    indices = np.linspace(0, n_steps - 1, 10, dtype=int)
    
    # Create axes if not provided
    if ax_bounded is None or ax_autoscaled is None:
        fig, (ax_bounded, ax_autoscaled) = plt.subplots(1, 2, figsize=(16, 5))
        
        # Set title if creating a new figure
        if title == '':
            title = f'{params}'
        fig.suptitle(title, fontsize=16, y=0.99, va='top')

    # Plot on both axes
    for ax_idx, ax in enumerate([ax_bounded, ax_autoscaled]):
        ax.plot(predator_pos[:, 0], predator_pos[:, 1], label='Predator', color='red')
        ax.plot(prey_pos[:, 0], prey_pos[:, 1], label='Prey', color='green')
        ax.scatter(predator_pos[0, 0], predator_pos[0, 1], color='red', marker='o', label='Predator Start')
        ax.scatter(prey_pos[0, 0], prey_pos[0, 1], color='green', marker='o', label='Prey Start')
        ax.scatter(predator_pos[-1, 0], predator_pos[-1, 1], color='red', marker='x', label='Predator End')
        ax.scatter(prey_pos[-1, 0], prey_pos[-1, 1], color='green', marker='x', label='Prey End')

        for idx in indices:
            ax.plot(
                [predator_pos[idx, 0], prey_pos[idx, 0]],
                [predator_pos[idx, 1], prey_pos[idx, 1]],
                color='blue', linestyle='--', linewidth=0.8
            )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True)

        if ax_idx == 0:
            ax.set_xlim(-bound, bound)
            ax.set_ylim(-bound, bound)
            ax.set_title('Bounded View')
            ax.legend(loc='best')
        elif ax_idx == 1:
            all_positions = np.vstack([predator_pos, prey_pos])
            x_min, y_min = np.min(all_positions, axis=0)
            x_max, y_max = np.max(all_positions, axis=0)
            ax.set_xlim(x_min - 1, x_max + 1)
            ax.set_ylim(y_min - 1, y_max + 1)
            ax.set_title('Full View (Auto-scaled)')
            
    return ax_bounded, ax_autoscaled


def plot_metrics(state, params: dict, analysis_fns=None, fig=None, gs=None, row_start=0):
    """
    Plot metrics derived from the state data.
    
    Parameters:
    -----------
    state : numpy.ndarray
        System state array with shape (time_steps, num_agents, features, dimensions)
    params : dict
        Dictionary of simulation parameters
    analysis_fns : list, optional
        List of tuples (function, title, y_label) where function takes state and returns a metric
    fig : matplotlib.figure.Figure, optional
        Figure to plot on. If None, a new figure will be created.
    gs : matplotlib.gridspec.GridSpec, optional
        GridSpec to place the plots on. If None, a new gridspec will be created.
    row_start : int, optional
        Starting row in the gridspec
        
    Returns:
    --------
    list
        List of axes objects created for the metric plots
    """    
    if analysis_fns is None:
        analysis_fns = []
        
    n_metrics = len(analysis_fns)
    if n_metrics == 0:
        return []
        
    # Time array for x-axis
    time = np.arange(state.shape[0])
    
    # Create figure and gridspec if not provided
    if fig is None or gs is None:
        fig = plt.figure(figsize=(16, 2 * n_metrics))
        gs = plt.GridSpec(n_metrics, 1)
        
    # Create metric plots
    axes = []
    for i, (fn, metric_title, y_label) in enumerate(analysis_fns):
        ax = fig.add_subplot(gs[row_start + i, :])
        metric = fn(state)

        # Ensure metric is a 1D array of correct length
        metric = np.asarray(metric).squeeze()
        if metric.ndim != 1 or metric.shape[0] != time.shape[0]:
            raise ValueError(f"Metric function '{metric_title}' returned invalid shape: {metric.shape}")

        ax.plot(time, metric, color='blue' if i%2 == 0 else 'black')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.0, alpha=0.5)
        ax.set_ylabel(y_label)
        ax.set_title(metric_title)
        ax.grid(True)
        
        axes.append(ax)

        if i < len(analysis_fns) - 1:
            ax.label_outer()  # Hide x-tick labels

    # Label the x-axis only on the last subplot
    if axes:
        axes[-1].set_xlabel('Time')
        formatter = ticker.FuncFormatter(lambda x, _: f"{x * params['dt']}")
        axes[-1].xaxis.set_major_formatter(formatter)
    
    return axes


def plot_combined_analysis(state, params: dict, bound: int = 20, title='', analysis_fns=None):
    """
    Combined plotting function that shows:
    - Top row: Two trajectory plots side by side (bounded and auto-scaled)
    - Bottom section: Variable number of metric plots passed in via analysis_fns
    
    Parameters:
    -----------
    state : numpy.ndarray
        System state array with shape (time_steps, num_agents, features, dimensions)
    params : dict
        Dictionary of simulation parameters
    bound : int, optional
        Boundary for the bounded view plot
    title : str, optional
        Title for the plots
    analysis_fns : list, optional
        List of tuples (function, title, y_label) where function takes state and returns a metric
    """
    if analysis_fns is None:
        analysis_fns = []

    n_metrics = len(analysis_fns)
    total_rows = 1 + n_metrics

    # Create figure and gridspec
    fig = plt.figure(figsize=(16, 3 + 2.5 * total_rows))
    gs = gridspec.GridSpec(total_rows, 2, height_ratios=[2.5] + [1.0] * n_metrics)
    
    # Set title
    if title == '':
        title = f'{params}'
    fig.suptitle(title, fontsize=16, y=0.99, va='top')
    
    # Create trajectory plots
    ax_bounded = fig.add_subplot(gs[0, 0])
    ax_autoscaled = fig.add_subplot(gs[0, 1])
    plot_trajectory(state, params, bound, title='', ax_bounded=ax_bounded, ax_autoscaled=ax_autoscaled)
    
    # Create metric plots
    plot_metrics(state, params, analysis_fns, fig=fig, gs=gs, row_start=1)
    
    # Adjust layout
    fig.subplots_adjust(hspace=0.25, top=0.94, bottom=0.06, left=0.08, right=0.95)
    plt.show()