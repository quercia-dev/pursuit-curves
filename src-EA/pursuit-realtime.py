import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# time: t in {0, ... n_steps}
# agents: 0=predator, 1=prey;
# feature: 0=x, 1=x',2=x''
# dims 2D


def get_trajectory(deg: float, params: dict):

    state = np.zeros((params["n_steps"], 2, 3, 2), dtype=float)

    # initial predator pos, vel, acc
    state[0, 0, 0] = [0, 0]
    state[0, 0, 1] = [0, 0]
    state[0, 0, 2] = [0, 0]

    rad = np.deg2rad(deg)

    # initial prey pos, vel, acc
    state[0, 1, 0] = [1, 0]
    state[0, 1, 1] = [np.sin(rad) * params["11_max"], np.cos(rad) * params["11_max"]]
    state[0, 1, 2] = [0, 0]

    return state


def normalize(v):
    length = np.linalg.norm(v)
    if length == 0:
        return np.zeros_like(v)
    return v / length


def rotate_right_90(v):
    """Rotate a 2D vector 90 degrees to the right (clockwise)."""
    return np.array([v[1], -v[0]])


def step(state, t, dt, pred_maxspeed, prey_maxspeed):
    """
    Update position and velocity for timestep t based on previous state.
    Applies max speed limits separately for predator (agent 0) and prey (agent 1).
    """
    # Position update
    state[t, :, 0, :] = state[t - 1, :, 0, :] + state[t - 1, :, 1, :] * dt

    # Velocity update
    state[t, :, 1, :] = state[t - 1, :, 1, :] + state[t - 1, :, 2, :] * dt

    # Limit predator speed
    pred_speed = np.linalg.norm(state[t, 0, 1, :])
    if pred_speed > pred_maxspeed:
        state[t, 0, 1, :] = state[t, 0, 1, :] / pred_speed * pred_maxspeed

    # Limit prey speed
    prey_speed = np.linalg.norm(state[t, 1, 1, :])
    if prey_speed > prey_maxspeed:
        state[t, 1, 1, :] = state[t, 1, 1, :] / prey_speed * prey_maxspeed

    return state


def simulate(state, params):
    """
    Run the simulation and return the final state tensor.
    state: (n_steps, 2 agents, 3 features, 2 dims)
    params: dictionary containing simulation parameters
    """
    n_steps = state.shape[0]
    dt = params["dt"]

    for t in range(1, n_steps):
        distance = np.linalg.norm(state[t - 1, 0, 0] - state[t - 1, 1, 0])

        if distance < params["R_kill"]:
            # Cut the tensor: freeze everything after t
            # Set velocities and accelerations to zero at capture
            state[t, :, 1, :] = 0  # zero velocities
            state[t, :, 2, :] = 0  # zero accelerations

            # Cut the tensor after capture
            state = state[:t]
            break
        elif distance < params["R_react"]:
            # Prey reacts: rotate predator-prey vector 90° right
            direction = state[t - 1, 1, 0] - state[t - 1, 0, 0]
            state[t - 1, 1, 2, :] = (
                normalize(rotate_right_90(direction)) * params["12_max"]
            )

        # Predator always steers toward prey
        direction = state[t - 1, 1, 0] - state[t - 1, 0, 0]
        state[t - 1, 0, 2, :] = normalize(direction) * params["02_max"]

        # Step positions and velocities
        step(state, t, dt, params["01_max"], params["11_max"])

    return state


# Configurable parameters
params = {
    "dt": 0.01,
    "n_steps": 10000,
    "11_max": 1.0,
    "01_max": 10.0,
    "12_max": 1.0,
    "02_max": 1.0,
    "R_kill": 0.001,
    "R_react": 1.0,
}
angle_deg = 36

# Run simulation
state = get_trajectory(angle_deg, params)
sim_state = simulate(state, params)

# 2) Extract trajectories & kinematics
pred_traj = sim_state[:, 0, 0]
prey_traj = sim_state[:, 1, 0]
pred_vel = sim_state[:, 0, 1]
prey_vel = sim_state[:, 1, 1]
pred_acc = sim_state[:, 0, 2]
prey_acc = sim_state[:, 1, 2]


# 3) Precompute metrics
dt = params["dt"]
t = np.arange(len(pred_traj)) * dt

# Unsigned distance
sep_vec = prey_traj - pred_traj
dist = np.linalg.norm(sep_vec, axis=1)
d_dist = np.gradient(dist, dt)

# Velocities
prey_vel_array = prey_vel  # (N,2) array from sim_state
pred_vel_array = pred_vel

# Signed distance via 2D cross‐product sign
distance_unsigned = dist
relative_position = pred_traj - prey_traj
# compute z‐component of 3D cross([vx,vy,0], [rx,ry,0])
cross_z = (
    prey_vel_array[:, 0] * relative_position[:, 1]
    - prey_vel_array[:, 1] * relative_position[:, 0]
)
sign = cross_z
signed = distance_unsigned * sign
d_signed = np.gradient(signed, dt)

# Velocity dot‐product
vel_dot = np.einsum("ij,ij->i", pred_vel_array, prey_vel_array)

# 4) Create 3×2 grid (full)
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
ax0, ax1, ax2, ax3, ax4, ax5 = axes.flatten()

# —— Ax0: chase + velocity/acceleration quivers ——
ax0.set_title("Predator vs Prey with Vel & Acc")
all_pts = np.vstack([pred_traj, prey_traj])
xmin, ymin = all_pts.min(axis=0) - 0.5
xmax, ymax = all_pts.max(axis=0) + 0.5
ax0.set_xlim(xmin, xmax)
ax0.set_ylim(ymin, ymax)
ax0.set_xlabel("X")
ax0.set_ylabel("Y")
ax0.grid(True)

scatter_pred = ax0.scatter([], [], c="red", s=50, label="Predator")
scatter_prey = ax0.scatter([], [], c="green", s=50, label="Prey")

pred_vel_q = ax0.quiver(
    [], [], [], [], angles="xy", scale_units="xy", scale=1, color="black", width=0.004
)
pred_acc_q = ax0.quiver(
    [], [], [], [], angles="xy", scale_units="xy", scale=1, color="gray", width=0.004
)
prey_vel_q = ax0.quiver(
    [], [], [], [], angles="xy", scale_units="xy", scale=1, color="blue", width=0.004
)
prey_acc_q = ax0.quiver(
    [], [], [], [], angles="xy", scale_units="xy", scale=1, color="cyan", width=0.004
)

ax0.legend(loc="upper right")

# —— Ax1: distance ——
ax1.set_title("Distance ||prey–pred||")
ax1.set_xlim(0, t[-1])
ax1.set_ylim(dist.min() * 0.9, dist.max() * 1.1)
ax1.set_xlabel("Time")
ax1.set_ylabel("Distance")
(line_dist,) = ax1.plot([], [], lw=2)

# —— Ax2: d(distance)/dt ——
ax2.set_title("d(Distance)/dt")
ax2.set_xlim(0, t[-1])
ax2.set_ylim(d_dist.min() * 1.1, d_dist.max() * 1.1)
ax2.set_xlabel("Time")
ax2.set_ylabel("dDist/dt")
(line_d_dist,) = ax2.plot([], [], lw=2)

# —— Ax3: signed distance ——
ax3.set_title("Signed Distance")
ax3.set_xlim(0, t[-1])
ax3.set_ylim(signed.min() * 1.1, signed.max() * 1.1)
ax3.set_xlabel("Time")
ax3.set_ylabel("Signed dist")
(line_signed,) = ax3.plot([], [], lw=2)

# —— Ax4: velocity dot product ——
ax4.set_title("v_pred · v_prey")
ax4.set_xlim(0, t[-1])
ax4.set_ylim(vel_dot.min() * 1.1, vel_dot.max() * 1.1)
ax4.set_xlabel("Time")
ax4.set_ylabel("Dot product")
(line_dot,) = ax4.plot([], [], lw=2)

# —— Ax5: d(signed distance)/dt ——
ax5.set_title("d(Signed Distance)/dt")
ax5.set_xlim(0, t[-1])
ax5.set_ylim(d_signed.min() * 1.1, d_signed.max() * 1.1)
ax5.set_xlabel("Time")
ax5.set_ylabel("dSigned/dt")
(line_d_signed,) = ax5.plot([], [], lw=2)

plt.tight_layout()


# 5) Animation init & update
def init():
    scatter_pred.set_offsets(pred_traj[0])
    scatter_prey.set_offsets(prey_traj[0])
    # zero out all arrows
    for q in (pred_vel_q, pred_acc_q, prey_vel_q, prey_acc_q):
        q.set_offsets(np.array([[0, 0]]))
        q.set_UVC(0, 0)
    # reset metric lines
    for ln in (line_dist, line_d_dist, line_signed, line_dot, line_d_signed):
        ln.set_data([], [])
    return [
        scatter_pred,
        scatter_prey,
        pred_vel_q,
        pred_acc_q,
        prey_vel_q,
        prey_acc_q,
        line_dist,
        line_d_dist,
        line_signed,
        line_dot,
        line_d_signed,
    ]


def update(i):
    i = i * 1
    # update positions
    scatter_pred.set_offsets(pred_traj[i])
    scatter_prey.set_offsets(prey_traj[i])
    # update vectors
    pred_vel_q.set_offsets(np.array([pred_traj[i]]))
    pred_vel_q.set_UVC(pred_vel[i, 0], pred_vel[i, 1])
    pred_acc_q.set_offsets(np.array([pred_traj[i]]))
    pred_acc_q.set_UVC(pred_acc[i, 0], pred_acc[i, 1])
    prey_vel_q.set_offsets(np.array([prey_traj[i]]))
    prey_vel_q.set_UVC(prey_vel[i, 0], prey_vel[i, 1])
    prey_acc_q.set_offsets(np.array([prey_traj[i]]))
    prey_acc_q.set_UVC(prey_acc[i, 0], prey_acc[i, 1])
    # update metrics
    line_dist.set_data(t[:i], dist[:i])
    line_d_dist.set_data(t[:i], d_dist[:i])
    line_signed.set_data(t[:i], signed[:i])
    line_dot.set_data(t[:i], vel_dot[:i])
    line_d_signed.set_data(t[:i], d_signed[:i])
    return [
        scatter_pred,
        scatter_prey,
        pred_vel_q,
        pred_acc_q,
        prey_vel_q,
        prey_acc_q,
        line_dist,
        line_d_dist,
        line_signed,
        line_dot,
        line_d_signed,
    ]


ani = FuncAnimation(
    fig, update, frames=int(len(t)), init_func=init, interval=0, blit=True
)
plt.show()