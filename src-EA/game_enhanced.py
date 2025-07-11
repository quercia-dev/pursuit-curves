import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
from numba.experimental import jitclass
from numba import float64, int32, boolean

enhanced_prey_spec = [
    ('x', float64),
    ('y', float64),
    ('vx', float64),
    ('vy', float64),
    ('max_speed', float64),
    ('react_radius', float64),
    ('evasion_angle', float64),
    ('evasion_time', float64),
    ('alive', boolean),
    ('boundary_x', float64),
    ('boundary_y', float64),
    ('time_since_evasion', float64),
    ('zigzag_direction', float64),
    ('base_flee_angle', float64),
    ('evasion_active', boolean),
    ('energy', float64),
    ('max_energy', float64),
    ('energy_regen_rate', float64),
    ('energy_cost_per_acceleration', float64),
]

enhanced_predator_spec = [
    ('x', float64),
    ('y', float64),
    ('vx', float64),
    ('vy', float64),
    ('max_speed', float64),
    ('catch_radius', float64),
    ('boundary_x', float64),
    ('boundary_y', float64),
    ('energy', float64),
    ('max_energy', float64),
    ('energy_regen_rate', float64),
    ('energy_cost_per_acceleration', float64),
    ('pursuit_strength', float64),
]

@jitclass(enhanced_prey_spec)
class EnhancedPrey:
    def __init__(self, x, y, max_speed=3.0, react_radius=15.0, evasion_angle=np.pi/3, evasion_time=2.0, boundary_x=100.0, boundary_y=100.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = max_speed
        self.react_radius = react_radius
        self.evasion_angle = evasion_angle
        self.evasion_time = evasion_time
        self.alive = True
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        self.time_since_evasion = 0.0
        self.zigzag_direction = 1.0 
        self.base_flee_angle = 0.0
        self.evasion_active = False
        
        # Energy system
        self.max_energy = 100.0
        self.energy = self.max_energy
        self.energy_regen_rate = 1.0  # energy per second when not accelerating hard
        self.energy_cost_per_acceleration = 25.0  # energy cost for high acceleration
    
    def update(self, predator_x, predator_y, dt=0.1):
        if not self.alive:
            return
        
        # Define base steering strength at the beginning
        base_steering_strength = 10.0
        
        # Calculate distance to predator
        dx = self.x - predator_x
        dy = self.y - predator_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        self.time_since_evasion += dt
        
        # Store previous velocity for acceleration calculation
        prev_vx, prev_vy = self.vx, self.vy
        
        if distance < self.react_radius and distance > 0:
            self.evasion_active = True
            self.base_flee_angle = np.arctan2(dy, dx)

            if self.time_since_evasion >= self.evasion_time:
                self.zigzag_direction *= -1.0

            zigzag_angle = self.base_flee_angle + (self.zigzag_direction * self.evasion_angle)
            target_vx = np.cos(zigzag_angle) * self.max_speed
            target_vy = np.sin(zigzag_angle) * self.max_speed
            
            # Energy-dependent steering strength
            energy_factor = max(0.1, self.energy / self.max_energy)  # Never go below 10% effectiveness
            steering_strength = base_steering_strength * energy_factor
            
            if self.time_since_evasion >= self.evasion_time:
                self.vx += (target_vx - self.vx) * steering_strength * dt
                self.vy += (target_vy - self.vy) * steering_strength * dt
                self.time_since_evasion = 0.0
            else:
                current_angle = self.base_flee_angle + (self.zigzag_direction * self.evasion_angle)
                target_vx = np.cos(current_angle) * self.max_speed
                target_vy = np.sin(current_angle) * self.max_speed
                
                gentle_steering = 2.0 * energy_factor
                self.vx += (target_vx - self.vx) * gentle_steering * dt
                self.vy += (target_vy - self.vy) * gentle_steering * dt
        else:
            self.evasion_active = False
            self.vx = 0.5
            self.vy = 0.5
            self.time_since_evasion = self.evasion_time
        
        # Calculate acceleration magnitude for energy cost
        accel_x = (self.vx - prev_vx) / dt if dt > 0 else 0
        accel_y = (self.vy - prev_vy) / dt if dt > 0 else 0
        acceleration_magnitude = np.sqrt(accel_x*accel_x + accel_y*accel_y)
        
        # Energy consumption based on acceleration
        if acceleration_magnitude > 1.5:  # Only high accelerations cost energy
            energy_cost = self.energy_cost_per_acceleration * (acceleration_magnitude / base_steering_strength) * dt
            self.energy = max(0.0, self.energy - energy_cost)
        else:
            # Regenerate energy when not accelerating hard
            self.energy = min(self.max_energy, self.energy + self.energy_regen_rate * dt)
        
        # Apply drag
        drag_factor = 0.95
        self.vx *= drag_factor
        self.vy *= drag_factor
        
        # Limit speed based on energy
        current_max_speed = self.max_speed * max(0.3, self.energy / self.max_energy)
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > current_max_speed:
            self.vx = (self.vx / speed) * current_max_speed
            self.vy = (self.vy / speed) * current_max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Boundary collision with bounce
        if self.x < 0 or self.x > self.boundary_x:
            self.vx *= -0.8
            self.x = max(0, min(self.boundary_x, self.x))
            self.zigzag_direction *= -1.0
        
        if self.y < 0 or self.y > self.boundary_y:
            self.vy *= -0.8
            self.y = max(0, min(self.boundary_y, self.y))
            self.zigzag_direction *= -1.0

@jitclass(enhanced_predator_spec)
class EnhancedPredator:
    def __init__(self, x, y, max_speed=4.0, catch_radius=2.5, pursuit_strength=4.0, boundary_x=100.0, boundary_y=100.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = max_speed
        self.catch_radius = catch_radius
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
        
        # Energy system
        self.max_energy = 120.0
        self.energy = self.max_energy
        self.energy_regen_rate = 15.0
        self.energy_cost_per_acceleration = 20.0
        self.pursuit_strength = pursuit_strength
    
    def update(self, prey_x, prey_y, prey_react_radius, dt=0.1):
        # Store previous velocity for acceleration calculation
        prev_vx, prev_vy = self.vx, self.vy
        
        # Chase behavior - move towards prey
        dx = prey_x - self.x
        dy = prey_y - self.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Energy-dependent pursuit strength
            base_pursuit_strength = self.pursuit_strength
            energy_factor = max(0.2, self.energy / self.max_energy)
            pursuit_energy = base_pursuit_strength * energy_factor
            
            self.vx += (dx / distance) * pursuit_energy
            self.vy += (dy / distance) * pursuit_energy
        
        # Calculate acceleration magnitude for energy cost
        accel_x = (self.vx - prev_vx) / dt if dt > 0 else 0
        accel_y = (self.vy - prev_vy) / dt if dt > 0 else 0
        acceleration_magnitude = np.sqrt(accel_x*accel_x + accel_y*accel_y)
        
        # Energy consumption based on acceleration
        if distance < prey_react_radius:
            if acceleration_magnitude > 3.0:
                energy_cost = self.energy_cost_per_acceleration * (acceleration_magnitude / 40.0) * dt
                self.energy = max(0.0, self.energy - energy_cost)
            else:
                # Regenerate energy when not accelerating hard
                self.energy = min(self.max_energy, self.energy + self.energy_regen_rate * dt)
        
        # Apply drag
        self.vx *= 0.6
        self.vy *= 0.6
        
        # Limit speed based on energy
        current_max_speed = self.max_speed * max(0.4, self.energy / self.max_energy)
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > current_max_speed:
            self.vx = (self.vx / speed) * current_max_speed
            self.vy = (self.vy / speed) * current_max_speed
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Boundary collision with bounce
        if self.x < 0 or self.x > self.boundary_x:
            self.vx *= -0.8
            self.x = max(0, min(self.boundary_x, self.x))
        
        if self.y < 0 or self.y > self.boundary_y:
            self.vy *= -0.8
            self.y = max(0, min(self.boundary_y, self.y))
    
    def check_catch(self, prey_x, prey_y):
        distance = np.sqrt((self.x - prey_x)**2 + (self.y - prey_y)**2)
        return distance < self.catch_radius

@njit
def calculate_distance(x1, y1, x2, y2):
    """Fast distance calculation using njit"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class EnhancedPredatorPreyGame:
    def __init__(self, width=100, height=100, prey_genome=None, predator_genome=None):
        self.width = width
        self.height = height
        
        # Initialize prey and predator at random positions
        if prey_genome is None:
            self.prey = EnhancedPrey(
                x=15,
                y=15,
                boundary_x=width,
                boundary_y=height
            )
        else:
            self.prey = EnhancedPrey(
                x=15,
                y=15,
                max_speed = 3.0,
                react_radius = prey_genome.react_radius,
                evasion_angle = prey_genome.evasion_angle,
                evasion_time = prey_genome.evasion_time,
                boundary_x=width,
                boundary_y=height
            )
        
        if predator_genome is None:
            self.predator = EnhancedPredator(
                x=0,
                y=0,
                boundary_x=width,
                boundary_y=height
            )
        else:
            self.predator = EnhancedPredator(
                x=0,
                y=0,
                max_speed = 4.0,
                catch_radius = 2.5,
                boundary_x=width,
                boundary_y=height,
                pursuit_strength = predator_genome.pursuit_strength
            )

        
        self.game_over = False
        self.catch_time = None
        self.time_elapsed = 0
        self.initial_prey_position = (self.prey.x, self.prey.y)
        
        # For visualization
        self.prey_trail = [(self.prey.x, self.prey.y)]
        self.predator_trail = [(self.predator.x, self.predator.y)]
        self.max_trail_length = 200
    
    def update(self, dt=0.1):
        if self.game_over:
            return
        
        self.time_elapsed += dt
        
        # Update entities
        self.prey.update(self.predator.x, self.predator.y, dt)
        self.predator.update(self.prey.x, self.prey.y, self.prey.react_radius, dt)
        
        # Check for catch
        if self.predator.check_catch(self.prey.x, self.prey.y):
            self.prey.alive = False
            self.game_over = True
            self.catch_time = self.time_elapsed
        
        # Update trails
        self.prey_trail.append((self.prey.x, self.prey.y))
        self.predator_trail.append((self.predator.x, self.predator.y))
        
        # Limit trail length
        if len(self.prey_trail) > self.max_trail_length:
            self.prey_trail.pop(0)
        if len(self.predator_trail) > self.max_trail_length:
            self.predator_trail.pop(0)
    
    def reset(self):
        """Reset the game with new random positions"""
        self.prey = EnhancedPrey(
            x=15,
            y=15,
            boundary_x=self.width,
            boundary_y=self.height
        )
        
        self.predator = EnhancedPredator(
            x=0,
            y=0,
            boundary_x=self.width,
            boundary_y=self.height
        )
        
        self.game_over = False
        self.catch_time = None
        self.time_elapsed = 0
        self.prey_trail = [(self.prey.x, self.prey.y)]
        self.predator_trail = [(self.predator.x, self.predator.y)]

    def get_survival_time(self):
        return self.catch_time if self.catch_time else self.time_elapsed


def visualize_game():
    """Visualize the predator-prey game with matplotlib animation"""
    game = EnhancedPredatorPreyGame()
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, game.width)
    ax.set_ylim(0, game.height)
    ax.set_aspect('equal')
    ax.set_title('Predator-Prey Pursuit Game', fontsize=16)
    
    # Create plot elements
    prey_dot, = ax.plot([], [], 'go', markersize=8, label='Prey')
    predator_dot, = ax.plot([], [], 'ro', markersize=10, label='Predator')
    prey_trail_line, = ax.plot([], [], 'g-', alpha=0.5, linewidth=1)
    predator_trail_line, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)
    
    # Fear radius circle
    fear_circle = plt.Circle((0, 0), game.prey.react_radius, 
                           fill=False, color='green', alpha=0.3, linestyle='--')
    ax.add_patch(fear_circle)
    
    # Catch radius circle
    catch_circle = plt.Circle((0, 0), game.predator.catch_radius, 
                            fill=False, color='red', alpha=0.3, linestyle='--')
    ax.add_patch(catch_circle)
    
    ax.legend()
    
    # Status text
    status_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def animate(frame):
        game.update()
        
        # Update positions
        prey_dot.set_data([game.prey.x], [game.prey.y])
        predator_dot.set_data([game.predator.x], [game.predator.y])
        
        # Update trails
        if len(game.prey_trail) > 1:
            prey_x, prey_y = zip(*game.prey_trail)
            prey_trail_line.set_data(prey_x, prey_y)
        
        if len(game.predator_trail) > 1:
            pred_x, pred_y = zip(*game.predator_trail)
            predator_trail_line.set_data(pred_x, pred_y)
        
        # Update circles
        fear_circle.center = (game.prey.x, game.prey.y)
        catch_circle.center = (game.predator.x, game.predator.y)
        
        # Update status
        if game.game_over:
            status_text.set_text(f'CAUGHT! Time: {game.catch_time:.1f}s\nPress R to reset')
            # Auto-reset after 3 seconds
            if game.time_elapsed - game.catch_time > 3.0:
                game.reset()
        else:
            distance = calculate_distance(game.prey.x, game.prey.y, 
                                        game.predator.x, game.predator.y)
            zigzag_status = "EVADING" if game.prey.evasion_active else "wandering"
            zigzag_dir = "LEFT" if game.prey.zigzag_direction < 0 else "RIGHT"
            
            status_text.set_text(f'Time: {game.time_elapsed:.1f}s\n'
                               f'Distance: {distance:.1f}\n'
                               f'Status: {zigzag_status}\n'
                               f'Zigzag: {zigzag_dir}\n'
                               f'Evasion Angle: {np.degrees(game.prey.evasion_angle):.1f}°\n'
                               f'Evasion Time: {game.prey.evasion_time:.1f}s\n'
                               f'React Radius: {game.prey.react_radius:.1f}\n'
                               f'Prey Energy: {game.prey.energy:.1f}/{game.prey.max_energy:.1f}\n'
                               f'Predator Energy: {game.predator.energy:.1f}/{game.predator.max_energy:.1f}')
        return prey_dot, predator_dot, prey_trail_line, predator_trail_line
    
    def on_key(event):
        if event.key == 'r' or event.key == 'R':
            game.reset()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    print("Starting Predator-Prey Pursuit Game...")
    print("Green circle: Prey (tries to escape)")
    print("Red circle: Predator (chases prey)")
    print("Dashed circles show fear/catch radius")
    print("Press 'R' to reset the game")
    print("Game will auto-reset 3 seconds after catch")
    
    # Run the visualization
    animation_obj = visualize_game()

