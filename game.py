import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
from numba.experimental import jitclass
from numba import float64, int32, boolean

# Define the spec for Numba jitclass
prey_spec = [
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
    ('time_since_evasion', float64)
]

predator_spec = [
    ('x', float64),
    ('y', float64),
    ('vx', float64),
    ('vy', float64),
    ('max_speed', float64),
    ('catch_radius', float64),
    ('boundary_x', float64),
    ('boundary_y', float64)
]

@jitclass(prey_spec)
class Prey:
    def __init__(self, x, y, max_speed=3.0, react_radius=15.0, evasion_angle=np.pi/2, evasion_time=1.0, boundary_x=100.0, boundary_y=100.0):
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
    
    def update(self, predator_x, predator_y, dt=0.1):
        if not self.alive:
            return
        
        # Calculate distance to predator
        dx = self.x - predator_x
        dy = self.y - predator_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        self.time_since_evasion += dt
        
        if distance < self.react_radius and distance > 0:
            # Flee behavior - move away from predator
            if self.time_since_evasion >= self.evasion_time:
                #Evasion
                evasion_vx = np.cos(self.evasion_angle)
                evasion_vy = np.sin(self.evasion_angle)
                self.vx += evasion_vx * self.max_speed
                self.vy += evasion_vy * self.max_speed
                self.time_since_evasion = 0.0
            else:
                # Continue current movement
                self.vx *= 0.95
                self.vy *= 0.95
        else:
            # Random wandering when not threatened
            self.vx += (np.random.random() - 0.5) * 0.5
            self.vy += (np.random.random() - 0.5) * 0.5
        
        # Apply drag
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Limit speed
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed
        
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


@jitclass(predator_spec)
class Predator:
    def __init__(self, x, y, max_speed=2.5, catch_radius=5.0, boundary_x=100.0, boundary_y=100.0):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.max_speed = max_speed
        self.catch_radius = catch_radius
        self.boundary_x = boundary_x
        self.boundary_y = boundary_y
    
    def update(self, prey_x, prey_y, dt=0.1):
        # Chase behavior - move towards prey
        dx = prey_x - self.x
        dy = prey_y - self.y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            # Pursue the prey
            pursuit_strength = 3.0
            self.vx += (dx / distance) * pursuit_strength
            self.vy += (dy / distance) * pursuit_strength
        
        # Apply drag
        self.vx *= 0.9
        self.vy *= 0.9
        
        # Limit speed
        speed = np.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed
        
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

class PredatorPreyGame:
    def __init__(self, width=100, height=100, prey_genome=None):
        self.width = width
        self.height = height
        
        # Initialize prey and predator at random positions
        if prey_genome is None:
            self.prey = Prey(
                x=50,
                y=50,
                boundary_x=width,
                boundary_y=height
            )
        else:
            self.prey = Prey(
                x=50,
                y=50,
                max_speed = 3.0,
                react_radius = prey_genome.react_radius,
                evasion_angle = prey_genome.evasion_angle,
                evasion_time = prey_genome.evasion_time,
                boundary_x=width,
                boundary_y=height
            )
        
        self.predator = Predator(
            x=0,
            y=0,
            boundary_x=width,
            boundary_y=height
        )
        
        self.game_over = False
        self.catch_time = None
        self.time_elapsed = 0
        
        # For visualization
        self.prey_trail = [(self.prey.x, self.prey.y)]
        self.predator_trail = [(self.predator.x, self.predator.y)]
        self.max_trail_length = float('inf')
    
    def update(self, dt=0.1):
        if self.game_over:
            return
        
        self.time_elapsed += dt
        
        # Update entities
        self.prey.update(self.predator.x, self.predator.y, dt)
        self.predator.update(self.prey.x, self.prey.y, dt)
        
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
        self.prey = Prey(
            x=np.random.uniform(100, self.width-100),
            y=np.random.uniform(100, self.height-100),
            boundary_x=self.width,
            boundary_y=self.height
        )
        
        self.predator = Predator(
            x=np.random.uniform(50, self.width-50),
            y=np.random.uniform(50, self.height-50),
            boundary_x=self.width,
            boundary_y=self.height
        )
        
        self.game_over = False
        self.catch_time = None
        self.time_elapsed = 0
        self.prey_trail = [(self.prey.x, self.prey.y)]
        self.predator_trail = [(self.predator.x, self.predator.y)]


def visualize_game():
    """Visualize the predator-prey game with matplotlib animation"""
    game = PredatorPreyGame()
    
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
            status_text.set_text(f'Time: {game.time_elapsed:.1f}s\n'
                               f'Distance: {distance:.1f}\n'
                               f'Prey Speed: {np.sqrt(game.prey.vx**2 + game.prey.vy**2):.1f}\n'
                               f'Predator Speed: {np.sqrt(game.predator.vx**2 + game.predator.vy**2):.1f}')
        
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