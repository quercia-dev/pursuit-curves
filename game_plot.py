import matplotlib.pyplot as plt


# Visualize evolution progress
def visualize_evolution(fitness_history):
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Evolution of Prey Fitness")
    plt.show()


# Visualize a simulation with the best individual
def visualize_best_simulation(game):
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(0, game.width)
    ax.set_ylim(0, game.height)
    ax.set_aspect('equal')
    ax.set_title('Predator-Prey Pursuit Game with Evolved Prey', fontsize=16)
    
    # Plot prey and predator trails
    prey_x, prey_y = zip(*game.prey_trail)
    pred_x, pred_y = zip(*game.predator_trail)
    ax.plot(prey_x, prey_y, 'g-', alpha=0.5, linewidth=1, label='Prey Trail')
    ax.plot(pred_x, pred_y, 'r-', alpha=0.5, linewidth=1, label='Predator Trail')
    
    # Plot final positions
    ax.plot(game.prey.x, game.prey.y, 'go', markersize=8, label='Prey')
    ax.plot(game.predator.x, game.predator.y, 'ro', markersize=10, label='Predator')
    
    # Fear radius circle
    fear_circle = plt.Circle((game.prey.x, game.prey.y), game.prey.react_radius, 
                           fill=False, color='green', alpha=0.3, linestyle='--', label='Fear Radius')
    ax.add_patch(fear_circle)
    
    # Catch radius circle
    catch_circle = plt.Circle((game.predator.x, game.predator.y), game.predator.catch_radius, 
                            fill=False, color='red', alpha=0.3, linestyle='--', label='Catch Radius')
    ax.add_patch(catch_circle)
    
    ax.legend(loc='upper right')
    plt.show()


def visualize_trait_distributions(best_prey_history):
    """
    Visualizes the distribution of prey traits over generations using individual plots.
    """
    # Extract trait values from best prey history
    react_radii = [prey.react_radius for prey in best_prey_history]
    evasion_angles = [prey.evasion_angle for prey in best_prey_history]
    evasion_times = [prey.evasion_time for prey in best_prey_history]

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Plot React Radius
    axes[0].plot(react_radii)
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('React Radius')
    axes[0].set_title('React Radius Over Generations')

    # Plot Evasion Angle
    axes[1].plot(evasion_angles)
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Evasion Angle')
    axes[1].set_title('Evasion Angle Over Generations')

    # Plot Evasion Time
    axes[2].plot(evasion_times)
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Evasion Time')
    axes[2].set_title('Evasion Time Over Generations')

    plt.tight_layout()
    plt.show()