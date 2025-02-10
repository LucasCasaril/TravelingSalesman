import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 5  # Number of cities
eta = 0.01  # Learning rate
time_steps = 1000  # Number of iterations

def distance_matrix(N):
    """Generate a random distance matrix for N cities."""
    cities = np.random.rand(N, 2)  # Random coordinates for cities
    dist_matrix = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(axis=2))
    return dist_matrix, cities

def energy(V, D):
    """Compute the energy function of the network."""
    term1 = np.sum((np.sum(V, axis=1) - 1) ** 2)
    term2 = np.sum((np.sum(V, axis=0) - 1) ** 2)
    term3 = np.sum(D * np.roll(V, shift=-1, axis=1))
    return term1 + term2 + term3

def hopfield_tsp(D, N, time_steps=1000, eta=0.01):
    """Solve TSP using a Hopfield Neural Network with numpy."""
    V = np.random.rand(N, N)  # Random initial state
    
    for _ in range(time_steps):
        # Calculate energy
        loss = energy(V, D)
        
        # Compute gradients (simple finite difference approximation)
        grad = np.zeros_like(V)
        epsilon = 1e-4  # Small epsilon for finite differences
        for i in range(N):
            for j in range(N):
                V_perturbed = V.copy()
                V_perturbed[i, j] += epsilon
                grad[i, j] = (energy(V_perturbed, D) - loss) / epsilon
        
        # Update state (gradient descent)
        V -= eta * grad
        
        # Normalize the state with softmax
        V = np.exp(V) / np.sum(np.exp(V), axis=1, keepdims=True)
    
    return np.argmax(V, axis=1)

def plot_tsp_solution(route, cities):
    """Plot the TSP route with arrows indicating travel direction at the midpoint."""
    ordered_cities = cities[route]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])  # Complete the cycle
    
    plt.figure(figsize=(8, 6))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'o-', markersize=8, label='Route', color='black')
    
    for i in range(len(ordered_cities) - 1):
        x_start, y_start = ordered_cities[i]
        x_end, y_end = ordered_cities[i + 1]
        
        # Compute midpoint
        x_mid, y_mid = (x_start + x_end) / 2, (y_start + y_end) / 2
        
        # Draw arrow at midpoint
        plt.arrow(x_mid, y_mid, (x_end - x_start) * 0.2, (y_end - y_start) * 0.2, 
                  head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    
    for i, (x, y) in enumerate(ordered_cities[:-1]):
        plt.scatter(x, y, color='red', s=100, edgecolors='black')  # Draw city as circle
        plt.text(x, y + 0.02, str(route[i]), fontsize=12, color='black', ha='center')  # Label above city
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('TSP Solution - Hopfield Network')
    plt.legend()
    plt.show()

# Run TSP Solver
dist_matrix, cities = distance_matrix(N)
route = hopfield_tsp(dist_matrix, N, time_steps, eta)
plot_tsp_solution(route, cities)
