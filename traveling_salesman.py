"""
Author: Lucas Casaril
Contact: eng@lucascasaril.me
Updated: 11/Feb/2025
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters for the Problem
N = 8  # Number of cities
eta = 0.01  # Learning rate
time_steps = 100 # Number of iterations

def distance_matrix(N):
    # Generating a random cities and calculating the distance matrix
    cities = np.random.rand(N, 2)
    dist_matrix = np.sqrt(((cities[:, None] - cities[None, :]) ** 2).sum(axis=2))
    return dist_matrix, cities

def energy(V, D):
    term1 = np.sum((np.sum(V, axis=1) - 1) ** 2)
    term2 = np.sum((np.sum(V, axis=0) - 1) ** 2)
    term3 = np.sum(D * np.roll(V, shift=-1, axis=1)) # TSP cost - ensuring shorter tours
    return term1 + term2 + term3

def hopfield_tsp(D, N, time_steps, eta, system_energy):
    V = np.random.rand(N, N)  # Random initial state
    
    for _ in range(time_steps):
        # Calculate energy for the step
        loss = energy(V, D)
        system_energy.append(loss.item())
        
        # Gradients
        grad = np.zeros_like(V)
        epsilon = 1e-4  # Finite differences

        for i in range(N):
            for j in range(N):

                V_perturbed = V.copy()
                V_perturbed[i, j] += epsilon
                grad[i, j] = (energy(V_perturbed, D) - loss) / epsilon
        
        # Gradient descent
        V -= eta * grad
        
        # Normalizing
        V = np.exp(V) / np.sum(np.exp(V), axis=1, keepdims=True)
    
    return np.argmax(V, axis=1)

def plot_tsp_solution(route, cities):
    ordered_cities = cities[route]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])
    
    plt.figure()
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

# Main Code
dist_matrix, cities = distance_matrix(N)
system_energy = []
route = hopfield_tsp(dist_matrix, N, time_steps, eta, system_energy)
print(system_energy)
plot_tsp_solution(route, cities)
