import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
    """Compute the energy function using Equation (5)."""
    A, B, C, D_const = 1.0, 1.0, 1.0, 1.0  # Penalty coefficients
    term1 = A / 2 * torch.sum((torch.sum(V, dim=1) - 1) ** 2)  # Row constraint
    term2 = B / 2 * torch.sum((torch.sum(V, dim=0) - 1) ** 2)  # Column constraint
    term3 = C / 2 * (torch.sum(V) - V.shape[0]) ** 2  # Stability constraint
    term4 = D_const / 2 * torch.sum(D * torch.roll(V, shifts=-1, dims=1))  # Distance minimization
    return term1 + term2 + term3 + term4

def hopfield_tsp(D, N, time_steps=1000, eta=0.01):
    """Solve TSP using a Hopfield Neural Network in PyTorch with Equation (6) update rule."""
    V = torch.rand(N, N, requires_grad=True)  # Random initial state
    D = torch.tensor(D, dtype=torch.float32)  # Convert distance matrix to a PyTorch tensor
    
    for _ in range(time_steps):
        dV_dt = - (torch.sum(V, dim=1, keepdim=True) - 1) - (torch.sum(V, dim=0, keepdim=True) - 1) - torch.matmul(D, torch.roll(V, shifts=-1, dims=1))
        V = V + eta * dV_dt  # Gradient update using Equation (6)
        V.data = F.softmax(V, dim=1)  # Normalize using softmax
    
    return torch.argmax(V, dim=1).detach().numpy()

def plot_tsp_solution(route, cities):
    """Plot the TSP route with arrows indicating travel direction at the midpoint."""
    ordered_cities = cities[route]
    ordered_cities = np.vstack([ordered_cities, ordered_cities[0]])  # Complete the cycle
    
    plt.figure(figsize=(8, 6))
    plt.plot(ordered_cities[:, 0], ordered_cities[:, 1], 'k-', alpha=0.6, label='Route')
    
    for i in range(len(ordered_cities) - 1):
        x_start, y_start = ordered_cities[i]
        x_end, y_end = ordered_cities[i + 1]
        
        # Compute midpoint
        x_mid, y_mid = (x_start + x_end) / 2, (y_start + y_end) / 2
        
        # Draw arrow at midpoint
        plt.arrow(x_mid, y_mid, (x_end - x_start) * 0.2, (y_end - y_start) * 0.2, 
                  head_width=0.02, head_length=0.02, fc='blue', ec='blue')
    
    for i, (x, y) in enumerate(cities):
        plt.scatter(x, y, color='red', s=100, edgecolors='black')  # Draw city as circle
        plt.text(x, y + 0.03, str(i), fontsize=12, color='black', ha='center')  # Label above city
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('TSP Solution - Hopfield Network')
    plt.legend()
    plt.grid()
    plt.show()

# Run TSP Solver
dist_matrix, cities = distance_matrix(N)
route = hopfield_tsp(dist_matrix, N, time_steps, eta)
plot_tsp_solution(route, cities)