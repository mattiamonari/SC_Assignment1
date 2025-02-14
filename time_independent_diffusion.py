import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def sor_iteration(c_k, c_init):
    omega = 1.8

    # Interior points
    c_k[1:-1, 1:-1] = 0.25 * omega * (
        c_k[2:, 1:-1] +    # down
        c_k[:-2, 1:-1] +   # up
        c_k[1:-1, 2:] +  # right
        c_k[1:-1, :-2]   # left
    ) + (1 - omega) * c_k[1:-1, 1:-1]
    
    # Apply periodic boundary condition along y-axis (left-right)
    c_k[1:-1, 0] = 0.25 * omega * (
        c_k[2:, 0] +     # down
        c_k[:-2, 0] +    # up
        c_k[1:-1, 1] +   # right
        c_k[1:-1, -1]    # left (periodic)
    ) + (1 - omega) * c_k[1:-1, 0]
    
    c_k[1:-1, -1] = 0.25 * omega * (
        c_k[2:, -1] +    # down
        c_k[:-2, -1] +   # up
        c_k[1:-1, 0] +   # right (periodic)
        c_k[1:-1, -2]    # left
    ) + (1 - omega) * c_k[1:-1, -1]

    return np.maximum(c_k, c_init)

def gauss_seidel_iteration(c_k, c_init):
    # Interior points
    c_k[1:-1, 1:-1] = 0.25 * (
        c_k[2:, 1:-1] +    # down
        c_k[:-2, 1:-1] +   # up
        c_k[1:-1, 2:] +  # right
        c_k[1:-1, :-2]   # left
    )
    
    # Apply periodic boundary condition along y-axis (left-right)
    c_k[1:-1, 0] = 0.25 * (
        c_k[2:, 0] +     # down
        c_k[:-2, 0] +    # up
        c_k[1:-1, 1] +   # right
        c_k[1:-1, -1]    # left (periodic)
    )
    
    c_k[1:-1, -1] = 0.25 * (
        c_k[2:, -1] +    # down
        c_k[:-2, -1] +   # up
        c_k[1:-1, 0] +   # right (periodic)
        c_k[1:-1, -2]    # left
    )

    return c_k

def jacobi_iteration(c_k, c_init):
    c_k_new = np.zeros_like(c_k)
    
    # Interior points
    c_k_new[1:-1, 1:-1] = 0.25 * (
        c_k[2:, 1:-1] +    # down
        c_k[:-2, 1:-1] +   # up
        c_k[1:-1, 2:] +  # right
        c_k[1:-1, :-2]   # left
    )
    
    # Apply periodic boundary condition along y-axis (left-right)
    c_k_new[1:-1, 0] = 0.25 * (
        c_k[2:, 0] +     # down
        c_k[:-2, 0] +    # up
        c_k[1:-1, 1] +   # right
        c_k[1:-1, -1]    # left (periodic)
    )
    
    c_k_new[1:-1, -1] = 0.25 * (
        c_k[2:, -1] +    # down
        c_k[:-2, -1] +   # up
        c_k[1:-1, 0] +   # right (periodic)
        c_k[1:-1, -2]    # left
    )
    
    return np.maximum(c_k_new, c_init)

def solve_diffusion(c_k, method, tol=1e-6, max_iter=100000):
    c_init = np.copy(c_k)
    for iter in range(max_iter):
        c_k_new = method(c_k.copy(), c_init)
        error = np.linalg.norm(c_k_new - c_k)
        
        if error < tol:
            print(f"Converged after {iter} iterations")
            return c_k_new
            
        c_k = c_k_new
    
    print("Warning: Maximum iterations reached without convergence")
    return c_k

if __name__ == "__main__":
    x_len = 100
    y_len = 100
    
    # Initialize solution array
    c_k = np.zeros((x_len, y_len))
    
    # Set initial boundary conditions
    c_k[0, :] = 1.0  # top boundary
    c_k[-1, :] = 0.0 # bottom boundary
    
    methods = [jacobi_iteration, gauss_seidel_iteration, sor_iteration]

    for method in methods:
        # Solve the diffusion equation
        solution = solve_diffusion(c_k, method)
        
        # Plot the solution
        fig, ax = plt.subplots()
        # extent shows [left, right, bottom, top] true size of the grid
        im = ax.imshow(solution, cmap='Reds',extent=[0, 1, 0, 1])
        plt.colorbar(im)
        plt.title(f'Diffusion Solution with {method.__name__}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()