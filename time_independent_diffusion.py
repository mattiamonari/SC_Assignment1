import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import animation


def sor_iteration(c_k, c_init, omega=1):
    
    # Interior points - using in-place updates
    for i in range(1, c_k.shape[0]-1):
        for j in range(1, c_k.shape[1]-1):
            c_k[i,j] = (1 - omega) * c_k[i,j] + 0.25 * omega * (
                c_k[i+1,j] +    # down
                c_k[i-1,j] +    # up
                c_k[i,j+1] +    # right
                c_k[i,j-1]      # left
            )
    
    # Apply periodic boundary condition along y-axis (left-right)
    for i in range(1, c_k.shape[0]-1):
        # Left boundary
        c_k[i,0] = (1 - omega) * c_k[i,0] + 0.25 * omega * (
            c_k[i+1,0] +     # down
            c_k[i-1,0] +     # up
            c_k[i,1] +       # right
            c_k[i,-1]        # left (periodic)
        )
        
        # Right boundary
        c_k[i,-1] = (1 - omega) * c_k[i,-1] + 0.25 * omega * (
            c_k[i+1,-1] +    # down
            c_k[i-1,-1] +    # up
            c_k[i,0] +       # right (periodic)
            c_k[i,-2]        # left
        )
    
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

def solve_diffusion(c_k, method, omega, tol=1e-6, max_iter=100000):
    c_init = np.copy(c_k)
    errors = []
    if method.__name__ == "sor_iteration":
        kwargs = {"omega": omega}
    else:
        kwargs = {}

    for iter in tqdm(range(max_iter)):
        c_k_new = method(c_k.copy(), c_init, **kwargs)
        error = np.linalg.norm(c_k_new - c_k)
        errors.append(error)
        if error < tol:
            print(f"Converged after {iter} iterations")
            return c_k_new, errors
            
        c_k = c_k_new
    
    print("Warning: Maximum iterations reached without convergence")
    return c_k, errors

def solve_diffusion_with_comparison(grid_size=(100, 100), tol=1e-6, max_iter=100000):
    # Initialize solution array
    c_k = np.zeros(grid_size)
    
    # Set initial boundary conditions
    c_k[0, :] = 1.0  # top boundary
    c_k[-1, :] = 0.0 # bottom boundary
    
    # Define methods and their parameters
    method_configs = [
        {'method': jacobi_iteration, 'name': 'Jacobi', 'omega': 1, 'color': 'blue', 'linestyle': '-'},
        {'method': gauss_seidel_iteration, 'name': 'Gauss-Seidel', 'omega': 1, 'color': 'green', 'linestyle': '-'},
    ]
    
    # Add SOR with different omega values
    sor_omegas = np.linspace(0.5, 1.9, 8)  # Test 8 different omega values
    for omega in sor_omegas:
        method_configs.append({
            'method': sor_iteration,
            'name': f'SOR (ω={omega:.2f})',
            'omega': omega,
            'color': plt.cm.autumn(omega/2),  # Color gradient based on omega
            'linestyle': '--' if omega > 1 else ':'
        })
    
    # Store results
    all_results = []
    
    # Run all methods
    for config in tqdm(method_configs, desc="Running methods"):
        solution, errors = solve_diffusion(
            c_k.copy(), 
            config['method'], 
            config['omega'], 
            tol=tol, 
            max_iter=max_iter
        )
        all_results.append({
            **config,
            'solution': solution,
            'errors': errors,
            'iterations': len(errors)
        })
    
    # Create scientific plot
    plt.figure(figsize=(12, 8))
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Plot error convergence for each method
    for result in all_results:
        plt.semilogy(
            result['errors'], 
            color=result['color'],
            linestyle=result['linestyle'],
            label=f"{result['name']} ({result['iterations']} iter)"
        )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error (L2 norm)', fontsize=12)
    plt.title('Convergence Comparison of Different Methods', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    # Add scientific notation to y-axis
    plt.gca().yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    plt.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    return all_results, plt.gcf()

if __name__ == "__main__":
    # Run comparison
    results, fig = solve_diffusion_with_comparison()
    plt.show()
    
    # Plot final solutions for selected methods
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    # Select interesting results to display
    methods_to_show = ['Jacobi', 'Gauss-Seidel', 'SOR (ω=1.25)', 'SOR (ω=1.75)']
    
    for ax, method_name in zip(axes, methods_to_show):
        result = next(r for r in results if r['name'] == method_name)
        im = ax.imshow(result['solution'], cmap='Reds', extent=[0, 1, 0, 1])
        ax.set_title(f'{method_name}\n({result["iterations"]} iterations)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()