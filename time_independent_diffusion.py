import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit


__all__ = ['solve_diffusion_with_comparison', 'find_optimal_omega', 
           'create_rectangle_mask', 'create_circle_mask']

@jit(nopython=True)
def core_jacobi(c, c_new, c_init, objects_masks=None):
    """Core Jacobi iteration calculations with mask handling"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            # Skip calculation if point is in any mask
            skip = False
            if objects_masks:
                for mask in objects_masks:
                    if not mask[i,j]:
                        c_new[i,j] = 0
                        skip = True
                        break
            if skip:
                continue
            
            c_new[i,j] = 0.25 * (c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1])
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,0]:
                    c_new[i,0] = 0
                    skip = True
                    break
        if not skip:
            c_new[i,0] = 0.25 * (c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1])
            
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,-1]:
                    c_new[i,-1] = 0
                    skip = True
                    break
        if not skip:
            c_new[i,-1] = 0.25 * (c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2])
    
    # Apply initial conditions as minimum values
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c_new[i,j]:
                c_new[i,j] = c_init[i,j]

@jit(nopython=True)
def core_gauss_seidel(c, c_init, objects_masks=None):
    """Core Gauss-Seidel iteration calculations with mask handling"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            # Skip calculation if point is in any mask
            skip = False
            if objects_masks:
                for mask in objects_masks:
                    if not mask[i,j]:
                        c[i,j] = 0
                        skip = True
                        break
            if skip:
                continue
                
            c[i,j] = 0.25 * (c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1])
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,0]:
                    c[i,0] = 0
                    skip = True
                    break
        if not skip:
            c[i,0] = 0.25 * (c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1])
            
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,-1]:
                    c[i,-1] = 0
                    skip = True
                    break
        if not skip:
            c[i,-1] = 0.25 * (c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2])
    
    # Apply initial conditions as minimum values
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c[i,j]:
                c[i,j] = c_init[i,j]

@jit(nopython=True)
def core_sor(c, c_init, omega, objects_masks=None):
    """Core SOR iteration calculations with mask handling"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            # Skip calculation if point is in any mask
            skip = False
            if objects_masks:
                for mask in objects_masks:
                    if not mask[i,j]:
                        c[i,j] = 0
                        skip = True
                        break
            if skip:
                continue
                
            c[i,j] = (1 - omega) * c[i,j] + 0.25 * omega * (
                c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1]
            )
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,0]:
                    c[i,0] = 0
                    skip = True
                    break
        if not skip:
            c[i,0] = (1 - omega) * c[i,0] + 0.25 * omega * (
                c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1]
            )
            
        skip = False
        if objects_masks:
            for mask in objects_masks:
                if not mask[i,-1]:
                    c[i,-1] = 0
                    skip = True
                    break
        if not skip:
            c[i,-1] = (1 - omega) * c[i,-1] + 0.25 * omega * (
                c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2]
            )
    
    # Apply initial conditions as minimum values
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c[i,j]:
                c[i,j] = c_init[i,j]

def solve_diffusion(grid_size=(100, 100), method='jacobi', omega=1.0, tol=1e-6, max_iter=100000, objects_masks=None):
    """Solve diffusion equation using specified method"""
    # Initialize arrays
    objects_masks = objects_masks or [np.ones(grid_size, dtype=np.bool_)]
    c = np.zeros(grid_size)
    c_new = np.zeros(grid_size)
    
    # Set initial boundary conditions
    c[0, :] = 1.0  # top boundary

    c[-1, :] = 0.0 # bottom boundary
    c_init = c.copy()
    
    errors = []
    
    # Main iteration loop
    for iteration in range(max_iter):
        if method == 'jacobi':
            core_jacobi(c, c_new, c_init, objects_masks)
            error = np.linalg.norm(c_new - c)
            c, c_new = c_new.copy(), c
        elif method == 'gauss-seidel':
            c_old = c.copy()
            core_gauss_seidel(c, c_init, objects_masks)
            error = np.linalg.norm(c - c_old)
        else:  # SOR
            c_old = c.copy()
            core_sor(c, c_init, omega, objects_masks)
            error = np.linalg.norm(c - c_old)
        
        # Apply object masks if provided
        # for object_mask in objects_masks:
        #     for i in range(c.shape[0]):
        #         for j in range(c.shape[1]):
        #             if not object_mask[i, j]:
        #                 c[i, j] = 0  # Sink has zero concentration

        errors.append(error)
        
        if error < tol:
            return c, np.array(errors), iteration + 1
    
    print("Warning: Maximum iterations reached without convergence")
    return c, np.array(errors), max_iter

def solve_diffusion_with_comparison(grid_size=(50, 50), tol=1e-6, max_iter=100000, objects_masks=[]):
    # Define methods
    methods = [
        {'name': 'Jacobi', 'method': 'jacobi', 'omega': 1.0, 'color': 'blue', 'linestyle': '-'},
        {'name': 'Gauss-Seidel', 'method': 'gauss-seidel', 'omega': 1.0, 'color': 'green', 'linestyle': '-'},
    ]

    objects_added = "_objects" if objects_masks else ""
    objects_title = " with Objects" if objects_masks else ""
    
    # Add SOR with different omega values
    for i, omega in enumerate(np.linspace(1.7, 1.98, 4)):
        methods.append({
            'name': f'SOR (ω={omega:.2f})',
            'method': 'sor',
            'omega': omega,
            'color': plt.cm.Reds((i+1) / 5),
            'linestyle': '--'
        })
    
    # Run all methods
    all_results = []
    for config in tqdm(methods, desc="Running comparison..."):
        print(f"\nRunning {config['name']}...")
        solution, errors, iters = solve_diffusion(
            grid_size=grid_size,
            method=config['method'],
            omega=config['omega'],
            tol=tol,
            max_iter=max_iter,
            objects_masks=objects_masks
        )

        # fig, ax = plt.subplots(1, 1)
        # im = ax.imshow(create_rectangle_mask(grid_size, (20, 1), (50, 50)), cmap='Reds', extent=[0, 1, 0, 1])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.colorbar(im, ax=ax)
        # plt.show()

        all_results.append({
            **config,
            'solution': solution,
            'errors': errors,
            'iterations': iters
        })
        print(f"Converged after {iters + 1} iterations")
    
    # Create comparison plot
    fig = plt.figure(figsize=(10, 6))
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    for result in all_results:
        plt.semilogy(
            result['errors'],
            color=result['color'],
            linestyle=result['linestyle'],
            label=f"{result['name']} ({result['iterations']} iter)"
        )
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Error (L2 norm)', fontsize=14)
    plt.title(f'Convergence Comparison of Different Methods{objects_title}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'./Images/convergence_comparison{objects_added}.pdf')
    
    return all_results, fig

def find_optimal_omega(tol=1e-6, max_iter=100000, N_values=None, objects_masks=None):
    # Use a wider range of N values
    N_values = N_values or np.logspace(1, 2, 20, dtype=int)  # N from 10 to 100
    omega_values = np.linspace(1.5, 1.99, 50)  # More omega values for better resolution
    objects_added = "_objects" if objects_masks else ""
    # Store optimal omega and its convergence iterations for each N
    optimal_results = []
    
    # For each grid size N
    for N in tqdm(N_values, desc="Finding optimal omega..."):
        print(f"\nTesting grid size N={N}")
        min_iterations = float('inf')
        best_omega = None
        
        # Test each omega value
        for omega in omega_values:
            _, errors, iterations = solve_diffusion(
                grid_size=(N, N),
                method='sor',
                omega=omega,
                tol=tol,
                max_iter=max_iter,
                objects_masks=objects_masks
            )
            
            # If this omega gives faster convergence, store it
            if iterations < min_iterations:
                min_iterations = iterations
                best_omega = omega
        
        optimal_results.append({
            'N': N,
            'omega': best_omega,
            'iterations': min_iterations
        })
        print(f"N={N}: Best omega={best_omega:.4f} with {min_iterations} iterations")
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot optimal omega vs N
    N_plot = [r['N'] for r in optimal_results]
    omega_plot = [r['omega'] for r in optimal_results]
    iterations_plot = [r['iterations'] for r in optimal_results]
    
    # Plot 1: Optimal omega vs N
    ax.plot(N_plot, omega_plot, 'o-')
    # Calculate theoretical optimal omega for comparison
    theoretical_omega = [2 / (1 + np.sin(np.pi/(N + 1))) for N in N_plot]
    ax.plot(N_plot, theoretical_omega, 'r--', label='Theoretical')
    
    ax.set_xlabel('Grid Size (N)', fontsize=14)
    ax.set_ylabel('Optimal ω', fontsize=14)
    ax.set_title('Optimal ω vs Grid Size', fontsize=14)
    ax.legend()
    ax.grid(True)
    
    # # Plot 2: Convergence iterations vs N
    # ax2.loglog(N_plot, iterations_plot, 'o-')
    # ax2.loglog(N_plot, (np.array(N_plot) + 1) / (2*np.pi) * 13.8, 'r--')
    # ax2.set_xlabel('Grid Size (N)', fontsize=12)
    # ax2.set_ylabel('Number of Iterations', fontsize=12)
    # ax2.set_title('Convergence Speed vs Grid Size', fontsize=14)
    # ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'./images/optimal_omega_vs_N{objects_added}.pdf')
    plt.show()
    
    # Print numerical results
    print("\nFinal Results:")
    print("N\tOptimal ω\tIterations")
    print("-" * 40)
    for result in optimal_results:
        print(f"{result['N']}\t{result['omega']:.4f}\t{result['iterations']}")
    
    return optimal_results, fig

def create_rectangle_mask(grid_size, object_size, start=None):
    """Creates a mask for the sink (zero concentration region)."""
    mask = np.ones(grid_size, dtype=np.bool_)
    if not start:
        start = (grid_size[0] // 2 - object_size[0] // 2, grid_size[1] // 2 - object_size[1] // 2)
    mask[start[0]:start[0]+object_size[0], start[1]:start[1]+object_size[1]] = False
    return mask

def create_circle_mask(grid_size, radius, center=None):
    """Creates a mask for the sink (zero concentration region)."""
    mask = np.ones(grid_size, dtype=np.bool_)
    center = center or (grid_size[0] // 2, grid_size[1] // 2)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                mask[x, y] = False
    return mask

if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})

    # Run comparison
    grid_size = (50, 50)
    results, fig = solve_diffusion_with_comparison(grid_size=grid_size)
    plt.show()

    results, fig = solve_diffusion_with_comparison(
        grid_size=grid_size,
        objects_masks=[
                create_rectangle_mask(grid_size, (10, 10), (10, 10)),
                # create_rectangle_mask(grid_size, (1, 20), (40, 40)),
                # create_rectangle_mask(grid_size, (20, 1), (20, 60)),
                create_circle_mask(grid_size, 5, (15, 35)),
                # create_rectangle_mask(grid_size, (20, 20), (65, 15))
                ]
    )
    plt.show()
    
    print(np.all(np.isclose(results[0]['solution'], results[1]['solution'], atol=1e-4) == True))
    print(np.all(np.isclose(results[0]['solution'], results[-1]['solution'], atol=1e-4) == True))
    # # Plot final solutions
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    methods_to_show = ['Jacobi', 'Gauss-Seidel', 'SOR (ω=1.80)', 'SOR (ω=1.98)']
    
    for i, ax in enumerate(axes):
        result = results[i]
        im = ax.imshow(result['solution'], cmap='Reds', extent=[0, 1, 0, 1])
        ax.set_title(f"{result['method']}, ω = {result['omega']:.2f} ({result['iterations']} iterations)")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

    # Plot a single solution
    plt.figure(figsize=(10, 6))
    plt.imshow(results[-1]['solution'], cmap='Reds', extent=[0, 1, 0, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.title('Equilibrium Solution with Objects')
    plt.tight_layout()
    plt.savefig('./images/final_solution_objects.pdf')
    plt.show()

    ###### J
    resuts, fig = find_optimal_omega()
    plt.tight_layout()
    plt.show()


