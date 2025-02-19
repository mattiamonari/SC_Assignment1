import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def core_jacobi(c, c_new, c_init, objects_masks):
    """Core Jacobi iteration calculations"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            c_new[i,j] = 0.25 * (c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1])
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        c_new[i,0] = 0.25 * (c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1])
        c_new[i,-1] = 0.25 * (c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2])
    
    # Apply initial conditions as minimum values CORRECT????
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c_new[i,j]:
                c_new[i,j] = c_init[i,j]

    for object_mask in objects_masks:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if not object_mask[i, j]:
                    c_new[i, j] = 0  # Sink has zero concentration

@jit(nopython=True)
def core_gauss_seidel(c, c_init, objects_masks=[]):
    """Core Gauss-Seidel iteration calculations"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            c[i,j] = 0.25 * (c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1])
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        c[i,0] = 0.25 * (c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1])
        c[i,-1] = 0.25 * (c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2])
    
    # Apply initial conditions as minimum values
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c[i,j]:
                c[i,j] = c_init[i,j]

    for object_mask in objects_masks:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if not object_mask[i, j]:
                     c[i, j] = 0  # Sink has zero concentration

@jit(nopython=True)
def core_sor(c, c_new, c_init, omega, objects_masks):
    """Core SOR iteration calculations"""
    # Interior points
    for i in range(1, c.shape[0]-1):
        for j in range(1, c.shape[1]-1):
            c_new[i,j] = (1 - omega) * c[i,j] + 0.25 * omega * (
                c[i+1,j] + c[i-1,j] + c[i,j+1] + c[i,j-1]
            )
    
    # Periodic boundary condition along y-axis
    for i in range(1, c.shape[0]-1):
        c_new[i,0] = (1 - omega) * c[i,0] + 0.25 * omega * (
            c[i+1,0] + c[i-1,0] + c[i,1] + c[i,-1]
        )
        c_new[i,-1] = (1 - omega) * c[i,-1] + 0.25 * omega * (
            c[i+1,-1] + c[i-1,-1] + c[i,0] + c[i,-2]
        )
    
    # Apply initial conditions as minimum values
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if c_init[i,j] > c_new[i,j]:
                c_new[i,j] = c_init[i,j]
    
    for object_mask in objects_masks:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if not object_mask[i, j]:
                     c_new[i, j] = 0  # Sink has zero concentration

def solve_diffusion(grid_size=(100, 100), method='jacobi', omega=1.0, tol=1e-6, max_iter=100000):
    """Solve diffusion equation using specified method"""
    # Initialize arrays
    c = np.zeros(grid_size)
    c_new = np.zeros(grid_size)
    
    # Set initial boundary conditions
    c[0, :] = 1.0  # top boundary
    c[-1, :] = 0.0 # bottom boundary
    c_init = c.copy()
    
    errors = []
    
    # Main iteration loop
    for iteration in tqdm(range(max_iter)):
        if method == 'jacobi':
            core_jacobi(c, c_new, c_init)
            error = np.linalg.norm(c_new - c)
            c, c_new = c_new.copy(), c
        elif method == 'gauss-seidel':
            c_old = c.copy()
            core_gauss_seidel(c, c_init)
            error = np.linalg.norm(c - c_old)
        else:  # SOR
            core_sor(c, c_new, c_init, omega)
            error = np.linalg.norm(c_new - c)
            c, c_new = c_new.copy(), c
        
        errors.append(error)
        
        if error < tol:
            return c, np.array(errors), iteration + 1
    
    print("Warning: Maximum iterations reached without convergence")
    return c, np.array(errors), max_iter

def solve_diffusion_with_comparison(grid_size=(100, 100), tol=1e-6, max_iter=100000):
    # Define methods
    methods = [
        {'name': 'Jacobi', 'method': 'jacobi', 'omega': 1.0, 'color': 'blue', 'linestyle': '-'},
        {'name': 'Gauss-Seidel', 'method': 'gauss-seidel', 'omega': 1.0, 'color': 'green', 'linestyle': '-'},
    ]
    
    # Add SOR with different omega values
    for omega in np.linspace(1.7, 2.0, 4):
        methods.append({
            'name': f'SOR (ω={omega:.2f})',
            'method': 'sor',
            'omega': omega,
            'color': plt.cm.autumn(omega/2),
            'linestyle': '--'
        })
    
    # Run all methods
    all_results = []
    for config in methods:
        print(f"\nRunning {config['name']}...")
        solution, errors, iters = solve_diffusion(
            grid_size=grid_size,
            method=config['method'],
            omega=config['omega'],
            tol=tol,
            max_iter=max_iter
        )
        all_results.append({
            **config,
            'solution': solution,
            'errors': errors,
            'iterations': iters
        })
        print(f"Converged after {iters + 1} iterations")
    
    # Create comparison plot
    fig = plt.figure(figsize=(12, 8))
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
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
    
    return all_results, fig

def find_optimal_omega(tol=1e-6, max_iter=100000):
    methods = []
    omega_values = np.linspace(1.7, 2.0, 10)
    N_values = np.logspace(1,1.5,10, dtype=int)

    # Add SOR with different omega values
    for omega in omega_values:
        methods.append({
            'name': f'SOR (ω={omega:.2f})',
            'method': 'sor',
            'omega': omega,
            'color': plt.cm.autumn(omega/2),
            'linestyle': '--'
        })
    
    # Run all methods
    all_results = []
    for N in N_values:
        print(f"\nRunning all config with N={N}...")
        for config in methods:
            solution, errors, iters = solve_diffusion(
                grid_size=(N, N),
                method=config['method'],
                omega=config['omega'],
                tol=tol,
                max_iter=max_iter
            )
            all_results.append({
                **config,
                'solution': solution,
                'errors': errors,
                'iterations': iters,
                'N': N
            })


    # Create comparison plot
    fig = plt.figure(figsize=(12, 8))
    plt.grid(True, which="both", ls="-", alpha=0.2)

    optimal_omega = []
    for N in N_values:
        print("Unordered",[el['iterations'] for el in all_results if el['N'] == N])
        n_results = sorted([el for el in all_results if el['N'] == N], key=lambda d: d['iterations'])
        print("Ordered",[el['iterations'] for el in n_results])
        optimal_omega.append(n_results[0]['omega'])
    plt.semilogx(
        N_values,
        optimal_omega, 
        # color=result['color'],
        # linestyle=result['linestyle'],
        # label=f"{result['name']} ({result['iterations']} iter)"
    )
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Error (L2 norm)', fontsize=12)
    plt.title('Convergence Comparison of Different Methods', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    return all_results, fig

def create_object_mask(grid_size, object_size, start=None):
    """Creates a mask for the sink (zero concentration region)."""
    mask = np.ones(grid_size, dtype=np.bool_)
    if not start:
        start = (grid_size[0] // 2 - object_size // 2, grid_size[1] // 2 - object_size // 2)
    mask[start[0]:start[0]+object_size, start[1]:start[1]+object_size] = False
    return mask

if __name__ == "__main__":
    # Run comparison
    # results, fig = solve_diffusion_with_comparison()
    # plt.show()
    
    # # Plot final solutions
    # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # axes = axes.ravel()
    
    # methods_to_show = ['Jacobi', 'Gauss-Seidel', 'SOR (ω=1.70)', 'SOR (ω=2.00)']
    
    # for ax, method_name in zip(axes, methods_to_show):
    #     result = next(r for r in results if r['name'] == method_name)
    #     im = ax.imshow(result['solution'], cmap='Reds', extent=[0, 1, 0, 1])
    #     ax.set_title(f'{method_name}\n({result["iterations"]} iterations)')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     plt.colorbar(im, ax=ax)
    
    # plt.tight_layout()
    # plt.show()

    ####### J
    results, fig = find_optimal_omega()
    plt.show()

    ###### K
    #Centered
    mask1 = create_object_mask((100, 100), (20, 20), (0,0))
    #Not centerdx
    mask2 = create_object_mask((100, 100), (20, 20))

