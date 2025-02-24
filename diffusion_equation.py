import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from scipy.special import erfc

__all__ = ['diffusion_equation', 'plot_diffusion', 'compare_solution', 'plot_multiple_concentrations',
           'diffusion_equation_until_equilibrium', 'animate_diffusion']

def diffusion_equation(N=50, D=1.0, T=1.0, dt=0.0001, save_interval=100, output_file='diffusion_data.npy'):
    """
    Simulates the 2D time-dependent diffusion equation using the explicit finite difference scheme.

    Parameters:
    - N: Number of grid intervals (grid size = N+1)
    - D: Diffusion coefficient
    - T: Total simulation time
    - dt: Time step
    - save_interval: Interval for saving results
    - output_file: File to save the solution data

    Returns:
    - times: Array of time points where the solution was saved
    - solutions: 3D array of concentration profiles over time
    """
    dx = 1.0 / N

    # Stability condition check
    if 4 * D * dt / dx**2 > 1:
        raise ValueError("Stability condition violated. Reduce dt or increase N.")

    # Initialize concentration field: c(y=1)=1 (top boundary), c(y=0)=0 (bottom boundary)
    c = np.zeros((N+1, N+1))
    c[-1, :] = 1

    times = [0]
    solutions = [c.copy()]
    coef = D * dt / dx**2
    t = 0
    iteration = 0

    while t < T:
        c_new = c.copy()

        # Interior points update (five-point stencil)
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + coef * (
            c[2:, 1:-1] + c[:-2, 1:-1] + c[1:-1, 2:] + c[1:-1, :-2] - 4 * c[1:-1, 1:-1]
        )

        # Periodic boundary in x-direction
        c_new[:, 0] = c_new[:, -2]    # Left boundary from right interior
        c_new[:, -1] = c_new[:, 1]    # Right boundary from left interior

        # Fixed boundary conditions in y-direction
        c_new[0, :] = 0               # Bottom boundary (y=0)
        c_new[-1, :] = 1              # Top boundary (y=1)

        c = c_new
        t += dt
        iteration += 1

        if iteration % save_interval == 0:
            times.append(t)
            solutions.append(c.copy())
            # print(f"Saved at iteration {iteration}, time {t:.5f}")

    # Save the entire dataset
    # np.save(output_file, {'times': np.array(times), 'solutions': np.array(solutions)})
    # print(f"Simulation complete. Data saved to {output_file}.")

    return np.array(times), np.array(solutions)

def plot_diffusion(times, solutions, time_index=-1, cmap='viridis', equilibrium_label=False):
    """
    Plot the 2D concentration field at a specific time index.

    Parameters:
    - times: Array of time points.
    - solutions: 3D array of concentration profiles.
    - time_index: Index of the time snapshot to plot (default: -1 for the last time point).
    - cmap: Colormap for visualization (default: 'viridis').
    - equilibrium_label: If True, updates the title for equilibrium display.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(solutions[time_index], cmap=cmap, origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label="Concentration")

    # Title adjusted for equilibrium visualization
    if equilibrium_label:
        plt.title(f"2D Concentration Field at Equilibrium (t={times[time_index]:.3f})")
    else:
        plt.title(f"Concentration Field at t={times[time_index]:.3f}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def analytical_solution(x, t, D=1.0, terms=50):
    """
    Compute the analytical solution for the diffusion equation.

    Parameters:
    - x: Position array (0 to 1).
    - t: Time point (or array of time points).
    - D: Diffusion coefficient (default 1).
    - terms: Number of terms in the infinite series.

    Returns:
    - c: Analytical concentration profile at time t.
    """
    c = np.zeros_like(x)
    for i in range(terms):
        c += erfc((1 - x + 2 * i) / (2 * np.sqrt(D * t))) - erfc((1 + x + 2 * i) / (2 * np.sqrt(D * t)))
    return c

def compare_solution(times, solutions, D=1.0, selected_times=[0.001, 0.01, 0.1, 1.0]):
    """
    Comparison for vertical concentration profiles.
    """
    y_points = np.linspace(0, 1, solutions[-1].shape[0])
    time_indices = [np.argmin(np.abs(times - t)) for t in selected_times]

    plt.figure(figsize=(8, 6))
    for idx, t in zip(time_indices, selected_times):
        numerical = solutions[idx][:, solutions[idx].shape[1] // 2]  # Vertical slice at mid x
        analytical = analytical_solution(y_points, t, D, terms=200)  # Increased terms

        plt.plot(y_points, numerical, '-', label=f'Numerical t={t:.3f}')
        plt.plot(y_points, analytical, '--', label=f'Analytical t={t:.3f}')

    plt.xlabel('y')
    plt.ylabel('Concentration c(y)')
    plt.title('Comparison: Numerical vs Analytical')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_concentrations(times, solutions, selected_times=[0, 0.001, 0.01, 0.1, 1.0]):
    """
    Plot 2D concentration fields at specified time points.

    Parameters:
    - times: Array of time points from the simulation.
    - solutions: Array of 2D concentration profiles corresponding to 'times'.
    - selected_times: Times at which to show the 2D concentration field.
    """
    # Find closest indices in numerical data for selected times
    time_indices = [np.argmin(np.abs(times - t)) for t in selected_times]

    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
    c_min, c_max = 0, 1  # Fixed color scale for consistent comparison
    axs = axs.flatten()
    for ax, idx, t in zip(axs, time_indices, selected_times):
        print(ax)
        im = ax.imshow(solutions[idx], origin='lower', cmap='viridis',
                        extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
        ax.set_title(f't = {t:.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Concentration')

    plt.suptitle('2D Concentration Fields at Different Times')
    plt.show()

def diffusion_equation_until_equilibrium(N=50, D=1.0, dt=0.0001, save_interval=100,
                                        output_file='diffusion_data_equilibrium.npy',
                                        tolerance=1e-6, max_iterations=1000000):
    """
    Simulates the 2D time-dependent diffusion equation until equilibrium is reached,
    based on a specified convergence tolerance.

    Parameters:
    - N: Number of grid intervals (grid size = N+1)
    - D: Diffusion coefficient
    - dt: Time step
    - save_interval: Interval for saving results
    - output_file: File to save the solution data
    - tolerance: Threshold for detecting equilibrium (max difference between steps)
    - max_iterations: Safety limit to avoid infinite loops

    Returns:
    - times: Array of time points where the solution was saved
    - solutions: 3D array of concentration profiles over time
    - final_time: The time at which equilibrium was reached
    """
    dx = 1.0 / N

    # Stability condition check
    if 4 * D * dt / dx**2 > 1:
        raise ValueError("Stability condition violated. Reduce dt or increase N.")

    # Initialize concentration field: c(y=1)=1 (top boundary), c(y=0)=0 (bottom boundary)
    c = np.zeros((N+1, N+1))
    c[-1, :] = 1

    times = [0]
    solutions = [c.copy()]
    coef = D * dt / dx**2
    t = 0
    iteration = 0

    print(f"Starting simulation with tolerance={tolerance} and max_iterations={max_iterations}...")

    # Iterate until equilibrium (based on tolerance)
    while iteration < max_iterations:
        c_new = c.copy()

        # Interior points update (five-point stencil)
        c_new[1:-1, 1:-1] = c[1:-1, 1:-1] + coef * (
            c[2:, 1:-1] + c[:-2, 1:-1] + c[1:-1, 2:] + c[1:-1, :-2] - 4 * c[1:-1, 1:-1]
        )

        # Periodic boundary implementation
        c_new[:, 0] = c_new[:, -2]    # Left boundary from right interior
        c_new[:, -1] = c_new[:, 1]    # Right boundary from left interior

        # Fixed boundary conditions in y-direction
        c_new[0, :] = 0               # Bottom boundary (y=0)
        c_new[-1, :] = 1              # Top boundary (y=1)

        # Check for equilibrium (max change below tolerance)
        max_change = np.max(np.abs(c_new - c))
        if max_change < tolerance:
            print(f"Equilibrium reached at t={t:.5f} after {iteration} iterations with max change {max_change:.2e}.")
            times.append(t)
            solutions.append(c_new.copy())
            break

        c = c_new
        t += dt
        iteration += 1

        # Save intermediate results
        if iteration % save_interval == 0:
            times.append(t)
            solutions.append(c.copy())
            # print(f"Saved at iteration {iteration}, time {t:.5f}, max change {max_change:.2e}")

    else:
        print(f"Maximum iterations reached ({max_iterations}) without convergence.")

    # Save the entire dataset
    # np.save(output_file, {'times': np.array(times), 'solutions': np.array(solutions)})
    # print(f"Simulation complete. Data saved to {output_file}.")

    return np.array(times), np.array(solutions), t

def animate_diffusion(times, solutions, interval=200, time_multiplier=10,
                      save_animation=False, filename='diffusion_equilibrium_slow.gif'):
    """
    Create a slowed-down animated plot of the 2D diffusion equation until equilibrium.

    Parameters:
    - times: Array of time points from the simulation.
    - solutions: Array of 2D concentration fields corresponding to 'times'.
    - interval: Delay between frames in milliseconds (higher = slower playback).
    - time_multiplier: Factor by which to multiply time for slower time scaling in the title.
    - save_animation: If True, saves the animation as a .gif file.
    - filename: Filename for saving the animation.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    c_min, c_max = 0, 1
    img = ax.imshow(solutions[0], cmap='viridis', origin='lower', extent=[0, 1, 0, 1], vmin=c_min, vmax=c_max)
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Concentration')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title = ax.set_title(f'Time: {times[0] * time_multiplier:.3f} (scaled)')

    def update(frame):
        img.set_array(solutions[frame])
        title.set_text(f'Time: {times[frame] * time_multiplier:.3f} (scaled)')
        return img, title

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=interval, blit=True)

    # Save as .gif if requested
    if save_animation:
        anim.save(filename, writer='pillow', fps=10, dpi=150)  # Lower fps for slower playback
        print(f"Animation saved as {filename}.")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    plt.rcParams.update({'font.size': 14})

    # Run the simulation
    times, solutions = diffusion_equation(N=50, T=1.0, dt=0.0001)

    # Plot the final 2D concentration field
    plot_diffusion(times, solutions)

    # Compare numerical and analytical solutions
    compare_solution(times, solutions, D=1.0, selected_times=[0.001, 0.01, 0.1, 1.0])

    # Plot the results at specified time points
    plot_multiple_concentrations(times, solutions, selected_times=[0, 0.001, 0.01, 0.1, 1.0])

    # Run the simulation until equilibrium
    eq_times, eq_solutions, eq_time = diffusion_equation_until_equilibrium(N=50, dt=0.0001, tolerance=1e-6)

    # Plot the final 2D concentration field at equilibrium
    plot_diffusion(eq_times, eq_solutions, equilibrium_label=True)

    # Create and display a slowed-down animation
    animate_diffusion(eq_times, eq_solutions, interval=300, time_multiplier=100, save_animation=True,
                    filename='diffusion_to_equilibrium_slow.gif')