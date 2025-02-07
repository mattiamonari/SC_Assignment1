import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

__all__ = ['solve_wave_equation', 'plot_wave_solution']

def solve_wave_equation(L=1.0, N=100, c=1.0, T=2.0, dt=0.001, initial_condition=lambda x: np.sin(2*np.pi*x)):
    """
    L: Length of string
    N: Number of spatial points
    c: Wave speed
    T: Total simulation time
    dt: Time step
    """
    # Spatial grid
    dx = L/N
    x = np.linspace(0, L, N+1)
    
    # Stability condition check (CFL condition). WHY?
    if c*dt/dx > 1:
        raise ValueError("Solution will be unstable - decrease dt or increase dx")
    
    # Initialize solution arrays
    u = np.zeros((3, N+1))  # 3 time levels: n, n+1, n+2
    
    # Set initial conditions (example with sin(2πx))
    u[1,:] = initial_condition(x)  # Initial displacement, Ψ(x, t = 0)
    u[0,:] = u[1,:]  # Initial velocity Ψ′(x, t = 0) = 0.
    
    # Courant number
    r = (c*dt/dx)**2
    
    # Main time stepping loop
    t = 0
    time_points = []
    solutions = []
    
    while t < T:
        # Save current solution
        time_points.append(t)
        solutions.append(u[1,:].copy())
        
        # Update interior points
        u[2,1:-1] = 2*u[1,1:-1] - u[0,1:-1] + r*(u[1,2:] - 2*u[1,1:-1] + u[1,:-2])
        
        # Apply boundary conditions (fixed ends)
        u[2,0] = 0
        u[2,-1] = 0
        
        # Update time levels
        u[0,:] = u[1,:]
        u[1,:] = u[2,:]
        
        t += dt
    
    return np.array(time_points), np.array(solutions)

# Example usage and plotting
def plot_wave_solution(times, solutions):
    # Create animation frames
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('Position')
    ax.set_ylabel('Displacement')
    
    line, = ax.plot([], [], 'b-')
    
    def update(frame):
        line.set_data(np.linspace(0, 1, len(solutions[frame])), solutions[frame])
        ax.set_title(f'Time: {times[frame]:.3f}')
        return line,
    
    return fig, update, len(times)


times, solutions = solve_wave_equation(T=1, initial_condition= lambda x: np.sin(2*np.pi*x))
fig, update, num_frames = plot_wave_solution(times, solutions)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
plt.show()

times, solutions = solve_wave_equation(initial_condition= lambda x: np.sin(5*np.pi*x))
fig, update, num_frames = plot_wave_solution(times, solutions)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
plt.show()

initial_condition = lambda x: np.where((np.array(x) > 1/5) & (np.array(x) < 2/5), np.sin(5*np.pi*x), 0)
times, solutions = solve_wave_equation(initial_condition = initial_condition)
fig, update, num_frames = plot_wave_solution(times, solutions)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
plt.show()