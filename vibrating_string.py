import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

__all__ = ['solve_wave_equation', 'plot_wave_solution']

def solve_wave_equation(L=1.0, N=1000, c=1.0, T=1.0, dt=0.001, initial_condition=lambda x: np.sin(2*np.pi*x)):
    """
    L: Length of string
    N: Number of spatial points
    c: Wave speed
    T: Total simulation time
    dt: Time step
    """
    
    dx = L/N
    x = np.linspace(0, L, N+1)
    
    # Check stability condition
    if c*dt/dx > 1:
        raise ValueError("Solution will be unstable - decrease dt or increase dx")
    
    u = np.zeros((3, N+1))  # 3 time levels: n-1, n, n+1
    
    u[0,:] = initial_condition(x) # Initial displacement, Ψ(x, t = 0)
    u[1,:] = u[0,:] # Initial displacement, Ψ(x, t = 1)
    
    # Courant number
    C = (c*dt/dx)**2
    
    # Main time stepping loop
    t = 0
    time_points = []
    solutions = []
    
    while t < T:
        time_points.append(t)
        solutions.append(u[1,:].copy())
        
        # Update interior points
        u[2,1:-1] = 2*u[1,1:-1] - u[0,1:-1] + C*(u[1,2:] - 2*u[1,1:-1] + u[1,:-2])
        
        # Apply boundary conditions
        u[2,0] = 0
        u[2,-1] = 0
        
        # Update on time
        u[0,:] = u[1,:]
        u[1,:] = u[2,:]
        
        t += dt
    
    return np.array(time_points), np.array(solutions)

def plot_wave_solution(times, solutions):
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

if __name__ == '__main__':
    #Increase font 
    plt.rcParams.update({'font.size': 14})

    times, solutions = solve_wave_equation(initial_condition= lambda x: np.sin(2*np.pi*x))
    fig, update, num_frames = plot_wave_solution(times, solutions)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
    plt.tight_layout()
    plt.show()

    times, solutions = solve_wave_equation(T=5, initial_condition= lambda x: np.sin(5*np.pi*x))
    fig, update, num_frames = plot_wave_solution(times, solutions)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
    plt.tight_layout()
    plt.show()

    initial_condition = lambda x: np.where((np.array(x) > 1/5) & (np.array(x) < 2/5), np.sin(5*np.pi*x), 0)
    times, solutions = solve_wave_equation(T=2, initial_condition = initial_condition)
    fig, update, num_frames = plot_wave_solution(times, solutions)
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1, blit=True)
    FFwriter = animation.FFMpegWriter(fps=60)
    plt.tight_layout()
    ani.save('animation.mp4', writer = FFwriter, dpi=600)
    plt.show()