import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
c = 1  # Wave velocity
L = 1  # Length of the string
T = 1  # Total time
N = 100  # Number of grid points
dx = L / (N - 1)  # Grid spacing
dt = 0.001  # Time step

# Initialize grid
x = np.linspace(0, L, N)
u = np.zeros(N)
u_prev = np.sin(np.pi * x / L)

# Define the wave equation with boundary conditions
def wave_equation(t, u):
    u_next = np.zeros(N)
    u_next[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + (c * dt / dx) ** 2 * (u[:-2] - 2 * u[1:-1] + u[2:])
    
    # Boundary conditions
    u_next[0] = 0  # Left boundary condition: u(0, t) = 0
    u_next[-1] = 0  # Right boundary condition: u(L, t) = 0
    
    u_prev[:] = u[:]
    return u_next

# Solve the wave equation using solve_ivp
solution = solve_ivp(wave_equation, (0, T), u_prev, t_eval=np.arange(0, T, dt))

# Plot the results
plt.figure()
plt.imshow(solution.y, aspect='auto', cmap='coolwarm', extent=[0, T, 0, L])
plt.colorbar(label='Displacement')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Wave Equation Simulation with Boundary Conditions')
plt.show()
