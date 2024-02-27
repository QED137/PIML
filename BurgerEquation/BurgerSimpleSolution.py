import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BurgersEquationSolver:
    def __init__(self, nt, nx, nu, dt):
        self.nt = nt
        self.nx = nx
        self.nu = nu
        self.dt = dt

    def solve(self):
        # Initialize arrays
        x = np.linspace(-1, 1, self.nx)
        dx = abs(x[0] - x[1])
        u = -np.sin(np.pi * x)
        uf = np.zeros((self.nt, self.nx))
        uf[0, :] = u

        # Solve Burgers' equation
        for n in range(1, self.nt):
            un = u.copy()
            for i in range(1, self.nx - 1):
                u[i] = un[i] - un[i] * (self.dt / dx) * (un[i] - un[i - 1]) + \
                        (self.nu * self.dt / (dx ** 2)) * (un[i + 1] - 2 * un[i] + un[i - 1])
                uf[n, i] = u[i]

        return x, u, uf

if __name__ == "__main__":
    # Parameters
    nt = 1001
    nx = 1001
    nu = 0.01 / np.pi  # Adjusted viscosity coefficient
    dt = 0.0001  # Reduced time step

    # Solve Burgers' equation
    solver = BurgersEquationSolver(nt, nx, nu, dt)
    x, u, uf = solver.solve()

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(x, u, label="u Solution")
    plt.title("Final Time Step Solution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

    plt.plot(uf[0, :])
    plt.plot(uf[-1, :])
    plt.grid(True)
    plt.show()
    uf_cleaned = np.nan_to_num(uf)

# Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(uf_cleaned.T, cmap='jet')
    plt.xlabel('Time Step')
    plt.ylabel('Spatial Index')
    plt.title('Heatmap of Burgers\' Equation Solution')
    plt.show()