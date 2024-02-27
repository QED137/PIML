import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from tqdm import tqdm

# Constants
Nx = 200
tmax = 1
viscosity_coeff = 0.02
dt = 0.002
x = np.linspace(-1, 1, 200)
dx = abs(x[1] - x[0])
nt = int(tmax / dt)
uf = np.zeros((nt, Nx))
u = -1 * np.sin(math.pi * x)

def calculate_solution():
    for i in range(1, nt):
        u1 = u + dt * RHS(u, dx, viscosity_coeff)
        u = 0.5 * u + 0.5 * (u1 + dt * RHS(u1, dx, viscosity_coeff))
        uf[i, :] = u

def plot_initial_condition():
    plt.figure(figsize=(8, 6))
    plt.plot(x, uf[0], '-o', color='b')
    plt.title("Initial Condition")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def plot_final_solution():
    plt.figure(figsize=(8, 6))
    plt.plot(x, uf[-1], '-o', color='b')
    plt.title("Final Solution")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

def plot_solution_progress():
    plt.figure(figsize=(8, 6))
    plt.plot(x, uf[0], '-o', color='b', label='Initial')
    plt.plot(x, uf[-1], '-o', color='r', label='Final')
    plt.title("Solution Progress")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def RHS(u, dx, viscosity_coeff):
    diffusion_term = viscosity_coeff * (np.roll(u, 1) - 2 * u + np.roll(u, -1)) / dx ** 2
    ux = minmod((u - np.roll(u, 1)) / dx, (np.roll(u, -1) - u) / dx)
    uL = np.roll(u - 0.5 * dx * ux, 1)
    uR = u - 0.5 * dx * ux
    fL, fpL = f(uL)
    fR, fpR = f(uR)
    a = np.maximum(np.abs(fpL), np.abs(fpR))
    H = 0.5 * (fL + fR - a * (uR - uL))
    conv_term = -(np.roll(H, -1) - H) / dx
    y = conv_term + diffusion_term
    return y

def f(u):
    y = 0.5 * u ** 2
    yp = u
    return y, yp

def minmod(a, b):
    return 0.5 * (np.sign(a) + np.sign(b)) * np.minimum(np.abs(a), np.abs(b))

def prepare_data():
    tf = np.linspace(0, 1, nt)
    xf = x
    tf_tensor = torch.tensor(tf)
    xf_tensor = torch.tensor(xf)
    combined_tensor_x_train = torch.empty((len(tf) * len(xf), 2), dtype=torch.float32)
    index = 0
    for i in range(len(tf)):
        for j in range(len(xf)):
            combined_tensor_x_train[index][0] = xf_tensor[j]
            combined_tensor_x_train[index][1] = tf_tensor[i]
            index += 1
    your_tensor = torch.tensor(uf, dtype=torch.float32)
    flattened_tensor_y_train = your_tensor.view(-1)
    flattened_tensor_y_train = flattened_tensor_y_train.unsqueeze(1)
    return combined_tensor_x_train, flattened_tensor_y_train

def train_neural_network(X_train, y_train):
    class NN(nn.Module):
        def __init__(self):
            super(NN, self).__init__()
            self.net = torch.nn.Sequential(
                nn.Linear(2, 20),
                nn.Tanh(),
                nn.Linear(20, 30),
                nn.Tanh(),
                nn.Linear(30, 30),
                nn.Tanh(),
                nn.Linear(30, 20),
                nn.Tanh(),
                nn.Linear(20, 20),
                nn.Tanh(),
                nn.Linear(20, 1)
            )

        def forward(self, x):
            out = self.net(x)
            return out

    class Net:
        def __init__(self):
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model = NN().to(device)
            self.h = 0.1
            self.k = 0.1
            x = torch.arange(-1, 1 + self.h, self.h)
            t = torch.arange(0, 1 + self.k, self.k)
            self.X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
            self.X_train = X_train
            self.y_train = y_train
            self.X = self.X.to(device)
            self.X.requires_grad = True
            self.X_train = self.X_train.to(device)
            self.y_train = self.y_train.to(device)
            self.adam = torch.optim.Adam(self.model.parameters())
            self.criterion = torch.nn.MSELoss()
            self.iter = 1

        def loss_func(self):
            self.adam.zero_grad()
            y_pred = self.model(self.X_train)
            loss_data = self.criterion(y_pred, self.y_train)
            u = self.model(self.X)
            du_dX = torch.autograd.grad(
                u,
                self.X,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )[0]
            du_dt = du_dX[:, 1]
            du_dx = du_dX[:, 0]
            du_dXX = torch.autograd.grad(
                du_dX,
                self.X,
                grad_outputs=torch.ones_like(du_dX),
                create_graph=True,
                retain_graph=True
            )[0]
            du_dxx = du_dXX[:, 0]
            loss_pde = self.criterion(du_dt + 1 * u.squeeze() * du_dx, (0.02) * du_dxx)
            loss = loss_pde + loss_data
            loss.backward()
            if self.iter % 100 == 0:
                print(self.iter, loss.item())
            self.iter = self.iter + 1
            return loss

        def train(self):
            self.model.train()
            for i in range(3000):
                self.adam.step(self.loss_func)

        def eval_(self):
            self.model.eval()

    net = Net()
    net.train()
    net.model.eval()
    return net

def visualize_result(net):
    h = 0.01
    k = 0.01
    x = torch.arange(-1, 1, h)
    t = torch.arange(0, 1, k)
    X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T
    X = X.to(net.X.device)
    model = net.model
    model.eval()
    with torch.no_grad():
        y_pred = model(X)
        y_pred = y_pred.reshape(len(x), len(t)).cpu().numpy()

    # Plotting
    plt.plot(y_pred_inverse[:, 0], label='Initial Condition')
    plt.plot(y_pred_inverse[:, -1], label='Final Solution (TVD)')
    plt.plot(y_pred[:, -1], label='Final Solution (PINNs)')
    plt.title("Solution Comparison")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

    sns.set_style("white")
    plt.figure(figsize=(5, 3), dpi=3000)
    sns.heatmap(y_pred, cmap='jet')
    plt.show()

def main():
    calculate_solution()
    plot_initial_condition()
    plot_final_solution()
    plot_solution_progress()
    X_train, y_train = prepare_data()
    trained_net = train_neural_network(X_train, y_train)
    visualize_result(trained_net)

if __name__ == "__main__":
    main()
