#Generating NUmerical Data
import numpy as np
import matplotlib.pyplot as plt

# Define the parameters and spatial domain
E = 1.0
A = 1.0
N = 10000  # Increase the number of data points
x = np.linspace(0, 1, N)
dx = x[1] - x[0]

# Initialize arrays for u and p
u = np.zeros(N)
p = 4 * np.pi**2 * np.sin(2 * np.pi * x)

# Set boundary conditions
u[0] = 0
u[-1] = 0

# Perform finite difference discretization and solve
for i in range(1, N - 1):
    u[i] = (u[i - 1] + u[i + 1] - p[i] * dx**2 / (E * A)) / 2

# Save the data points to a CSV file with column names 'x' and 'u'
data = np.column_stack((x, u))
np.savetxt("data_points_10000.csv", data, delimiter=",", header="x,u", comments="")

print(f"Saved {N} data points to data_points_10000.csv")

# Create a plot of u(x)
plt.figure(figsize=(8, 6))
plt.plot(x, u, label='u(x)')
plt.xlabel('x')
plt.ylabel('u')
plt.title('Solution u(x)')
plt.legend()
plt.grid(True)
plt.show()
###Model
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Load your data from the CSV file and normalize the input
data = pd.read_csv("data_points_10000.csv")
x_values = data['x'].values
u_values = data['u'].values
x_train = torch.tensor(x_values, dtype=torch.float32, requires_grad=True).view(-1, 1)
u_train = torch.tensor(u_values, dtype=torch.float32, requires_grad=True).view(-1, 1)

# Define the neural network architecture
def buildModel(input_dim, hidden_dim, output_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_dim, output_dim)
    )
    return model

# Define the physics-informed loss function
def physics_informed_loss(model, x, EA, p, u_true):
    u = model(x)
    u_x = torch.autograd.grad(u, x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    EAu_xx = torch.autograd.grad(EA(x) * u_x, x, torch.ones(x.size()[0], 1), create_graph=True, retain_graph=True)[0]
    f = EAu_xx - p(x)
    loss_mse = torch.mean(f**2)  # Mean squared error loss
    loss_data = torch.mean((u - u_true)**2)  # Data fitting loss
    loss = loss_mse + loss_data
    return loss

# Create the model
model = buildModel(1, 10, 1)

# Define functions EA and p based on your physics equations
EA = lambda x: torch.tensor(1.0, requires_grad=True) + torch.zeros_like(x, requires_grad=True)
p = lambda x: 4 * torch.tensor(math.pi**2, requires_grad=True) * torch.sin(2 * math.pi * x)

# Define an optimizer (e.g., Adam optimizer) with learning rate scheduling
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Training loop with early stopping
num_epochs = 1000
best_loss = float('inf')
tolerance = 1e-6
early_stopping_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    loss = physics_informed_loss(model, x_train, EA, p, u_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
        
        # Early stopping check
        if loss < best_loss - tolerance:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_epochs:
                print("Early stopping. Loss did not improve.")
                break

"""# Visualize the predicted wavefunction
x_values = torch.linspace(0, 1, 1000).view(-1, 1)
predicted_u = model(x_values).detach().numpy()

plt.figure(figsize=(8, 6))
plt.plot(x_values, predicted_u, label='Predicted u(x)')
plt.scatter(x_values, u_values, label='Data', color='red', marker='o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Predicted Wavefunction')
plt.legend()
plt.grid(True)
plt.show()"""

# Visualize the predicted wavefunction
x_values = torch.linspace(0, 1, 1000).view(-1, 1)
predicted_u = model(x_values).detach().numpy()

plt.figure(figsize=(8, 6))
plt.plot(x_values, predicted_u, label='Predicted u(x)')
plt.scatter(x_values, predicted_u, label='Data', color='red', marker='o')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Predicted Wavefunction')
plt.legend()
plt.grid(True)
plt.show()

