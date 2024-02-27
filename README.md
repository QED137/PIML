# PDE Solver with Physics-Informed Neural Networks (PINNs)

Physics-Informed Neural Networks (PINNs) are a powerful scientific machine-learning technique used to solve problems involving Partial Differential Equations (PDEs). Unlike traditional numerical methods, PINNs approximate PDE solutions by training a neural network to minimize a loss function. This loss function includes terms reflecting the initial and boundary conditions along the space-time domain’s boundary and the PDE residual at selected points in the domain, known as collocation points.

## Key Concept

-Residual Network: PINNs incorporate a residual network that encodes the governing physics equations, allowing them to learn the underlying physics of the problem.
    
-Unsupervised Training: PINNs operate as an unsupervised strategy, eliminating the need for labeled data or prior simulations.
    
-Mesh-Free Technique: PINNs transform the problem of directly solving PDEs into a loss function optimization problem, making them a mesh-free technique.
    
-Physics-Driven Learning: By integrating the mathematical model into the network and reinforcing the loss function with a residual term from the governing equation, PINNs leverage structured prior knowledge about the solution.


