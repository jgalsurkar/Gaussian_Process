# Gaussian_Process
Implement the Gaussian process model for regression from scratch

The Gaussian process treats a set of N observations $(x_1, y_1), . . . , (x_N , y_N)$, with $x_i ∈ R^d$ and $y_i ∈ R$, as being generated from a multivariate Gaussian distribution as follows,
$y∼Normal(0, σ^2I + K), K_{ij} =K(x_i,x_j)$  
I will be using the RBF Kernel: $exp^{−1/b ∥x_i−x_j∥^2}$