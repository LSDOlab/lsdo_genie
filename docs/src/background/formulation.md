# Formulation

Bsplines

$$
\phi(x,y,z) = \sum_{i,j,k=1}^{n_i,n_j,n_k} \mathbf{C}_\phi \mathbf{B}_{i,d}(x) \mathbf{B}_{j,d}(y) \mathbf{B}_{k,d}(z)
$$

Energy terms

$$
\begin{align}
    \mathcal{E}_p &= \frac{1}{N_\Gamma} \sum^{N_\Gamma}_{i=1} \phi(\mathbf{p}_i)^2 \\
    \mathcal{E}_n &= \frac{1}{N_\Gamma} \sum^{N_\Gamma}_{i=1}\left\|\nabla \phi(\mathbf{p}_i) + \vec{\mathbf{n}}_i\right\|^2 \\
    \mathcal{E}_r &= \frac{1}{|V|} \int_V \left\| \nabla^2 \phi(\mathbf{x}) \right\|_F^2 dV
\end{align}
$$

To evaluate the integral in $\mathcal{E}_r$, we setup a simple quadrature rule using the Bspline control points within the domain of interest.

$$
\mathcal{E}_r \approx \frac{1}{N} \sum^N_{i=1} \left\| \nabla^2 \phi(\mathbf{x}_{i}) \right\|_F^2,
$$

Energy minimization problem

$$
\begin{align}
    \begin{array}{r l}
        \text{minimize}            & f = \mathcal{E}_p + \lambda_n \mathcal{E}_n + \lambda_r \mathcal{E}_r\\
        \text{with respect to}     & \mathbf{C}_\phi
    \end{array}
\end{align}
$$

Unconstrained quadratic programming problem

$$
\begin{array}{r l}
    \text{minimize}            & \frac{1}{2} \mathbf{C}_\phi^T \tilde{A} \mathbf{C}_\phi - \tilde{b}^T \mathbf{C}_\phi
\end{array}
$$

Solution

$$
\tilde{A} \mathbf{C}_\phi = \tilde{b},
$$
where $\tilde{A}\in\mathbb{R}^{N_{c},N_{c}}$ is a sparse, symmetric, positive definite matrix. An example of its sparsity is shown below
![text](/src/images/sparsity.png)
