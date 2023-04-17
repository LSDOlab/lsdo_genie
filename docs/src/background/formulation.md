# Formulation

Our formulation is based on creating an implicit function defined by a B-spline.
For geometric shapes in 2D, a B-spline surface is used, and in 3D, a B-spline volume is used.
A B-spline volume is given by

$$
\phi(x,y,z) = \sum_{i,j,k=1}^{n_i,n_j,n_k} \mathbf{C}_\phi \mathbf{B}_{i,d}(x) \mathbf{B}_{j,d}(y) \mathbf{B}_{k,d}(z),
$$
where $\mathbf{C}_\phi$ is the value of the control points and $\mathbf{B}_{i,d}(x)$ is the basis function defined
by the Cox de-Boor algorithm in the $i$ direction with $d$ polynomial degree.
$N_c$ is the total number of control points, which is built up by uniformly distributed control points along the $i,j,k$ directions with
the number of control points $n_i,n_j,n_k$ in each direction, respectively.

Energy terms are defined for the implicit function with respect to the user-defined oriented point cloud.
An oriented point cloud is a set of ordered pairs $\{(\mathbf{p}_i,\vec{\mathbf{n}}_i):i=1,\dots, N_\Gamma\}$,
which represents any arbitrary geometric shape.
The formulation uses 3 energy terms that measure the 
accuracy of the zero level set ($\mathcal{E}_p$), 
accuracy of the gradient field ($\mathcal{E}_n$), 
and smoothness ($\mathcal{E}_r$) of the function.

$$
\begin{align}
    \mathcal{E}_p &= \frac{1}{N_\Gamma} \sum^{N_\Gamma}_{i=1} \phi(\mathbf{p}_i)^2 \\
    \mathcal{E}_n &= \frac{1}{N_\Gamma} \sum^{N_\Gamma}_{i=1}\left\|\nabla \phi(\mathbf{p}_i) + \vec{\mathbf{n}}_i\right\|^2 \\
    \mathcal{E}_r &= \frac{1}{|V|} \int_V \left\| \nabla^2 \phi(\mathbf{x}) \right\|_F^2 dV
\end{align}
$$

To evaluate the integral in $\mathcal{E}_r$, a simple quadrature rule is used.
The quadrature points are the Bspline control points within the domain of interest, of which there are $N$ of.

$$
\mathcal{E}_r \approx \frac{1}{N} \sum^N_{i=1} \left\| \nabla^2 \phi(\mathbf{x}_{i}) \right\|_F^2,
$$

The underlying energy minimization is then defined as follows.

$$
\begin{align}
    \begin{array}{r l}
        \text{minimize}            & f = \mathcal{E}_p + \lambda_n \mathcal{E}_n + \lambda_r \mathcal{E}_r\\
        \text{with respect to}     & \mathbf{C}_\phi
    \end{array}
\end{align}
$$

The defined problem is a well-posed unconstrained quadratic programming problem.
It's form may be reduced to the following.

$$
\begin{array}{r l}
    \text{minimize}            & \frac{1}{2} \mathbf{C}_\phi^T \tilde{A} \mathbf{C}_\phi + \tilde{b}^T \mathbf{C}_\phi
\end{array}
$$

It is then clear to see that the exact solution reduces to a linear system of equations in the form

$$
\tilde{A} \mathbf{C}_\phi = -\tilde{b},
$$
where $\tilde{A}\in\mathbb{R}^{N_c,N_c}$ is a sparse, symmetric, positive definite matrix. 
An example of its sparsity is shown below
![text](/src/images/sparsity.png)
