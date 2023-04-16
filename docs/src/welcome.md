# Welcome to LSDO GeNIe

![text](/src/images/lsdolab.png "Large-Scale Design Optimization Lab")

The LSDO Lab's Gemoetric Non-Interference (GeNIe) constraint formulation is an efficient and scalable method to enforce non-interference constraints in gradient-based optimization. We define geometric non-interference as a constraint to enforce such that the design body does not interfere with any other geometric shape in the environment. Non-interference constraints appear in layout optimization, optimal path planning optimization, and shape optimization problems.

In the diagram below, geometric non-interference is enforced between the design body and the infeasible space outside of the grey geometric shape. In gradient-based optimization, a constraint function $\phi(\mathbf{x})\geq0$, where $\mathbf{x}$ is the design variables, must be enforced to prevent the design body from interfering with the infeasible region. It is required that $\phi$ is continuous and differentiable for gradient-based optimization and desired that it is fast-to-evaluate, scalable, and an accurate representation of the boundary $\Gamma$ between the feasible $\phi(\mathbf{x})>0$ and infeasible $\phi(\mathbf{x})<0$ spaces. The focus of this package is on generating the geometric non-interferencee constraint function $\phi$.

![text](/src/images/arbitrarydiagram.png "Geometric Non-interference on arbitrary geometric shapes")

This package is more efficient formulation to the original energy minimization formulation presented in a previous paper. As an unconstrained quadratic programming problem, the solution to this formulation reduces to a solution to a sparse linear system; however, the original implementation was done using a BFGS approximation using a gradient-based optimizer. The original implementation, `lsdo_noninterference`, can be found [here](https://github.com/LSDOlab/lsdo_noninterference), but we recommend this package.

# Cite the original work
Pending review and revisions...
```none
"Scalable Enforcement of Geometric Non-interference Constraints for Gradient-Based Optimization"
Ryan C. Dunn, Anugrah Jo Joshy, Jui-Te Lin, Cedric Girerd, Tania K. Morimoto, John T. Hwang 
Springer Nature's Structural and Multidisciplinary Optimization Journal
```

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
src/tutorials
src/examples
src/api
src/bibilography
```