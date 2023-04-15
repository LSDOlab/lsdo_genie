# Welcome to LSDO GeNIe

![alt text](/src/images/lsdolab.png "LSDO Lab")

The LSDO Lab's Gemoetric Non-Interference (GeNIe) constraint formulation is an efficient and scalable method to enforce non-interference constraints in gradient-based optimization. We define geometric non-interference as a constraint to enforce such that the design body does not interfere with any other geometric shape in the environment. Non-interference constraints appear in layout optimization, optimal path planning optimization, and shape optimization problems.

![alt text](/src/images/arbitrarydiagram.png "Generic design optimization enforcing geometric non-interference")

This package is more efficient formulation to the original energy minimization formulation presented in a previous paper. As an unconstrained quadratic programming problem, the solution to this formulation reduces to a solution to a sparse linear system; however, the original implementation was done using a BFGS approximation using a gradient-based optimizer. The original implementation, `lsdo_noninterference`, can be found [here](https://github.com/LSDOlab/lsdo_noninterference), but we recommend this package.


# Cite the original work
Pending review and revisions...
```none
"Scalable Enforcement of Geometric Non-interference Constraints for Gradient-Based Optimization"
Ryan C. Dunn, Anugrah Jo Joshy, Jui-Te Lin, Cedric Girerd, Tania K. Morimoto, John T. Hwang 
Springer Nature's Structural and Multidisciplinary Optimization Journal
```

<!-- Remove/add custom pages from/to toc as per your package's requirement -->

```{toctree}
:maxdepth: 1
:hidden:

src/getting_started
src/background
```

<!-- src/tutorials -->
<!-- src/custom_1 -->
<!-- src/custom_2 -->
<!-- src/api -->