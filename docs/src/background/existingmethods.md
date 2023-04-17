# Existing Methods

## Problems involving geometric non-interference constraints

Wind Farm Layout Optimization Problem (2D) with boundary constraints
by 
Risco et al. {cite:p}`Risco_et_al`

Robotic Design Optimization (3D) with anatomical constraints
by 
Lin et al. {cite:p}`Lin_et_al`
and
Bergeles et al. {cite:p}`Bergeles_et_al`

Aerodynamic Shape Optimization (3D) with spatial integration constraints
by
Brelje et al. {cite:p}`Brelje_et_al`

## Implicit surface reconstruction formulation
Our method is based on an implicit surface reconstruction method, Smooth Signed Distance (SSD) by Calakli and Taubin {cite:p}`Calakli_and_Taubin`

## Explicit formulation
We compare ourselves to an explicit formulation for the signed distance function by Hicken and Kaur {cite:p}`Hicken_and_Kaur`.
Their method is continuous and differentiable but much like many other methods, scales in computational complexity with the number of points in the input point cloud.
The function interpolates data points via piecewise linear signed distance functions defined by local hyperplanes.
The piecewise functions are then smoothly combined with KS-aggregation.

$$
\phi_{H}(\mathbf{x}) = \frac{\sum^{N_\Gamma}_{i=1} d_i(\mathbf{x})e^{-\rho (\Delta_i(\mathbf{x}) - \Delta_{\text{min}})}}
    {\sum^{N_\Gamma}_{j=1} e^{-\rho (\Delta_j(\mathbf{x}) - \Delta_{\text{min}})}}
$$
where $d_i(\mathbf{x})$ is the signed distance to the hyperplane defined by the point and normal vector pair
in the point cloud $(\mathbf{p}_i,\vec{\mathbf{n}}_i)$, $\Delta_i(x)$ is the Euclidean distance to the point $\mathbf{p}_i$, $\Delta_\text{min}$ 
is the Euclidean distance to the nearest neighbor, and $\rho$ is 
a smoothing parameter.

## Prior gradient-based optimziation formulations
Three main formulations were previously used in gradient-based optimization.
Risco et al. {cite:p}`Risco_et_al` presents a generic 2D formulation that is continuous and non-differentiable, because it uses the nearest neighbor to calculate the distance.
While it has worked in practice, it is not considered a sufficient solution because it is non-differentiable, not generic to 3D shapes, and scales with $\mathcal{O}(N_\Gamma)$
Brelje et al. {cite:p}`Brelje_et_al` presents a generic 3D constraint function that is continuous and differentiable, but scales with $\mathcal{O}(N_\Gamma)$.
Their implementation is parallelized using graphics processing units (GPUs), but is not considered a sufficient solution due to scaling.
Additionally, the constraint function is not defined when the design body is entirely infeasible.
Lin et al. {cite:p}`Lin_et_al` presents a generic 3D constraint function that is continuous and differentiable.
Their formulation prioritizes smoothness over accuracy in the function, making ti a poor representation of the signed distance. 
Additionally, the constraint function scales with $\mathcal{O}(N_\Gamma)$.