# Overview
This repository contains the codebase used in the paper "Efficient algebraic preconditioning for elastic wave scattering based on $\mathcal{H}^2$-matrices in a distributed setting", including the input files used during the experiments. 

# Dependencies
The following dependencies are required in order to build the code:
- BLAS
- LAPACK
- OpenMP
- MPI
- Eigen (https://github.com/PX4/eigen)

# Build source
- We provide a cmake-file that per default links Eigen against OpenBLAS und should be able to compile on any sytem that satisfies the above dependencies.
- The file can be run with: \
  `mkdir build && cd build && cmake .. && cmake --build .`
- Alternatively on Tsubame 4.0 we link against Intel MKL by passing the `-DUSE_MKL=ON` option as follows:\
`mkdir build && cd build && cmake .. -DUSE_MKL=ON && cmake --build .`\
 (Note that this follows the TSUBAME 4.0 instructions for loading and linking against IntelMKL and has not been verified on other systems).
- After the compilation is finished an exexutable named `main.app` will be created in the `build` directory.

# Run
* `main.app` creates two $\mathcal{H}^2$-matrices from the sytem matrix corresponding to the single layer potential in elastodynamics. The preconditioner matrix eploys HiDR and uses a factorization basis while the $\mathcal{H}^2_\varepsilon$-matrix is a conventional $\mathcal{H}^2$-matrix used for calculating the matrix-vector products in a restarted GMRES iterative solver.
* Example: \
MPI launch **does not** need to have process number be a strict power of 2, \
yet power of 2 process numbers is encouraged to have better performance (1 for serial run, 2, 4, 8, 16 etc.). \
`mpirun -n 8 ./main.app 5697 1 1.303 128 2 128 0 1e-8 2 10 20 64 64` \
\
Use 8 processes to solve the mesh from a single sphere with 34 170 DoF (see table below) where $\omega = 1.303$. and both $\mathcal{H}^2$-matrices use a maximum leaf size of 128 elements. The preconditioner is created with a fixed rank of 128 (without rank increase) using HiDR ($s_1=64, s_2=64$) and $admis = 2$, while $\mathcal{H}^2_\varepsilon$ used a target accuracy of 1e-8 and $admis = 2$ as well. The iterative solver is a preconditioned GMRES, which is restarted after 10 iterations until either the maximum number of restarts (i.e. 20) is achieved or target accuracy of 1e-8 has been achieved.
* Using provided scripts: \
`cd scripts && . run.sh` \
Running this script generates results for **O(N) serial factor time** and **Strong Scaling factor time** for very small problem sizes. \
The results is stored in `nbd/log` folder by default, and containing the plots, the raw output logs, and the parsed csv results.

# Runtime parameters
The order of the Runtime parameters is as follows `./main.app M geom omega leaf_size admis_precon rank leveled_rank epsilon admis inner_iter max_iter s1, s2 write_result read_folder runs`
- M: Number of nodes (not elements) in the input file (default is 5697).
- geom: Type of the geometry stored in the input file (default is 1). Together, M and geom define which input mesh is used by the program. The following table provides an overview over the available files (note that the identifiers for the salt model are slightly different as the number of nodes is $\approx M * 1000$):

| M | geom |  Mesh |  DoF  |
|----------:|:-------------:|:------:|------:|
| 2455 | 1 | single sphere | 14718 |
| 5697 | 1 | single sphere | 34 170 |
| 15520 | 1 | single sphere | 93 108 |
| 50034 | 1 | single sphere | 300 192 |
| 61542 | 1 | single sphere | 369 240 |
| 101814 | 1 | single sphere | 610 872 |
| 169798 | 1 | single sphere | 1 018 776 |
| 4937 | 2 | two spheres | 29 598 |
| 9871 | 4 | four spheres | 59 178 |
| 19736 | 8 | eight spheres | 118 320 |
| 39535 | 16 | 16 spheres | 237 018 |
| 78982 | 32 | 32 spheres | 473 508 |
| 158787 | 64 | 64 spheres | 951 954 |
| 200 | 3 | salt model | 1 217 358 |

- omega: The angular frequency (default is 1). 
- leaf_size: The maximum number of elements contained in each leaf-level node (default is 32).
- admis_precon: The admissibility condition used for the preconditioner (default is 2).
- rank: The fixed rank used for the factorization basis in the preconditioner (default is 32).
- leveled_rank: The rank increase per level. The default is 0 which corresponds to no rank increase. This option is mainly intended to be used with a HSS preconditioner.
- epsilon: The target accuracy for the $\mathcal{H}^2_\varepsilon$-matrix used for accurate matrix-vector products in the iterative solver (default is $10^{-8}$).
- admis: The admissibility of the $\mathcal{H}^2_\varepsilon$-matrix used for accurate matrix-vector products in the iterative solver (default is 2).
- inner_iter: Number of inner iterations in the restarted GMRES solver (default is 10).
- max_iter: Maximum number of outer iterations (i.e. restarts) to compute (default is 50).
- s1, s2: HiDR parameters (number of sample points for each sweep of the cluster tree). Default is 0, which corresponds to no sampling at all (i.e. HiDR is not employed).
- write_result: If enabled, the obtained solution will be written to the `results`-folder
- read_folder: Since the creation of $\mathcal{H}^2_\varepsilon$ is expensive, it can be seralized to the `read_folder` and read directly from there to save time. The default value is an empty string, which means no de-/serialization takes place.
- runs: How often the experiment (starting from the factorization) should be repeated (to get reliable timings). The default is 1.

# Other Scripts
- We provide two examples of job scripts on TSUBAME 4.0 in the `scripts` directory:
  - `solve_sphere.sh` is set to solve the largest single sphere problem, while
  - `solve_salt.sh` is set to solve the salt model.
- Additionally we provide the files used to create the spherical meshes in `gmsh` in the `scripts/mesh` directory as:
  - `sphere.geo` -> meshing a single unit sphere
  - `two_spheres.geo` -> meshing two unit spheres
  - `four_spheres.geo` -> meshing four unit spheres
  - `eight_spheres.geo` -> meshing eight unit spheres
  - `16_spheres.geo` -> meshing 16 unit spheres
  - `32_spheres.geo` -> meshing 32 unit spheres
  - `64_spheres.geo` -> meshing 32 unit spheres
- and our custom converter to that processes `ply2` files and returns `inp` files in the format expected by our code (`mesh_converter.py`).
- The script `create_meshfile.sh` combines those two steps into a single script, taking the name of the `.geo` file to be processed and the `clscale` (i.e. size of mesh elements) as inputs.
- Finally, `salt_200k.ply2` contains the remeshed version of the modified SEG/EAEG salt model used during the experiments.

