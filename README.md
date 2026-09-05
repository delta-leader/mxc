

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
- The default cmake-file links Eigen against OpenBLAS and can be run with: \
  `mkdir build && cd build && cmake .. && cmake --build .`
- On Tsubame 4.0
* Both compile will create an executable named `main.app`

# Run / Obtain Results
* Binary lorasp: solves a complete H^2-matrix system and verify answer through dense matrix vector multiplication.
* Example: \
MPI launch **does not** need to have process number be a strict power of 2, \
yet power of 2 process numbers is encouraged to have better performance (1 for serial run, 2, 4, 8, 16 etc.). \
`mpirun -n 16 ./lorasp 20000 2 256 1.e-10 100 2000` \
\
Use 16 processes to solve a 3-D Laplacian H^2 matrix of dimension 20000 by 20000 under strong admissibility configuration theta=2, \
and Low-rank approximation tolerance 1.e-10 and a maximum compressed rank of 100, sampling 2,000 particles per box to compress.
* Using provided scripts: \
`cd scripts && . run.sh` \
Running this script generates results for **O(N) serial factor time** and **Strong Scaling factor time** for very small problem sizes. \
The results is stored in `nbd/log` folder by default, and containing the plots, the raw output logs, and the parsed csv results.

* Plotting script requires Python3.

# Runtime parameters
- M: Number of nodes (not elements) in the input file (default is 5697).
- geom: Type of the geometry stored in the input file (default is 1). Together, M and geom define which input mesh is used by the program. The following table provides an overview over the available files (note that the identifiers for the salt model are slightly different as the number of nodes is $\approx M * 1000$):
| M | geom |  Mesh |  DoF  |
|----------:|:-------------:|:------:|------:|
| 2455 | 1 | single sphere | 14718 |
| 5697 | 1 | single sphere | 34 170 |
| 15520 | 1 | single sphere | 93108 |
| 50034 | 1 | single sphere | 300 192 |
| 61542 | 1 | single sphere | 369240 |
| 101814 | 1 | single sphere | 610872 |
| 169798 | 1 | single sphere | 1 018 776 |
| 4937 | 2 | two spheres | 29 598 |
| 9871 | 4 | four spheres | 59 178 |
| 19736 | 8 | eight spheres | 118 320 |
| 39535 | 16 | 16 spheres | 237 018 |
| 78982 | 32 | 32 spheres | 473 508 |
| 158787 | 64 | 64 spheres | 951 954 |
| 200 | 3 | salt model | 1217358 |

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

