sphere: 
Nodes   Elements       DOFs
12611      25218      75654
100 processes, 100 per node
M = 12611, geom = 1
Omega = 1, Leaf-size = 128, admis_precon = 1, rank = 128, leveled rank = 20, epsilon = 1e-08, theta = 1, inner iter = 10, max iter = 50
N = 25218, Leaf = 128, Levels = 8, #Leafs = 256, #Cells = 511
Elements per leaf: 98
Level 8
H^2-Matrix Construct Err: 1.54614e-09
H^2-Matrix Construct Time: 1161.9, 301.792
H^2-Matvec Time: 1.24593, 0.0224375
Dense Matvec Time: 596.814, 0
Level 8
5.46785e+07 dense elements stored
Level 7
Level 6
Level 5
Level 4
Level 3
Level 2
Level 1
Level 0
6.17631e+07 total elements stored
H^2-Preconditioner Construct Err: 0.0566742
H^2-Preconditioner Construct Time: 492.625, 67.0188
H^2-Matvec Time: 0.245798, 0.0160322
H^2-Matrix Factorization Time: 5.71285, 1.20125
H^2-Matrix Substitution Time: 0.321849, 0.0378346
H^2-Matrix Substitution Err: 5913.8
GMRES Residual: 3.29948e-11, Iters: 2
GMRES Time: 8.66774, Comm: 1.80602
iter 0: 1
iter 1: 7.25699e-07
iter 2: 3.29948e-11
Actual Residual: 1.5453e-09

50034     100064     300192
100 processes, 100 per node
M = 50034, geom = 1
Omega = 1, Leaf-size = 128, admis_precon = 1, rank = 128, leveled rank = 20, epsilon = 1e-08, theta = 1, inner iter = 10, max iter = 50
N = 100064, Leaf = 128, Levels = 10, #Leafs = 1024, #Cells = 2047
Elements per leaf: 97
Level 10
2.18808e+08 dense elements stored
Level 9
Level 8
Level 7
ERROR (out of memory?) did not run on two nodes either (exact same place)
It did not run with five nodes either
It seems the problem is indeed the memory on a single node, even though I don't understand it fully. Keeping the number of total processes constant, I was able
to get to level 6 on two nodes (50 processes per node)
Increasing the admissibility to two, it worked for the 3 different settings below on a single node:
f_node=1
50 processes, 50 per node
M = 50034, geom = 1
Omega = 1, Leaf-size = 128, admis_precon = 2, rank = 128, leveled rank = 20, epsilon = 1e-08, theta = 2, inner iter = 10, max iter = 50
N = 100064, Leaf = 128, Levels = 10, #Leafs = 1024, #Cells = 2047
Elements per leaf: 97
50 processes, 50 per node
M = 50034, geom = 1
Omega = 1, Leaf-size = 64, admis_precon = 2, rank = 64, leveled rank = 10, epsilon = 1e-08, theta = 2, inner iter = 10, max iter = 50
N = 100064, Leaf = 64, Levels = 11, #Leafs = 2048, #Cells = 4095
Elements per leaf: 48
50 processes, 50 per node
M = 50034, geom = 1
Omega = 1, Leaf-size = 64, admis_precon = 3, rank = 64, leveled rank = 10, epsilon = 1e-08, theta = 3, inner iter = 10, max iter = 50
N = 100064, Leaf = 64, Levels = 11, #Leafs = 2048, #Cells = 4095
Elements per leaf: 48


77751 did not run on 5 processes
node_f=5
500 processes, 20 per node
M = 77751, geom = 1
Omega = 1, Leaf-size = 256, admis_precon = 2, rank = 256, leveled rank = 40, epsilon = 1e-08, theta = 2, inner iter = 10, max iter = 50
N = 155498, Leaf = 256, Levels = 10, #Leafs = 1024, #Cells = 2047
Elements per leaf: 151
Level 10
1.68271e+09 dense elements stored
Level 9
Level 8
Level 7
Level 6
Level 5
Level 4
Level 3
Level 2
Level 1
Level 0
1.70276e+09 total elements stored
H^2-Matrix Construct Err: 9.97031e-10
H^2-Matrix Construct Time: 1089.71, 410.405
H^2-Matvec Time: 0.860769, 0.717863
Dense Matvec Time: 918.091, 0
Level 10
1.68271e+09 dense elements stored
Level 9
Level 8


M = 12611, geom = 1
Omega = 1, Leaf-size = 128, admis_precon = 1, rank = 128, leveled rank = 0, epsilon = 1e-08, theta = 1, inner iter = 10, max iter = 50
N = 25218, Leaf = 128, Levels = 8, #Leafs = 256, #Cells = 511
Elements per leaf: 98
Total size (all processes) on level 8: 4.20849e+09 bytes   
Total size (all processes) on level 7: 8.75392e+09 bytes
Total size (all processes) on level 6: 1.08866e+10 bytes
Total size (all processes) on level 5: 8.91413e+09 bytes
Total size (all processes) on level 4: 3.6741e+09 bytes
Total size (all processes) on level 3: 2.26154e+09 bytes
Total size (all processes) on level 2: 1.16954e+09 bytes
Total size (all processes) on level 1: 0 bytes   
precon:
Total size (all processes) on level 8: 3.20978e+09 bytes
Total size (all processes) on level 7: 1.573e+09 bytes
Total size (all processes) on level 6: 8.96532e+08 bytes
Total size (all processes) on level 5: 4.88112e+08 bytes
Total size (all processes) on level 4: 2.38289e+08 bytes
Total size (all processes) on level 3: 1.81142e+08 bytes
Total size (all processes) on level 2: 1.05251e+08 bytes
Total size (all processes) on level 1: 5.47226e+07 bytes
Total size (all processes) on level 0: 2.52314e+07 bytes


M = 50034, geom = 1
Omega = 1, Leaf-size = 128, admis_precon = 1, rank = 128, leveled rank = 10, epsilon = 1e-08, theta = 1, inner iter = 10, max iter = 50
N = 100064, Leaf = 128, Levels = 10, #Leafs = 1024, #Cells = 2047
Elements per leaf: 97
Total size (all processes) on level 10: 1.38984e+10 bytes
Total size (all processes) on level 8: 3.04934e+10 bytes
Total size (all processes) on level 7: 3.58518e+10 bytes
Total size (all processes) on level 6: 3.74175e+10 bytes
Total size (all processes) on level 5: 2.76993e+10 bytes
Total size (all processes) on level 4: 8.93933e+09 bytes
Total size (all processes) on level 3: 3.55095e+09 bytes
Total size (all processes) on level 2: 1.63147e+09 bytes
