Sample points:
 - I managed to reconstruct the QR with full rank and a resulting error of 1e-16
   - but I cannot use this as an error measurement
   - Ma proposed to use matvec instead
     - implemented the matvec and when using full rank (48), we get an error of 1e-16
     - this doesn't really solve my problem, since the sample still has smaller dimensions
     - I probably didn't finish this yet, what about the part after ortho?

Sparsity:
 - check if there are many small elements
   - the scaling seems to take the majority of the compute
   - is the number of significant elemtns per row constant?
   - Take the norm per row
     - verify sparsity for HSS and H2
     - we do not include the dense parts in the row norm



Sample points:
The WSA basically took the role of identifying the bodies of the far field.
For this the WSA had the same level structure as the H2matrix and at each level wsa(i) would return the far field of the ith node (basically, it stored a vector of vectors)

HiDR
Xi is the set of points corresponding to node i
Yi is the farfield of node i

Bottom-up sweep:
  -Leaf nodes already contain O(1) points so Xi = Xi*, but we could also select less
  - Parent node Sp = union of Xi* from all the children Xp* = DataReduct(Sp)
Top-down sweep:
  - starting from the top node in the tree with a non-empty interaction list
  - Ti = union Yp (p denotes the parent of i) and Xj* for all nodes j in the interaction list of i
  - Yi* = DataReduct(Ti)

What is the interaction list?
For each node i, the interaction list of i consists of nodes j such that Xj is well-separated from Xi, but for the parent p of j, Xp is not well separeated from Xi

Implement this in a similar datastructure to WSA (but we just need the index sets for now).
Each node, needs to store both X* and Y*
I can just parse the tree after it has been fused
for now, I just implement an empty DataReduct method that returns the same set, i.e. it will be the same as the current implementation


TODO:
 - today finish Ma's requests for sparsity -> DONE, remember that the density is calculated based on the #rows, so less rows can lead to less denisty
   - I'm not sure if those results are good or bad, but it seems about 2/3 of the values are small(ish)
   - Actually I could not run this for larger matrices, so I need to debug
 - add data structure for storing the reduced sets DONE
 - implement bottom up sweep DONE
 - implement the top down swep DONE
 - Testing debugging (without data reduct) it seems to work, the far field indices should be stored correctly and in order
   - add a construction function that takes the hidr-structure as an input an reorders the matrix according to the indices
     - comment out the density output
     - debug the current main 
       - it failed because I was trying to factorize a matrix with fixed accuracy instead of fixed rank
       - now it is all fine again
     - test with hidr only for the upper levels
       - fails for fixed epsilon
       - the middle levels are not the same, because they are built from the ranks of the lower levels
     - test with hidr for the leaf levels DONE
 - Add data reduct - random sampling 19/7
 - Add data reduct - farthest point sampling? other techniques

