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
 - Add data reduct - random sampling DONE
   - add random sampling to
     - initialization DONE
     - bottom up sweep DONE
     - top down sweep DONE
 - Add data reduct - farthest point sampling? other techniques
   - farthest point sampling (only at the leaf level in the bottom up sweep) seems to perform better than the using the original matrix
     - check whether you made any errors 
       - I found one bug, I did not pass the right offsets in the points array 
         (updated the results below after I fixed it)
       - with >= 3 points selected on the leaf level, it seems to perform well for the smallest matrix size (only 1 or two points need slightly more iterations each)
         - so it seems to behave as expected
     - check the behavior if you add it throught all of the bottom up sweep
       - we need a slightly larger sample to get good results if we reduce throughout all of the
         bottom up sweep.
         - for the smallest matric sizes >= 5 points yielded good results
     - check the behavior if added to the top down sweep
       - I am not sure yet that this is error free, the behavior is not exactly as expected
         - I tried to manually inspect the points, but could not spot anything wrong
         - However, the construction error outperforms the full matrix, so something should be wrong?
       - but at least the results seem to be good, even for larger matrix sizes (with constant r1, r2)
 - Added grid point sampling
   - basically I'm distributing r points uniformly on the surface of a unit sphere
   - then the sample set consists of the r point where ri is the closest point to grid point ri
     - Sometimes a point could be selected multiple times, so I do not consider previously selected points'
       in the distance calculation
   - I could try sampling from inside a sphere instead of the surface DONE
     - performance seems to be somewhat comparable

Sparse SVD approximation
 - Encode Far field as sparse matrix - DONE
   - But I fail to see how this could be an order N algorithm
   - Interestingly, if I decrease the threshold to 1e-1, it takes one less outer GMRES iteration
 - randomized SVD to create basis
   - compute basis from randomized SVD with the dense far field
     - implemented, but there are a couple of thins I am unsure about
       - I replaced the whole ID with the V basis
         - this means the basis is already orthogonal
         - but it also means we do not actually sample the matrix anymore, so how do I compute the non-leaf level bases?
     - changed the implementation to compute a randomized ID instead, and now it works

Matrix creation from coordinates
 - Look at Matsumoto-sensei's code for creating the matrix entries from the points
 - Integrate it into the code (similar to how the kernel function was previously integrated)
 - how does it work with the scaling factor?
   - you could get the diagonal values from Matsumoto-senseis code and use them for scaling (complex values, use abs() and then sqrt())
 - try to run it on MPI
 - I see two options to try to tackle the matrix creation:
   - from coordinates as specified above
     - needs to somehow do the scaling during creation 
   - saved to a file (might become large)
     - can save scaled version

TODO
 - update the mesh reading functions - DONE
 - use Matsumoto-sensei's code to create the matrix and the right hand side
   - compare them to the data from file


Experiments
 - For now focus on construction
 - Create 2 matrices, one for factorization on for matvec (target 1e-10 to 1e-12)
 - iterate to the same residual size, independent of the matrix size
   - for a finer mesh GMRES iterations increase because the matrix becomes more ill-conditioned
   - it would be ideal if we can maintain a constant number of iterations for finer meshes


 Matrix valued ACA?
   - might converge too slowly -> ng

 Clearify further goals