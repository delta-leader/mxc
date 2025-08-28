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
   - I need to read the nodals and elements in their original Fortran format because that's what the matrix creation expects as input
   - but then I don't have the coordinates available for sorting in C++
     - I guess the nodel_point struct has some of the information I am lookin for?
       - the xc field stores to coordinates of the nodes
       - I believe the xc field of the elements also stores the center of the element
    - I wrote functions to read the Nodes and elements from the fortran code, but the fortran is currently
      not built when compiling the project I need to add the compilation -> FIXED
   - compare them to the data from file
     - Verfied that the xc field stores the coordinates for the nodes
     - verified that the xc field stores the coordinates of the center for the elements
   - tree creation
     - we can freely re-order the nodes and elements inside their respective vectors
     - write this tree creation (only fused trees make sense) method and compare
       the order to the index shuffle you already have
       - works for the nodes tree
       - works for the element tree
     - added a boolean value that indicates if a cell stores nodes or elements
   - H2 creation
     - CSR should be fine as it does not access the Bodies information
     - we copy the local bodies into S [which I could do as long as I know the offset]
     - we generate the marrices from the bodies
     - we don't access the bodies anywhere else in the original code
     - first we compare the created dense matrix and right-hand side
       - note that the stored matrices/rhs use the setting omega=1
       - Matsumoto sensei's code groups the x,y and z elements together instead
         - would that be advantageous for any reason?
         - actually, he has both versions
       - RHS works (without shuffling), Diff is 1.03617e-15
         - the sorting seems to be messed up - is it different between the two versions?
           - what could be different?
             - check level by level, i.e. modify Nleaf
             - I verified the sorting of the nodes by checking the indices and it is the same
             - I think the problem is that the nodes reference the elems by their indices (and vice versa),
               however, those indices change by re-ordering
               - if we just reorder the nodes, it should be fine though no?
               - try to genterate the matrix when switching just two nodes
               - The matrix does not seem to work (without shuffling)?
                 - the freaking problem was that I used the wrong omega (2 instead of 1)
                   - however, even with the right omega, the error seems kind of large (1e-11)
                   - there is still something wrong, maybe it is the column vs row major?
                     - that doesn't seem to be the problem, maybe I just need more accuracy?
                     - I should compare to matsumoto-sensei's code directly
                       - it seems the differences are due to this, when comparing to Matsumoto-sensei's code they are identical
                   - comparing to Matsumoto sensei's code, the results are identical, 
                     so unshuffled results are fine
               - I can generate the sorted matrix, however, it would be better to pass two index arrays, one for the nodes and one for the elements
                 -> DONE
               - Next, how to handle the scaling?  
                 - Wrote some functions to get the max diagonal value of both elements and nodes
                   -> tested and seems to work
                 - I integrated the scaling into the matrix construction and it seems to work,
                   the difference is < 1e-16
                - So, the way to do the scaling is to solve the system SAS y = Sb and then recover x = Sy
                - Next steps: reordering and scaling of the right hand side DONE
                - Next steps: clean up the code and start experiments
                  - the matrix solver currently takes the all_sorted index array, what is this used for?

               - it seems the matrix construction is currently not using OpenMP
     - pass the new cell array in addition to the old code and compare the far matrices that are created
       - for the leaf level
         - we construct the far field matrices from alist of xnodes, xelems and ynodes , yelems

Paper:
 - PMCW calderon preconditionioning does not work
 - piecewise linear basis for both displacement and traction works, but mixed basis strategy the calderon preconditioner 
 - dense linear solver -> natural but not sure if it is used broadely or not
 - show that no preconditioner needs many more iterations.
 - save the updated formulation for next time
 - make an overleaf project and share 



Experiments
 - For now focus on construction
 - Create 2 matrices, one for factorization on for matvec (target 1e-10 to 1e-12)
 - iterate to the same residual size, independent of the matrix size
   - for a finer mesh GMRES iterations increase because the matrix becomes more ill-conditioned
   - it would be ideal if we can maintain a constant number of iterations for finer meshes


 Matrix valued ACA?
   - might converge too slowly -> ng

 Clearify further goals

 Next meeting:
 - test the HiDR approach with the previous kernel matrices (Helmholtz) to confirm that we
   can achieve good accuracy even with constant sample points - test high accuracy settings
   - compare to sparse - for Frobenius norm constant threshold should be okay
     -> switch to Infinity norm

Ideas for our contribution
  - first hierarchical preconditioner for this problem
  - compare to other common preconditioners for this problem
    - without preconditioner - analytical preconditioner cannot be applied because the linear/constant bases are mixed (HPC preconditioner would already be novel)
  - submit to Computer Physics Communication