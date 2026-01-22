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
                    -> I think we only needed that for HiDR, so I removed it
                  - cleaned up most of the print outs, what is left is to make the selection of the matrix a cmd line argument DONE
                  - Started doing experiments, my idea is to test the different matrix sizes for omegas from 1-50
                    - 160 DONE
                    - 634 DONE
                    - 2530
                  - In the meantime, I want to clean up the tree construction, so that we only need the fortran mesh from now on
                    - I think I found a way of doing it without reordering the actual points, but I still need to test it
                      - It doesn't work because get bounds accesses the nodes sequentially
                      - fixing the get_bound function to use the indices solved the problem
                    - DONE, removed everything that referenced the non-fortran mesh from the code

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
    - I downloaded their remplate and made a sample project
    - I did not find any page limit, but most articles in the last two issues have 10-12 pages, although I also saw up to 20
  - what will we write about
    - I don't really understand the physics, so I'd love to limit that discussion to the introduction/background
    - focus on the H2 preconditioner instead
      - construction:
        - making two trees and fusing them
      - factorization
        - selecting the corresponding matrix rowsin the upper levels?
      - investigations:
        - which tree/sorting works best
        - restarted vs non-restarted GMRES?
        - omega/leaf-size/rank?
        - comparison to HSS/other preconditioners
        - other geometries?
        - multi-node implementation?

Further experiments:
  - add growing ranks for larger matrices
  - 32 * 3 = 96 = 32
  One Leaf 32 x 32 nodes = 96 x 96 matrix
  - increase rank to 40 or more
  - timings for matrix assembly vs construction
  - omega: set 10 as the limit

This is my current idea for making this application use MPI:
  - we cannot construct the dense matrix for all processes, instead, we follow the partition of the leaf level
    - i.e. each process only constructs the rows it is responsible for and then creates only the local h-matrix approx
    - What do we need?
      - all processes need the complete nodes and elements list -> we can just read them from the file
      - We partition the points locally, but how exactly?
      - Also, currently we construct the H2 from the lower triangle, we would need to change that to the upper triangle
      - I don't really understand the current communicator
        - First, change the code so that each H2 matrix on the leaf level stores it's corresponding rows
        - construction needs to be changed to use the rows instead of the columns
        - we need to communicate the selected rows from the child to the parent (if they are not on the same node)
        - factorization and preconditioning should inherently work
        - we need to calculate the matrix vector product inside GMRES on each node locally
        - it seems that everything else is already local, although I don't really understand it
      FIRST STEP
        - construct the matrix on each leaf level and check if we can broadcast them together
          - NOTE: even if alpha is not used in the C code, it needs to be set in Fortran? not sure
          - I rewrote the matrix generator so that it now saves all the nodes and elements
            -> construction of the rhs seems to work, but there is an error for the matrix
              - all matrix elements seem to be zero, I need to dig deeper
              - found the issue, the 'analysis_condition' file is only read after the mesh, so I needed
                to set mu0 and mu1 later in the code
        - actually the first step should be to build row bases instead of column bases and test
          - DONE, for the small matrix, this actually achieved slightly better accuracy
        - constructed the matrix for each cell on the leaf-level in a row-wise fashion
        - wrote a method to do a dense matrix-vector with the matrices from the leaf level
          - seems to work
          - need to check copying back of the result
            - checked, the function currently takes a vector of local length and broadcasts it before calculating only the local part of the restul
              -> not sure if this is the best version
          - need to test in a multi-process environment
            - doesn't work and I don't understand why
              - it seems that the neighbor broadcast doesn't do anything
              - forthermore, it seems that xlen is only the number of cells on this level for one process, so we never allocate enough storage?
              - all of this currently does not make much sense to me
                - maybe the new tree breaks something in the communicator
                - ask Ma about details tomorrow
                  - all reduce change to bitwise OR (future reference)
                - it seems the real issue was that I was using HSS and the communicator was not working as expected
                  because there is no neighbor communication happening in HSS
                  - the neighbor communicator for HSS is on the first level, which makes sense, we actually don't need to communicate
                    on the lower level, we just communicate on the level where the data is split into different processes, but how to find that level?
              - Managed to get the matrix-vector multiplication to work
                - I'm questioning whether it would make more sense to allocate all Matrices on a single node into the same matrix
              - Upper level far fields
                - S_ind just stores the indices for each row in a cell (globally numbered)
                - When building the basis, we permute the indices, so that the selected ones are in front

Updates from the 9/15 meeting:
  - it seems that my idea of re-using the previously created matrix to extract the far field was not
    considered favourably
  - instead, the idea was to create the matrix at each level from scratch, using the indices from the    previous level
  - additionally, it seemed like the consens was to use the whole far field instead of the sampled one (which I think is going to introduce some heavy computations)
  - I think this approach should be considerably easier to implement
    - Write a wrapper function to generate the matrix from matrix indices instead of node/element indices DONE
      - the whole matrix is correct
        - the matrix was only correct because it was being created sequentially, I found another bug and it is fixed now
      - The sequence of S_ind now also seems to be correct
      - I get the same results for construction, factorization and GMRES solve as with the previous full matrix version
    - use this to do a full construction/factorization/gmres run
      - Construction DONE
      - Factorization DONE
      - GMRES
        - with full matrix DONE
        - with distributed matvec DONE
      - everythin works now on a single node
      - distributed execution leads to discrepancies
        - construction error does not match
          - checked that the reference matvec has the same norm on 1 and 2 processes
          - How to check where the distributed construction goes wrong?
          - Works now for the 1 level case (the passed in vectors were wrong)
            - unless I oversuscribe (i.e. more processes than leaf nodes), then the allgatherv crashes
          - For the 2 level case it still breaks, so I need to check that
            - Fixed that, the loop to select the far field was running only over the local nodes
            - It works, but not for HSS, because there the tree is clearly split on the lower level and not all processes are in the communicator
              - each cell stores the begin and end locations (local bodies) so I know which points to exclude
    - think about optimizations only after this is done

    - The current meshes are all unit spheres
      - If I want to increase the number of unknowns, I can just use translation to add more spheres

TODO:
  - test for multiple spheres
    - currently it does not seem to work for multiple spheres (i.e doesn't converge)
  - serialize the matrix
    - tested for the small matrix, seems to work
    - but not sure if it actually saves memory?
    - at least it seems to match what I calculated
  - change to row major?
    - Mat is now read and stored in row major order!
    - that should actually be all that is needed
    - I could optimize some calls now if multiple leafs are stored on the same node (i.e. remove loops)
    - add caching to generation of A and test -> DONE
  - check my mesh generation script if it adds the .inp postfix
  - run tests for larger matrices using Tsubame
    - compile on tsubame
    - how to best serialize the matrix on multiple processes?
      - MPI_File_write_at and reat_at?
      - basically, I want an application to just serialize a matrix with as many processes as possible (i.e. as fast as possible)
      - the experiments should then use that serialized matrix
      - after completing the experiments we delete the matrix again
      - I think the writing only makes sense if it is row major, otherwise I would have to read non-consecutive chunks of memory
      -> changing to row major should be the first priority DONE
      - So I create an executable that generates the dense matrix on n processes
        - each process creates total/n rows and writes them to a file
        - but what about the scale factor - we get it from the diagonal, so it's okay
        - we can also serialize it with the matrix
        - but the nodes have to be sorted ...
          - we can do that as long as I know the leaf size
        - the create_matrix_sorted function has not really been written to support arbitrary splits,
          it currently only supports splits entirely in the nodes or element dimension
          - to support arbitrary splits -> DONE but not tested
            - if start < num_nodes write all the rows until either num_rows or num_nodes run out
              - if num_rows runs out -> we are finished and can return
              if num_nodes runs out -> we need to continue to the elements
            if start > num_nodes -> just write elements until nrows (should be safe)
          - how to test the writing?
          - I read and compared the matrix and it seems to work, next check for multiple processes

  - paper from here https://arxiv.org/pdf/2509.19986

  - two sphere mesh
    - Matsumoto's code - mesh size is insufficient -> try to increase the number of elements (around 10 000)
    - verify that the Dense LU actually converges (i.e. condition number is not too large, Matsumoto sensei says two spheres should not be too ill conditioned) -> Dense LU converges fast
      - find out if the problem is on the numerical side or the physics side
    - Use group jh240021
    - Check if there are no duplicates in the mesh! -> Matsumoto-san's code removes duplicates in remove_dn_sort_nn_non_global
    - Future: more complex geometry (not a toy problem)
      - not a regular shape, but still a structured mesh (e.g. submarine, twisted cylinder, flower petal, ...)


- Creating the matrix does not work for arbitrary processes
 - each process gets a starting row and a number of rows to write (target matrix[0-num_rows]) reading from [start-start+num_rows]
 - if start < num_nodes
   write starting from 0 until either num_nodes have been written or start+rows_written is larger than num_nodes
 - calculate remaining rows as num_rows - rows_written
 - write the remaining rows starting from rows_written until num_rows
 -> DONE this seems to work now

 Mesh suggestions:
  - submarine hull
  - other papers for elastic wave scattering / elastodynamics

Two scatterers:
- send screenshots of geometry
- check original ordering
- compute singular vectors (should be the same for symmetric matrix) U*VT 
  - one sphere vs two sphere
  - where is the major error coming from
 - try with k=100 to k=1100 (for single and double sphere)


My current goal is to run large scale experiments with the salt model on tsubame
  - if I get a solution in sufficient time -> publish
  - otherwise I need to invest more time

Store matrix in /gs/fs/jh240021/thomas

fix the current bug in writing the salt model
 - the error seems to come from somewhere within the fortran code
 - it seems to be first caused by node 72
 - my current assumption is that there is some problem with the data for that node
 - I can use the show() function to print that data and maybe compar it?
   - It seems the max number of elements that a node can be part of is hardcoded. I can increase the number, but there is now easy way to know if it was enough. Therefore I wrote a simple check() function, that checks for each node whether it is part of more elements than allowed
   - I had to double the number to make it work
check the rhs for the sphere case
   - writing the RHS and matrix seems to work for 8 processes and the sphere
 - Now I am trying to write the 50k salt matrix again
    -> it worked with 8 processes on the lab server (50k problem)
now we need to read from the file
  - read the rhs (for each process)
  - read the matrix rows on the leaf level (for each process)
  - read or recalculate the rows on the upper level

  Ideas for reading:
    - provide the matrix generator with the filename and then get the data from there instead of the fortran code
    - the matrix has already been sorted and scaled
      - read the scale from the file
    - when is the matge called in the construction?
      - get the number of nodes/elements
      - get the dense matrices on the leaf level
      - gen_matrix_sorted(Mat[i], start, num_rows, omega scale (reads consecutively in multiples of three) DONE
      - gen_matrix_idx_element(Far, S_ind, F_ind)
        - gets the matrix from the nodes/elements indices
        - could leave this for later
        - reading from file DONE
      - gen_matrix_element(S_ind, M, S_ind)
        - this generates the matrix from nodes/elements indices
        - file read would have to be one by one, so it's probably not worth it
        - I'll leave this for later (at least for now)

when reading the file it just seems to be all zeroes
-> Found the issue and FIXED it
- read the right hand side and challenge the salt model
  -> Read rhs DONE

Test just LU factorization of the SALT model
try torus (donut) shape
   

Current research summary:
Ma's version:
  - failed convergence is primarily coming from the non-symmetric properties of the low-rank components
  - even though the ISC paper produces very large errors, it works when coupled with a Krylov solver because the hierarchical decomposition decomposes the matrix into different spectrums, so that the Krylov iterations can effectively reduce the residual effectively (similar to a multi-grid fashion)
  - the newer geometries are not working because of the limitations of the implementation that requires the per level H2 matrix to be formulated in a numerically symmetric way, but Matsumoto-sensei' code does not meet that criterion
  - we conducted some experiments on the dense matrices and it seems only the sphere has an acceptable symmetric property which enables the sphere to run
  - so we either need active development to drop the assumption on complete numerical symmetry during factorization or to not use the double-layer or hypersingular potential (the latter makes the BEM less convincing to readers)
- Ma spends about half a way per week to work on the non-symmetric parts, but the H2 currently does not converge as good as HSS (even on a single node)
  - he switched the ID to SVD so HSS converges much faster, but it degrades the H2 performance, but we don't know why
- option two: reduce the problem complexity (will make the paper less convincing)

His suggestions:
- sparse matrix like LoRaSP, because many of them have good numerical symmetry
- dig more on the ISC theory part, going more into the applied math part, trying to get more info on convergence properties, limitations, etc. 
- H2 multigrid is based on symmetric matrices so it is going to be more challenging, 
  - this means for multigrid we need SPD matrices, i.e. symmetric
  - Helmholtz single layer potential is not SPD, but it seems that it is still able to extract the lower frequencies

    complex geometry - single layer potential / let us change the problem util it works
  look for a matrix that we can solve for a complex geometry 
  it is already novel even if we do single layer potential and make the matrix symmetric, because the method is novel

  Try single layer potential:
   - standard graph creation with splitting (storing the indices)
     - DONE, but can't really test
   - modified kernel functions
     - too many functions, which ones should I modify?
     - I think the best way to do this is still over the file,
       so for now I only modify the file reading/writing functions
   - seems to be working now

Experiments:
 Steps:
   1) save the matrix to file using write.app
     - fully parallel
     - takes the following arguments
       - number of nodes (the number in the filename)
       - geometriy
         1 - single sphere
         2 - two spheres
         3 - salt model
         4 - four spheres
         5 - torus
         8 - eight spheres
       - omega
       - leaf_size (for the ordering)
   2) run the solver (main.app)
     - takes the following arguments
        - number of nodes (number in the filename)
        - geometry (see above)
        - omega
        - leaf-size
        - theta (admissibility condition)
        - rank
        - leveled_rank
        - accurcy until which to iterate
        - number of inner GMRES iterations
        - max number of outer GMRES iterations

Salt model is very large, but we use unit length wavenumber -> for large mesh, this results in a high frequency problem
 so scaling down the mesh helps with this issue

 check tsubame schedule

 experiments
  - no preconditioner
  - HSS
  - matrices get more ill-conditioned as the size increases, so more GMRES iterations are necessary -> confirm this behavior
  - with good preconditioner we expect almost same iteration number between small DOF and large DOF

tested the 12611 file (leaf size = 32), it does not converge if we don't allow for level growth (used 16 in the end)

Upscaling on Tsubame:
- we have 10TB of storage on the HDD filesystem /gs/bs/
- The 50034 matrix has 100 000 elements, i.e. 3e5 * 3e5 * 16 = 1.44e12, estimated 1.44 TB
 - final filesize 1.343 TB
 - this matrix has 300 000 degrees of freedom
 - was able to run the solver on this matrix using 8 nodes and 40 processes per node, however, it converged very slowly
  - settings: leaf = 128, rank=100, leveld_rank = 10, iters=10/50
- Using the same assessment, 1M DOFs would take 1e6 * 1e6 * 16 = 16e12, estimated 16 TB, but we only have 10 TB storage
 - what is the largest matrix I can realistically store? - 750 000 DOFs
 - everything else I would need to calculate on-the-fly
 - the largest file I currently have is ~600 000
 - 

Current matrices:
         Nodes   Elements       DOFs
sphere:    160        316        948
           568       1132       3396
          1489       2974       8922
         12611      25218      75654
         50034     100064     300192
         77751     155498     466494
        138201     276398     829194
        157772     315540     946620
        169798     339592    1018776
        198027     396050    1188150 
        309365     618726    1856178     

salt:      334        664       1992       1k
          3328       6652      19956      10k
          6644      13284      39852      20k
         16541      33078      99234      50k
         32959      65914     197742     100k
         65357     130710     392130     200k
         97196     194338     583014     300k
        130012     260020     780060     400k
        162352     324700     974100     500k

Basically I need to find settings where the salt model converges with 1M DOFs
 - I tried to get the sphere to converge with 1M DOFs first
   - tried to do it iteratively, but it is just too slow
   - also, from the ISC paper, it seems there is not necessarily a correlation between
   - rank and convergence
 - I want to do more experiments and I want to do them faster/in parallel
   - reduce printing of the output to only the essentials
   - integrate all necessary information in th output to be able to reconstruct the settings
   - also, creating the matrix first is not going to work for 1M DOFs, so I need code
     that create the matrix from scratch again

Writing the paper:
- wrote the introduction, but it is currently mostly ripped off
- did not find much related work
- copied the formulations from Matsumoto-sensei's notes
- don't really know how to continue from here
  - continue ripping off Matsumot-sensei's paper
  - introduce H^2 matrices (reuse previous papers)

#pages in previous articles
28
16
18
18
16
20
28
27
16

Frequencies in the 1
if diameter is d and wavenumber is k we should fix k/d
if we use a 100 times larger mesh, we use 10times the wavenumber
keep wavenumber constant first, if it works try to increase the wavenumber

- check if we can get a good accuracy for the H2 on the smaller problem sizes
- clarify the ordering in the paper (we don't exactly use the ordering from Matsumoto senseis formula) -> still needs to be done, but I want to wait for results first
- use the sphere and just keep increasing the number of points

- fixed the near field kernels to not use the dense matrix anymore
  - still need to fix the far field on the leaf-level for H2
- still need to H2 fr field on the upper levels

TODO's
 - check that the ranks decrease and the accuracy stays stable for H2 with strong admissibility
   - increase admis condition DONE
   - multi-node
     - construction error does not seem to be exactly the same for multiple nodes, but that might be summation order
   - completely remove the storage for the dense matrix
     - removed from the H-matrix, however now I'm missing the dense matvec - FIXED
     - did not remove completely (for testing reasons), but don't allocate it anymore
 - find a way to calculate the total memory consumption
   - calculate the number of elements stored DONE
 - test H2 with strong admis for large scale problems
   - currently in queue

 - enable the full solver with 2 Hmatrices
   - Got it working, but there seems to still be a problem when calculating the accurate H2-matrix on multiple nodes (accuracy degrades)
   - it seems the far field on the upper levels was not calculated accurately on multiple nodes
     - in line 1015 it did not use the correct offset into the cell array
     - it seems I was on the right track, I managed to fix/mitigate the issue for two processes bot for more it is still off. At least I know where to look now
     - the HSS also seemed to be off
     - it seems it is impossible to know the entire far field on a node (since the tree might be split further up)
       - maybe I can construct the far field by excluding the near field (just like I did in the leaf level case)
       this seems to have worked!
       - just noticed that this fix never built a factorization basis, so the preconditioner was off?
         - that seems to be correct, the preconditioner with the factorization basis has a larger construction error, but a smaller factorization error
        - made a stupid mistake when fixing this, the first near field cell != diagonal cell
          - cleaned up the code for the leaf-level too
          - after that the results for N=160 were slightly better?
          - on multi-nodes, now I get different storage but identical results?
            - the storage for the bases is not accurate, since it does not account for splitting of the tree, in which case not all the levels are stored on a node, should probably use a neighbor communication instead
          - turns out I stored the wrong results, so everything was alright


 - benchmark that solver
 - HiDR
 - single precision 

 Writing the paper:
   - Ma said to aim for 20 pages
   - should we add th ULV factorization algorithm
     - I kind of don't want to repeat it
     - we could explain the factorization basis in more detail than
       in the last paper, along with solves and matvecs
       - I want to introduce the notation and do a very quick rundown of the ULV-algorithm and explain it's problems with strong admissibility
       - or maybe give the rundown of the factorization basis directly

       - try to summarize notation and dense block break down into a single picture and add to background DONE
       - then in methods, explain the factorization basis with a picture o f both HSS and H2 - made the graphic, still need the text
   - Methods
     - how do we deal with matrix valued functions
       - creation
       - factorization
       - solves
       - GMRES
     - how to distribute

Next meeting:
  - confirm with Ma if we use ID or rank revealing QR
  - get Ma to give me an explanation of the factorization algorithm


my current idea is to see how far I can go with a single node on tsubame and
then slowely increase the node count from there to find the optimal setting
  Current matrices:
         Nodes   Elements       DOFs
sphere:    160        316        948
           568       1132       3396
          1489       2974       8922
         12611      25218      75654 DONE
         50034     100064     300192 -> currently trying this 
         77751     155498     466494
        138201     276398     829194
        157772     315540     946620
        169798     339592    1018776
        198027     396050    1188150 
        309365     618726    1856178     

IMPORTANT:
 - before I touch the implementation again, I want to be able
   to run a large scale problem
