mpi-kmeans
==========

Parallel implementation of the k-means algorithm using MPI. Data points are 
created at random.


To run:
-------
1. Install an MPI implementation. (I used [MPICH](http://www.mpich.org/downloads/))
2. Run: mpiexec -n *number_of_processes* ./kmeansTest  *n_data_points*  *n_attributes_per_point* *n_clusters*
