#include "kmeans.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define threshold 0.001
//double threshold = 0.01;

double euclidean_distance(double *v1, double *v2, int length){

  int i = 0;
  double dist = 0;

  for(i=0; i<length; i++){
    dist += (v1[i] - v2[i])*(v1[i] - v2[i]); 
  }

  return(dist);
}


void kmeans_process(data_struct *data_in, data_struct *clusters, double *newCentroids, double* SumOfDist, double *sse){

  int i, j, k;
  double tmp_dist = 0;
  int tmp_index = 0;
  double min_dist = 0;
  double *dataset = data_in->dataset;
  double *centroids = clusters->dataset;
  unsigned int *Index = data_in->members;
  unsigned int *cluster_size = clusters->members;

  //SumOfDist[0] = 0;
    
  for(i=0; i<clusters->secondary_dim; i++){
    cluster_size[i] = 0;
  }

  for(i=0; i<data_in->secondary_dim; i++){
    tmp_dist = 0;
    tmp_index = 0;
    min_dist = FLT_MAX;

    /*find nearest center*/
    for(k=0; k<clusters->secondary_dim; k++){
      tmp_dist = euclidean_distance(dataset+i*data_in->leading_dim, centroids+k*clusters->leading_dim, data_in->leading_dim);
      if(tmp_dist<min_dist){
	      min_dist = tmp_dist;
	      tmp_index = k;
      }
    }
   
    Index[i] = tmp_index;
    SumOfDist[0] += min_dist;
	sse[0] += min_dist*min_dist;
    cluster_size[tmp_index]++;
    
	for(j=0; j<data_in->leading_dim; j++){
      newCentroids[tmp_index * clusters->leading_dim + j] += dataset[i * data_in->leading_dim + j]; 
    }
   
  }

  /*update cluster centers*/
  for(k=0; k<clusters->secondary_dim; k++){
    for(j=0; j<data_in->leading_dim; j++){
      centroids[k * clusters->leading_dim + j] = newCentroids[k * clusters->leading_dim + j];
    }
  }

}



void cluster(data_struct *data_in, data_struct *clusters, int max_iterations){ 

  int iter, i, j, k, dest;
  double SumOfDist = 0, new_SumOfDist = 0, part_SumOfDist, sse = 0, psse;
  double *newCentroids, *partCentroids;
  unsigned int *part_size;
  int endcond = 0;

  
  int rank, NumTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status stat;


  part_size = (unsigned int*) malloc(clusters->secondary_dim*sizeof(unsigned int));
  newCentroids = (double*)malloc(clusters->leading_dim*clusters->secondary_dim*sizeof(double));
  partCentroids = (double*)malloc(clusters->leading_dim*clusters->secondary_dim*sizeof(double));


  for(iter=0; iter<max_iterations; iter++){
						
    // All processes call this function and they split 
    // to follow different execution paths.
		
    if(rank==0) {
	
        new_SumOfDist=0;
	
        // set partial cluster's size array to zero
        // this array is use to sum cluster sizes that
        // are received from the other processes
        for(k=0; k<clusters->secondary_dim; k++) {
				part_size[k] = (unsigned int) 0;
				clusters->members[k] = 0;
			}

         // Alternative way to send clusters' centers
			/*	
	    for(dest = 1; dest<NumTasks; dest++) {
	      MPI_Send(clusters->dataset, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, dest, 2, MPI_COMM_WORLD);
	    }
			*/
	
        // Broadcast the new clusters' centers to other processes 
        MPI_Bcast(clusters->dataset, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	    for(i=0; i<clusters->secondary_dim; i++){
	      for(j=0; j<clusters->leading_dim; j++){
	      	newCentroids[i * clusters->leading_dim + j] = 0;
	      }
	    }

        // Alternative way to receive the partial SumOfDist, computed by the other processes 
		/*
	    for(dest = 1; dest<NumTasks; dest++) {
	      MPI_Recv(&part_SumOfDist, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, &stat);
	      new_SumOfDist += part_SumOfDist;
	    }
		*/

        // Reduce to take the sum of the partially computed sum of distance
        MPI_Reduce(&part_SumOfDist, &new_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			

        // compute the new centroids by summing up the parial new centroids values reported by
        // the other processes
	    /*for(dest = 1; dest<NumTasks; dest++) {
	      MPI_Recv(partCentroids, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, dest, 4, MPI_COMM_WORLD, &stat);
	      
	      for(k=0; k<clusters->secondary_dim; k++){
	        for(j=0; j<clusters->leading_dim; j++){
	          newCentroids[k * clusters->leading_dim + j] += partCentroids[k * clusters->leading_dim + j];
	        }
	      }
	    }
*/
		MPI_Reduce(partCentroids,newCentroids, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	    
        // sum up partial clusters' size. that is received from other processes
	  	for(dest = 1; dest<NumTasks; dest++) {
	
	      MPI_Recv(part_size, clusters->secondary_dim, MPI_UNSIGNED, dest, 5, MPI_COMM_WORLD, &stat);
	
	      for(k=0; k<clusters->secondary_dim; k++){
	          clusters->members[k] += part_size[k];
	      }
	    }


         // compute the new center for each cluster
         for(k=0; k<clusters->secondary_dim; k++) {
           for(j=0; j<clusters->leading_dim; j++) {
             clusters->dataset[k * clusters->leading_dim + j] = newCentroids[k * clusters->leading_dim + j] / (double) clusters->members[k];
           }
         }
	
        // check whether SumOfDist has stabilized
	    if(fabs(SumOfDist - new_SumOfDist)<threshold){
				
          // if yes, then broadcast 1 to inform other processes to exit
          // the loop		
          endcond = 1;
          MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
	      break;

	    } else {
			
				// else broadcast 0
				MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
			
			}
	
	    SumOfDist = new_SumOfDist;
	
	    printf("Sum of Distances of iteration %d: %f\n",iter, new_SumOfDist);

  } else {

        // This is the other path that is followed by all processes except that with rank 0
			
		
        part_SumOfDist = 0;
        psse = 0;
	
			// Another way to receive clusters->dataset, but braodcast is a bit better
			//MPI_Recv(clusters->dataset, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &stat);
	    
			// Receive new clusters' centers from master process
        MPI_Bcast(clusters->dataset, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
			// set new part centroid to zero
	    for(i=0; i<clusters->secondary_dim; i++){
				for(j=0; j<clusters->leading_dim; j++){
					partCentroids[i * clusters->leading_dim + j] = 0;
	      }
	    }
	
        // run kmeans for this part of the dataset
	    kmeans_process(data_in, clusters, partCentroids, &part_SumOfDist, &psse);
	
        // Alternate way to reduce the partial sum of distance
	    // MPI_Send(&part_SumOfDist, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	
        // Reduce to compute total sum of dist on master
        MPI_Reduce(&part_SumOfDist, &new_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Each slave process sends the partial centroids computed
 //       MPI_Send(partCentroids, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);

        MPI_Reduce(partCentroids,newCentroids, clusters->leading_dim*clusters->secondary_dim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);	
        
		// Each slave process sends the partial clusters' sizes
        MPI_Send(clusters->members, clusters->secondary_dim, MPI_UNSIGNED, 0, 5, MPI_COMM_WORLD);


	
        // Each slave process receives the condition whether sum of distance has stabilized
        MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
        if(endcond)
          break;

    }
  }


  if(rank==0) {
		
    // Alternative way to reduce
    /*
    for(dest = 1; dest<NumTasks; dest++) {
	
      MPI_Recv(&psse, 1, MPI_DOUBLE, dest, 11, MPI_COMM_WORLD, &stat);
      sse += psse;
    }
    */

    // Reduce partial computed sse
    MPI_Reduce(&psse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("Finished after %d iterations\n", iter);
    printf("SSE equals to %f\n", sse);

  } else {
    // alternative way to reduce
    //MPI_Send(&sse, 1, MPI_DOUBLE, 0, 11, MPI_COMM_WORLD);


    MPI_Reduce(&psse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

	free(newCentroids);
	free(partCentroids);
	free(part_size);
	
}
