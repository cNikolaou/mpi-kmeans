#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kmeans.h"
#include "cluster.h"
#include  <time.h>
#include <sys/time.h>
#include "mpi.h"


#define max_iterations 50


void error_message(){

char *help = "Error using kmeans: Three arguments required\n"
  "First: number of elements\n"
  "Second: number of attributes (dimensions)\n"
  "Third: numder of clusters\n";

  printf(help);

}

void random_initialization(data_struct *data_in){

  int i, j = 0;
  int n = data_in->leading_dim;
  int m = data_in->secondary_dim;
  double *tmp_dataset = data_in->dataset;
  unsigned int *tmp_Index = data_in->members;


  //srand(time(NULL)); // generate different random numbers
   srand(0); // generate the same random numbers on every run
  // random floating points [0 1]
  for(i=0; i<m; i++){
    tmp_Index[i] = 0;
    for(j=0; j<n; j++){
      tmp_dataset[i*n + j] = (double) rand() / RAND_MAX; 
    }
  }

}


void initialize_clusters(data_struct *data_in,data_struct *cluster_in){

  int i, j, pick = 0;
  int n = cluster_in->leading_dim;
  int m = cluster_in->secondary_dim;
  int Objects = data_in->secondary_dim;
  double *tmp_Centroids = cluster_in->dataset;
  double *tmp_dataset = data_in->dataset;
  unsigned int *tmp_Sizes = data_in->members;

  int step = Objects / m;

  /*randomly pick initial cluster centers*/
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      tmp_Centroids[i*n + j] = tmp_dataset[pick * n + j];
    }
    pick += step; 
  }

}

void print(data_struct* data2print){

  int i, j = 0;
  int n = data2print->leading_dim;
  int m = data2print->secondary_dim;
  double *tmp_dataset = data2print->dataset;

  
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      printf("%f ", tmp_dataset[i*n + j]);
    }
    printf("\n");
  }
  
}


void save(data_struct* data2save, char *filename1, char *filename2){

  int i, j = 0;
  FILE *outfile;
  int n = data2save->leading_dim;
  int m = data2save->secondary_dim;
  double *tmp_dataset = data2save->dataset;
  unsigned int *tmp_members = data2save->members;

  printf("Saving data to files: "); printf(filename1); printf(" and "); printf(filename2); printf("\n");

  /*===========Save to file 1===========*/
  if((outfile=fopen(filename1, "wb")) == NULL){
    printf("Can't open output file\n");
  }

  fwrite(tmp_dataset, sizeof(double), m*n, outfile);

  fclose(outfile);

  /*===========Save to file 2========*/

  if((outfile=fopen(filename2, "wb")) == NULL){
    printf("Can't open output file\n");
  }

  fwrite(tmp_members, sizeof(unsigned int), m, outfile);

  fclose(outfile);

}

void clean(data_struct* data1){

  free(data1->dataset);
  free(data1->members);
}

int main(int argc, char **argv){
	
	// Pass arguments to MPI procceses 	
  MPI_Init(&argc, &argv);

  struct timeval first, second, lapsed;
  struct timezone tzp;

  if(argc<4){
    error_message();
    return 0;
    //printf("Error using kmeans: Three arguments required\n");
  }

  int numObjects = atoi(argv[1]);
  int numAttributes = atoi(argv[2]);
  int numClusters = atoi(argv[3]);
  int i =0 ;

  char *file1_0 = "centroidsP.bin";
  char *file1_1 = "ClusterSizeP.bin";
  char *file2_0 = "datasetP.bin";
  char *file2_1 = "IndexP.bin"; 

  data_struct data_in;
  data_struct clusters;

  /*=======Initialize processes=======*/
  int rank, NumOfTasks;
  
  MPI_Comm_size(MPI_COMM_WORLD, &NumOfTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int NumOfSlaves = NumOfTasks-1;
  
  // Compute (floor) number of objects per task
  unsigned int parNumObjects;
  parNumObjects = numObjects/NumOfSlaves;
  
  // number of object that will not be included
  // if all processes get parNumObjects objects
  int remain = numObjects - parNumObjects*NumOfSlaves;

  // compute final number of objects per task
  int procNumObjects[NumOfTasks];
  procNumObjects[0] = numObjects;

  for (i=1; i<NumOfTasks; i++) {

    if(i<=remain)
      procNumObjects[i] = parNumObjects+1;
    else
      procNumObjects[i] = parNumObjects; 
  }

  /*=======Memory Allocation=========*/

  // Allocate the appropriate memory
  if (rank == 0) {

    data_in.leading_dim = numAttributes;
    data_in.secondary_dim = numObjects;
    data_in.dataset = (double*)malloc(numObjects*numAttributes*sizeof(double));
    data_in.members = (unsigned int*)malloc(numObjects*sizeof(unsigned int));

  } else {
    
    numObjects = procNumObjects[rank];

    data_in.leading_dim = numAttributes;
    data_in.secondary_dim = numObjects;
    data_in.dataset = (double*)malloc(numObjects*numAttributes*sizeof(double));
    data_in.members = (unsigned int*)malloc(numObjects*sizeof(unsigned int));

  }
  
 // printf("Process %i will take %ld number of objects.\n", rank, numObjects);
  

  clusters.leading_dim = numAttributes;
  clusters.secondary_dim = numClusters;
  clusters.dataset = (double*)malloc(numClusters*numAttributes*sizeof(double));
  clusters.members = (unsigned int*)malloc(numClusters*sizeof(unsigned int)); 
  

  /*=============Get Dataset==========*/

  if (rank==0) {
    random_initialization(&data_in);
    initialize_clusters(&data_in, &clusters);
    printf("Data initiallized!\n");
  }

  /*=================================*/


  /*=============Send Dataset============*/


  // send dataset to other process.
  // will only send the appropriate part
  // of the dataset
  int dest;
  double *sendBuf = (double *) malloc(numAttributes*procNumObjects[1]*sizeof(double));
  double *recvBuf = (double *) malloc(numAttributes*procNumObjects[rank]*sizeof(double));

  if (rank==0) {
  
	// holds the start position to copy elements to sendBuf
    int offset = 0;
    for(dest=1; dest<NumOfTasks; dest++) {
     
      int cp;
      for(cp = 0; cp<procNumObjects[dest]*numAttributes; cp++) {
        sendBuf[cp] = data_in.dataset[offset+cp];
      }
	  // update offset
      offset += procNumObjects[dest]*numAttributes;
      
	  // send dataset
      MPI_Send(sendBuf, procNumObjects[dest]*numAttributes, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
    } 
  } else {

    MPI_Status stat;
    
	// Receive dataset
    MPI_Recv(recvBuf, procNumObjects[rank]*numAttributes, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,&stat);

	int cp;
    
	// copy elements from recvB to data_in.dataset array
	for(cp=0; cp<procNumObjects[rank]*numAttributes; cp++) {
		data_in.dataset[cp] = recvBuf[cp];
	}	
  }


  /*===============Cluster Data====================*/
  gettimeofday(&first, &tzp);

  cluster(&data_in, &clusters, max_iterations);


  /*===========Save final cluster indexes========*/

  // Return final cluster indexes of elements from slave processes
  if(rank==0) {
	
    MPI_Status stat;
    int offset = 0;
			
    unsigned int *rBuff = (unsigned int*) malloc(procNumObjects[1]*sizeof(unsigned int));

	for(dest=1; dest<NumOfTasks; dest++) {
	
		// Receive indexes of elements
		MPI_Recv(rBuff, procNumObjects[dest], MPI_UNSIGNED, dest, 10, MPI_COMM_WORLD, &stat);
				
		for(i=0; i<procNumObjects[dest]; i++) { 
			data_in.members[offset+i] = rBuff[i];
		}
					
		offset += procNumObjects[dest];
	}

	free(rBuff);

  } else {
  
	  // Send indexes of elements
	  MPI_Send(data_in.members, procNumObjects[rank], MPI_UNSIGNED, 0, 10, MPI_COMM_WORLD);		
  }

  gettimeofday(&second, &tzp);


  if(rank==0) {
  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("Time elapsed: %d.%06dsec\n", lapsed.tv_sec, lapsed.tv_usec); 

  /*========save data============*/
  save(&clusters, file1_0, file1_1);
  save(&data_in, file2_0, file2_1);

  }

  /*============clean memory===========*/
  clean(&data_in);
  clean(&clusters);

  MPI_Finalize();
}
