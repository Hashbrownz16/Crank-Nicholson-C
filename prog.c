#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_lapacke.h>

/* Define structure that holds band matrix information */
struct band_mat{
  long ncol;        /* Number of columns in band matrix */
  long nbrows;      /* Number of rows (bands in original matrix) */
  long nbands_up;   /* Number of bands above diagonal */
  long nbands_low;  /* Number of bands below diagonal */
  double *array;    /* Storage for the matrix in banded format */
  /* Internal temporary storage for solving inverse problem */
  long nbrows_inv;  /* Number of rows of inverse matrix */
  double *array_inv;/* Store the inverse if this is generated */
  int *ipiv;        /* Additional inverse information */
};
/* Define a new type band_mat */
typedef struct band_mat band_mat;

int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
  bmat->nbrows = nbands_lower + nbands_upper + 1;
  bmat->ncol   = n_columns;
  bmat->nbands_up = nbands_upper;
  bmat->nbands_low= nbands_lower;
  bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
  bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
  bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
  bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
  if (bmat->array==NULL||bmat->array_inv==NULL) {
    return 0;
  }  
  /* Initialise array to zero */
  long i;
  for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
    bmat->array[i] = 0.0;
  }
  return 1;
};

double *getp(band_mat *bmat, long row, long column) {
  int bandno = bmat->nbands_up + row - column;
  if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
    printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
    exit(1);
  }
  return &bmat->array[bmat->nbrows*column + bandno];
}

/* Set an element of a band matrix to a desired value based on the pointer
   to a location in the band matrix, using the row and column indexes
   of the full matrix.           */
double setv(band_mat *bmat, long row, long column, double val) {
  *getp(bmat,row,column) = val;
  return val;
}

int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
  /* Copy bmat array into the temporary store */
  int i,bandno;
  for(i=0;i<bmat->ncol;i++) { 
    for (bandno=0;bandno<bmat->nbrows;bandno++) {
      bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
    }
    x[i] = b[i];
  }

  long nrhs = 1;
  long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
  int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
  return info;
}

void MemSwap(double **next, double **current){
  double * temp = * current;
  * current = * next;
  * next = temp;
}

int active_or_inactive(long i, long j, long * i_index, long * j_index, long active_grid_cells, long * matrix_column){
  int counter5;
  for (counter5 = 0; counter5 < active_grid_cells; counter5++){
    if (i_index[counter5] == i && j_index[counter5] == j){
      *matrix_column = counter5;
      return 1;
    }
  }
  return 0;
}

void main(){
    
    /* Reading the data from input.txt, sorting in to variables */
    FILE * fptr;
    fptr = fopen("input.txt","r");

    long x_grid_points;
    long y_grid_points;
    long active_grid_cells;
    double length_domain_x;
    double length_domain_y;
    double final_time;
    double lambda;
    double diagnostic_timestep;

    fscanf(fptr, "%ld", &x_grid_points);
    fscanf(fptr, "%ld", &y_grid_points);
    fscanf(fptr, "%ld", &active_grid_cells);
    fscanf(fptr, "%lf", &length_domain_x);
    fscanf(fptr, "%lf", &length_domain_y);
    fscanf(fptr, "%lf", &final_time);
    fscanf(fptr, "%lf", &lambda);
    fscanf(fptr, "%lf", &diagnostic_timestep);

    fclose(fptr);

    /* Taking coefficients and placing in to arrays with only active cell details. */
    FILE * fptr2;
    fptr2 = fopen("coefficients.txt","r");
    long * i_index = malloc(active_grid_cells*sizeof(long));
    long * j_index = malloc(active_grid_cells*sizeof(long));
    double * initial_value = malloc(active_grid_cells*sizeof(double));
    long current_i, current_j;
    double current_initial_value;
    int i = 0;
    while (i < active_grid_cells){
      fscanf(fptr2,"%ld", &current_i);
      fscanf(fptr2,"%ld", &current_j);
      fscanf(fptr2,"%lf", &current_initial_value);
      i_index[i] = current_i;
      j_index[i] = current_j;
      initial_value[i] = current_initial_value;
      i = i + 1;
    }

    /* Performing operations on input*/
    long total_grid_cells = x_grid_points * y_grid_points;
    double dx = length_domain_x / x_grid_points;
    double dy = length_domain_y / y_grid_points;
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    /* Create band matrix: */

    band_mat band_matrix;
    long nbands_low = y_grid_points;
    long nbands_up = y_grid_points;
    init_band_mat(&band_matrix, nbands_low, nbands_up, active_grid_cells);
    
    /* Choose timestep*/
    double dt = diagnostic_timestep * 0.1;

    /*Create boundary conditions and fill in matrix deets etc.*/
    double unit_time_x = diagnostic_timestep / dx2;
    double unit_time_y = diagnostic_timestep / dy2;
    int j = 0;
    for (j=0; j<active_grid_cells;j++){
      long matrix_column;
      double inactive_x = 0;
      double inactive_y = 0;

      if (active_or_inactive(i_index[j]+1,j_index[j],i_index,j_index, active_grid_cells,&matrix_column)){
        setv(&band_matrix,matrix_column,j, unit_time_x);
      }
      else{
        inactive_x = inactive_x + 1;
      }
      if (active_or_inactive(i_index[j]-1,j_index[j],i_index,j_index, active_grid_cells,&matrix_column)){
        setv(&band_matrix,matrix_column,j, unit_time_x);
      }
      else{
        inactive_x = inactive_x + 1;
      }
      if (active_or_inactive(i_index[j],j_index[j]+1,i_index,j_index, active_grid_cells,&matrix_column)){
        setv(&band_matrix,matrix_column,j, unit_time_y);
      }
      else{
        inactive_y = inactive_y + 1;
      }
      if (active_or_inactive(i_index[j],j_index[j]-1,i_index,j_index, active_grid_cells,&matrix_column)){
        setv(&band_matrix,matrix_column,j, unit_time_y);
      }
      else{
        inactive_y = inactive_y + 1;
      }
      setv(&band_matrix, j, j, -1 + diagnostic_timestep * (((-2 + inactive_x)/dx2) + ((-2 + inactive_y)/dy2)));
    }

    double * next_u = malloc(active_grid_cells*sizeof(long));
    double * vector_u = malloc(active_grid_cells*sizeof(long));
    double current_time = 0;

    FILE * fptr3;
    fptr3 = fopen("output.txt","w");
    long diagnostic_timestep_iterations = ceil(final_time / diagnostic_timestep);
    long dt_iterations = ceil(diagnostic_timestep / dt);
    long diagnostic_timestep_number = 0;

    for(int counter6 = 0; counter6<active_grid_cells;counter6++){
      fprintf(fptr3,"%lf %ld %ld %lf \n", current_time, i_index[counter6],j_index[counter6], initial_value[counter6]);
    }

    for(int counter3 = 0; counter3<diagnostic_timestep_iterations;counter3++){

      for(int counter2 = 0; counter2 < dt_iterations;counter2++){

        for(int counter = 0; counter < active_grid_cells; counter++){
          vector_u[counter] = initial_value[counter] * (dt * (initial_value[counter] * initial_value[counter] - lambda) -1);
        }
      solve_Ax_eq_b(&band_matrix, next_u, vector_u);
      MemSwap(&initial_value,&next_u);
      }
      current_time = current_time + diagnostic_timestep;
      for(int counter4 = 0; counter4 < active_grid_cells; counter4++){
        fprintf(fptr3,"%lf %ld %ld %lf \n",current_time, i_index[counter4], j_index[counter4], initial_value[counter4]);
      }
    }
    fclose(fptr3);
    free(i_index);
    free(j_index);
    free(next_u);
    free(vector_u);
    free(initial_value);
}