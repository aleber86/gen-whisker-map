#ifndef MATH_AUXILIARY_FUNCTIONS
#define MATH_AUXILIARY_FUNCTIONS

/* Auxiliary functions */



void Flat_to_zero(double *A_in, int size, double tollerance);
void vector_to_matrix(double3 v, double *Out);
double3 Matrix_product_vector(double *A, double3 vector);
void Matrix_product(double *A, double *B, double *C);
void Transpose(double *A, double *T);
void HQR(double *A, double *Q, double *R);

#endif


