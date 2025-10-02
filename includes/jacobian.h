#ifndef JACOBIAN_FUNCTIONS
#define JACOBIAN_FUNCTIONS


double3 jacobian(double x, double t, double y, double lambda_1,
       double const inv_lambda_1, double lambda_2, double v, double3 ptr);


void vector_jacobian(double x, double t, double y, double lambda_1,
        double inv_lambda_1, double lambda_2, double v, double3 *ptr);



#endif
