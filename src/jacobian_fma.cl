
void jacobian(double x, double t, double y, double lambda_1,
        double const inv_lambda_1, double lambda_2, double v, double *ptr){
    double3 X_1, X_2, X_3, res;

    double abs_argument, inv_abs_argument, signed_argument, argument;     
    argument = fma(inv_lambda_1,sin(t), y) - v*inv_lambda_1*cos(x);
    abs_argument = fabs(argument);
    inv_abs_argument = 1./abs_argument;
    signed_argument = sign(argument); 
    X_1 = (double3)( 1. - lambda_2*v*inv_lambda_1*signed_argument*inv_abs_argument*sin(x), 
            -lambda_2*inv_lambda_1*signed_argument*inv_abs_argument*cos(t), 
            -lambda_2*signed_argument*inv_abs_argument);
    X_2 =(double3) (-v*signed_argument*inv_abs_argument*sin(x), 
            1. - signed_argument*inv_abs_argument*cos(t),  
            -signed_argument*lambda_1*inv_abs_argument);
    X_3 = (double3)(v*inv_lambda_1 * sin(x), inv_lambda_1*cos(t), 1.);

    ptr[0] = X_1.x; ptr[1] = X_1.y; ptr[2] = X_1.z;
    ptr[3] = X_2.x; ptr[4] = X_2.y; ptr[5] = X_2.z;
    ptr[6] = X_3.x; ptr[7] = X_3.y; ptr[8] = X_3.z;

}
