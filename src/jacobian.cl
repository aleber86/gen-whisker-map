
double3 jacobian(double x, double t, double y, double lambda_1,
       double const inv_lambda_1, double lambda_2, double v, double3 ptr){
    double3 X_1, X_2, X_3, res;
    /*
     *  Evolution of the tangent vector.
     *
     *  x,t,y: Map coordinates
     *  lambda_1: Parameter lambda_1
     *  inv_lambda_1 : 1/lambda_1
     *  lambda_2 : lambda_1 * omega_2
     *  v : upsilon parameter
     *  ptr : tangent vector 
     *
     *  Returns : (double3) evolution of tangent vector
     */

    double signed_argument, argument;     
    double cos_x, cos_t, sin_x, sin_t;
    sin_t = sincos(t, &cos_t);
    sin_x = sincos(x, &cos_x);
    argument = y + inv_lambda_1*sin_t - v*inv_lambda_1*cos_x;
    double const inv_abs_argument = (double)1./fabs(argument);
    signed_argument = sign(argument); 
    X_1 = (double3)( 1. - lambda_2*v*inv_lambda_1*signed_argument*inv_abs_argument*sin_x, 
            -lambda_2*inv_lambda_1*signed_argument*inv_abs_argument*cos_t, 
            -lambda_2*signed_argument*inv_abs_argument);
    X_2 =(double3) (-v*signed_argument*inv_abs_argument*sin_x, 
            1. - signed_argument*inv_abs_argument*cos_t,  
            -signed_argument*lambda_1*inv_abs_argument);
    X_3 = (double3)(v*inv_lambda_1 * sin_x, inv_lambda_1*cos_t, 1.);

    res.x = dot(X_1, ptr);
    res.y = dot(X_2, ptr);
    res.z = dot(X_3, ptr);

    return res;
}

void vector_jacobian(double x, double t, double y, double lambda_1,
        double inv_lambda_1, double lambda_2, double v, double3 *ptr){

    double3 X_1, X_2, X_3;

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

    *ptr = (double3)(dot(X_1,*ptr), dot(X_2, *ptr), dot(X_3, *ptr));

}

