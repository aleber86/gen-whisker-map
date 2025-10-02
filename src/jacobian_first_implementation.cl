
/*
 * FIRST IMPLEMENTATION OF JACOBIAN CODE
 * */
double3 jacobian(double x, double t, double y, double lambda_1,
        double lambda_2, double v, double3 ptr){
    double3 X_1, X_2, X_3, res;

    double abs_argument, signed_argument, argument;     
    argument = y + 1./lambda_1*sin(t) - v/lambda_1*cos(x);
    abs_argument = fabs(argument);
    signed_argument = sign(argument); 
    X_1 = (double3)( 1. - lambda_2*v/lambda_1*signed_argument/abs_argument*sin(x), 
            -lambda_2/lambda_1*signed_argument/abs_argument*cos(t), 
            -lambda_2*signed_argument/abs_argument);
    X_2 =(double3) (-v*signed_argument/abs_argument*sin(x), 
            1. - signed_argument/abs_argument*cos(t),  
            -signed_argument*lambda_1 / abs_argument);
    X_3 = (double3)(v/lambda_1 * sin(x), 1./lambda_1*cos(t), 1.);

    res.x = dot(X_1, ptr);
    res.y = dot(X_2, ptr);
    res.z = dot(X_3, ptr);

    return res;
}
