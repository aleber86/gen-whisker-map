#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64

#include"modulus.h"
#include"jacobian.h"

__kernel void gen_whisker_map(__global double *initial_conditions, 
        __global double *output_matrix, 
        __global double *max_width_matrix,
        __global double *min_width_matrix,
        __global double *lambda_1_arr,
        __global double *lambda_2,
        __global double *v, 
        __global double *eta,
        __global double* omega_2, 
        int dimension,
        __global double* mu,
        int GWM_FLAG,
        int ONE_ETA_FLAG){

    int gid_0 = get_global_id(0);
    int gsz_0 = get_global_size(0);
    int gid_1 = get_global_id(1);
    int gsz_1 = get_global_size(1);
    int gid_2 = get_global_id(2);
    int gsz_2 = get_global_size(2);
    double x,t,y;
    double3 tangent; 
    double  lyapunov, norma, eta_element;
    double v_element, omega_2_element, lambda_2_element; 
    double mu_element;
    double max_width = 0.0, min_width = 0.0;
    double lambda_1;
    double dpi = 8.0*atan(1.0);

    lyapunov = 0.0;
    x = (double)initial_conditions[gid_0*3];
    t = (double)initial_conditions[gid_0*3+1];
    y = (double)initial_conditions[gid_0*3+2];
    //mu_element = mu[gid_1]; NOT USED
    /* IF ONE_ETA_FLAG FALSE, USES GLOBAL DIM 1 
     * IF ONE_ETA_FLAG TRUE, TAKES ARGUMENT FROM FILE
     * */
    if(ONE_ETA_FLAG){
        eta_element = eta[gid_2];
    }
    else{
        eta_element = eta[gid_1];
    }
    if(GWM_FLAG){
        v_element = v[gid_2];
        omega_2_element = omega_2[0];
    }
    else{
        v_element = 0.;
        omega_2_element = 0.;
    }
    lambda_1 = lambda_1_arr[gid_2];
    double const inv_lambda_1 = 1./lambda_1;
    lambda_2_element = lambda_2[gid_2];
    tangent = (double3)(sin((double)(gid_2 + gsz_1)), 
                        cos((double)(gid_1 + gsz_0)), 
                        tan((double)(gid_0 + gsz_2)));
    tangent = normalize(tangent);
//#################################
    for (int i = 0; i < dimension; i++){
        y = y + inv_lambda_1*sin(t) - v_element*inv_lambda_1 * cos(x);
        t = t - lambda_1 * log(fabs(y)) + eta_element;
        if(GWM_FLAG){
            x = x - lambda_2_element * log(fabs(y)) + omega_2_element*eta_element;
            x = modulus(x, dpi);
        }
        else{
            x = 0.;
        }
        t = modulus(t, dpi);

        tangent = jacobian(x, t, y, lambda_1, inv_lambda_1, lambda_2_element, v_element, tangent);
        norma = length(tangent);
        lyapunov += log(norma);
        tangent = normalize(tangent);
        if (max_width < y) max_width = y;
        if (min_width > y) min_width = y;
    }
    lyapunov = lyapunov/(double)(dimension);
    output_matrix[(gid_0*gsz_1 + gid_1)*gsz_2 + gid_2] = lyapunov;
    max_width_matrix[(gid_0*gsz_1 + gid_1)*gsz_2 + gid_2] = max_width;
    min_width_matrix[(gid_0*gsz_1 + gid_1)*gsz_2 + gid_2] = min_width;

}

