#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64

#include"modulus.h"
#include"jacobian.h"
//#define MAXITER  
//#define GWM_FLAG 
//#define ONE_ETA_FLAG 
//#define EXPLICIT_ETA

__kernel void gen_whisker_map(__constant double *initial_conditions, 
        __global double *output_matrix, 
        __global double *max_width_matrix,
        __global double *min_width_matrix,
        __constant double *lambda_1_arr,
        __constant double *lambda_2,
        __constant double *v, 
        __constant double *eta,
        __constant double* omega_2
        //int dimension,
){
        //int GWM_FLAG,
        //int ONE_ETA_FLAG,
        //int EXPLICIT_ETA){

    uint gid_0 = get_global_id(0);
    uint gsz_0 = get_global_size(0);
    uint gid_1 = get_global_id(1);
    uint gsz_1 = get_global_size(1);
    uint gid_2 = get_global_id(2);
    uint gsz_2 = get_global_size(2);
    if(gid_0 >= gsz_0 || gid_1 >= gsz_1 || gid_2>= gsz_2) return;
    
    double3 tangent; 
    double  lyapunov = 0.0, norma, eta_element;
    double v_element, omega_2_element, lambda_2_element; 
    
    
    
    
    /* IF ONE_ETA_FLAG FALSE, USES GLOBAL DIM 1 
     * IF ONE_ETA_FLAG TRUE, TAKES ARGUMENT FROM FILE
     * */
	
    if(ONE_ETA_FLAG){
        eta_element = eta[gid_2];
    }
    else{
        eta_element = eta[gid_1];
    }
    if(EXPLICIT_ETA){
        eta_element = eta[0];
    }
	double x,t,y;
	t = (double)initial_conditions[gid_0*3+1];
    y = (double)initial_conditions[gid_0*3+2];
    if(GWM_FLAG){
        v_element = v[gid_2];
        omega_2_element = omega_2[0];
	    x = (double)initial_conditions[gid_0*3];
		tangent = (double3)(sin((double)(gid_2 + gsz_1)), 
                        cos((double)(gid_1 + gsz_0)), 
                        tan((double)(gid_0 + gsz_2)));
    }
    else{
        v_element = 0.;
        omega_2_element = 0.;
		x = 0.;
		tangent = (double3)(0., 
                        cos((double)(gid_1 + gsz_0)), 
                        tan((double)(gid_0 + gsz_2)));
    }
	double lambda_1;
    lambda_1 = lambda_1_arr[gid_2];
    double const inv_lambda_1 = 1./lambda_1;
    lambda_2_element = lambda_2[gid_2];
	
    
    tangent = normalize(tangent);
//#################################
    double max_width = 0.0, min_width = 0.0;
    for (uint i = 0; i < MAXITER; i++){
        y = y + inv_lambda_1*sin(t) - v_element*inv_lambda_1 * cos(x);
        t = t - lambda_1 * log(fabs(y)) + eta_element;
	    t = modulus(t, 2*M_PI);
        if(GWM_FLAG){
            x = x - lambda_2_element * log(fabs(y)) + omega_2_element*eta_element;
            x = modulus(x, 2*M_PI);
        }
        else{
            x = 0.;
        }
        

        tangent = jacobian(x, t, y, lambda_1, 1./lambda_1, lambda_2_element, v_element, tangent);
        norma = length(tangent);
        lyapunov += log(norma);
        tangent = normalize(tangent);
        if (max_width < y) max_width = y;
        if (min_width > y) min_width = y;
    }
    
	uint index_global = (gid_0*gsz_1 + gid_1)*gsz_2 + gid_2;
    output_matrix[index_global] = lyapunov/(double)MAXITER;
    max_width_matrix[index_global] = max_width;
    min_width_matrix[index_global] = min_width;

}

