#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64
//#pragma OPENCL EXTENSION cl_amd_assembly_program : enable 


#define flag_gwm (1)
#define MAXITER (100000000)
#define dim_ang (2048)
#define dim_y (4096)
//const ushort flag_gwm = 1;
//const uint MAXITER = 10000000;
//const uint dim_ang = 2048;
//const uint dim_y = 4096;

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

double modulus(double arg, double _fmod){
    /*
     * Floating point modulus operation. Allways positive value.
     *
     * arg : argument of function modulo
     * _fmod : congruent value
     *
     * returns (double) modulo
     */
    double _modulus = fmod(arg, _fmod);
    if (_modulus < 0){
        _modulus = _modulus + _fmod;
    }
    return _modulus;
}

void index_finder(int offset, double spread_value, 
        double coordenate, int array_dimension, uint *index_found){

    double differential_value, differential_value_plus;
    int count;
    

    count = (int)(floor(coordenate/spread_value));
    if((count+offset)<0){
        *index_found =(uint) 0;

    }
    else if((count + offset)>=offset+array_dimension){
        *index_found = (uint)(offset + array_dimension - 1);
    }
    else{
        *index_found = (uint)(count + offset);

    }
}

__kernel void gen_whisker_map(
        __constant  double *initial_conditions, 
        __global double *output_matrix, 
        __global double *max_width_matrix,
        __global double *min_width_matrix,
        __global double *lambda_1_arr,
        __global double *lambda_2,
        __constant  double *v, 
        __constant  double *eta,
        __constant  double* omega_2, 
        //uint dimension,
        //uint dim_ang,
        //uint dim_y,
        //uint dimension_sample,
        __constant  double *_half,
        __global double *mLCE,
        __global uint *LCE_MAP,
        __global uint *LCE_MAP_x,
        __global uint *partition_y,
        uint offset_lambda,
        __global uint *partition_tau,
        __global uint *partition_x,
        __local uint *partition_tau_local,
        __local uint *partition_x_local,
        __local uint *partition_y_local){
       // uint flag_gwm){

    /**
     * @brief OpenCL kernel that calculates a Generalized Whisker Map.
     * @param initial_conditions an Input buffer in global memory
     * @param output_matrix an Output buffer in global memory
     * @return void
     */

    uint gid_0 = get_global_id(0);
    uint gsz_0 = get_global_size(0);
    //uint gid_1 = get_global_id(1);
    //uint gsz_1 = get_global_size(1);
    uint gid_2 = get_global_id(2);
    uint gsz_2 = get_global_size(2);
    uint lid_0 = get_local_id(0);
    uint lsz_0 = get_local_size(0);
    uint lid_2 = get_local_id(2);
    uint lsz_2 = get_local_size(2);
    uint LOCAL_SIZE = (uint)(dim_ang/lsz_0);
    uint LOCAL_SIZE_y = (uint)(dim_y/lsz_0);

    if(gid_0 >= gsz_0 || gid_2>= gsz_2) return;

    for(uint index_local = 0; index_local<LOCAL_SIZE; index_local++){
        partition_tau_local[lid_2*dim_ang + lid_0*LOCAL_SIZE + index_local] = 0;
        if(flag_gwm){
            partition_x_local[lid_2*dim_ang + lid_0*LOCAL_SIZE + index_local] = 0;
        }

    }

    for(uint index_local = 0; index_local<LOCAL_SIZE_y; index_local++)
        partition_y_local[lid_2*dim_ang + lid_0*LOCAL_SIZE_y + index_local] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index_global = gid_2 + offset_lambda ;
    double x,t,y;
    double3 tangent; 
    //uint angular, angular_x, y_wide; 
    uint angular, y_wide; 
    double v_element, omega_2_element, lambda_2_element, lambda_1;
    double dpi = 8.0*atan(1.0);
    double ang = dpi/(double)dim_ang, y_scale = 2.*_half[index_global]/(double)dim_y; 
    int y_offset = (int)(dim_y / 2);
    //uint index_buff_c, index_buff_c_x, index_buff_p, index_buff_p_x;

    double Y_m_n = 0., MEGNO = 0.;
    double  lyapunov = 0., eta_element;
    double max_width = 0.0, min_width = 0.0;
    x = initial_conditions[gid_0*3 + 0];
    //WATCHOUT DEFINED FOR WM:
    //*******************************************
    t = initial_conditions[gid_0*3 + 1];
    y = initial_conditions[gid_0*3 + 2];
    //WATCHOUT DEFINED FOR WM:
    //eta_element = 3.09108;
    //*******************************************
    lambda_2_element = lambda_2[index_global];
    lambda_1 = lambda_1_arr[index_global];
    if(flag_gwm==0){
        x = (double) 0.;
        v_element = (double) 0.;
        omega_2_element = (double)0.;
    }
    else{
        eta_element = eta[index_global];
        v_element = v[index_global];
        omega_2_element = omega_2[index_global];

    }
    

    //WATCHOUT! CONDITIONS FOR WM:
    //__private double const inv_lambda_1 = (double)1./lambda_1;
    double log_f_y;
    tangent = (double3)(sin((double)(gid_0 + gsz_0)), 
                            cos((double)(gid_2 + gsz_0)), 
                            tan((double)(gid_0 + gsz_2)));
    for (long i = 1; i < MAXITER+1; i++){
        y = y + (1./lambda_1)* sin(t) - v_element*(1./lambda_1) * cos(x);
        log_f_y = log(fabs(y)); //<-----------------This register will change bellow!
        t = t - lambda_1 * log_f_y +   eta_element;
        //WATCHOUT CONDITIONS FOR WM:
        if(flag_gwm==0){
            x = (double) 0.;
        }
        else{
            x = x - lambda_2_element* log_f_y + omega_2_element*eta_element;
            x = modulus(x, dpi);

        }
        //*****************************************************************
        t = modulus(t, dpi);
        tangent = jacobian(x, t, y, lambda_1, (1./lambda_1), lambda_2_element, v_element, tangent);
        //norma = length(tangent);
        //Reuse of register log_f_y <---------------------WATCHOUT
        log_f_y = log(length(tangent));
        tangent = normalize(tangent);
        lyapunov += log_f_y; 
        MEGNO += 2.*log_f_y*((double)i);
        Y_m_n += MEGNO/((double)i); 
        if (max_width < y){
            max_width = y;
        }
        if (min_width >= y){
            min_width = y;
        }
        //Find t index
        index_finder(0, ang, t, dim_ang, &angular);
        //Find y of t index
        index_finder(y_offset, y_scale, y, y_offset, &y_wide);


            //index_buff_c = (uint)(((gid_2*dim_y + y_wide )*dim_ang + angular));
            //index_buff_p = (uint)((gid_2*dim_ang + angular)*gsz_0 + gid_0);
//            atom_add(LCE_MAP+index_buff_c,1);
            atom_add(partition_tau_local + lid_2*dim_ang + angular,1 );
            atom_add(partition_y_local + lid_2*dim_ang + y_wide, 1);
            //LCE_MAP[index_buff_c] ++ ;
            //partition_tau[index_buff_p] ++;
            if(flag_gwm == 1){
                //Find x index
                //index_finder(0, ang, x, dim_ang, &angular_x);
                index_finder(0, ang, x, dim_ang, &angular);
                //index_buff_c_x = (uint)(((gid_2*dim_y + y_wide )*dim_ang + angular_x));
                //index_buff_p_x = (uint)((gid_2*dim_ang + angular_x)*gsz_0 + gid_0);
                //LCE_MAP_x[index_buff_c_x] ++;
//                atom_add(LCE_MAP_x + index_buff_c_x, 1);
                //atom_add(partition_x_local + lid_2*dim_ang + angular_x, 1);
                atom_add(partition_x_local + lid_2*dim_ang + angular, 1);
                //partition_x[index_buff_p_x] ++;
            }
    }

    //index_global <----------CHANGE
    index_global = index_global * gsz_0 + gid_0;
    output_matrix[index_global] = lyapunov/(double)MAXITER;
    mLCE[index_global] = Y_m_n / (double)MAXITER;
    max_width_matrix[index_global] = max_width;
    min_width_matrix[index_global] = min_width;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    uint grp_id_0 = get_group_id(0);
    uint grp_sz_0 = get_num_groups(0);
    for(uint index_local = 0; index_local<LOCAL_SIZE; index_local++){
        partition_tau[((gid_2*lsz_2+lid_2)*dim_ang + lid_0*LOCAL_SIZE + index_local)*grp_sz_0 + grp_id_0] = partition_tau_local[lid_2*dim_ang + lid_0*LOCAL_SIZE + index_local];
        if(flag_gwm){
            partition_x[((gid_2*lsz_2+lid_2)*dim_ang + lid_0*LOCAL_SIZE + index_local)*grp_sz_0 + grp_id_0] = partition_x_local[lid_2*dim_ang + lid_0*LOCAL_SIZE + index_local];
        }


}
    for(uint index_local = 0; index_local<LOCAL_SIZE_y; index_local++){
        partition_y[((gid_2*lsz_2+lid_2)*dim_ang + lid_0*LOCAL_SIZE_y + index_local)*grp_sz_0 + grp_id_0] = partition_y_local[lid_2*dim_ang + lid_0*LOCAL_SIZE + index_local];
    }

}

__kernel void copy_map(__global uint *LCE_MAP,
                     __global uint *LCE_MAP_x,
                     __global uint *MAP_OUT,
                     __global uint *MAP_OUT_x,
                     //uint dim_ang,
                     //uint dim_y,
                     uint offset_lambda,
                     uint dimension_sample,
                     uint dimension_matrix){


    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);

    double LCE_value_in_cell;
    for(uint i = gid_1; i < dim_y; i+=gsz_1 ){
        for(uint j = gid_0; j < dim_ang; j+=gsz_0){
                LCE_value_in_cell = LCE_MAP[((i * dim_ang + j)*dimension_matrix)];
                MAP_OUT[((i * dim_ang + j)*dimension_sample + offset_lambda)] = LCE_value_in_cell ;
                LCE_value_in_cell = LCE_MAP_x[((i * dim_ang + j)*dimension_matrix)];
                MAP_OUT_x[((i * dim_ang + j)*dimension_sample + offset_lambda)] = LCE_value_in_cell ;
            
        }


    }
    
}

__kernel void from_matrix_to_array(__global uint *LCE_MAP,
                                   __global uint *LCE_MAP_x, 
                                   __global float *MAP_OUT,
                                   __global float *MAP_OUT_x,
                                   //uint dim_ang, uint dim_y,
                                   __global double *half_width,
                                   __global double *lambda_1,
                                   uint offset_size, // g_2/lamb_r_m_o 
                                   uint full_offset, //g_2 size
                                   uint index_offset){

    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);

    double y_scale = 2.*lambda_1[full_offset*index_offset+offset_size*gid_2]*
        half_width[full_offset*index_offset+offset_size*gid_2] / (double)dim_y;
    double x_scale = 8.0*atan(1.0) / (double)dim_ang;
    double LCE_value_in_cell;
    float s_x, s_y;
    uint index, index_out;
    
    for(uint i = gid_1; i < dim_y; i+=gsz_1 ){
        for(uint j = gid_0; j < dim_ang; j+=gsz_0){
                //index = ((i * dim_ang + j)*full_offset + offset_size*gid_2);
                index = (((gid_2)*dim_y + i )*dim_ang +j );
                s_x = (float)(x_scale*(double)j);
                s_y = (float)(y_scale*(double)((double)i - (double)(dim_y/2.)));
                index_out = ((i * dim_ang + j)*3)*gsz_2 + gid_2;
                MAP_OUT[index_out] = s_x;
                MAP_OUT_x[index_out] = s_x;
                index_out = ((i * dim_ang + j)*3 +1)*gsz_2 + gid_2;
                MAP_OUT[index_out] = s_y;
                MAP_OUT_x[index_out] = s_y;
                index_out = ((i * dim_ang + j)*3 +2)*gsz_2 + gid_2;
                LCE_value_in_cell = (float)LCE_MAP[index];
                MAP_OUT[index_out] =  LCE_value_in_cell ;
                LCE_value_in_cell = (float)LCE_MAP_x[index];
                MAP_OUT_x[index_out] =  LCE_value_in_cell;
        }
    }
    /*
    for(uint i = gid_1; i < dim_y; i+=gsz_1 ){
        for(uint j = gid_0; j < dim_ang; j+=gsz_0){
                index = ((i * dim_ang + j)*full_offset + offset_size*gid_2);
                s_x = (float)(x_scale*(double)j);
                s_y = (float)(y_scale*(double)((double)i - (double)(dim_y/2)));
                index_out = ((i * dim_ang + j)*3)*gsz_2 + gid_2;
                MAP_OUT[index_out] = s_x;
                MAP_OUT_x[index_out] = s_x;
                index_out = ((i * dim_ang + j)*3 +1)*gsz_2 + gid_2;
                MAP_OUT[index_out] = s_y;
                MAP_OUT_x[index_out] = s_y;
                index_out = ((i * dim_ang + j)*3 +2)*gsz_2 + gid_2;
                LCE_value_in_cell = (float)LCE_MAP[index];
                MAP_OUT[index_out] =  LCE_value_in_cell ;
                LCE_value_in_cell = (float)LCE_MAP_x[index];
                MAP_OUT_x[index_out] =  LCE_value_in_cell;
        }
    }
    */


}

__kernel void Shannon_entropy(__global uint *partition_tau,
                              __global uint *partition_x,
                              __local uint *counter_partition_tau,
                              __local uint *counter_partition_x,
                              __global double *counter_information_tau,
                              __global double *counter_information_x,
                              uint partition_size,
                              uint group_1,
                              uint group_2
                              ){

    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);

    
    for(uint j = gid_0; j<partition_size; j+=gsz_0){
    double counter = 0.;
    double counter_x = 0.;
    //double porosity = 0.; //ADDED BE AWARE
    //uint c_counter = 0;
    counter_information_tau[j*gsz_2 + gid_2] = 0;
    counter_information_x[j*gsz_2 + gid_2] = 0;
        for(uint i = 0; i<group_1; i++){
            for(uint k = 0; k<group_2; k++){
            //counter += (double)partition_tau[(i*partition_size+ j)*gsz_2 + gid_2];
            //counter_x += (double)partition_x[(i*partition_size+ j)*gsz_2 + gid_2];
            counter += (double)partition_tau[((gid_2*group_2+k)* partition_size + j) * group_1 + i ];
            counter_x += (double)partition_x[((gid_2*group_2+k) * partition_size + j) * group_1 + i ];
            //partition_tau[(i*partition_size+ j)*gsz_2 + gid_2] = 0;
            //partition_x[(i*partition_size+ j)*gsz_2 + gid_2] = 0;
            partition_tau[((gid_2*group_2+k) * partition_size + j) * group_1 + i ] = 0;
            partition_x[((gid_2*group_2+k) * partition_size + j) * group_1 + i ] = 0;
            }
        }
        //if(counter*partition_size/(MAXITER*128)<=0.9){
        //    c_counter++;
        //    printf("%u", c_counter);
        //}
    if(counter!=0){
        counter_information_tau[j*gsz_2 + gid_2] = counter * log(counter);
    }
    else{
        counter_information_tau[j*gsz_2 + gid_2] = 0.;
    }
    if(counter_x!=0){
    counter_information_x[j*gsz_2 + gid_2] = counter_x * log(counter_x) ;
    }
    else{
    counter_information_x[j*gsz_2 + gid_2] = 0.;
    }
    //DEBUG
    //counter_information_tau[j*gsz_2 + gid_2] = counter;
    //counter_information_x[j*gsz_2 + gid_2] = counter_x  ;
    }
}

__kernel void reduction(__global uint *LCE_MAP, 
                        __global uint *LCE_MAP_x,
                        __global uint *counter_array,
                        __local uint *counter_local,
                        __global uint *counter_array_x,
                        __local uint *counter_local_x,
                        //uint dim_ang, 
                        //uint dim_y,
                        long dimension,
                        __constant double *_half,
                        uint offset_lambda,
                        __local uint *counter_colision_tau,
                        __local uint *counter_colision_x,
                        __global uint *counter_colision_array,
                        __global uint *counter_colision_array_x
                        ){

    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);
    uint lid_0 = get_local_id(0);
    uint lid_1 = get_local_id(1);
    uint lid_2 = get_local_id(2);
    uint lsz_0 = get_local_size(0);
    uint lsz_1 = get_local_size(1);
    uint lsz_2 = get_local_size(2);
    double dpi = 8.0*atan(1.0);
    uint total_count = 0, total_count_x = 0;
    uint value_in_cell, value_in_cell_x;
    double cell_value_y = 2.*_half[gid_2 + offset_lambda]/(double)dim_y;
    double cell_value_x = dpi/(double)dim_ang;
    double inv_dimension = (double)(1./(double)dimension);
    uint counter = 0, counter_x = 0, colision = 0, colision_x = 0;
    uint half_size;
    uint index;

    counter_local[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = 0;
    counter_local_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = 0;
    counter_colision_tau[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = 0;
    counter_colision_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i = gid_1; i < dim_y; i+=gsz_1 ){
        for(uint j = gid_0; j < dim_ang; j+=gsz_0){
            index = (gid_2 * dim_y + i)*dim_ang + j;
//            value_in_cell = LCE_MAP[((i * dim_ang + j)*gsz_2 + gid_2) ];
//            value_in_cell_x = LCE_MAP_x[((i * dim_ang + j)*gsz_2 + gid_2)];
            value_in_cell = LCE_MAP[index];
            value_in_cell_x = LCE_MAP_x[index];
            LCE_MAP[index] = 0;
            LCE_MAP_x[index] = 0;
            colision += value_in_cell;  
            colision_x += value_in_cell_x;  

            if(value_in_cell > 0){
                total_count ++;
            }
            if(value_in_cell_x > 0){
                total_count_x++;
            }

        }

    }
    counter_local[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = total_count;    
    counter_local_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = total_count_x;    
    counter_colision_tau[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = colision;    
    counter_colision_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] = colision_x;    
    barrier(CLK_LOCAL_MEM_FENCE);
    half_size = (uint)(lsz_0*lsz_1/2);
    for(uint index = half_size ; index >0; index/=2){
        if( (lid_0*lsz_1 + lid_1)*lsz_2 + lid_2 <index){
            counter_local[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] +=  counter_local[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2 + index];
            counter_local_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] +=  counter_local_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2 + index];
            counter_colision_tau[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] +=  counter_colision_tau[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2 + index];
            counter_colision_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2] +=  counter_colision_x[(lid_0*lsz_1 + lid_1)*lsz_2 + lid_2 + index];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint grp_id_0 = get_group_id(0);
    uint grp_id_1 = get_group_id(1);
    uint grp_sz_0 = get_num_groups(0);
    uint grp_sz_1 = get_num_groups(1);
    if(lid_0 == 0 && lid_1 == 0){
        counter_array[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_local[0];
        counter_array_x[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_local_x[0];
        counter_colision_array[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_colision_tau[0];
        counter_colision_array_x[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_colision_x[0];
    }
}


