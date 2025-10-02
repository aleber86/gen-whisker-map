#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64
//#pragma OPENCL EXTENSION cl_amd_assembly_program : enable 

#include"jacobian.h"
#include"modulus.h"


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
        uint dimension,
        uint dim_ang,
        uint dim_y,
        uint dimension_sample,
        __constant  double *_half,
        __global double *mLCE,
        __global uint *LCE_MAP,
        __global uint *LCE_MAP_x,
        uint offset_lambda,
        __global uint *partition_tau,
        __global uint *partition_x,
        ushort flag_gwm){


    uint gid_0 = get_global_id(0);
    uint gsz_0 = get_global_size(0);
    uint gid_1 = get_global_id(1);
    uint gsz_1 = get_global_size(1);
    uint gid_2 = get_global_id(2);
    uint gsz_2 = get_global_size(2);
     

    uint index_global = gid_2 + offset_lambda ;
    double x,t,y;
    double3 tangent; 
    double  lyapunov = 0., eta_element, MLCE = 0.;
    double v_element, omega_2_element, lambda_2_element, lambda_1;
    double max_width = 0.0, min_width = 0.0;
    double dpi = 8.0*atan(1.0);
    uint angular, angular_x, y_wide; 
    double ang = dpi/(double)dim_ang, y_scale = 2.*_half[index_global]/(double)dim_y; 
    int y_offset = (int)(dim_y / 2);
    uint index_buff_c, index_buff_c_x, index_buff_p, index_buff_p_x;

    double Y_m_n = 0., MEGNO = 0.;
    x = initial_conditions[gid_0*3 + 0];
    //WATCHOUT DEFINED FOR WM:
    //*******************************************
    t = initial_conditions[gid_0*3 + 1];
    y = initial_conditions[gid_0*3 + 2];
    eta_element = eta[index_global];
    //WATCHOUT DEFINED FOR WM:
    //eta_element = 3.09108;
    v_element = v[index_global];
    //*******************************************
    omega_2_element = omega_2[index_global];
    if(flag_gwm==0){
        x = (double) 0.;
        v_element = (double) 0.;
        omega_2_element = (double)0.;
    }
    lambda_2_element = lambda_2[index_global];
    lambda_1 = lambda_1_arr[index_global];
    

    //WATCHOUT! CONDITIONS FOR WM:
    __private double const inv_lambda_1 = (double)1./lambda_1;
    double log_f_y;
    tangent = (double3)(sin((double)(gid_0 + gsz_1)), 
                            cos((double)(gid_1 + gsz_0)), 
                            tan((double)(gid_1 + gsz_1)));
    for (long i = 1; i < dimension+1; i++){
        y = y + inv_lambda_1* sin(t) - v_element*inv_lambda_1 * cos(x);
        log_f_y = log(fabs(y)); //<-----------------This register will change bellow!
        t = t - lambda_1 * log_f_y +   eta_element;
        //WATCHOUT CONDITIONS FOR WM:
        if(flag_gwm==1){
            x = x - lambda_2_element* log_f_y + omega_2_element*eta_element;
            x = modulus(x, dpi);
        }
        else{

        x = (double) 0.;
        }
        //*****************************************************************
        t = modulus(t, dpi);
        tangent = jacobian(x, t, y, lambda_1, inv_lambda_1, lambda_2_element, v_element, tangent);
        //norma = length(tangent);
        //Reuse of register log_f_y <---------------------WATCHOUT
        log_f_y = log(length(tangent));
        tangent = normalize(tangent);
        lyapunov += log_f_y; 
        MEGNO += 2.*log_f_y*((double)i);
        Y_m_n += MEGNO/((double)i); 
        //Find t index
        index_finder(0, ang, t, dim_ang, &angular);
        //Find y of t index
        index_finder(y_offset, y_scale, y, y_offset, &y_wide);


        if (max_width < y){
            max_width = y;
        }
        if (min_width >= y){
            min_width = y;
        }
            index_buff_c = (uint)(((gid_2*dim_y + y_wide )*dim_ang + angular));
            index_buff_p = (uint)((gid_2*dim_ang + angular)*gsz_0 + gid_0);
            LCE_MAP[index_buff_c] ++ ;
            partition_tau[index_buff_p] ++;
            if(flag_gwm == 1){
                //Find x index
                index_finder(0, ang, x, dim_ang, &angular_x);
                index_buff_c_x = (uint)(((gid_2*dim_y + y_wide )*dim_ang + angular_x));
                index_buff_p_x = (uint)((gid_2*dim_ang + angular_x)*gsz_0 + gid_0);
                LCE_MAP_x[index_buff_c_x] ++;
                partition_x[index_buff_p_x] ++;
            }
    }
    MLCE = Y_m_n / (double)dimension;

    //index_global <----------CHANGE
    index_global = index_global * gsz_0 + gid_0;
    output_matrix[index_global] = lyapunov/(double)dimension;
    mLCE[index_global] = MLCE;
    max_width_matrix[index_global] = max_width;
    min_width_matrix[index_global] = min_width;
}

__kernel void copy_map(__global uint *LCE_MAP,
                     __global uint *LCE_MAP_x,
                     __global uint *MAP_OUT,
                     __global uint *MAP_OUT_x,
                     uint dim_ang,
                     uint dim_y,
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
                                   uint dim_ang, uint dim_y,
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
                              uint dim_essamble
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
    counter_information_tau[j*gsz_2 + gid_2] = 0;
    counter_information_x[j*gsz_2 + gid_2] = 0;
        for(uint i = 0; i<dim_essamble; i++){
            //counter += (double)partition_tau[(i*partition_size+ j)*gsz_2 + gid_2];
            //counter_x += (double)partition_x[(i*partition_size+ j)*gsz_2 + gid_2];
            counter += (double)partition_tau[(gid_2 * partition_size + j) * dim_essamble + i ];
            counter_x += (double)partition_x[(gid_2 * partition_size + j) * dim_essamble + i ];
            //partition_tau[(i*partition_size+ j)*gsz_2 + gid_2] = 0;
            //partition_x[(i*partition_size+ j)*gsz_2 + gid_2] = 0;
            partition_tau[(gid_2 * partition_size + j) * dim_essamble + i ] = 0;
            partition_x[(gid_2 * partition_size + j) * dim_essamble + i ] = 0;
        }
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
                        uint dim_ang, 
                        uint dim_y,
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
    uint grp_id_0 = get_group_id(0);
    uint grp_id_1 = get_group_id(1);
    uint grp_sz_1 = get_num_groups(1);
    uint grp_sz_0 = get_num_groups(0);
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

    if(lid_0 == 0 && lid_1 == 0){
        counter_array[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_local[0];
        counter_array_x[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_local_x[0];
        counter_colision_array[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_colision_tau[0];
        counter_colision_array_x[(grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2] = counter_colision_x[0];
    }
}

