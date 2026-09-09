#pragma OPENCL EXTENSION cl_khr_fp64 : enable    // enable Float64

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

    /**
     * @brief OpenCL kernel that calculates a Generalized Whisker Map.
     * @param initial_conditions an Input buffer in global memory
     * @param output_matrix an Output buffer in global memory
     * @return void
     */
__kernel void gen_whisker_map(
        __constant  double *initial_conditions, 
        __global double *output_matrix, 
        __global double *max_width_matrix,
        __global double *min_width_matrix,
        __global double *lambda_1_arr,
        __global double *lambda_2,
        __constant  double *v, 
        __constant  double *eta,
        __constant  double *omega_2, 
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


    uint gid_0 = get_global_id(0);
    uint gsz_0 = get_global_size(0);
    uint gid_2 = get_global_id(2);
    uint gsz_2 = get_global_size(2);
    uint lid_0 = get_local_id(0);
    uint lsz_0 = get_local_size(0);
    uint lid_2 = get_local_id(2);
    uint lsz_2 = get_local_size(2);
    

    //Segfault protection for thread dispatch
    if(gid_0 >= gsz_0 || gid_2>= gsz_2) return;


	uint LOCAL_SIZE = (uint)(DIM_ANG/lsz_0);
	uint LOCAL_SIZE_y = (uint)(DIM_Y/lsz_0);

    for(uint index_local = 0; index_local<LOCAL_SIZE; index_local++){
        partition_tau_local[lid_2*DIM_ANG + lid_0*LOCAL_SIZE + index_local] = 0;
        if(GWM_FLAG){
            partition_x_local[lid_2*DIM_ANG + lid_0*LOCAL_SIZE + index_local] = 0;
        }

    }
    for(uint index_local = 0; index_local<LOCAL_SIZE_y; index_local++)
        partition_y_local[lid_2*DIM_Y + LOCAL_SIZE_y + index_local] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint index_global = gid_2 + offset_lambda ;
    double x,t,y;
    double v_element, omega_2_element, lambda_2_element, lambda_1;
     
    double const ANGULAR_SCALE = 2.*M_PI/(double)DIM_ANG;
    double const Y_SCALE = 2.*_half[index_global]/(double)DIM_Y;
    

    double Y_m_n = 0., MEGNO = 0.;
    double  lyapunov = 0., eta_element;
    double max_width = 0.0, min_width = 0.0;
	t = initial_conditions[gid_0*3 + 1];
    y = initial_conditions[gid_0*3 + 2];

    double3 tangent;
	if(GWM_FLAG){
        x = initial_conditions[gid_0*3 + 0];
        v_element = v[index_global];
        omega_2_element = omega_2[index_global];
        tangent = (double3)(sin((double)(gid_0 + gsz_0)), 
                            cos((double)(gid_2 + gsz_0)), 
                            tan((double)(gid_0 + gsz_2)));
	}
	else{
        x = (double) 0.;
        v_element = (double) 0.;
        omega_2_element = (double)0.;
        tangent = (double3)(0., 
                            cos((double)(gid_2 + gsz_0)), 
                            tan((double)(gid_0 + gsz_2)));
    }
    
    lambda_2_element = lambda_2[index_global];
    lambda_1 = lambda_1_arr[index_global];
    eta_element = eta[index_global];
    
    //WATCHOUT! CONDITIONS FOR WM:
    double log_f_y;
    for (uint i = 1; i < MAXITER+1; i++){
        y = y + 1./lambda_1* sin(t) - v_element/lambda_1 * cos(x);
        log_f_y = log(fabs(y)); //<-----------------This register will change bellow!
        t = t - lambda_1 * log_f_y +   eta_element;
        t = modulus(t, 2.*M_PI);
        //WATCHOUT CONDITIONS FOR WM:
        if(GWM_FLAG){
            x = x - lambda_2_element* log_f_y + omega_2_element*eta_element;
            x = modulus(x,2.*M_PI);
        }
        else{
        x = (double) 0.;
        }
        //*****************************************************************
        tangent = jacobian(x, t, y, lambda_1, (double)(1./lambda_1), lambda_2_element, v_element, tangent);
        //Reuse of register log_f_y <---------------------WATCHOUT
        log_f_y = log(length(tangent));
        tangent = normalize(tangent);
        lyapunov += log_f_y; 
        MEGNO += 2.*log_f_y*((double)i);
        Y_m_n += MEGNO/((double)i); 
        //Find t index
	    uint angular, y_wide;
        index_finder(0, ANGULAR_SCALE, t, DIM_ANG, &angular);
        //Find y of t index
        index_finder(Y_OFFSET, Y_SCALE, y, Y_OFFSET, &y_wide);


        if (max_width < y){
            max_width = y;
        }
        if (min_width > y){
            min_width = y;
        }
            atomic_add(partition_tau_local + lid_2*DIM_ANG + angular,1 );
            atomic_add(partition_y_local + lid_2*DIM_Y + y_wide, 1);
            if(GWM_FLAG && FULL_POROSITY_COUNT==0){
                index_finder(0, ANGULAR_SCALE, x, DIM_ANG, &angular);
                atomic_add(partition_x_local + lid_2*DIM_ANG + angular, 1);
            }
            if(FULL_POROSITY_COUNT){
                uint index_buff_c = (uint)(((gid_2*DIM_Y + y_wide )*DIM_ANG + angular));
                atomic_add(LCE_MAP + index_buff_c,1);
                index_buff_c-=angular;
                if(GWM_FLAG){
                    index_finder(0, ANGULAR_SCALE, x, DIM_ANG, &angular);
                    atomic_add(partition_x_local + lid_2*DIM_ANG + angular, 1);
                    index_buff_c+=angular;
                    atomic_add(LCE_MAP_x + index_buff_c, 1);
                }
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
    
    for(uint index_local = 0 ; index_local<LOCAL_SIZE; index_local++){
		
        partition_tau[((gid_2*lsz_2)*DIM_ANG+ lid_2*DIM_ANG + lid_0*LOCAL_SIZE + index_local)*grp_sz_0 + grp_id_0] =
                partition_tau_local[lid_2*DIM_ANG + lid_0*LOCAL_SIZE + index_local];
        if(GWM_FLAG){
			partition_x[((gid_2*lsz_2)*DIM_ANG+ lid_2*DIM_ANG + lid_0*LOCAL_SIZE+ index_local)*grp_sz_0 + grp_id_0] = 
                partition_x_local[lid_2*DIM_ANG + lid_0*LOCAL_SIZE+index_local];
        }


}
	
    for(uint index_local = 0; index_local<LOCAL_SIZE_y; index_local++){
        partition_y[((gid_2*lsz_2+lid_2)*DIM_Y + lid_0*LOCAL_SIZE_y + index_local)*grp_sz_0 + grp_id_0] = 
            partition_y_local[lid_2*DIM_Y + lid_0*LOCAL_SIZE + index_local];
    }

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


}

__kernel void Shannon_entropy(__global uint *partition_tau,
                              __global uint *partition_x,
                              __global double *counter_information_tau,
                              __global double *counter_information_x,
                              uint group_1,
                              uint group_2
                              ){

    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);

    
    for(uint j = gid_0; j<DIM_ANG; j+=gsz_0){
    double counter = 0.;
    double counter_x = 0.;
    counter_information_tau[j*gsz_2 + gid_2] = 0;
    counter_information_x[j*gsz_2 + gid_2] = 0;
        for(uint i = 0; i<group_1; i++){
            for(uint k = 0; k<group_2; k++){
            //counter += (double)partition_tau[(i*DIM_ANG+ j)*gsz_2 + gid_2];
            //counter_x += (double)partition_x[(i*DIM_ANG+ j)*gsz_2 + gid_2];
            counter += (double)partition_tau[((gid_2*group_2+k)* DIM_ANG + j) * group_1 + i ];
            counter_x += (double)partition_x[((gid_2*group_2+k) * DIM_ANG + j) * group_1 + i ];
            //partition_tau[(i*DIM_ANG+ j)*gsz_2 + gid_2] = 0;
            //partition_x[(i*DIM_ANG+ j)*gsz_2 + gid_2] = 0;
            partition_tau[((gid_2*group_2+k) * DIM_ANG + j) * group_1 + i ] = 0;
            partition_x[((gid_2*group_2+k) * DIM_ANG + j) * group_1 + i ] = 0;
            }
        }
        //if(counter*DIM_ANG/(MAXITER*128)<=0.9){
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
    uint total_count = 0, total_count_x = 0;
    uint value_in_cell, value_in_cell_x;
    double cell_value_y = 2.*_half[gid_2 + offset_lambda]/(double)DIM_Y;
    double cell_value_x = 2.*M_PI/(double)DIM_ANG;
    uint counter = 0, counter_x = 0, colision = 0, colision_x = 0;
    uint index;
    uint index_local = (lid_0*lsz_1 + lid_1)*lsz_2 + lid_2;

    counter_local[index_local] = 0;
    counter_local_x[index_local] = 0;
    counter_colision_tau[index_local] = 0;
    counter_colision_x[index_local] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint i = gid_1; i < DIM_Y; i+=gsz_1 ){
        for(uint j = gid_0; j < DIM_ANG; j+=gsz_0){
            index = (gid_2 * DIM_Y + i)*DIM_ANG + j;
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
    counter_local[index_local] = total_count;    
    counter_local_x[index_local] = total_count_x;    
    counter_colision_tau[index_local] = colision;    
    counter_colision_x[index_local] = colision_x;    
    barrier(CLK_LOCAL_MEM_FENCE);
    uint half_size = (uint)(lsz_0*lsz_1/2);
    for(uint index = half_size ; index >0; index/=2){
        if( index_local <index){
            counter_local[index_local] +=  counter_local[index_local + index];
            counter_local_x[index_local] +=  counter_local_x[index_local + index];
            counter_colision_tau[index_local] +=  counter_colision_tau[index_local + index];
            counter_colision_x[index_local] +=  counter_colision_x[index_local + index];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint grp_id_0 = get_group_id(0);
    uint grp_id_1 = get_group_id(1);
    uint grp_sz_1 = get_num_groups(1);
    uint grp_sz_0 = get_num_groups(0);
    uint index_array_group = (grp_id_0*grp_sz_1 + grp_id_1)*gsz_2 + gid_2;
    if(lid_0 == 0 && lid_1 == 0){
        counter_array[index_array_group] = *counter_local;
        counter_array_x[index_array_group] = *counter_local_x;
        counter_colision_array[index_array_group] = *counter_colision_tau;
        counter_colision_array_x[index_array_group] = *counter_colision_x;
    }
}


__kernel void approximated_porosity(__global uint *input_partition_tau, 
                                    __global uint *input_partition_x,
                                    __global uint *input_partition_y, 
                                    __global uint *output_porosity_map_t,
                                    __global uint *output_porosity_map_x,
                                    uint group_1,
                                    uint group_2,
                                    uint initial_condition_size){

    uint gid_0 = get_global_id(0);
    uint gid_1 = get_global_id(1);
    uint gid_2 = get_global_id(2);
    uint gsz_0 = get_global_size(0);
    uint gsz_1 = get_global_size(1);
    uint gsz_2 = get_global_size(2);
    uint global_index = (gid_2*gsz_1 + gid_1)*gsz_0 + gid_0;

    uint full_counter_t = 0;
    uint full_counter_x = 0;
    output_porosity_map_t[global_index] = 0;
    output_porosity_map_x[global_index] = 0;
    double cut_t_x = 1.1;
    double cut_y = 1.1;
    for(uint partition_index = gid_0; partition_index < DIM_ANG; partition_index+=gsz_0){
        double counter_tau = 0.;
        double counter_x = 0.;
        for(uint i = 0; i<group_1; i++){
            for(uint k = 0; k<group_2; k++){
            counter_tau += (double)input_partition_tau[((gid_2*group_2+k)* DIM_ANG + partition_index) * group_1 + i ];
            counter_x += (double)input_partition_x[((gid_2*group_2+k) * DIM_ANG + partition_index) * group_1 + i ];
            }
        }
        for(uint partition_index_long = gid_1; partition_index_long < DIM_Y; partition_index_long+=gsz_1){
        double counter_y = 0.;
            for(uint i = 0; i<group_1; i++){
                for(uint k = 0; k<group_2; k++){
                counter_y += (double)input_partition_y[((gid_2*group_2+k) * DIM_Y + partition_index_long) * group_1 + i ];
                }
            }
            if(counter_y*DIM_Y/MAXITER/initial_condition_size*counter_x*DIM_ANG/MAXITER/initial_condition_size>cut_t_x*cut_y){
                full_counter_x+=1;

            }            
            if(counter_y*DIM_Y/MAXITER/initial_condition_size*counter_tau*DIM_ANG/MAXITER/initial_condition_size>cut_t_x*cut_y){
                full_counter_t+=1;

            }
        }
    }
    
    output_porosity_map_t[global_index] = full_counter_t;
    output_porosity_map_x[global_index] = full_counter_x;
}
