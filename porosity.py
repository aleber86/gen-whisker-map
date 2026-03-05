"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
import time



_wp = np.float64 # Working Precision (scientific calc)
_wpf = np.float32 # Working Precision (map graphics)
_wpi = np.int32 # Integer Precision (iteration and others)
_wpl = np.int64 # Iteneger Long
_wpui = np.uint32 # Unsigned Integer, for Counting map
_wpushort = np.uint16 # Unsigned half precision integer, for flags


def information_shannon_entropy(partition_cardinal : int, number_of_orbits : int,
                                iteration_time : int, count_of_points : np.array,
                                ratio : bool = True, _axis : int = 0) -> np.array:
    """Information calculation using the Shannon entropy applied to dynamical systems.
    The ration between Information of all orbits in the map, and the Information of
    a random orbit gives an estimation of correlations between fase variables of the map.

    Implementation from:
    P. M. Cincotta, C. M. Giordano. Phase correlations in chaotic dynamics A Shannon entropy measure.
    Celest Mech Dyn Astr 130, 74 (2018)

    entropy: see eq. (7)
    information: see eq. (11)
    INFORMARTION_RANDOM: see eq. (12)


    Args:
        partition_cardinal : Number of cells (phase space cell partition)
        number_of_orbits : Initial conditions in the ensemble
        count_of_points: Array of size <partition_cardinal> with n_k*log(n_k)

    Returns:
        information_array : information of the phase variable of the map (if ratio = True, returns
        information / information of a random orbit)


    """

    #Information of a random orbit
    INFORMARTION_RANDOM = _wp(partition_cardinal) /\
        (2.*_wp(iteration_time)*_wp(number_of_orbits)*np.log(partition_cardinal))

    entropy = np.log(_wp(number_of_orbits)*_wp(iteration_time))-\
    np.sum(count_of_points, axis=_axis)/(_wp(number_of_orbits)*_wp(iteration_time))

    ones = np.ones((count_of_points.shape[1],), dtype = _wp)

    information_array = ones - 1./np.log(_wp(partition_cardinal))*entropy

    if ratio: information_array = information_array / INFORMARTION_RANDOM

    return information_array

def save_output_to_file(array_to_be_saved : np.array, file_name : str):

    headers_elements = ["lambda_1", "lambda_2", "v", "mu",
                        "count_tau", "count_x", "eta", "I_tau", "I_x", "<mLCE(tan_map)>",
                        "mLCE_max(MEGNO)", "mLCE_sum_mean(MEGNO)",
                        "mLCE_mean(MEGNO)","mLCE(MEGNO)", "<h>(all)",
                        "std_h(all)", "max_hw", "min_hw", "min mLCE(tan_map)",
                        "max mLCE(tan_map)", "std mLCE(tan_map)",
                        "mean_hw(orbits)", "min_hw(max eta)", "max_hw(max eta)",
                        "std_width(orbits)", "min_widht(all)"]
    headers = "#"
    for index_h, head in enumerate(headers_elements):
        headers = f"{headers}\t({index_h+1}){head}"
    headers = f"{headers}\n"
    with open(file_name, "w") as file_to_save:
        file_to_save.write(headers)
        np.savetxt(file_to_save, array_to_be_saved)


def main():
    directory = "data"
    _it_step = 5 # Iteration time as powers of 10
    suffix = f"gwm_{_it_step}_1_pre_catched_256"
    _random_seed = 34567890
    #_random_seed = 547891248
    np.random.seed(_random_seed)
    SPREAD = _wp(1.e-7) #Spread of ensemble
    _max_iter = _wpui(10**_it_step) #Iteration time function of _it_step
    _dim_ensemble = _wpui(256) #Ensemble size
    _lambda_1_range = _wpui(1536) #Use 128 multiple
    _lambda_1_range_map_out = _wpui(1) #Maps to save as graphics! This could skyrocket the memory usage
    _common_gid_2_size = 128 # Size of chunks in lambda space. Change at will
    _save_maps = False # Save map flag. True, saves _lambda_1_range_map_out number per chunk
    _save_collisions = False # Save collision values flag. True, saves the difference between total points and rastered.
    _gwm_flag = _wpushort(1) # GENERALIZED WHISKER MAP. True, calc (x,t,y). False (t,y)


    #FIRST KERNEL ATRIBS
    _g_size_0 = int(_dim_ensemble)  #First Kernel
    _g_size_1 = 1   #First Kernel
    _g_size_2 = _common_gid_2_size   #First Kernel
    #_g_size_2 = 100   #First Kernel
    _dim_ang = _wpui(2048) # PHASE RASTER SIZE
    _dim_y = _wpui(4096) # ACTION-LIKE RASTER SIZE
    #Local problem size of the Whisker map iteration
    _local_id_0 = 4 #First Kernel
    _local_id_1 = 1 #First Kernel
    _local_id_2 = 4 #First Kernel

    #COPY KERNEL

    _g_size_0_c = 128
    _g_size_1_c = 128
    _g_size_2_c = int(_lambda_1_range_map_out)
    _local_id_0_c = 16 #Second Kernel
    _local_id_1_c = 16 #Second Kernel
    #_local_id_2 = 10
    _local_id_2_c = 1 #Second Kernel

    #SECOND KERNEL ATRIBS
    _g_size_0_s = 128
    _g_size_1_s = 128
    _g_size_2_s = _common_gid_2_size
    _local_id_0_s = 16 #Second Kernel
    _local_id_1_s = 16 #Second Kernel
    _local_id_2_s = 1 #Second Kernel
    _grp_sz_0 = int(_g_size_0_s/_local_id_0_s)
    _grp_sz_1 = int(_g_size_1_s/_local_id_1_s)

    #THIRD KERNEL ATRIBS
    index_value = _dim_ensemble
    if index_value > 256:
        index_value = 256
    _g_size_0_t = index_value
    _g_size_1_t = 1
    _g_size_2_t = _common_gid_2_size
    #_g_size_2_t = 100
    _local_id_0_t = index_value
    _local_id_1_t = 1
    _local_id_2_t = 1

    #******************************************************************************************


    #lambda chunks
    lambda_offset = _wpui(_lambda_1_range/_g_size_2)

    #CPU side objects
    #********************************************************************************************
    #Ensemble of initial conditions. Set to dimension 3
    array_initial_conditions = np.array(np.random.uniform(-1,1, (_dim_ensemble, 3)), dtype=_wp) * _wp(SPREAD)

    #Read model paramenters from file
    with open('pre_cached_gwm_omega_2_2.5_128_elements.dat', 'r') as file:
        array_file_initial_conditions = np.loadtxt(file, dtype = _wp)

    array_lambda_1 = array_file_initial_conditions[:,0].copy()
    array_lambda_2 = array_file_initial_conditions[:,1].copy()
    array_omega_2 =  array_file_initial_conditions[:,2].copy()
    array_mu = array_file_initial_conditions[:,3].copy()
    array_initial_conditions_eta = array_file_initial_conditions[:,4].copy()
    array_v = array_file_initial_conditions[:,5].copy()
    array_half = array_file_initial_conditions[:,6].copy()
    #Copy is necesary to align the memory blocks

    del array_file_initial_conditions #Redundant object

    if ~_gwm_flag:
        array_omega_2 = np.zeros_like(array_omega_2, dtype = _wp)
        array_v = np.zeros_like(array_v, dtype=_wp)


    array_to_file = np.zeros((_lambda_1_range, 26), _wp) #Creates array to save final output
    if array_lambda_1.shape[0] != _lambda_1_range:
        print(f"Range of lambda: {_lambda_1_range} not equal to model parameters list size: {array_lambda_1.shape[0]}.")
        exit(-1)

    output_matrix = np.zeros((_lambda_1_range, _dim_ensemble), dtype=_wp)
    max_width_matrix = np.zeros((_lambda_1_range, _dim_ensemble), dtype=_wp)
    min_width_matrix = np.zeros((_lambda_1_range, _dim_ensemble), dtype=_wp)
    counter_array = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
    counter_array_x = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
    counter_array_collision = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
    counter_array_collision_x = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
    CONSTANT_MAX_POINTS_ADDED = np.ones(_g_size_2_s, dtype = _wp) * _wp(_dim_ensemble) * _wp(_max_iter)
    CONSTANT_MAX_POINTS_ADDED = CONSTANT_MAX_POINTS_ADDED.astype(_wp)
    LCE_MAP = np.zeros(( _g_size_2,_dim_y,_dim_ang), dtype = _wpui)
    LCE_MAP_x = np.zeros(( _g_size_2,_dim_y,_dim_ang), dtype = _wpui)
    mLCE = np.zeros((_lambda_1_range, _dim_ensemble), dtype = _wp)
    partition_tau = np.zeros((_g_size_2_t, _dim_ang, _dim_ensemble), dtype = _wpui)
    partition_x = np.zeros((_g_size_2_t, _dim_ang, _dim_ensemble), dtype = _wpui)
    counter_information_tau = np.zeros((_dim_ang, _g_size_2_t), dtype=_wp)
    counter_information_x = np.zeros((_dim_ang, _g_size_2_t), dtype=_wp)
    #WATCH OUT  !!!!!!!!
    MAP_OUT = np.zeros((int(_dim_y)*int(_dim_ang), 3, _lambda_1_range_map_out), dtype = _wpf)
    MAP_OUT_x = np.zeros((int(_dim_y)*int(_dim_ang), 3, _lambda_1_range_map_out), dtype = _wpf)
    #**************************************************************************************************

    OCL_Object = OpenCL_Object() #OpenCL object handles host -> device -> host operations

    #Buffer CPU -> GPU
    #Sets pointer to global and local memory in GPU
    OCL_Object.buffer_global(array_half, "half", False)
    OCL_Object.buffer_global(array_initial_conditions, "initial_conditions", False)
    OCL_Object.buffer_global(array_initial_conditions_eta, "initial_conditions_eta", False)
    OCL_Object.buffer_global(array_omega_2, "omega_2", False)
    OCL_Object.buffer_global(array_v, "v", False)
    OCL_Object.buffer_global(array_lambda_2, "lambda_2", False)
    OCL_Object.buffer_global(array_lambda_1, "lambda_1", False)
    OCL_Object.buffer_global(output_matrix, "output_matrix")
    OCL_Object.buffer_global(max_width_matrix, "max_width_matrix")
    OCL_Object.buffer_global(min_width_matrix, "min_width_matrix")
    OCL_Object.buffer_global(counter_array, "counter_array")
    OCL_Object.buffer_global(counter_array_x, "counter_array_x")
    OCL_Object.buffer_global(mLCE, "mLCE")
    OCL_Object.buffer_global(LCE_MAP, "LCE_MAP")
    OCL_Object.buffer_global(LCE_MAP_x, "LCE_MAP_x")
    OCL_Object.buffer_global(MAP_OUT, "MAP_OUT")
    OCL_Object.buffer_global(MAP_OUT_x, "MAP_OUT_x")
    OCL_Object.buffer_global(partition_tau, "partition_tau")
    OCL_Object.buffer_global(partition_x, "partition_x")
    OCL_Object.buffer_global(counter_information_tau, "counter_information_tau")
    OCL_Object.buffer_global(counter_information_x, "counter_information_x")
    OCL_Object.buffer_global(counter_array_collision, "counter_array_collision")
    OCL_Object.buffer_global(counter_array_collision_x, "counter_array_collision_x")
    OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter")
    OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_x")
    OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_collision_tau")
    OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_collision_x")
    OCL_Object.buffer_local(_local_id_0_s, 4, "counter_partition_tau")
    OCL_Object.buffer_local(_local_id_0_s, 4, "counter_partition_x")
    #*******************************************************************************************

    #To unroll loops explicit declartion of maximum iteration step
    with open('one_kernel_form.cl', 'r') as file_to_change:
        script = file_to_change.read()
        script = script.replace("#define MAXITER", f"#define MAXITER {_max_iter}")
    with open('one_kernel.cl', 'w') as file:
        file.write(script)
    #Uses a form to create the final script file
    #Appends to the end of the script file necesary functions
    OCL_Object.program(['one_kernel.cl', 'src/jacobian.cl', 'src/modulus.cl'], ['-I ./includes'])

    print("Mem. Buffer OK")

    total_time = 0.

    for index_offset in np.arange(lambda_offset):
        start_time = time.time()
        print(f"Start time: {time.strftime('%H:%M:%S')}")

        lambda_offset_it = _wpui(index_offset*_g_size_2) #Iteration offset per chunk

        #Evolution of the system. Half-width, mLCE, Rasterization
        ev1 = OCL_Object.kernel.gen_whisker_map(OCL_Object.queue,
                                            (_g_size_0, _g_size_1, _g_size_2),
                                            (_local_id_0, _local_id_1, _local_id_2),
                                            OCL_Object.initial_conditions_device,
                                            OCL_Object.output_matrix_device,
                                            OCL_Object.max_width_matrix_device,
                                            OCL_Object.min_width_matrix_device,
                                            OCL_Object.lambda_1_device,
                                            OCL_Object.lambda_2_device, OCL_Object.v_device,
                                            OCL_Object.initial_conditions_eta_device,
                                            OCL_Object.omega_2_device, _max_iter, _dim_ang,
                                            _dim_y, _lambda_1_range,
                                            OCL_Object.half_device,
                                            OCL_Object.mLCE_device,
                                            OCL_Object.LCE_MAP_device,
                                            OCL_Object.LCE_MAP_x_device,
                                            lambda_offset_it,
                                            OCL_Object.partition_tau_device,
                                            OCL_Object.partition_x_device,
                                            _gwm_flag
                                            )
        #Wait for the evolution of the whisker map
        cl.wait_for_events([ev1])
        print("First Kernel Finished")
        #This kernel transform the rasterization back to vector, but in float32 (less mem usage)
        ev_copy_map = OCL_Object.kernel.from_matrix_to_array(OCL_Object.queue,
                                                             (_g_size_0_c, _g_size_1_c, _g_size_2_c),
    #                                                         None,
                                                             (_local_id_0_c, _local_id_1_c, _local_id_2_c),
                                                               OCL_Object.LCE_MAP_device,
                                                               OCL_Object.LCE_MAP_x_device,
                                                               OCL_Object.MAP_OUT_device,
                                                               OCL_Object.MAP_OUT_x_device,
                                                                _dim_ang, _dim_y,
                                                             OCL_Object.half_device,
                                                             OCL_Object.lambda_1_device,
                                                             _wpui(_g_size_2/_lambda_1_range_map_out),
                                                             _wpui(_g_size_2),
                                                             _wpui(index_offset),
                                                                wait_for=[ev1])
        cl.wait_for_events([ev_copy_map])
        #Sum of the rastered cells. Creates collision difference and porosity.
        ev2 = OCL_Object.kernel.reduction(OCL_Object.queue,
                                           (_g_size_0_s,_g_size_1_s,_g_size_2_s),
                                           (_local_id_0_s,_local_id_1_s,_local_id_2_s),
                                           OCL_Object.LCE_MAP_device,
                                           OCL_Object.LCE_MAP_x_device,
                                           OCL_Object.counter_array_device,
                                           OCL_Object.counter_device,
                                           OCL_Object.counter_array_x_device,
                                           OCL_Object.counter_x_device,
                                           _dim_ang,
                                           _dim_y,
                                           _wpl(_max_iter)*_wpl(_dim_ensemble),
                                           OCL_Object.half_device,
                                           lambda_offset_it,
                                          OCL_Object.counter_collision_tau_device,
                                          OCL_Object.counter_collision_x_device,
                                          OCL_Object.counter_array_collision_device,
                                          OCL_Object.counter_array_collision_x_device,
                                           wait_for=[ev_copy_map])
        cl.wait_for_events([ev2])
        print("Porosity count")
        #Sum over the partitions of the Shannon entropy from element to ensemble.
        ev_shannon = OCL_Object.kernel.Shannon_entropy(OCL_Object.queue,
                                                       (_g_size_0_t, _g_size_1_t, _g_size_2_t),
                                                       (_local_id_0_t, _local_id_1_t, _local_id_2_t),
                                                       OCL_Object.partition_tau_device,
                                                       OCL_Object.partition_x_device,
                                                       OCL_Object.counter_partition_tau_device,
                                                       OCL_Object.counter_partition_x_device,
                                                       OCL_Object.counter_information_tau_device,
                                                       OCL_Object.counter_information_x_device,
                                                       _wpui(_dim_ang),
                                                       _wpui(_dim_ensemble))


        cl.wait_for_events([ev_shannon])
        print("Count * log (count) Shannon_entropy (Sum argument)")
        #Copy every value out from the GPU
        ev_copy_6 = cl.enqueue_copy(OCL_Object.queue, counter_array, OCL_Object.counter_array_device)
        ev_copy_8 = cl.enqueue_copy(OCL_Object.queue, counter_array_x, OCL_Object.counter_array_x_device)
        ev_inform_tau = cl.enqueue_copy(OCL_Object.queue, counter_information_tau, OCL_Object.counter_information_tau_device)
        ev_inform_x = cl.enqueue_copy(OCL_Object.queue, counter_information_x, OCL_Object.counter_information_x_device)
        ev_collision_tau = cl.enqueue_copy(OCL_Object.queue, counter_array_collision, OCL_Object.counter_array_collision_device)
        ev_collision_x = cl.enqueue_copy(OCL_Object.queue, counter_array_collision_x, OCL_Object.counter_array_collision_x_device)
        cl.wait_for_events([ ev_copy_6, ev_copy_8, ev_inform_tau, ev_inform_x, ev_collision_tau, ev_collision_x ])



        #INFORMATION OF THE SHANNON ENTROPY******************************************************
        info_tau = information_shannon_entropy(_dim_ang, _dim_ensemble, _max_iter, counter_information_tau)
        info_x = information_shannon_entropy(_dim_ang, _dim_ensemble, _max_iter, counter_information_x)
        #*******************************************************************************************
        #Count of cells occupied********************************************************************
        count = np.sum(counter_array, axis=0)
        count_x = np.sum(counter_array_x, axis=0)

        col_step = np.sum(counter_array_collision, axis = 0).astype(_wp) - CONSTANT_MAX_POINTS_ADDED
        col_step_x = np.sum(counter_array_collision_x, axis = 0).astype(_wp) - CONSTANT_MAX_POINTS_ADDED

        #*******************************************************************************************
        # ACOMODATES ON VECTOR FOR HDD COPY AT THE END OF THE PROGRAM

        low_index = index_offset*_g_size_2_s
        high_index = low_index + _g_size_2_s
        info_tau_re_shape = np.reshape(info_tau,(_g_size_2_t,1))
        info_x_re_shape = np.reshape(info_x,(_g_size_2_t,1))
        c_re_shape = np.reshape(count, (_g_size_2_s, 1))
        c_x_re_shape = np.reshape(count_x, (_g_size_2_s, 1))
        lambda_1_re_shape = np.reshape(array_lambda_1[low_index : high_index ], (_g_size_2_s, 1))
        lambda_2_re_shape = np.reshape(array_lambda_2[low_index : high_index], (_g_size_2_s, 1))
        v_re_shape = np.reshape(array_v[low_index : high_index], (_g_size_2_s, 1))
        mu_re_shape = np.reshape(array_mu[low_index : high_index], (_g_size_2_s, 1))
        eta_re_shape = np.reshape(array_initial_conditions_eta[low_index : high_index], (_g_size_2_s, 1))
        v_stack = np.column_stack((lambda_1_re_shape,
                                   lambda_2_re_shape,
                                   v_re_shape,
                                   mu_re_shape,
                                   c_re_shape,
                                   c_x_re_shape,
                                   eta_re_shape,
                                   info_tau_re_shape,
                                   info_x_re_shape))

        array_to_file[low_index: high_index ,:9] = v_stack #Set the results in the array output
        #*********************************************************************************************
        end_time = (time.time() - start_time)/3600 # Time estimate
        total_time = total_time + end_time # Total time per chunk
        print(f"Total time: {end_time}")

        if _save_maps:
            ev_copy_MAP = cl.enqueue_copy(OCL_Object.queue, MAP_OUT , OCL_Object.MAP_OUT_device)
            ev_copy_MAP_x = cl.enqueue_copy(OCL_Object.queue, MAP_OUT_x , OCL_Object.MAP_OUT_x_device)
            cl.wait_for_events([ev_copy_MAP, ev_copy_MAP_x])
            for name in range(_lambda_1_range_map_out):
                file_map = f"{directory}/map_\
                    {array_lambda_1[index_offset*_g_size_2 + name*int(_g_size_2/_lambda_1_range_map_out)]}_{suffix}.map"
                file_map_x = f"{directory}/map_x_\
                    {array_lambda_1[index_offset*_g_size_2+ name*int(_g_size_2/_lambda_1_range_map_out)]}_{suffix}.map"

                with open(file_map, "w") as file, open(file_map_x, "w") as file_x:
                    np.savetxt(file, MAP_OUT[:,:,name])
                    np.savetxt(file_x, MAP_OUT_x[:,:,name])

        if _save_collisions:
            file_collision = f"{directory}/collision_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
            file_collision_x = f"{directory}/collision_x_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
            with open(file_collision, "w") as file_col, open(file_collision_x, "w") as file_col_x:
                np.savetxt(file_col, col_step)
                np.savetxt(file_col_x, col_step_x)

    ev_copy_4 = cl.enqueue_copy(OCL_Object.queue, mLCE, OCL_Object.mLCE_device)
    ev_copy_5 = cl.enqueue_copy(OCL_Object.queue, max_width_matrix , OCL_Object.max_width_matrix_device)
    ev_copy_6 = cl.enqueue_copy(OCL_Object.queue, min_width_matrix , OCL_Object.min_width_matrix_device)
    ev_copy_7 = cl.enqueue_copy(OCL_Object.queue, output_matrix , OCL_Object.output_matrix_device)

    print("TOTAL_TIME: ", total_time)

    cl.wait_for_events([ev_copy_4, ev_copy_5, ev_copy_6, ev_copy_7])

    #Free GPU global memory pointers*********************************************************
    OCL_Object.free_buffer("half_device")
    OCL_Object.free_buffer("initial_conditions_device")
    OCL_Object.free_buffer("initial_conditions_eta_device")
    OCL_Object.free_buffer("omega_2_device")
    OCL_Object.free_buffer("v_device")
    OCL_Object.free_buffer("lambda_2_device")
    OCL_Object.free_buffer("lambda_1_device")
    OCL_Object.free_buffer("output_matrix_device")
    OCL_Object.free_buffer("max_width_matrix_device")
    OCL_Object.free_buffer("min_width_matrix_device")
    OCL_Object.free_buffer("counter_array_x_device")
    OCL_Object.free_buffer("counter_array_device")
    OCL_Object.free_buffer("mLCE_device")
    OCL_Object.free_buffer("LCE_MAP_x_device")
    OCL_Object.free_buffer("LCE_MAP_device")
    OCL_Object.free_buffer("MAP_OUT_x_device")
    OCL_Object.free_buffer("MAP_OUT_device")
    OCL_Object.free_buffer("partition_tau_device")
    OCL_Object.free_buffer("partition_x_device")
    OCL_Object.free_buffer("counter_information_tau_device")
    OCL_Object.free_buffer("counter_information_x_device")
    OCL_Object.free_buffer("counter_array_collision_device")
    OCL_Object.free_buffer("counter_array_collision_x_device")
    #*****************************************************************************************

    mlce =2.*np.fabs(np.mean(mLCE, axis=1) - 2.)/_wp(_max_iter)
    mLCE_M = 2.*np.fabs(np.sum(mLCE, axis = 1)/_wp(_dim_ensemble) - 2.)/_wp(_max_iter)
    mLCE_mean = 2.*np.fabs(np.mean(mLCE,axis=1)- 2.)/_wp(_max_iter)
    min_width = np.min(min_width_matrix, axis=1)
    max_width = np.max(max_width_matrix, axis=1)
    half_width_vector = max_width - min_width
    full_width_vector = max_width_matrix - min_width_matrix
    metric_entropy_vector = output_matrix * full_width_vector / 2.
    for lam in np.arange(_lambda_1_range):
        half__ = half_width_vector[lam]/2
        mlce_max = np.max(mLCE[lam, :])
        min_tan_map_L = np.min(output_matrix[lam, :] )
        max_tan_map_L = np.max(output_matrix[lam, :] )
        std_tan_map_L = np.std(output_matrix[lam, :] )
        std_width = np.std(full_width_vector[lam, :] )
        h_metric_mean = np.mean(metric_entropy_vector[lam,:]*array_lambda_1[lam])
        h_metric_std = np.std(metric_entropy_vector[lam, :]*array_lambda_1[lam])
        min_widht_orbit = np.min(full_width_vector[lam,:])
        array_to_file[lam, 9:] = np.array([
                                          np.mean(output_matrix[lam, :], axis=0),
                                          mlce_max,
                                          mLCE_M[lam],
                                          mLCE_mean[lam],
                                          mlce[lam],
                                          h_metric_mean,
                                          h_metric_std,
                                          half_width_vector[lam]/2,
                                          half__,
                                          min_tan_map_L,
                                          max_tan_map_L,
                                          std_tan_map_L,
                                          np.mean(full_width_vector[lam,:]/2, axis=0),
                                          min_width[lam]/2.,
                                          max_width[lam]/2.,
                                          std_width,
                                          min_widht_orbit
                                          ])

    file_name = f"{directory}/data_{suffix}.dat"
    save_output_to_file(array_to_file, file_name)

if __name__ == '__main__':
    main()
