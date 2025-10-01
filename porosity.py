"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
import time

TIMEIT = []
TIME_START = time.perf_counter()
directory = "data"
_it_step = 7
suffix = f"gwm_{_it_step}_128_eta_fixed"
_wp = np.float64
_wpf = np.float32
_wpi = np.int32
_wpl = np.int64
_wpui = np.uint32
_wpushort = np.uint16
_random_seed = 34567890
np.random.seed(_random_seed)
SPREAD = _wp(1.e-7)
_pi = 4.0*np.arctan(1.0)
_max_iter = _wpui(10**_it_step)
_dim_essamble = _wpui(128)
_omega_2_range = _wpui(1)
_lambda_1_range = _wpui(512)
_lambda_1_range_map_out = _wpui(1)
_common_gid_2_size = 128
_save_maps = False
_save_collisions = False
_gwm_flag = _wpushort(1)


#FIRST KERNEL ATRIBS
_g_size_0 = int(_dim_essamble)  #First Kernel
_g_size_1 = 1   #First Kernel
_g_size_2 = _common_gid_2_size   #First Kernel
#_g_size_2 = 100   #First Kernel
_step = _wp(0.01)
_dim_ang = _wpui(2048)
_dim_y = _wpui(4096)
#_dim_y = _wpui(4096)
_local_id_0 = 4 #First Kernel
_local_id_1 = 1 #First Kernel
_local_id_2 = 8 #First Kernel

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
index_value = _dim_essamble
if index_value > 256:
    index_value = 256
_g_size_0_t = index_value
_g_size_1_t = 1
_g_size_2_t = _common_gid_2_size
#_g_size_2_t = 100
_local_id_0_t = index_value
_local_id_1_t = 1
_local_id_2_t = 1
_grp_sz_0_t = int(_g_size_0_t/_local_id_0_t)
_grp_sz_1_t = int(_g_size_1_t/_local_id_1_t)

#******************************************************************************************

INFORMARTION_RANDOM = _wp(_dim_ang) / (2.*_wp(_max_iter)*_wp(_dim_essamble)*np.log(_dim_ang))
print("Information RANDOM: ", INFORMARTION_RANDOM)

lambda_offset = _wpui(_lambda_1_range/_g_size_2)

initial_conditions = np.array(np.random.uniform(-1,1, (_dim_essamble, 3)), dtype=_wp) * _wp(SPREAD)

lambda_1_list = []
lambda_2_list = []
omega_2_list = []
mu_list = []
eta_list = []
v_list = []
half_list = []

all_readed = [lambda_1_list, lambda_2_list, omega_2_list, mu_list, eta_list, v_list, half_list]

with open('aux.dat', 'r') as file:
    status = True
    while status:
        line = file.readline().split()
        if line != []:
            for index, value in enumerate(all_readed):
                value.append(line[index])
        else:
            status = False



array_lambda_1 = np.array(lambda_1_list, dtype=_wp)
array_initial_conditions = np.array(initial_conditions, dtype=_wp)
array_initial_conditions_eta = np.array(eta_list, dtype=_wp)
array_lambda_2 = np.array(lambda_2_list, dtype=_wp)
array_mu = np.array(mu_list, dtype=_wp)
if _gwm_flag:
    array_omega_2 = np.array(omega_2_list, dtype=_wp)
    array_v = np.array(v_list, dtype=_wp)
else:
    array_omega_2 = np.zeros((len(omega_2_list),), dtype = _wp)
    array_v = np.zeros((len(v_list), ), dtype=_wp)
#output_matrix -> CPU

output_matrix = np.zeros((_lambda_1_range, _dim_essamble), dtype=_wp)
max_width_matrix = np.zeros((_lambda_1_range, _dim_essamble), dtype=_wp)
min_width_matrix = np.zeros((_lambda_1_range, _dim_essamble), dtype=_wp)
array_half = np.array(half_list, dtype = _wp)
counter_array = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
counter_array_x = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
counter_array_colision = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
counter_array_colision_x = np.zeros((_grp_sz_0*_grp_sz_1, _g_size_2_s), dtype=_wpui)
CONSTANT_MAX_POINTS_ADDED = np.ones(_g_size_2_s, dtype = _wp) * _wp(_dim_essamble) * _wp(_max_iter)
CONSTANT_MAX_POINTS_ADDED = CONSTANT_MAX_POINTS_ADDED.astype(_wp)
LCE_MAP = np.zeros(( _g_size_2,_dim_y,_dim_ang), dtype = _wpui)
LCE_MAP_x = np.zeros(( _g_size_2,_dim_y,_dim_ang), dtype = _wpui)
mLCE = np.zeros((_lambda_1_range, _dim_essamble), dtype = _wp)
partition_tau = np.zeros((_g_size_2_t, _dim_ang, _dim_essamble), dtype = _wpui)
partition_x = np.zeros((_g_size_2_t, _dim_ang, _dim_essamble), dtype = _wpui)
counter_information_tau = np.zeros((_dim_ang, _g_size_2_t), dtype=_wp)
counter_information_x = np.zeros((_dim_ang, _g_size_2_t), dtype=_wp)
ones = np.ones((_g_size_2_t,1), dtype = _wp)
#WATCH OUT  !!!!!!!!
MAP_OUT = np.zeros((int(_dim_y)*int(_dim_ang), 3, _lambda_1_range_map_out), dtype = _wpf)
MAP_OUT_x = np.zeros((int(_dim_y)*int(_dim_ang), 3, _lambda_1_range_map_out), dtype = _wpf)
#MAP_OUT = np.zeros((int(_dim_y),int(_dim_ang), _lambda_1_range_map_out), dtype = _wpui)
#MAP_OUT_x = np.zeros((int(_dim_y),int(_dim_ang), _lambda_1_range_map_out), dtype = _wpui)
OCL_Object = OpenCL_Object()

#Buffer CPU -> GPU

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
OCL_Object.buffer_global(counter_array_colision, "counter_array_colision")
OCL_Object.buffer_global(counter_array_colision_x, "counter_array_colision_x")
OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter")
OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_x")
OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_colision_tau")
OCL_Object.buffer_local(_local_id_0_s*_local_id_1_s, 4, "counter_colision_x")
OCL_Object.buffer_local(_local_id_0_s, 4, "counter_partition_tau")
OCL_Object.buffer_local(_local_id_0_s, 4, "counter_partition_x")
OCL_Object.program('one_kernel.cl', ["-I ./includes"])

array_to_file = np.zeros((_lambda_1_range, 26), _wp)
print("Mem. Buffer OK")
index_start = 0
first = 0
#for index_offset in np.arange(1):
count_offset = _wpui(0)

total_time = 0.
for index_offset in np.arange(index_start,lambda_offset):
    start_time = time.time()
    print(f"Start time: {time.strftime('%H:%M:%S')}")
    lambda_offset_it = _wpui(index_offset*_g_size_2)
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
    cl.wait_for_events([ev1])
    print("First Kernel Finished")
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
                                       _wpl(_max_iter)*_wpl(_dim_essamble),
                                       OCL_Object.half_device,
                                       lambda_offset_it,
                                      OCL_Object.counter_colision_tau_device,
                                      OCL_Object.counter_colision_x_device,
                                      OCL_Object.counter_array_colision_device,
                                      OCL_Object.counter_array_colision_x_device,
                                       wait_for=[ev_copy_map])
    cl.wait_for_events([ev2])
    print("Porosity count")


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
                                                   _wpui(_dim_essamble))


    cl.wait_for_events([ev_shannon])
    print("Count * log (count) Shannon_entropy (Sum argument)")
    ev_copy_6 = cl.enqueue_copy(OCL_Object.queue, counter_array, OCL_Object.counter_array_device)
    ev_copy_8 = cl.enqueue_copy(OCL_Object.queue, counter_array_x, OCL_Object.counter_array_x_device)
    ev_inform_tau = cl.enqueue_copy(OCL_Object.queue, counter_information_tau, OCL_Object.counter_information_tau_device)
    ev_inform_x = cl.enqueue_copy(OCL_Object.queue, counter_information_x, OCL_Object.counter_information_x_device)
    ev_colision_tau = cl.enqueue_copy(OCL_Object.queue, counter_array_colision, OCL_Object.counter_array_colision_device)
    ev_colision_x = cl.enqueue_copy(OCL_Object.queue, counter_array_colision_x, OCL_Object.counter_array_colision_x_device)
    cl.wait_for_events([ ev_copy_6, ev_copy_8, ev_inform_tau, ev_inform_x, ev_colision_tau, ev_colision_x ])


    count = np.sum(counter_array, axis=0)
    count_x = np.sum(counter_array_x, axis=0)

    info_tau =1./np.log(_wp(_dim_ang))*(np.log(_wp(_dim_essamble)*_wp(_max_iter))-\
        np.sum(counter_information_tau, axis=0)/(_wp(_dim_essamble)*_wp(_max_iter)))
    info_x =1./np.log(_dim_ang)*(np.log(_wp(_dim_essamble)*_wp(_max_iter))-\
        np.sum(counter_information_x, axis=0)/(_wp(_dim_essamble)*_wp(_max_iter)))

    info_tau_re_shape =(ones- np.reshape(info_tau,(_g_size_2_t,1) ))/INFORMARTION_RANDOM
    info_x_re_shape = (ones-np.reshape(info_x,(_g_size_2_t,1) ))/INFORMARTION_RANDOM

    c_re_shape = np.reshape(count, (_g_size_2_s, 1))
    c_x_re_shape = np.reshape(count_x, (_g_size_2_s, 1))
    index_start_offset = index_start*_g_size_2_s

    col_step = np.sum(counter_array_colision, axis = 0).astype(_wp) - CONSTANT_MAX_POINTS_ADDED
    col_step_x = np.sum(counter_array_colision_x, axis = 0).astype(_wp) - CONSTANT_MAX_POINTS_ADDED
    #print(CONSTANT_MAX_POINTS_ADDED)

    lambda_1_re_shape = np.reshape(array_lambda_1[first+index_start_offset:first+index_start_offset + _g_size_2_s], (_g_size_2_s, 1))
    lambda_2_re_shape = np.reshape(array_lambda_2[first + index_start_offset:first+index_start_offset + _g_size_2_s], (_g_size_2_s, 1))
    v_re_shape = np.reshape(array_v[first + index_start_offset:first+index_start_offset + _g_size_2_s], (_g_size_2_s, 1))
    mu_re_shape = np.reshape(array_mu[first+index_start_offset:first+index_start_offset + _g_size_2_s], (_g_size_2_s, 1))
    eta_re_shape = np.reshape(array_initial_conditions_eta[first + index_start_offset:first+index_start_offset + _g_size_2_s], (_g_size_2_s, 1))
    v_stack = np.column_stack((lambda_1_re_shape,
                               lambda_2_re_shape,
                               v_re_shape,
                               mu_re_shape,
                               c_re_shape,
                               c_x_re_shape,
                               eta_re_shape,
                               info_tau_re_shape,
                               info_x_re_shape))
    array_to_file[first:first + _g_size_2_s ,:9] = v_stack
    first = first + _g_size_2_s
    end_time = (time.time() - start_time)/3600
    total_time = total_time + end_time
    print(f"Total time: {end_time}")
    ev_copy_MAP = cl.enqueue_copy(OCL_Object.queue, MAP_OUT , OCL_Object.MAP_OUT_device)
    ev_copy_MAP_x = cl.enqueue_copy(OCL_Object.queue, MAP_OUT_x , OCL_Object.MAP_OUT_x_device)
    cl.wait_for_events([ev_copy_MAP, ev_copy_MAP_x])
    if _save_maps:
        for name in range(_lambda_1_range_map_out):
            file_map = f"{directory}/map_{array_lambda_1[index_offset*_g_size_2 + name*int(_g_size_2/_lambda_1_range_map_out)]}_{suffix}.map"
            file_map_x = f"{directory}/map_x_{array_lambda_1[index_offset*_g_size_2+ name*int(_g_size_2/_lambda_1_range_map_out)]}_{suffix}.map"
            with open(file_map, "w") as file, open(file_map_x, "w") as file_x:
                np.savetxt(file_map, MAP_OUT[:,:,name])
                #np.savetxt(file_map_x, MAP_OUT_x[:,:,name])

    if _save_collisions:
        file_collision = f"{directory}/collision_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
        file_collision_x = f"{directory}/collision_x_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
        with open(file_collision, "w") as file_col, open(file_collision_x, "w") as file_col_x:
            np.savetxt(file_collision, col_step)
            #np.savetxt(file_collision_x, col_step_x)
ev_copy_4 = cl.enqueue_copy(OCL_Object.queue, mLCE, OCL_Object.mLCE_device)
ev_copy_5 = cl.enqueue_copy(OCL_Object.queue, max_width_matrix , OCL_Object.max_width_matrix_device)
ev_copy_6 = cl.enqueue_copy(OCL_Object.queue, min_width_matrix , OCL_Object.min_width_matrix_device)
ev_copy_7 = cl.enqueue_copy(OCL_Object.queue, output_matrix , OCL_Object.output_matrix_device)
ev_copy_MAP = cl.enqueue_copy(OCL_Object.queue, MAP_OUT , OCL_Object.MAP_OUT_device)
ev_copy_MAP_x = cl.enqueue_copy(OCL_Object.queue, MAP_OUT_x , OCL_Object.MAP_OUT_x_device)

print("TOTAL_TIME: ", total_time)

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
OCL_Object.free_buffer("porosity_array_x_device")
OCL_Object.free_buffer("counter_array_x_device")
OCL_Object.free_buffer("porosity_array_device")
OCL_Object.free_buffer("counter_array_device")
OCL_Object.free_buffer("LCE_MAP_x_device")
OCL_Object.free_buffer("LCE_MAP_device")
OCL_Object.free_buffer("entropy_x_device")
OCL_Object.free_buffer("counter_device")
OCL_Object.free_buffer("entropy_x_device")
OCL_Object.free_buffer("counter_x_device")


mlce =2.*np.fabs(np.mean(mLCE, axis=1) - 2.)/_wp(_max_iter)
mLCE_M = 2.*np.fabs(np.sum(mLCE, axis = 1)/_wp(_dim_essamble) - 2.)/_wp(_max_iter)
mLCE_mean = 2.*np.fabs(np.mean(mLCE,axis=1)- 2.)/_wp(_max_iter)
#half_width_vector = (max_width_matrix - min_width_matrix)/2.
min_width = np.min(min_width_matrix, axis=1)
max_width = np.max(max_width_matrix, axis=1)
half_width_vector = max_width - min_width
full_width_vector = max_width_matrix - min_width_matrix
metric_entropy_vector = output_matrix * full_width_vector / 2.
for lam in np.arange(_lambda_1_range):
    #half__ = np.min(half_width_vector[:,lam], axis = 0)
    half__ = half_width_vector[lam]/2
    #index = np.where(half_width_vector[:,lam] == half__ )[0][0]
#    mlce = mLCE[index,lam]
    mlce_max = np.max(mLCE[lam, :])
    index_mlce = np.where(mLCE[lam,:] == mlce_max)[0][0]
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
headers_elements = ["lambda_1", "lambda_2", "v", "mu",
                    "count_tau", "count_x", "eta", "I_tau", "I_x", "<mLCE(tan_map)>",
                    "mLCE_max(MEGNO)", "mLCE_sum_mean(MEGNO)",
                    "mLCE_mean(MEGNO)","mLCE(MEGNO)" ,"<h>(all)",
                    "std_h(all)", "max_hw", "min_hw", "min mLCE(tan_map)", "max mLCE(tan_map)",
                    "std mLCE(tan_map)","mean_hw(orbits)", "min_hw(max eta)", "max_hw(max eta)", "std_width(orbits)", "min_widht(all)"]
headers = "#"
for index_h, head in enumerate(headers_elements):
    headers = f"{headers}\t({index_h+1}){head}"
headers = f"{headers}\n"
with open(file_name, "w") as file_to_save:
    file_to_save.write(headers)
    np.savetxt(file_to_save, array_to_file)

TIME_END = time.perf_counter()
TIMEIT.append(TIME_END-TIME_START)
with open(f"time_data_{_dim_essamble}_{_max_iter}.time", "a") as time_file:
    np.savetxt(time_file, np.array(TIMEIT))
