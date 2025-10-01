"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
import time

STATUS = "gwm_128_eta_7_2.5"
start_time = time.time()
_random_seed = 34567890
np.random.seed(_random_seed)
_pi = 4.0*np.arctan(1.0)
_max_iter = 10**7
_dim_essamble = 128
_dim_eta = 1
_omega_2_range = 1
_lambda_1_range = 1536
_g_size_0 = _dim_essamble
_g_size_1 = _dim_eta
_g_size_2 = _lambda_1_range
_local = (8,1,4)
_wp = np.float64
_wpi = np.int32
_step = 0.01
#_omega_2_ini = _wp(np.sqrt(_pi/3.)+2.)
_omega_2_ini = _wp(np.sqrt(2.5))
_gwm = True
_GMW_FLAG = _wpi(0)
_v_zero = _wp(0.)
_ONE_ETA_FLAG = _wpi(1)
if _gwm:
    _v_zero = _wp(1.)
    _GWM_FLAG = _wpi(1)


lambda_1_list = []
lambda_2_list = []
omega_2_list = []
mu_list = []
eta_list = []
v_list = []
half_list = []

all_readed = [lambda_1_list, lambda_2_list, omega_2_list, mu_list, eta_list, v_list, half_list]

with open('aux_pre_cached.dat', 'r') as file:
    status = True
    while status:
        line = file.readline().split()
        if line != []:
            for index, value in enumerate(all_readed):
                value.append(line[index])
        else:
            status = False


array_initial_conditions_eta = np.array(eta_list, dtype=_wp)

#initial_conditions = (x,t,y)
initial_conditions = np.array(np.random.uniform(-1,1, (_dim_essamble, 3)), dtype = _wp)*_wp(10.**-7.)
array_initial_conditions = np.array(initial_conditions, dtype=_wp)
lambda_1 = np.array(lambda_1_list, dtype = _wp)
array_omega_2 = np.array(omega_2_list, dtype = _wp)
mu = np.array(mu_list, dtype = _wp)

array_lambda_2 = array_omega_2[0] * lambda_1
array_v = array_omega_2**2 *np.sinh(_pi*lambda_1/2.)/np.sinh(lambda_1*_pi/2.*array_omega_2) * _v_zero
array_lambda_1 = lambda_1
#output_matrix -> CPU


_to_file = np.zeros((_lambda_1_range, 7))

_to_aux_file = np.zeros((_lambda_1_range, 7))

output_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range))
max_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
min_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
OCL_Object = OpenCL_Object()
#Buffer CPU -> GPU
OCL_Object.buffer_global(array_initial_conditions, "initial_conditions", False)
OCL_Object.buffer_global(array_initial_conditions_eta, "initial_conditions_eta", False)
OCL_Object.buffer_global(array_omega_2, "omega_2", False)
OCL_Object.buffer_global(array_v, "v", False)
OCL_Object.buffer_global(array_lambda_2, "lambda_2", False)
OCL_Object.buffer_global(array_lambda_1, "lambda_1", False)
OCL_Object.buffer_global(output_matrix, "output_matrix")
OCL_Object.buffer_global(max_width_matrix, "max_width_matrix")
OCL_Object.buffer_global(min_width_matrix, "min_width_matrix")
OCL_Object.buffer_global(mu, "mu")
OCL_Object.program('kernel_lambda_1.cl', [])

_max_iter = _wpi(_max_iter)

ev_1 = OCL_Object.kernel.gen_whisker_map(OCL_Object.queue,(_g_size_0, _g_size_1, _g_size_2),_local,
                                    OCL_Object.initial_conditions_device,
                                    OCL_Object.output_matrix_device,
                                    OCL_Object.max_width_matrix_device,
                                    OCL_Object.min_width_matrix_device,
                                    OCL_Object.lambda_1_device,
                                    OCL_Object.lambda_2_device,
                                    OCL_Object.v_device,
                                    OCL_Object.initial_conditions_eta_device,
                                    OCL_Object.omega_2_device, _max_iter,
                                    OCL_Object.mu_device,
                                    _GWM_FLAG,
                                    _ONE_ETA_FLAG)
cl.wait_for_events([ev_1])
cl.enqueue_copy(OCL_Object.queue, output_matrix, OCL_Object.output_matrix_device)
cl.enqueue_copy(OCL_Object.queue, max_width_matrix, OCL_Object.max_width_matrix_device)
cl.enqueue_copy(OCL_Object.queue, min_width_matrix, OCL_Object.min_width_matrix_device)


file_name_aux = f"data/aux_eta_pre_cached_{_max_iter}_mu_size_{_dim_eta}"\
           +f"_rand_seed_{_random_seed}_omega_2_range_{_omega_2_range}_gwm_lambda_1_{STATUS}.dat"

file_aux = open(file_name_aux, 'w')
half_width_vector = np.max(max_width_matrix, axis=0) - np.min(min_width_matrix, axis=0)
for ind in np.arange(_lambda_1_range):

    half_width = np.min(half_width_vector[:,ind], axis=0)
    index_1 = np.where(half_width_vector == half_width)
    half_width = half_width/2.
    mLCE_vec = output_matrix[:, index_1[0], ind]
    mLCE = np.max(mLCE_vec)

    c = array_initial_conditions_eta[ind]
    mu_val = mu[index_1[0][0]]
    lambda_2 = array_lambda_2[ind]
    lambda_1_el = array_lambda_1[ind]
    omega_2 = array_omega_2[0]
    v = array_v[ind]
    print(f"Lambda_1:{lambda_1_el}  lambda_2: {lambda_2}  omega_2: {omega_2}")
    print(f"mLCE:{mLCE}  half: {half_width}  c: {c}  v: {v}")
    _to_aux_file[ind, :] = np.array([lambda_1_el, lambda_2, omega_2,  mu_val,c, v, half_width])


end_time = (time.time() - start_time)/3600
np.savetxt(file_aux, _to_aux_file)
#

print(f"Total time: {end_time}")
