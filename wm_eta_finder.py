"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
import time

STATUS = "eta_finder"
start_time = time.time()
_wp = np.float64 # Working Precision
_wpi = np.int32 # Integer precision (for OpenCL arguments)
_random_seed = 34567890
np.random.seed(_random_seed)
_pi = 4.0*np.arctan(1.0) # System definition of pi
_max_iter = 10**7 # Iteration time
_dim_essamble = 64 # Ensemble of iniitial conditions
#Change _dim_eta size of the eta uniform distribution
_dim_eta = 40
_omega_2_range = 1
#Change the _lambda_1_range for lambda calc. (use 128 multiple)
_lambda_1_range  = 1536
_SPREAD = _wp(1.e-7) # Spread around (0,0,0) of initial conditions
#************************************************************************************
# Do not change this. Global problem size (OpenCL)
_g_size_0 = _dim_essamble
_g_size_1 = _dim_eta
_g_size_2 = _lambda_1_range
_local = (4,4,4)
#************************************************************************************
_step = 0.01
_omega_2_ini = _wp(np.sqrt(_pi/3.)+2.) # Could let this unchanged if you use only WM
#************************************************************************************
# GENERALIZED WHISKER MAP FLAG!
#
# if _gwm = True, then it will calculate eta value for the generalized whisker map
_gwm = False
_ONE_ETA_FLAG = _wpi(0)
_v_zero = _wp(0.)
_GWM_FLAG = _wpi(0)
if _gwm:
    _v_zero = _wp(1.)
    _GWM_FLAG = _wpi(1)
#************************************************************************************

#***************************************************************************************
#initial_conditions = (x,t,y)
# Change lambda_ini for the starting lambda value !!!!
lambda_ini = _wp(5.0) # <----------------Change it if you want to calc. another interval
# *************************************************************************************
#
lambda_1 = np.array([lambda_ini + _step*lam for lam in np.arange(_lambda_1_range)])
epsilon = 1./lambda_1**2
denom = _wp(epsilon*np.sinh(0.5*_pi*lambda_1))
mu = np.array([np.random.uniform(1.e-12,1.e-8) for _ in np.arange(_dim_eta)], dtype=_wp)
initial_conditions = np.array(np.random.uniform(-1,1, (_dim_essamble, 3)), dtype = _wp)*_SPREAD

#Initial conditions eta random****************************************************************
initial_conditions_eta = np.random.uniform(0.01, 2.*_pi, size=_dim_eta)
initial_conditions_eta[np.where(initial_conditions_eta<0.)] = initial_conditions_eta[np.where(initial_conditions_eta<0.)] + 2.*_pi
#**********************************************************************************************
array_initial_conditions = np.array(initial_conditions, dtype=_wp)
array_initial_conditions_eta = np.array(initial_conditions_eta, dtype=_wp)
array_omega_2 = np.array([_omega_2_ini + _step * i for i in np.arange(_omega_2_range)], dtype=_wp)

array_lambda_2 = array_omega_2[0] * lambda_1
array_v = array_omega_2**2 *np.sinh(_pi*lambda_1/2.)/np.sinh(lambda_1*_pi/2.*array_omega_2)
array_v = array_v * _v_zero
array_lambda_1 = lambda_1


#output_matrix -> CPU
_to_file = np.zeros((_lambda_1_range, 7))
_to_aux_file = np.zeros((_lambda_1_range, 7))

output_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range))
max_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
min_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
#*****************************************************************************************************
#OpenCL Memory buffers
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
#***********************************************************************************************************
#Program load and build. If you want to add opt, do it inside list [], e.g. ["-cl-single-precision-constant"]
#Do not recomend using any 'fast math' optimization
OCL_Object.program(['src/kernel_lambda_1.cl', 'src/jacobian.cl', 'src/modulus.cl'], ['-I ./includes'])
#************************************************************************************************************
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


file_name_aux = f"data/aux_{_max_iter}_eta_size_{_dim_eta}"\
           +f"_rand_seed_{_random_seed}_omega_2_range_{_omega_2_range}_wm_lambda_{STATUS}.dat"

file_aux = open(file_name_aux, 'w')
#******************************************************************************************
#HALF-WITH OF THE LAYER
half_width_vector = np.max(max_width_matrix, axis=0) - np.min(min_width_matrix, axis=0)
#******************************************************************************************
for ind in np.arange(_lambda_1_range):

    half_width = np.min(half_width_vector[:,ind], axis=0)
    index_1 = np.where(half_width_vector == half_width)
    half_width = half_width/2.
    mLCE_vec = output_matrix[:, index_1[0], ind]
    mLCE = np.max(mLCE_vec)
    c = array_initial_conditions_eta[index_1[0][0]]
    mu_val = mu[index_1[0][0]]
    lambda_2 = array_lambda_2[ind]
    lambda_1_el = array_lambda_1[ind]
    omega_2 = array_omega_2[0]
    v = array_v[ind]
    print(f"Lambda_1:{lambda_1_el}  lambda_2: {lambda_2}  omega_2: {omega_2}")
    print(f"mLCE:{mLCE}  half: {half_width}  c: {c}  v: {v}")
    _to_aux_file[ind, :] = np.array([lambda_1_el, lambda_2, omega_2,  mu_val,c, v, half_width])
"""

THIS USES THE HALF-WIDTH AND mLCE OF THE LAST ORBIT INSIDE THE LAYER



half_width_vector = max_width_matrix - min_width_matrix
for ind in np.arange(_lambda_1_range):

    mLCE_vec = output_matrix[:, :, ind]
    mLCE = np.max(mLCE_vec)
    index = np.where(mLCE_vec[:,:] == mLCE)
    half_width = np.min(half_width_vector[index[0], :, ind])/2
    #COMMENT
    half_width = np.max(half_width_vector[:,:,ind], axis=0)
#    print(half_width_vector)
    half_width_min = np.min(half_width)
#    print(half_width_min)
    #index = np.where(half_width_vector[:,:,ind] == half_width)
    index_1 = np.where(half_width == half_width_min)
    half_width = half_width_min/2.
    #mLCE_vec = output_matrix[index[0], index[1], ind]
    mLCE_vec = output_matrix[:, index_1[0][0], ind]
    mLCE = np.max(mLCE_vec)

    #c = array_initial_conditions_eta[ind,index[1][0]]
    print(index_1[0][0])
    c = array_initial_conditions_eta[index_1[0][0]]
    #mu_val = mu[index[1][0]]
    mu_val = mu[index_1[0][0]]
#    mu_val = 0
    lambda_2 = array_lambda_2[ind]
    lambda_1_el = array_lambda_1[ind]
    omega_2 = array_omega_2[0]
    v = array_v[ind]
    print(f"Lambda_1:{lambda_1_el}  lambda_2: {lambda_2}  omega_2: {omega_2}")
    print(f"mLCE:{mLCE}  half: {half_width}  c: {c}  v: {v}")
    _to_file[ind, :] = np.array([lambda_1_el, lambda_2, omega_2,  mLCE, half_width, c, v])
    _to_aux_file[ind, :] = np.array([lambda_1_el, lambda_2, omega_2,  mu_val,c, v, half_width])
"""
end_time = (time.time() - start_time)/3600
np.savetxt(file_aux, _to_aux_file)
#


print(f"Total time: {end_time}")
