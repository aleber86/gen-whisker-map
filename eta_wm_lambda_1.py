"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
from mod_functions.object_class_functions import File_reader
import time

<<<<<<< HEAD
def main():
    STATUS = "gwm_128_eta_7_2.5"
    _wp = np.float64 # Working Precision
    _wpi = np.int32 # Integer precision for OpenCL kernel args
    start_time = time.time()
    _random_seed = 34567890
    np.random.seed(_random_seed)
    _pi = 4.0*np.arctan(1.0) # Pi definition
    _max_iter = 10**7 # Iteration time
    _dim_essamble = 128 # Ensemble size
    _dim_eta = 1 # Allways 1 <- Global size argument
    _lambda_1_range = 1536 # Lambda_1 number of items, use multiple of 128
    _g_size_0 = _dim_essamble
    _g_size_1 = _dim_eta
    _g_size_2 = _lambda_1_range
    _local = (8,1,4) # Local dimension. Change it for device saturation
    _step = 0.01
    #_omega_2_ini = _wp(np.sqrt(_pi/3.)+2.)
    _omega_2_ini = _wp(np.sqrt(2.5)) # Omega_2 value. Set on irrational
    # GENERALIZED WHISKER MAP FLAG***************************************
    _gwm = True
    _GWM_FLAG = _wpi(0)
    _v_zero = _wp(0.)
    _ONE_ETA_FLAG = _wpi(1)
    if _gwm:
        _v_zero = _wp(1.)
        _GWM_FLAG = _wpi(1)
    #*********************************************************************
=======
_wp = np.float64 # Working Precision
_wpi = np.int32 # Integer precision for OpenCL kernel args
#_LAMBDA_1 = _wp(25.)
#STATUS = f"wm_eta_linespace_it_8_{_LAMBDA_1}"
STATUS = "wm_eta_using_1024_"
_random_seed = 34567890
#_random_seed = 547891248
np.random.seed(_random_seed)
_pi = 4.0*np.arctan(1.0) # Pi definition
_max_iter = 10**7 # Iteration time
_dim_essamble = 640 # Ensemble size
_dim_eta = 1 # Allways 1 <- Global size argument
_lambda_1_range = 1536# Lambda_1 number of items, use multiple of 128
_g_size_0 = _dim_essamble
_g_size_1 = _dim_eta
_g_size_2 = _lambda_1_range
_local = (8,1,8) # Local dimension. Change it for device saturation
_step = 0.01
_SPREAD = _wp(10.**-7.)
#_omega_2_ini = _wp(np.sqrt(_pi/3.)+2.)
_omega_2_ini = _wp(np.sqrt(2.5)) # Omega_2 value. Set on irrational
# GENERALIZED WHISKER MAP FLAG***************************************
_gwm = False
_GWM_FLAG = _wpi(0)
_v_zero = _wp(0.)
_ONE_ETA_FLAG = _wpi(1)
if _gwm:
    _v_zero = _wp(1.)
    _GWM_FLAG = _wpi(1)
#*********************************************************************
_explicit_eta = False
>>>>>>> abba830 (Housekeeping)

    lambda_1_list = []
    lambda_2_list = []
    omega_2_list = []
    mu_list = []
    eta_list = []
    v_list = []
    half_list = []

<<<<<<< HEAD
    all_readed = [lambda_1_list, lambda_2_list, omega_2_list, mu_list, eta_list, v_list, half_list]
    reader = File_reader('aux_pre_cached.dat')
    reader.read_file()
    data_read = reader.get_data()
    for data_stored in data_read:
        for index, arguments in enumerate(all_readed):
            arguments.append(data_stored[index])
=======
all_readed = [lambda_1_list, lambda_2_list, omega_2_list, mu_list, eta_list, v_list, half_list]

with open('aux_10000000_256_wm_7_full.dat', 'r') as file:
    status = True
    while status:
        line = file.readline().split()
        if line != []:
            for index, value in enumerate(all_readed):
                value.append(line[index])
        else:
            status = False
>>>>>>> abba830 (Housekeeping)


    """
    with open('aux_pre_cached.dat', 'r') as file:
        status = True
        while status:
            line = file.readline().split()
            if line != []:
                for index, value in enumerate(all_readed):
                    value.append(line[index])
            else:
                status = False

<<<<<<< HEAD
    """
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
=======
#initial_conditions = (x,t,y)
initial_conditions = np.array(np.random.uniform(-1,1, (_dim_essamble, 3)), dtype = _wp)*_SPREAD
array_initial_conditions = np.array(initial_conditions, dtype=_wp)
lambda_1 = np.array(lambda_1_list, dtype = _wp)
#lambda_1 = np.ones((_lambda_1_range,), dtype=_wp) * _LAMBDA_1
array_omega_2 = np.array(omega_2_list, dtype = _wp)
array_omega_2 = np.ones(array_omega_2.shape, dtype = _wp) * _omega_2_ini
mu = np.array(mu_list, dtype = _wp)

array_lambda_2 = array_omega_2[0] * lambda_1
array_v = array_omega_2**2 *np.sinh(_pi*lambda_1/2.)/np.sinh(lambda_1*_pi/2.*array_omega_2) * _v_zero
array_lambda_1 = lambda_1

if _explicit_eta:
    eta_explicit = _wp(4.2)
    #eta_explicit = np.linspace(0,2*_pi,_lambda_1_range)
    #eta_explicit = lambda_1*_wp(3.57012) + np.ones(lambda_1.shape, dtype = _wp) * _wp(-11.189)

    #a               = 3.57012          +/- 0.0574       (1.608%)
    #b               = -11.189          +/- 0.7487       (6.691%)


    array_initial_conditions_eta = np.ones(array_initial_conditions_eta.shape, dtype=_wp)* eta_explicit
#output_matrix -> CPU
>>>>>>> abba830 (Housekeeping)


    _to_file = np.zeros((_lambda_1_range, 7))

    _to_aux_file = np.zeros((_lambda_1_range, 7))

<<<<<<< HEAD
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
    OCL_Object.program(['kernel_lambda_1.cl', 'src/jacobian.cl', 'src/modulus.cl'], ['-I ./includes'])
=======
output_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range))
max_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
min_width_matrix = np.zeros((_dim_essamble, _dim_eta, _lambda_1_range), dtype=_wp)
OCL_Object = OpenCL_Object()
start_time = time.time()
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
with open('kernel_lambda_1_form.cl', 'r') as file_to_change:
    script = file_to_change.read()
    script = script.replace("#define MAXITER", f"#define MAXITER {_max_iter}")
with open('kernel_lambda_1.cl', 'w') as file:
    file.write(script)
OCL_Object.program(['kernel_lambda_1.cl', 'src/jacobian.cl', 'src/modulus.cl'], ['-I ./includes'])
>>>>>>> abba830 (Housekeeping)

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


    file_name_aux = f"data/aux_eta_pre_cached_{_max_iter}_eta_size_{_dim_eta}"\
               +f"_rand_seed_{_random_seed}_{STATUS}.dat"

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


if __name__ == '__main__':
<<<<<<< Updated upstream
    main()
=======

    map_aguments = {'iteration_time' : 10**7,
                    'initial_condition_size' : 128,
                    'free_parameter_size' : 1,
                    'omega_2_size' : 1,
                    'lambda_1_size' : 128,
                    'lambda_1_ini' : _wp(5.0),
                    'lambda_1_step' : _wp(0.01),
                    'spread_from_center' : _wp(1.e-7),
                    'omega_2_initial_condition' : _wp(np.sqrt(2.5)),
                    'gen_whisker_map' : True,
                    'explicit_eta' : 4.852340625778730931e+00,
                    'pre_catched_eta' : True
                    }
    opencl_arguments_structure = {'global_size' : (map_aguments['initial_condition_size'],
                                                   1,
                                                   map_aguments['lambda_1_size']),
                                  'local_size' : (16,1,4)}
    date = time.strftime('%d-%m-%Y__%H:%M:%S')
    print(f"Start time: {date}")
    STATUS = f"data/wm_eta_found_{date}_gwm_{map_aguments['gen_whisker_map']}_it_time_\
{map_aguments['iteration_time']}_eta_size_\
{map_aguments['free_parameter_size']}_ensemble_size_\
{map_aguments['initial_condition_size']}.dat"


    input_file = "./data/wm_eta_found_26-07-2026__17:35:23_gwm_False_it_time_100000000_eta_size_40_ensemble_size_128.dat"
    Experiment_execution_instance = Experiment_execution_using_file(STATUS, map_aguments)
    Experiment_execution_instance.set_program_script('src/kernel_lambda_1_form.cl')
    Experiment_execution_instance.set_file_as_initial_conditions(input_file)
    start_time = time.time()
    Experiment_execution_instance.create_device_buffers()
    Experiment_execution_instance.execute_experiment(opencl_arguments_structure)
    Experiment_execution_instance.digest_statistics(verbose=True)
    end_time = (time.time() - start_time)/3600
    print("Time elapsed: ", end_time)
    Experiment_execution_instance.save_raw_data()
>>>>>>> Stashed changes
