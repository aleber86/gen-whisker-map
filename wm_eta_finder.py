"""Calculo de 'whisker map' generalizado"""

import pyopencl as cl
import numpy as np
from mod_opencl.opencl_class_device import OpenCL_Object
import time

#Environment definitions
_wp = np.float64 # Working Precision
_wpi = np.int32 # Integer precision (for OpenCL arguments)
_random_seed = 34567890
_pi = 4.0*np.arctan(1.0) # System definition of pi
np.random.seed(_random_seed)

class Evolution_eta_finder:
    _ONE_ETA_FLAG = _wpi(1)
    _GWM_FLAG = _wpi(0)
    _EXPLICIT_ETA = _wpi(1)
    def __init__(self,  arguments : dict):
        self._max_iter = arguments['iteration_time']
        self._dim_ensemble = arguments['initial_condition_size']
        self._dim_eta = arguments['free_parameter_size']
        self._omega_2_range = arguments['omega_2_size']
        self._lambda_1_range = arguments['lambda_1_size']
        self._lambda_1_ini = arguments['lambda_1_ini']
        self._SPREAD = arguments['spread_from_center']
        self._step = arguments['lambda_1_step']
        self._omega_2_ini = arguments['omega_2_initial_condition']
        self.is_gwm = arguments['gen_whisker_map']
        self.explicit_eta = arguments['explicit_eta']
        self.__FLAGS()
        self._lambda_1 = self.set_lambda_1_initial_conditions()
        self.initial_conditions = self.set_initial_condition_xty()
        self.initial_conditions_eta = self.set_initial_conditions_free_parameter()
        self.initial_conditions_omega_2 = self.set_initial_conditions_omega_2()
        self.upsilon = self.set_upsilon(self.is_gwm)
        self._lambda_2 = self._lambda_1 * self.initial_conditions_omega_2
        self.mu = self.set_mu()

    def __FLAGS(self):
        if self.is_gwm:
            self._GWM_FLAG = _wpi(1)
        if self.explicit_eta is None:
            self._ONE_ETA_FLAG = _wpi(0)
            self._EXPLICIT_ETA = _wpi(0)

    def set_lambda_1_initial_conditions(self) -> np.array:
        lambda_1 = np.array([self._lambda_1_ini + self._step*lam for lam in np.arange(self._lambda_1_range)])
        lambda_1.astype(_wp)
        return lambda_1


    def set_initial_condition_xty(self):

        initial_conditions = np.array(np.random.uniform(-1,1, (self._dim_ensemble, 3)), dtype = _wp)*self._SPREAD
        return initial_conditions

    def set_initial_conditions_free_parameter(self):

        if self.explicit_eta is None:
            initial_conditions_eta = np.random.uniform(0.01, 2.*_pi, size=self._dim_eta)
            initial_conditions_eta[np.where(initial_conditions_eta<0.)] = \
                initial_conditions_eta[np.where(initial_conditions_eta<0.)] + 2.*_pi
        else:
            initial_conditions_eta = np.ones((self._dim_eta,), dtype=_wp)
        initial_conditions_eta.astype(_wp)
        return initial_conditions_eta

    def set_initial_conditions_omega_2(self):
        array_omega_2 = np.array([self._omega_2_ini + self._step * i\
                                  for i in np.arange(self._omega_2_range)], dtype=_wp)
        return array_omega_2

    def set_upsilon(self, gen_whisker_map = True) -> np.array:
        _v_zero = 0.
        if gen_whisker_map:
            _v_zero = 1.
        array_upsilon = self.initial_conditions_omega_2**2 *np.sinh(_pi*self._lambda_1/2.)\
            /np.sinh(self._lambda_1*_pi/2.*self.initial_conditions_omega_2)

        array_upsilon = array_upsilon * _v_zero

        return array_upsilon

    def set_mu(self) -> np.array:

        mu = np.array([np.random.uniform(1.e-12,1.e-8)\
                       for _ in np.arange(self._dim_eta)], dtype=_wp)
        return mu


class Experiment_execution(Evolution_eta_finder):
    def __init__(self, name_file_output : str, arguments_of_the_map : dict):
        Evolution_eta_finder.__init__(self, arguments_of_the_map)
        self.file_output_name = name_file_output
        self.OCL_Object = OpenCL_Object()
        #Host side buffers
        self.output_mLCE_matrix = np.zeros((self._dim_ensemble,\
                                            self._dim_eta, self._lambda_1_range), dtype=_wp)

        self.max_width_matrix = np.zeros((self._dim_ensemble,\
                                          self._dim_eta, self._lambda_1_range), dtype=_wp)

        self.min_width_matrix = np.zeros((self._dim_ensemble,\
                                          self._dim_eta, self._lambda_1_range), dtype=_wp)
        self.__create_device_buffers()

    def __create_device_buffers(self):

        self.OCL_Object.buffer_global(self.initial_conditions, "initial_conditions", False)
        self.OCL_Object.buffer_global(self.initial_conditions_eta, "initial_conditions_eta", False)
        self.OCL_Object.buffer_global(self.initial_conditions_omega_2, "omega_2", False)
        self.OCL_Object.buffer_global(self.upsilon, "v", False)
        self.OCL_Object.buffer_global(self._lambda_2, "lambda_2", False)
        self.OCL_Object.buffer_global(self._lambda_1, "lambda_1", False)
        self.OCL_Object.buffer_global(self.output_mLCE_matrix, "output_matrix")
        self.OCL_Object.buffer_global(self.max_width_matrix, "max_width_matrix")
        self.OCL_Object.buffer_global(self.min_width_matrix, "min_width_matrix")
        self.OCL_Object.buffer_global(self.mu, "mu")

    def set_program_script(self, program_form : str = 'kernel_lambda_1_form.cl',\
                           inculuded =['kernel_lambda_1.cl',\
                                       'src/jacobian.cl', 'src/modulus.cl']) -> None:

        with open(program_form, 'r') as file_to_change:
            script = file_to_change.read()
            script = script.replace("#define MAXITER", f"#define MAXITER {self._max_iter}")
        with open('kernel_lambda_1.cl', 'w') as file:
            file.write(script)
        self.OCL_Object.program(inculuded, ['-I ./includes'])

    def execute_experiment(self, opencl_arguments : dict) -> None:
        _global_size = opencl_arguments['global_size']
        _local_size = opencl_arguments['local_size']

        _max_iter = _wpi(self._max_iter)
        ev_1 =self.OCL_Object.kernel.gen_whisker_map(self.OCL_Object.queue, _global_size, _local_size,
                                            self.OCL_Object.initial_conditions_device,
                                            self.OCL_Object.output_matrix_device,
                                            self.OCL_Object.max_width_matrix_device,
                                            self.OCL_Object.min_width_matrix_device,
                                            self.OCL_Object.lambda_1_device,
                                            self.OCL_Object.lambda_2_device,
                                            self.OCL_Object.v_device,
                                            self.OCL_Object.initial_conditions_eta_device,
                                            self.OCL_Object.omega_2_device, _max_iter,
                                            self.OCL_Object.mu_device,
                                            self._GWM_FLAG,
                                            self._ONE_ETA_FLAG,
                                            self._EXPLICIT_ETA)
        cl.wait_for_events([ev_1])
        ev_copy_1 = cl.enqueue_copy(self.OCL_Object.queue,\
                        self.output_mLCE_matrix, self.OCL_Object.output_matrix_device)
        ev_copy_2 = cl.enqueue_copy(self.OCL_Object.queue,\
                        self.max_width_matrix, self.OCL_Object.max_width_matrix_device)
        ev_copy_3 = cl.enqueue_copy(self.OCL_Object.queue,\
                        self.min_width_matrix, self.OCL_Object.min_width_matrix_device)

        cl.wait_for_events([ev_copy_1, ev_copy_2, ev_copy_3])

    def __mask_finder_ensamble(self, array_in : np.array,\
                               array_comp : np.array) -> np.array:

        """
        Function compares the values in array_comp with array_in to get the mask

        Args:
            array_in : array of values to be compared
            array_comp : array of values to comapre

        Returns:
            mask : array of bool
        """
        mask = False
        print(array_in.shape)
        print(array_comp[0].shape)
        mask = array_in == array_comp

        return mask

    def digest_statistics(self, ensemble_axis = 0, eta_axis = 0) -> np.array:

        #FULL-WITH OF THE LAYER
        #Collapse over ensemble axis, new shape (eta, lambda_1)
        start_time = time.time()
        print(f"Start time: {time.strftime('%H:%M:%S')}")
        full_width_vector_per_initial_condition = np.max(self.max_width_matrix, axis=ensemble_axis)\
            - np.min(self.min_width_matrix, axis=ensemble_axis)
        _to_aux_file = np.zeros((self._lambda_1_range, 7))
        for ind in np.arange(self._lambda_1_range):

            full_width = np.min(full_width_vector_per_initial_condition[:,ind], axis=0)
            index_1 = np.where(full_width_vector_per_initial_condition == full_width)
            half_width = full_width/2.
            mLCE_vec = self.output_mLCE_matrix[:, index_1[0], ind]
            mLCE = np.max(mLCE_vec)
            c = self.initial_conditions_eta[index_1[0][0]]
            mu_val = self.mu[index_1[0][0]]
            lambda_2 = self._lambda_2[ind]
            lambda_1_el = self._lambda_1[ind]
            omega_2 = self.initial_conditions_omega_2[0]
            v = self.upsilon[ind]
            print(f"Lambda_1:{lambda_1_el}  lambda_2: {lambda_2}  omega_2: {omega_2}")
            print(f"mLCE:{mLCE}  half: {half_width}  c: {c}  v: {v}")
            _to_aux_file[ind, :] = np.array([lambda_1_el, lambda_2, omega_2,  mu_val,c, v, half_width])
        end_time = (time.time() - start_time)/3600
        print("Time elapsed: ", end_time)
        np.savetxt(self.file_output_name, _to_aux_file)
        #

if __name__ == '__main__':
    map_aguments = {'iteration_time' : 10**7,
                    'initial_condition_size' : 256,
                    'free_parameter_size' : 40,
                    'omega_2_size' : 1,
                    'lambda_1_size' : 1536,
                    'lambda_1_ini' : _wp(5.0),
                    'lambda_1_step' : _wp(0.01),
                    'spread_from_center' : _wp(1.e-7),
                    'omega_2_initial_condition' : _wp(np.sqrt(2.5)),
                    'gen_whisker_map' : False,
                    'explicit_eta' : None,
                    }
    opencl_arguments_structure = {'global_size' : (map_aguments['initial_condition_size'],
                                                   map_aguments['free_parameter_size'],
                                                   map_aguments['lambda_1_size']),
                                  'local_size' : (4,4,4)}
    STATUS = "wm_eta_found.dat"
    Experiment_execution_instance = Experiment_execution(STATUS, arguments_of_the_map=map_aguments)
    Experiment_execution_instance.set_program_script('./kernel_lambda_1_form.cl')
    Experiment_execution_instance.execute_experiment(opencl_arguments_structure)
    Experiment_execution_instance.digest_statistics()
