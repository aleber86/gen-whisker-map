from wm_eta_finder import Experiment_execution
import numpy as np
import time

#Environment definitions
_wp = np.float64 # Working Precision
_wpi = np.int32 # Integer precision (for OpenCL arguments)
_random_seed = 34567890
_pi = 4.0*np.arctan(1.0) # System definition of pi
np.random.seed(_random_seed)

class Experiment_execution_using_file(Experiment_execution):
    def __init__(self,output_file : str, arguments_of_the_map : dict, input_file : str):
        Experiment_execution.__init__(self, output_file, arguments_of_the_map)
        self.input_file_path = input_file

    def set_file_as_initial_conditions(self):
        with open(self.input_file_path, "r") as file_input:
            array_input = np.loadtxt(file_input, dtype = _wp)
            self._lambda_1 = array_input[:,0].copy()
            self._lambda_2 = array_input[:,1].copy()
            self.initial_conditions_eta = array_input[:,4].copy()


        assert self.free_parameter_size == self.initial_conditions_eta.shape[0]
        assert self._lambda_1_range == self._lambda_1.shape[0]


if __name__ == '__main__':

    map_aguments = {'iteration_time' : 10**7,
                    'initial_condition_size' : 256,
                    'free_parameter_size' : 1,
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
                                  'local_size' : (16,1,16)}
    date = time.strftime('%d-%m-%Y__%H:%M:%S')
    print(f"Start time: {date}")
    STATUS = f"data/wm_eta_found_{date}_gwm_{map_aguments['gen_whisker_map']}_it_time_\
{map_aguments['iteration_time']}_eta_size_\
{map_aguments['free_parameter_size']}_ensemble_size_\
{map_aguments['initial_condition_size']}.dat"


    input_file = "./data/wm_eta_found_07-03-2026__21:08:00_gwm_False_it_time_100_eta_size_40_ensemble_size_256.dat"
    Experiment_execution_instance = Experiment_execution_using_file(STATUS, map_aguments, input_file)
    Experiment_execution_instance.set_program_script('./kernel_lambda_1_form.cl')

    start_time = time.time()
    Experiment_execution_instance.execute_experiment(opencl_arguments_structure)
    Experiment_execution_instance.digest_statistics()
    end_time = (time.time() - start_time)/3600
    print("Time elapsed: ", end_time)
    Experiment_execution_instance.save_raw_data()
