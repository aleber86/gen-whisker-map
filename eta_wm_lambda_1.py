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
    def __init__(self,output_file : str, arguments_of_the_map : dict):
        Experiment_execution.__init__(self, output_file, arguments_of_the_map)

    def set_file_as_initial_conditions(self, input_file : str ):
        with open(input_file, "r") as file_input:
            array_input = np.loadtxt(file_input, dtype = _wp)
            self._lambda_1 = array_input[:,0].copy()
            self._lambda_2 = array_input[:,1].copy()
            self.mu = array_input[:,3].copy()
            self.initial_conditions_eta = array_input[:,4].copy()

    def digest_statistics(self, ensemble_axis = 0, eta_axis = 0, verbose = False) -> np.array:

        #FULL-WITH OF THE LAYER
        #Collapse over ensemble axis, new shape (eta, lambda_1)
        full_width_vector_per_initial_condition = np.max(self.max_width_matrix, axis=ensemble_axis)\
            - np.min(self.min_width_matrix, axis=ensemble_axis)
        _to_aux_file = np.zeros((self._lambda_1_range, 8))


        full_width = np.min(full_width_vector_per_initial_condition, axis=0)
        index_1 = np.where(full_width_vector_per_initial_condition == full_width)
        half_width = full_width/2.
        mLCE_vec = self.output_mLCE_matrix[index_1[0],:,index_1[1]]
        omega_2 = self.initial_conditions_omega_2[0] * np.ones((self._lambda_1_range), dtype=_wp)

        _to_aux_file[:,] = np.column_stack([self._lambda_1, self._lambda_2, omega_2,\
                                            self.mu, self.initial_conditions_eta, self.upsilon,\
                                            half_width, mLCE_vec])
        np.savetxt(self.file_output_name, _to_aux_file)

        headers = ["lambda_1", "lambda_2", "omega_2", "mu", "tilde_eta", "upsilon", "y_hw", "mlce"]
        if verbose:
            for values in _to_aux_file:
                outstring = ""
                for heads, vals in zip(headers, values):
                    outstring = f"{outstring}{heads}: {vals} "
                print(outstring)


if __name__ == '__main__':

    map_aguments = {'iteration_time' : 10**4,
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


    input_file = "./data/wm_eta_found_07-03-2026__21:08:00_gwm_False_it_time_100_eta_size_40_ensemble_size_256.dat"
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
