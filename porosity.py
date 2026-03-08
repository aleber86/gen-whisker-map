"""
Script calculates a generalized separatrix map (generalized whisker map) proposed
by Chirikov (1979) to estimate the diffusion coefficient in Arnold's model (Arnold 1964).
The map has 2 phase variables (t,x) and 1 action-like variable (y) afected by a term \\upsilon that
depends heavily on a frequency \\omega_2.
Chirikov (1979) obtains the diffusion coefficient under the assumption that, when one of the
phase variables is correlated, the other is random. If \\upsilon << 1 and \\omega_2 > 1 then t is correlated,
and x is random. However, if \\upsilon >> 1 and 0 < \\omega_2 < 1 then x is correlated and t is random
(see Cincotta et al., 2022).



* The output of the model contains:

-   Porosity of the layer.
-   Half-width of the layer.
-   Maximal Lyapunov characteristic exponent.
-   Metric entropy.
-   Phase correlations using Shannon entropy (implementation of Information defined by Cincotta & Giordano, 2018).

*

As a generalized separatrix map we may use it to calculate the case of
\\omega_2 = 0, and get the pendullum separatrix map.


*****For an extended explanaiton of the code,  see:

http://sedici.unlp.edu.ar/handle/10915/189876
"Estudio del mapa de la separatriz generalizado", ch. 7.
                                                   *****


Python / OpenCL program.
Python : (pyopencl, numpy) HOST -> DEVICE (GPU) -> HOST transfers. Statistics calculations.
OpenCL : Paralelized heavy-duty work. Evolution of the dynamical system defined for the map.

Testd on AMD architectures: RDNA 2.0 (RX 6700 XT) and CGN 5.0 (Vega 20).

--------------------------------------------------------------------------------------------
Arnold, V. I. (1964). On the instability of dynamic systems with many degrees of
freedom. Dokl. Akad, Nauk SSSR.

Chirikov, B. V. (1979). A universal instability of many-dimensional oscillator sys-
tems. Physics Reports, 52(5):263–379.


Cincotta, P. M. y Giordano, C. M. (2018). Phase correlations in chaotic dyna-
mics: a Shannon entropy measure. Celestial Mechanics and Dynamical Astronomy,
130(11):74.

Cincotta, P. M., Giordano, C. M., y Shevchenko, I. I. (2022). Diffusion and Lyapunov
timescales in the Arnold model. Phys. Rev. E, 106:044205.

"""

import pyopencl as cl
import numpy as np
from wm_eta_finder import Experiment_execution
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
                        "std_width(orbits)", "min_widht(all)", "y_max_tangent_map"]

    headers = "#"
    for index_h, head in enumerate(headers_elements):
        headers = f"{headers}\t({index_h+1}){head}"
    headers = f"{headers}\n"
    with open(file_name, "w") as file_to_save:
        file_to_save.write(headers)
        np.savetxt(file_to_save, array_to_be_saved)


def mask_gather(array_in : np.array, array_comp : np.array) -> np.array:
    """
    Function compares the values in array_comp with array_in to get the mask

    Args:
        array_in : array of values to be compared
        array_comp : array of values to comapre

    Returns:
        mask : array of bool
    """

    mask = array_in == array_comp
    return mask


class Experiment_execution_full(Experiment_execution):
    def __init__(self,output_file : str, arguments_of_the_map : dict):
        Experiment_execution.__init__(self, output_file, arguments_of_the_map)

        self._save_maps = arguments_of_the_map['save_maps']
        self._save_collisions = arguments_of_the_map['save_collisions']
        self._lambda_1_range_map_out = arguments_of_the_map['map_out_lambda_range']
        self._lambda_1_range_map_out = _wpui(self._lambda_1_range_map_out)
        self._g_size_0 = arguments_of_the_map['first_kernel']['_g_size_0']
        self._g_size_1 = arguments_of_the_map['first_kernel']['_g_size_1']
        self._g_size_2 = arguments_of_the_map['first_kernel']['_g_size_2']
        self._local_id_0 = arguments_of_the_map['first_kernel']['_local_id_0']
        self._local_id_1 = arguments_of_the_map['first_kernel']['_local_id_1']
        self._local_id_2 = arguments_of_the_map['first_kernel']['_local_id_2']
        self._g_size_0_c = arguments_of_the_map['copy_kernel']['_g_size_0_c']
        self._g_size_1_c = arguments_of_the_map['copy_kernel']['_g_size_1_c']
        self._g_size_2_c = arguments_of_the_map['copy_kernel']['_g_size_2_c']
        self._local_id_0_c = arguments_of_the_map['copy_kernel']['_local_id_0_c']
        self._local_id_1_c = arguments_of_the_map['copy_kernel']['_local_id_1_c']
        self._local_id_2_c = arguments_of_the_map['copy_kernel']['_local_id_2_c']
        self._g_size_0_s = arguments_of_the_map['second_kernel']['_g_size_0_s']
        self._g_size_1_s = arguments_of_the_map['second_kernel']['_g_size_1_s']
        self._g_size_2_s = arguments_of_the_map['second_kernel']['_g_size_2_s']
        self._local_id_0_s = arguments_of_the_map['second_kernel']['_local_id_0_s']
        self._local_id_1_s = arguments_of_the_map['second_kernel']['_local_id_1_s']
        self._local_id_2_s = arguments_of_the_map['second_kernel']['_local_id_2_s']
        self._g_size_0_t = arguments_of_the_map['third_kernel']['_g_size_0_t']
        self._g_size_1_t = arguments_of_the_map['third_kernel']['_g_size_1_t']
        self._g_size_2_t = arguments_of_the_map['third_kernel']['_g_size_2_t']
        self._local_id_0_t = arguments_of_the_map['third_kernel']['_local_id_0_t']
        self._local_id_1_t = arguments_of_the_map['third_kernel']['_local_id_1_t']
        self._local_id_2_t = arguments_of_the_map['third_kernel']['_local_id_2_t']

        _grp_sz_0 = int(self._g_size_0_s/self._local_id_0_s)
        _grp_sz_1 = int(self._g_size_1_s/self._local_id_1_s)

        self._dim_ang = arguments_of_the_map['raster_size']['_dim_ang']
        self._dim_y = arguments_of_the_map['raster_size']['_dim_y']

        self._dim_ang = _wpui(self._dim_ang)
        self._dim_y = _wpui(self._dim_y)
        self._max_iter = _wpui(self._max_iter)
        self._lambda_1_range = _wpui(self._lambda_1_range)

        self.output_matrix = np.zeros((self._lambda_1_range, self._dim_ensemble), dtype=_wp)
        self.max_width_matrix = np.zeros((self._lambda_1_range, self._dim_ensemble), dtype=_wp)
        self.min_width_matrix = np.zeros((self._lambda_1_range, self._dim_ensemble), dtype=_wp)
        self.counter_array = np.zeros((_grp_sz_0*_grp_sz_1, self._g_size_2_s), dtype=_wpui)
        self.counter_array_x = np.zeros((_grp_sz_0*_grp_sz_1, self._g_size_2_s), dtype=_wpui)
        self.counter_array_collision = np.zeros((_grp_sz_0*_grp_sz_1, self._g_size_2_s), dtype=_wpui)
        self.counter_array_collision_x = np.zeros((_grp_sz_0*_grp_sz_1, self._g_size_2_s),\
                                                  dtype=_wpui)
        self.CONSTANT_MAX_POINTS_ADDED = np.ones(self._g_size_2_s, dtype = _wp) *\
            _wp(self._dim_ensemble) * _wp(self._max_iter)
        self.CONSTANT_MAX_POINTS_ADDED = self.CONSTANT_MAX_POINTS_ADDED.astype(_wp)
        self.LCE_MAP = np.zeros(( self._g_size_2,self._dim_y,self._dim_ang), dtype = _wpui)
        self.LCE_MAP_x = np.zeros(( self._g_size_2,self._dim_y, self._dim_ang), dtype = _wpui)
        self.mLCE = np.zeros((self._lambda_1_range, self._dim_ensemble), dtype = _wp)
        self.partition_tau = np.zeros((self._g_size_2_t, self._dim_ang, self._dim_ensemble), dtype = _wpui)
        self.partition_x = np.zeros((self._g_size_2_t, self._dim_ang, self._dim_ensemble), dtype = _wpui)
        self.counter_information_tau = np.zeros((self._dim_ang, self._g_size_2_t), dtype=_wp)
        self.counter_information_x = np.zeros((self._dim_ang, self._g_size_2_t), dtype=_wp)
        #WATCH OUT  !!!!!!!!
        self.MAP_OUT = np.zeros((int(self._dim_y)*int(self._dim_ang), 3, self._lambda_1_range_map_out), dtype = _wpf)
        self.MAP_OUT_x = np.zeros((int(self._dim_y)*int(self._dim_ang), 3, self._lambda_1_range_map_out), dtype = _wpf)
        self.array_to_file = np.zeros((self._lambda_1_range, 27), _wp) #Creates array to save final output
        self.array_half = np.empty_like(self._lambda_1, dtype=_wp)
        if ~self._GWM_FLAG:
            self.initial_conditions_omega_2 = np.zeros_like(self.initial_conditions_omega_2, dtype = _wp)
            self.upsilon = np.zeros_like(self.upsilon, dtype=_wp)

    def set_file_as_initial_conditions(self, input_file : str ):
        with open(input_file, "r") as file_input:
            array_input = np.loadtxt(file_input, dtype = _wp)
        self._lambda_1 = array_input[:,0].copy()
        self._lambda_2 = array_input[:,1].copy()
        self.initial_conditions_omega_2 =  array_input[:,2].copy()
        self.mu = array_input[:,3].copy()
        self.initial_conditions_eta = array_input[:,4].copy()
        self.upsilon = array_input[:,5].copy()
        self.array_half = array_input[:,6].copy()
    #Copy is necesary to align the memory blocks

    def create_device_buffers(self):

        self.OCL_Object.buffer_global(self.array_half, "half", False)
        self.OCL_Object.buffer_global(self.initial_conditions, "initial_conditions", False)
        self.OCL_Object.buffer_global(self.initial_conditions_eta, "initial_conditions_eta", False)
        self.OCL_Object.buffer_global(self.initial_conditions_omega_2, "omega_2", False)
        self.OCL_Object.buffer_global(self.upsilon, "v", False)
        self.OCL_Object.buffer_global(self._lambda_2, "lambda_2", False)
        self.OCL_Object.buffer_global(self._lambda_1, "lambda_1", False)
        self.OCL_Object.buffer_global(self.output_matrix, "output_matrix")
        self.OCL_Object.buffer_global(self.max_width_matrix, "max_width_matrix")
        self.OCL_Object.buffer_global(self.min_width_matrix, "min_width_matrix")
        self.OCL_Object.buffer_global(self.counter_array, "counter_array")
        self.OCL_Object.buffer_global(self.counter_array_x, "counter_array_x")
        self.OCL_Object.buffer_global(self.mLCE, "mLCE")
        self.OCL_Object.buffer_global(self.LCE_MAP, "LCE_MAP")
        self.OCL_Object.buffer_global(self.LCE_MAP_x, "LCE_MAP_x")
        self.OCL_Object.buffer_global(self.MAP_OUT, "MAP_OUT")
        self.OCL_Object.buffer_global(self.MAP_OUT_x, "MAP_OUT_x")
        self.OCL_Object.buffer_global(self.partition_tau, "partition_tau")
        self.OCL_Object.buffer_global(self.partition_x, "partition_x")
        self.OCL_Object.buffer_global(self.counter_information_tau, "counter_information_tau")
        self.OCL_Object.buffer_global(self.counter_information_x, "counter_information_x")
        self.OCL_Object.buffer_global(self.counter_array_collision, "counter_array_collision")
        self.OCL_Object.buffer_global(self.counter_array_collision_x, "counter_array_collision_x")
        self.OCL_Object.buffer_local(self._local_id_0_s*self._local_id_1_s, 4, "counter")
        self.OCL_Object.buffer_local(self._local_id_0_s*self._local_id_1_s, 4, "counter_x")
        self.OCL_Object.buffer_local(self._local_id_0_s*self._local_id_1_s, 4, "counter_collision_tau")
        self.OCL_Object.buffer_local(self._local_id_0_s*self._local_id_1_s, 4, "counter_collision_x")
        self.OCL_Object.buffer_local(self._local_id_0_s, 4, "counter_partition_tau")
        self.OCL_Object.buffer_local(self._local_id_0_s, 4, "counter_partition_x")

    def kernel_execution_gen_whisker_map(self, lambda_offset_it, wait = None) -> cl.Event:
        ev1 = self.OCL_Object.kernel.gen_whisker_map(self.OCL_Object.queue,
                                            (self._g_size_0, self._g_size_1, self._g_size_2),
                                            (self._local_id_0, self._local_id_1, self._local_id_2),
                                            self.OCL_Object.initial_conditions_device,
                                            self.OCL_Object.output_matrix_device,
                                            self.OCL_Object.max_width_matrix_device,
                                            self.OCL_Object.min_width_matrix_device,
                                            self.OCL_Object.lambda_1_device,
                                            self.OCL_Object.lambda_2_device, self.OCL_Object.v_device,
                                            self.OCL_Object.initial_conditions_eta_device,
                                            self.OCL_Object.omega_2_device, self._max_iter, self._dim_ang,
                                            self._dim_y, self._lambda_1_range,
                                            self.OCL_Object.half_device,
                                            self.OCL_Object.mLCE_device,
                                            self.OCL_Object.LCE_MAP_device,
                                            self.OCL_Object.LCE_MAP_x_device,
                                            _wpui(lambda_offset_it),
                                            self.OCL_Object.partition_tau_device,
                                            self.OCL_Object.partition_x_device,
                                            self._GWM_FLAG,
                                            wait_for = None
                                            )
        return ev1



    def kernel_execution_reduction(self, lambda_offset_it, wait = None) -> cl.Event:
        ev2 = self.OCL_Object.kernel.reduction(self.OCL_Object.queue,
                                           (self._g_size_0_s, self._g_size_1_s, self._g_size_2_s),
                                           (self._local_id_0_s, self._local_id_1_s, self._local_id_2_s),
                                           self.OCL_Object.LCE_MAP_device,
                                           self.OCL_Object.LCE_MAP_x_device,
                                           self.OCL_Object.counter_array_device,
                                           self.OCL_Object.counter_device,
                                           self.OCL_Object.counter_array_x_device,
                                           self.OCL_Object.counter_x_device,
                                           self._dim_ang,
                                           self._dim_y,
                                           _wpl(self._max_iter)*_wpl(self._dim_ensemble),
                                           self.OCL_Object.half_device,
                                           _wpui(lambda_offset_it),
                                          self.OCL_Object.counter_collision_tau_device,
                                          self.OCL_Object.counter_collision_x_device,
                                          self.OCL_Object.counter_array_collision_device,
                                          self.OCL_Object.counter_array_collision_x_device,
                                           wait_for=wait)
        return ev2

    def kernel_execution_form_matrix_to_array(self, index_offset, wait = None) -> cl.Event:
        ev_copy_map = self.OCL_Object.kernel.from_matrix_to_array(self.OCL_Object.queue,
                                                             (self._g_size_0_c, self._g_size_1_c, self._g_size_2_c),
    #                                                         None,
                                                             (self._local_id_0_c, self._local_id_1_c, self._local_id_2_c),
                                                               self.OCL_Object.LCE_MAP_device,
                                                               self.OCL_Object.LCE_MAP_x_device,
                                                               self.OCL_Object.MAP_OUT_device,
                                                               self.OCL_Object.MAP_OUT_x_device,
                                                                self._dim_ang, self._dim_y,
                                                             self.OCL_Object.half_device,
                                                             self.OCL_Object.lambda_1_device,
                                                             _wpui(self._g_size_2/self._lambda_1_range_map_out),
                                                             _wpui(self._g_size_2),
                                                             _wpui(index_offset),
                                                                wait_for=wait)
        return ev_copy_map


    def kernel_execution_Shannon_entropy(self, wait = None) -> cl.Event:

        ev_shannon = self.OCL_Object.kernel.Shannon_entropy(self.OCL_Object.queue,
                                                       (self._g_size_0_t, self._g_size_1_t, self._g_size_2_t),
                                                       (self._local_id_0_t, self._local_id_1_t, self._local_id_2_t),
                                                       self.OCL_Object.partition_tau_device,
                                                       self.OCL_Object.partition_x_device,
                                                       self.OCL_Object.counter_partition_tau_device,
                                                       self.OCL_Object.counter_partition_x_device,
                                                       self.OCL_Object.counter_information_tau_device,
                                                       self.OCL_Object.counter_information_x_device,
                                                       _wpui(self._dim_ang),
                                                       _wpui(self._dim_ensemble),
                                                        wait_for = wait)

        return ev_shannon


    def update_execution_events(self):
        total_time = 0.
        lambda_offset = _wpui(self._lambda_1_range/self._g_size_2)

        for index_offset in np.arange(lambda_offset):
            start_time = time.time()
            print(f"Start time: {time.strftime('%H:%M:%S')}")
            lambda_offset_it = _wpui(index_offset*self._g_size_2) #Iteration offset per chunk

            ev1 = self.kernel_execution_gen_whisker_map(lambda_offset_it)
            #Wait for the evolution of the whisker map
            cl.wait_for_events([ev1])
            print("First Kernel Finished")
            ev_copy_map = self.kernel_execution_form_matrix_to_array(index_offset, [ev1])
            cl.wait_for_events([ev_copy_map])
            ev2 = self.kernel_execution_reduction(lambda_offset_it,[ev_copy_map])
            cl.wait_for_events([ev2])
            ev_shannon = self.kernel_execution_Shannon_entropy([ev2])

            cl.wait_for_events([ev_shannon])
            print("Count * log (count) Shannon_entropy (Sum argument)")
            #Copy every value out from the GPU
            ev_copy_6 = cl.enqueue_copy(self.OCL_Object.queue, self.counter_array,\
                                        self.OCL_Object.counter_array_device)
            ev_copy_8 = cl.enqueue_copy(self.OCL_Object.queue, self.counter_array_x,\
                                        self.OCL_Object.counter_array_x_device)
            ev_inform_tau = cl.enqueue_copy(self.OCL_Object.queue,\
                                            self.counter_information_tau,\
                                            self.OCL_Object.counter_information_tau_device)
            ev_inform_x = cl.enqueue_copy(self.OCL_Object.queue,\
                                          self.counter_information_x,\
                                          self.OCL_Object.counter_information_x_device)
            ev_collision_tau = cl.enqueue_copy(self.OCL_Object.queue,\
                                               self.counter_array_collision,\
                                               self.OCL_Object.counter_array_collision_device)
            ev_collision_x = cl.enqueue_copy(self.OCL_Object.queue,\
                                             self.counter_array_collision_x,\
                                             self.OCL_Object.counter_array_collision_x_device)
            cl.wait_for_events([ ev_copy_6, ev_copy_8, ev_inform_tau,\
                                ev_inform_x, ev_collision_tau, ev_collision_x ])
            #INFORMATION OF THE SHANNON ENTROPY******************************************************
            info_tau = information_shannon_entropy(self._dim_ang, self._dim_ensemble,\
                                                   self._max_iter, self.counter_information_tau)
            info_x = information_shannon_entropy(self._dim_ang, self._dim_ensemble,\
                                                 self._max_iter, self.counter_information_x)
            #*******************************************************************************************
            #Count of cells occupied********************************************************************
            count = np.sum(self.counter_array, axis=0)
            count_x = np.sum(self.counter_array_x, axis=0)

            col_step = np.sum(self.counter_array_collision, axis = 0).astype(_wp) - self.CONSTANT_MAX_POINTS_ADDED
            col_step_x = np.sum(self.counter_array_collision_x, axis = 0).astype(_wp) - self.CONSTANT_MAX_POINTS_ADDED

            #*******************************************************************************************
            # ACOMODATES ON VECTOR FOR HDD COPY AT THE END OF THE PROGRAM

            low_index = index_offset*self._g_size_2_s
            high_index = low_index + self._g_size_2_s
            info_tau_re_shape = np.reshape(info_tau,(self._g_size_2_t,1))
            info_x_re_shape = np.reshape(info_x,(self._g_size_2_t,1))
            c_re_shape = np.reshape(count, (self._g_size_2_s, 1))
            c_x_re_shape = np.reshape(count_x, (self._g_size_2_s, 1))
            lambda_1_re_shape = np.reshape(self._lambda_1[low_index : high_index ], (self._g_size_2_s, 1))
            lambda_2_re_shape = np.reshape(self._lambda_2[low_index : high_index], (self._g_size_2_s, 1))
            v_re_shape = np.reshape(self.upsilon[low_index : high_index], (self._g_size_2_s, 1))
            mu_re_shape = np.reshape(self.mu[low_index : high_index], (self._g_size_2_s, 1))
            eta_re_shape = np.reshape(self.initial_conditions_eta[low_index : high_index], (self._g_size_2_s, 1))
            v_stack = np.column_stack((lambda_1_re_shape,
                                       lambda_2_re_shape,
                                       v_re_shape,
                                       mu_re_shape,
                                       c_re_shape,
                                       c_x_re_shape,
                                       eta_re_shape,
                                       info_tau_re_shape,
                                       info_x_re_shape))

            self.array_to_file[low_index: high_index ,:9] = v_stack #Set the results in the array output
            #*********************************************************************************************
            end_time = (time.time() - start_time)/3600 # Time estimate
            total_time = total_time + end_time # Total time per chunk


            if self._save_maps:
                ev_copy_MAP = cl.enqueue_copy(self.OCL_Object.queue, self.MAP_OUT , self.OCL_Object.MAP_OUT_device)
                ev_copy_MAP_x = cl.enqueue_copy(self.OCL_Object.queue,\
                                                self.MAP_OUT_x ,self.OCL_Object.MAP_OUT_x_device)
                cl.wait_for_events([ev_copy_MAP, ev_copy_MAP_x])

        print(f"Total time: {end_time}")
        ev_copy_4 = cl.enqueue_copy(self.OCL_Object.queue, self.mLCE, self.OCL_Object.mLCE_device)
        ev_copy_5 = cl.enqueue_copy(self.OCL_Object.queue, self.max_width_matrix , self.OCL_Object.max_width_matrix_device)
        ev_copy_6 = cl.enqueue_copy(self.OCL_Object.queue, self.min_width_matrix , self.OCL_Object.min_width_matrix_device)
        ev_copy_7 = cl.enqueue_copy(self.OCL_Object.queue, self.output_matrix , self.OCL_Object.output_matrix_device)
        print("TOTAL_TIME: ", total_time)

        cl.wait_for_events([ev_copy_4, ev_copy_5, ev_copy_6, ev_copy_7])
    """
    def save_auxilliary_data_map(self, index_offset : int, save_flag : bool = False) -> None:
        if save_flag:
            for name in np.arange(self._lambda_1_range_map_out):
                file_map = f"{directory}/map_\
                    {self._lambda_1[index_offset*self._g_size_2 + name*int(self._g_size_2/self._lambda_1_range_map_out)]}_{suffix}.map"
                file_map_x = f"{directory}/map_x_\
                    {self._lambda_1[index_offset*self._g_size_2+ name*int(self._g_size_2/sefl._lambda_1_range_map_out)]}_{suffix}.map"

                with open(file_map, "w") as file, open(file_map_x, "w") as file_x:
                    np.savetxt(file, MAP_OUT[:,:,name])
                    np.savetxt(file_x, MAP_OUT_x[:,:,name])

        if self._save_collisions:
            file_collision = f"{directory}/collision_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
            file_collision_x = f"{directory}/collision_x_{array_lambda_1[index_offset*_g_size_2]}_{suffix}.dat"
            with open(file_collision, "w") as file_col, open(file_collision_x, "w") as file_col_x:
                np.savetxt(file_col, col_step)
                np.savetxt(file_col_x, col_step_x)
    """
    def digest_statistics(self,file_name, _axis = 1, verbose = False) -> np.array:
        #Maximal Lyapunov characteristic exponent (MEGNO) shape: (_lambda_1_range)
        mlce =2.*np.fabs(np.mean(self.mLCE, axis=1) - 2.)/_wp(self._max_iter)

        #Maximal Lyapunov characteristic exponent (MEGNO) shape: (_lambda_1_range)
        mLCE_M = 2.*np.fabs(np.sum(self.mLCE, axis = _axis)/\
                            _wp(self._dim_ensemble) - 2.)/_wp(self._max_iter)

        #Maximal Lyapunov characteristic exponent (MEGNO) shape: (_lambda_1_range)
        mLCE_mean = 2.*np.fabs(np.mean(self.mLCE,axis= _axis)- 2.)\
            /_wp(self._max_iter)

        #Minimum action-like variable value per layer shape: (_lambda_1_range,)
        min_width = np.min(self.min_width_matrix, axis= _axis)
        #Maximum action-like variable value per layer shape: (_lambda_1_range,)
        max_width = np.max(self.max_width_matrix, axis= _axis)

        #2* y_{\\mathrm{hw}} shape: (_lambda_1_range)
        half_width_vector = max_width - min_width

        #2* y_{\\mathrm{b}} shape: (_lambda_1_range, _dim_ensemble)
        full_width_vector = self.max_width_matrix - self.min_width_matrix

        #L * y_{\\mathrm{b}} shape: (_lambda_1_range, _dim_ensemble)
        metric_entropy_vector = self.output_matrix * full_width_vector / 2.


        #y_{\\mathrm{hw}} shape: (_lambda_1_range)
        half__ = half_width_vector/2.
        mlce_max = np.max(self.mLCE, axis = _axis)
        min_tan_map_L = np.min(self.output_matrix, axis = _axis )
        max_tan_map_L = np.max(self.output_matrix, axis = _axis )
        std_tan_map_L = np.std(self.output_matrix, axis = _axis )
        std_width = np.std(full_width_vector, axis = _axis )
        h_metric_mean = np.mean(metric_entropy_vector, axis = _axis)*self._lambda_1
        h_metric_std = np.std(metric_entropy_vector, axis = _axis)*self._lambda_1
        min_widht_orbit = np.min(full_width_vector, axis = _axis)
        #Half-witdh of the max of mLCE (Tangent Map)
        mask_output_matrix = np.apply_along_axis(mask_gather, 0, self.output_matrix, (max_tan_map_L,))
        mask_output_matrix = mask_output_matrix[0]
        y_max_tangent_map = full_width_vector[mask_output_matrix] / 2.

        self.array_to_file[:, 9:] = np.column_stack([
                                          np.mean(self.output_matrix, axis=_axis),
                                          mlce_max,
                                          mLCE_M,
                                          mLCE_mean,
                                          mlce,
                                          h_metric_mean,
                                          h_metric_std,
                                          half_width_vector/2,
                                          half__,
                                          min_tan_map_L,
                                          max_tan_map_L,
                                          std_tan_map_L,
                                          np.mean(full_width_vector/2, axis=_axis),
                                          min_width/2.,
                                          max_width/2.,
                                          std_width,
                                          min_widht_orbit,
                                          y_max_tangent_map
                                          ])

        save_output_to_file(self.array_to_file, file_name)
    def free_all_global_buffers(self):

        self.OCL_Object.free_buffer("half_device")
        self.OCL_Object.free_buffer("initial_conditions_device")
        self.OCL_Object.free_buffer("initial_conditions_eta_device")
        self.OCL_Object.free_buffer("omega_2_device")
        self.OCL_Object.free_buffer("v_device")
        self.OCL_Object.free_buffer("lambda_2_device")
        self.OCL_Object.free_buffer("lambda_1_device")
        self.OCL_Object.free_buffer("output_matrix_device")
        self.OCL_Object.free_buffer("max_width_matrix_device")
        self.OCL_Object.free_buffer("min_width_matrix_device")
        self.OCL_Object.free_buffer("counter_array_x_device")
        self.OCL_Object.free_buffer("counter_array_device")
        self.OCL_Object.free_buffer("mLCE_device")
        self.OCL_Object.free_buffer("LCE_MAP_x_device")
        self.OCL_Object.free_buffer("LCE_MAP_device")
        self.OCL_Object.free_buffer("MAP_OUT_x_device")
        self.OCL_Object.free_buffer("MAP_OUT_device")
        self.OCL_Object.free_buffer("partition_tau_device")
        self.OCL_Object.free_buffer("partition_x_device")
        self.OCL_Object.free_buffer("counter_information_tau_device")
        self.OCL_Object.free_buffer("counter_information_x_device")
        self.OCL_Object.free_buffer("counter_array_collision_device")
        self.OCL_Object.free_buffer("counter_array_collision_x_device")

if __name__ == '__main__':
    _dim_ensemble = 256
    _common_gid_2_size = 128
    _lambda_1_range_map_out = 1
    index_value = _dim_ensemble
    if index_value > 256: index_value = 256

    map_aguments = {'iteration_time' : 10**4,
                    'initial_condition_size' : _dim_ensemble,
                    'free_parameter_size' : 1,
                    'omega_2_size' : 1,
                    'lambda_1_size' : 1536,
                    'lambda_1_ini' : _wp(5.0),
                    'lambda_1_step' : _wp(0.01),
                    'spread_from_center' : _wp(1.e-7),
                    'omega_2_initial_condition' : _wp(np.sqrt(2.5)),
                    'gen_whisker_map' : True,
                    'explicit_eta' : None,
                    'pre_catched_eta' : True,
                    'raster_size' : {'_dim_ang' : 512,
                                     '_dim_y' : 512},
                    'map_out_lambda_range' : _lambda_1_range_map_out,
                    'save_collisions' : False,
                    'save_maps' : False,
                    'first_kernel' : {'name' : None,
                                      '_g_size_0' : int(_dim_ensemble),
                                      '_g_size_1' : 1,
                                      '_g_size_2' : _common_gid_2_size,
                                      '_local_id_0' : 4,
                                      '_local_id_1' : 1,
                                      '_local_id_2' : 4,
                                    },
                    'copy_kernel' : {'name' : None,
                                     '_g_size_0_c' : 128,
                                     '_g_size_1_c' : 128,
                                     '_g_size_2_c' : int(_lambda_1_range_map_out),
                                     '_local_id_0_c' : 16,
                                     '_local_id_1_c' : 16,
                                     '_local_id_2_c' : 1
                                     },
                    'second_kernel' : {'name' : None,
                                       '_g_size_0_s' : 128,
                                       '_g_size_1_s' : 128,
                                       '_g_size_2_s' : _common_gid_2_size,
                                       '_local_id_0_s' : 16,
                                       '_local_id_1_s' : 16,
                                       '_local_id_2_s' : 1
                                       },
                    'third_kernel'  : {'name' : None,
                                       '_g_size_0_t' : index_value,
                                       '_g_size_1_t' : 1,
                                       '_g_size_2_t' : _common_gid_2_size,
                                       '_local_id_0_t' : index_value,
                                       '_local_id_1_t' : 1,
                                       '_local_id_2_t' : 1
                                       }
                    }
    date = time.strftime('%d-%m-%Y__%H:%M:%S')
    print(f"Start time: {date}")
    STATUS = f"data/full_exec_wm_{date}_gwm_{map_aguments['gen_whisker_map']}_it_time_\
{map_aguments['iteration_time']}_eta_size_\
{map_aguments['free_parameter_size']}_ensemble_size_\
{map_aguments['initial_condition_size']}.dat"


    input_file = "./data/wm_eta_found_07-03-2026__21:08:00_gwm_False_it_time_100_eta_size_40_ensemble_size_256.dat"
    Experiment_execution_instance = Experiment_execution_full(STATUS, map_aguments)
    Experiment_execution_instance.set_program_script('src/one_kernel_form.cl')
    Experiment_execution_instance.set_file_as_initial_conditions(input_file)
    start_time = time.time()
    Experiment_execution_instance.create_device_buffers()
    Experiment_execution_instance.update_execution_events()
    Experiment_execution_instance.digest_statistics(STATUS, verbose=True)
    end_time = (time.time() - start_time)/3600
    print("Time elapsed: ", end_time)

