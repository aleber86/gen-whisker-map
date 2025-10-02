import numpy as np
import pyopencl as cl

class Platform_Device_OpenCL():
    log_stats = ""
    attrib_for_exec = []
    def __init__(self,  _id : int = 0,
               debug : bool = False, log : bool = False,
                 context : cl.create_some_context = None ) -> dict:
        if context is None:
            self.ctx = cl.create_some_context()
        else:
            self.ctx = context

        try:
          self.device = self.ctx.devices[_id]
          self._device_attrib(debug, log)
        except IndexError:
          print(f"No device found with id: {_id}")
          exit(-1)

        self._values_to_use()

    def _device_attrib(self, debug = False, log = False):
        device_attr = dir(self.device)
        headers = [string for string in device_attr if not string.startswith("__")]
        if "persistent_unique_id" in headers:
          headers.remove("persistent_unique_id")
          headers.append("hashable_model_and_version_identifier")

        attrib = ""
        for value in headers:
          try:
            attrib = value
            self.__setattr__(f"device_{value}", eval(f"self.device.{value}"))
            if debug:
              print(f"{value} : ", eval(f"self.{value}"))

          except cl.Error as msg:
            if debug:
              print(msg)
              print(f"Attrib error: {attrib}")
            if log:
              self.log_stats = f"{self.log_stats}Atrrib error: {attrib}\n"

    def _values_to_use(self):
        self.attrib_for_exec.append(self.device_extensions.split(" "))
        self.attrib_for_exec.append(self.device_max_mem_alloc_size)
        self.attrib_for_exec.append(self.device_local_mem_size)
        self.attrib_for_exec.append(self.device_max_work_item_sizes)

        true_value = True
        dividend = 1
        local_size_values = []
        while true_value:
            division = int(self.device_max_work_group_size/dividend)
            if self.device_max_work_group_size%dividend==0 and division != 1:
                local_size_values.append(division)
                dividend*=2
            else:
                true_value = False
        self.attrib_for_exec.append(local_size_values)


#        return attrib_to_for_exec

class OpenCL_Object(Platform_Device_OpenCL):
    mem_flag = cl.mem_flags
    def __init__(self,  _id : int = 0,
               debug : bool = False, log : bool = False,
                 context : cl.create_some_context = None ) -> dict:
        Platform_Device_OpenCL.__init__(self, _id, debug, log, context)
        self.queue = cl.CommandQueue(self.ctx)

    def buffer_global(self,
                      matrix_to_device : "np.array",
                      assign_name : str, WRITE = True, size_buff = None):
        #Memory buffer structure. WRITE device access mode
        if WRITE:
            WRITE_FLAG = self.mem_flag.WRITE_ONLY
        elif WRITE is None:
            WRITE_FLAG = self.mem_flag.KERNEL_READ_AND_WRITE
        else:
            WRITE_FLAG = self.mem_flag.READ_ONLY

        if matrix_to_device is not None:
            buffer_matrix_device = cl.Buffer(self.ctx, WRITE_FLAG |
                                          self.mem_flag.COPY_HOST_PTR, hostbuf = matrix_to_device)
        else:
            buffer_matrix_device = cl.Buffer(self.ctx, WRITE_FLAG, size = size_buff , hostbuf=None)
        true_value_name = self.__new_parameter(assign_name)
        self.__assign_attrib(true_value_name, assign_name, matrix_to_device)
        self.__assign_attrib(true_value_name, assign_name, buffer_matrix_device, "_device")


    def buffer_local(self, size : int, byte_size : int, buffer_name: str):
      local_buffer = cl.LocalMemory(size*byte_size)
      true_value_name = self.__new_parameter(buffer_name)
      self.__assign_attrib(true_value_name, buffer_name, local_buffer, "_device")

    def program(self, kernel_file_name : str, bulid_options : list = []):
        kernel_read = ''
        if isinstance(kernel_file_name, list):
            for files in kernel_file_name:
                with open(f"{files}", "r") as file:
                    kernel_read = f"{kernel_read}\n{file.read()}"
        else:

            with open(f"{kernel_file_name}", "r") as file:
                kernel_read = file.read()

        prog = cl.Program(self.ctx, kernel_read).build( bulid_options )
        self.__setattr__("kernel", prog)

    def execute_compiled_program(self, program_name : str,  **kwarg):
        exec(f"self.kernel.{program_name}({kwarg})")

    def __new_parameter(self, name_of_attrib : str):
        #Reuse of variables if attrib exist, else creates new one.
        name_variable = self.__dict__.keys()
        if name_of_attrib in name_variable:
            return True
        else:
            return False

    def __assign_attrib(self, true_value_name, assign_name, value_to_assign, _parameter = ""):
        #Creates or reuses attrib
        if true_value_name:
            exec(f"self.{assign_name}{_parameter} = value_to_assign")
        else:
            self.__setattr__(f"{assign_name}{_parameter}", value_to_assign)

    def reset_attrib(self):
        #Kills every attrub except kernels, queues, cl.context
        attrib_name_keys = self.__dict__.copy()
        attrib_name_keys = attrib_name_keys.keys()

        for name in attrib_name_keys:
            if (name.count("kernel")==0 and
                    name.count("queue")==0 and
                    name.count("ctx")==0):
                self.__delattr__(name)

    def free_buffer(self, name):
        try:
            exec(f"self.{name}.release()")
        except AttributeError:
            print(f"{self}.{name} is not a global memory buffer in device")


    def return_attrib(self):
        return self.attrib_for_exec

if __name__ == "__main__" :
    OCL = OpenCL_Object()
    #OCL.reset_attrib()
    #print(OCL.__dict__.keys())
    print(OCL.attrib_for_exec)

