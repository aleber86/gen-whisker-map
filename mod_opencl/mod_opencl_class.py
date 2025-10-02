import numpy as np
import pyopencl as cl

global _pi, _wp, _vec_comp
_pi = np.float64(np.pi) #Def. PI en numpy
_wp = np.float64  #REAL*8
_bytes_sz = np.float64(0).nbytes
_vec_comp = np.int64(4) #Cuatro componentes espaciales (vectores)
#----------------------------------------------------------------------------------------------------
#BLOQUE DE CLASE PARA LA EJECUCION DEL KERNEL
#----------------------------------------------------------------------------------------------------
class Generador():
    def __init__(self, dim_vec_x_ens =128 , t_int =10**4, lsz = 64, debug = False):
        """
        La clase Generador define todos los parametros a utilizar por la API de OpenCL.
        Se espera que no se realicen modificaciones ya que el problema propuesto esta
        determinado. En caso de modificar los valores dimensionales de los vectores (4 -> X),
        se debe modificar el kernel adjunto como RK8_11.cl
        """
        self.DEBUG = debug
        self._dim = np.int32(dim_vec_x_ens)
        self._ensambles = np.int32(6)
        self.dim_print = np.int32(500)
        self._dim_tiempo_int = np.int64(t_int) #Siempre tiene que ser un np.int64(int)
        self.vector_in = []
        self.vector_out = []
        self.vec_print_3d = []
        self.exit_c = None
        self.index_vector_print = None
        ####################
        #Variables de OpenCL
        ####################
        self.contexto = None
        self.queue = None
        self.mem_flag = None
        self.vec_out_dev = None
        self.vec_in_dev = None
        self.kernel = None
        self.local_mem = None
        self.local_mem_k = None
        self.local_mem_y = None
        self.program = None
        self.exit_d = None
        self.local_sz = lsz
        self.vec_print_3d_dev = None
        self.index_vector_print_dev = None

    def valores_iniciales(self, cond_in = [np.float64(-1.69), np.float64(-1.13),
                                           np.float64(-0.173),np.float64( 0.053),
                                           np.float64(0.55),np.float64(1.55)]):
        #Los valores de la lista corresponden a los centros de los ensambles en
        #frecuencia angualar_2

        self._ensambles = len(cond_in) #Cntidad de ensambles en funcion de las cond_in.
        self.vector_in = np.array([]).astype(_wp)

        for val_in in cond_in:
            if self.DEBUG:
                #Opcion debug=True (cond. in = 1 p/toda componente y vector)
                vector_fab = np.array([[np.float64(1.0),np.float64(1.0),
                                        np.float64(1.0), np.float64(1.0)]
                                       for k in range(self._dim)]).astype(_wp)
            else:
                #Opcion debug=False (cond. in normales del problema)
                vector_fab = np.array([[np.float64(_pi), np.float64(0.0),
                                        np.float64(1.0) +
                                        np.float64((np.float64(-1))**j*((np.float64(1.0)-np.random.random())*np.float64(10**-6))),
                                       val_in + np.float64((np.float64(-1))**j*((1.0 - np.random.random())*np.float64(10**-6)))]
                                       for j in range(self._dim)]).astype(_wp)
            if len(self.vector_in) == 0:
                self.vector_in = vector_fab
            else:
                self.vector_in = np.vstack((self.vector_in, vector_fab))
        self.vector_in = np.matrix(self.vector_in).astype(_wp) #Forma matricial de los valores iniciales.

        self.vector_out_asign()
        del vector_fab #Se elimina la matriz auxiliar para reducir uso de memoria en ejecucion.

        self.reset_buffer_print()

    def vector_out_asign(self):
        self.vector_out = np.empty_like(self.vector_in).astype(_wp)

    def OpenCL_contexto(self, archivo = 'opencl_kernel/gpu.cl'):
        #Se genera el contexto como los buffer de memoria
        #Se establece el tamano de la memoria local


        self.contexto = cl.create_some_context()
        self.queue = cl.CommandQueue(self.contexto)
        self.mem_flag = cl.mem_flags
        self.buffer_write()
        self.buffer_read()
        if isinstance(archivo, list):
            kernel_script = ''
            for files in archivo:
                with open(files, 'r') as file_kernel:
                    kernel_script = f"{kernel_script}\n{file_kernel.read()}"

        else:

            with open(archivo, 'r') as file_kernel:
                kernel_script = file_kernel.read()


        self.program = cl.Program(self.contexto, kernel_script).build()
        ####################################################################
        ####################################################################

        print(cl.program_build_info)
        self.kernel = self.program.Arnold_H #Programa dentro del kernel de OpenCL

        #8 bytes por float64 son 128*6*4 (vec, ensa, comp. vec)---Espacio en memoria local_mem
        #del dispositivo.
        self.local_mem = cl.LocalMemory(self.local_sz*(5)*_bytes_sz)  #VER ejecucion_kernel() tamano mem_local
        self.local_mem_y = cl.LocalMemory(self.local_sz*np.int32(0).nbytes)
        self.local_mem_k = cl.LocalMemory(np.int32(0).nbytes)
    def buffer_write(self, arg = None):
        #Construye el buffer de salida; se espara un solo uso, pero se programa para el caso que se quiera
        #ejecutar otro kernel escrito y re-usar el buffer de memoria junto con el queue y el contexto

        #self.exit_c genera un conjunto de 1 por cada vector dentro de cada ensamble.
        #Control de integracion. Ver dopri.cl, error de integracion -> salida 1 en el vector
        #mal integrado.

        self.exit_c = np.asarray([[np.int32(0) for i in range(self._dim)]
                                  for j in range(self._ensambles)]).astype(np.int32)
        if arg is None:
            arg = self.vector_out
        self.exit_d = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                self.mem_flag.USE_HOST_PTR, hostbuf = self.exit_c)

        self.vec_out_dev = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                     self.mem_flag.USE_HOST_PTR, hostbuf = arg)

        self.vec_print_3d_dev = cl.Buffer(self.contexto, self.mem_flag.WRITE_ONLY |
                                          self.mem_flag.USE_HOST_PTR, hostbuf = self.vec_print_3d)
    def buffer_read(self, arg = None, arg2 = None, arg3 = None):
        #Construye el buffer de entrada y se usa para la re-entrada de datos procesados
        if arg is None:
            arg = self.vector_in

        if arg2 is None:
            arg2 = self.vec_print_3d

        if arg3 is None:
            arg3 = self.index_vector_print

        self.vec_in_dev =  cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                     self.mem_flag.COPY_HOST_PTR, hostbuf = arg)
        self.vec_print_3d_dev =  cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                           self.mem_flag.COPY_HOST_PTR, hostbuf = arg2)
        self.index_vector_print_dev = cl.Buffer(self.contexto, self.mem_flag.READ_ONLY |
                                                self.mem_flag.COPY_HOST_PTR, hostbuf = arg3)

    def enq_copy(self, ):
        #Copia los valores procesados por el dispositivo a la memoria del host
        #la funcion se ejecuta cada vez que se va a reiniciar un ciclo de calculo.

        cl.enqueue_copy(self.queue, self.vector_out, self.vec_out_dev )
        cl.enqueue_copy(self.queue, self.exit_c, self.exit_d)
        cl.enqueue_copy(self.queue, self.vec_print_3d, self.vec_print_3d_dev)
        cl.enqueue_copy(self.queue, self.index_vector_print, self.index_vector_print_dev)
#        self.vector_out.astype(_wp)


    def ejecucion_kernel(self, t_i = 0, t_f = 100):
        #Ejecucion del kernel.
        #Los valores iniciales pueden ser cambiados por argumentos desde la linea
        #de comandos.

        tiempo_i = np.float64(t_i)
        tiempo_f = np.float64(t_f)
        self.kernel(self.queue, (self._dim*self._ensambles,), (self.local_sz,), self.index_vector_print_dev, self._dim,
                    self.dim_print, self.vec_out_dev, self.vec_in_dev, self._dim_tiempo_int, tiempo_i, tiempo_f,
                    self.local_mem, self.vec_print_3d_dev, self.exit_d, self.local_mem_k, self.local_mem_y)

    def reset_buffer_print(self):
        self.vec_print_3d = np.zeros((self._ensambles*self._dim*self.dim_print,6)).astype(_wp)
        self.index_vector_print = np.zeros((self._ensambles*self._dim)).astype(np.int32)
