from abc import ABC, abstractmethod
#from classify_image import ClassifyImage
from tensorflow.python.eager import context
import numpy as np
import tensorflow as tf
import statistics
import time

class AI_Interface(ABC):
    @abstractmethod
    def measure(self, parameters):
        times = []
        
        for i in range (0, self.num_measures):
            #tf.profiler.experimental.start('logdir')
            self.reset()
            self.set_parameters(parameters)
            self.build_model()            
            t1_start = time.perf_counter()            
            with tf.device(self.device):
                self.run()
            t1_stop = time.perf_counter()
            #tf.profiler.experimental.stop()
            if i >= self.num_warmup_skips: # first few rounds are warmup only
                times.append(round((t1_stop-t1_start) * 1000))

        t_mean = round(statistics.mean(times))
        t_stdev = round(statistics.stdev(times))
        print("DEBUG: " + str(times) + " -> " + str(t_mean) + "±" + str(t_stdev) + " ms")
        return t_mean, t_stdev
    
    #@abstractmethod
    #def time_measure(self, parameters):
    #    self.reset()
    #    SetParameters().set_parameters(parameters)
    
    @abstractmethod
    def reset(self):
        print("Setting seeds")
        context._context = None
        context._create_context()
        tf.random.set_seed(130)
        np.random.seed(130)
        
    def set_parameters(self, parameters):    
                ### tf.config.threading.set_*_op_parallelism_threads()
        tf.config.threading.set_inter_op_parallelism_threads(parameters.get(
                "threading.inter_op_parallelism",
                tf.config.threading.get_inter_op_parallelism_threads()))
        tf.config.threading.set_intra_op_parallelism_threads(parameters.get(
                "threading.intra_op_parallelism",
                tf.config.threading.get_intra_op_parallelism_threads()))
        print("DEBUG: threading.inter_op_parallelism: "
                + str(tf.config.threading.get_inter_op_parallelism_threads()))
        print("DEBUG: threading.intra_op_parallelisms: "
                + str(tf.config.threading.get_intra_op_parallelism_threads()))

        ### tf.config.run_functions_eagerly()
        tf.config.run_functions_eagerly(parameters.get(
                "run_functions_eagerly",
                tf.config.functions_run_eagerly()))
        print("DEBUG: run_functions_eagerly: "
                + str(tf.config.functions_run_eagerly()))

        ### tf.config.experimental.enable_tensor_float_32_execution()
        tf.config.experimental.enable_tensor_float_32_execution(parameters.get(
                "experimental.tensor_float_32_execution",
                tf.config.experimental.tensor_float_32_execution_enabled()))
        print("DEBUG: experimental.tensor_float_32_execution: "
                + str(tf.config.experimental.tensor_float_32_execution_enabled()))
                ### tf.config.optimizer.set_jit()
        tf.config.optimizer.set_jit(parameters.get(
                "optimizer.jit",
                tf.config.optimizer.get_jit()))
        print("DEBUG: optimizer.jit: " + str(tf.config.optimizer.get_jit()))

        ### tf.config.optimizer.set_experimental_options()
        optimizer_options = tf.config.optimizer.get_experimental_options()
        for key in ["layout_optimizer", "constant_folding", "shape_optimization",
                    "remapping", "arithmetic_optimization",
                    "dependency_optimization", "loop_optimization",
                    "function_optimization", "debug_stripper",
                    "disable_model_pruning", "scoped_allocator_optimization",
                    "pin_to_host_optimization", "implementation_selector",
                    "auto_mixed_precision", "disable_meta_optimizer"]:
            param_key = "optimizer.experimental." + key
            if param_key in parameters:
                optimizer_options[key] = parameters[param_key]
        tf.config.optimizer.set_experimental_options(optimizer_options)
        print("DEBUG: optimizer.experimental: "
                + str(tf.config.optimizer.get_experimental_options()))


        gpus = tf.config.list_physical_devices("GPU")
        print("DEBUG: GPUs: " + str(gpus))

        ### tf.config.experimental.set_memory_growth() for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, parameters.get(
                    "experimental.gpu.memory_growth",
                    tf.config.experimental.get_memory_growth(gpu)))
            print("DEBUG: experimental.gpu.memory_growth for " + str(gpu.name)
                    + ": " + str(tf.config.experimental.get_memory_growth(gpu)))

        
    @abstractmethod    
    def build_model(self):
        pass
    
    @abstractmethod
    def run(self):
        print("Start")


#class Perform():
#    def perform(self, parameters):
#        a = ClassifyImage(AI_Interface)        
#        a.classifyImageCPU(parameters)