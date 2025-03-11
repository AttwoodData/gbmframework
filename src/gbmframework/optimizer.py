"""
Enhanced System Optimizer with separate thread allocation for training and SHAP
"""

class SystemOptimizer:
    """
    Central manager for system resource detection and optimization across the workflow.
    This class optimizes and coordinates resource utilization for model training, 
    hyperparameter optimization, and SHAP calculations with separate thread allocation
    strategies for each task.

    Usage example:
    --------------
    # Basic usage (adaptive by default):
    optimizer = SystemOptimizer()  
    
    # Adjust thread allocation aggressiveness:
    optimizer = SystemOptimizer(thread_aggressiveness=0.8)
    
    # Force specific thread count:
    optimizer = SystemOptimizer(force_threads=6)
	# Usage examples:
	# 1. Basic usage (adaptive by default):
	# optimizer = SystemOptimizer()
	#
	# 2. More aggressive thread allocation:
	# optimizer = SystemOptimizer(thread_aggressiveness=0.8)
	#
	# 3. Force specific thread count:
	# optimizer = SystemOptimizer(force_threads=6)
    """
    
    def __init__(self, enable_parallel=True, memory_safety=0.8, verbose=True, 
                 min_threads=2, force_threads=None, thread_aggressiveness=0.7, 
                 suppress_numpy_info=True, separate_shap_threads=True):
        """
        Initialize the system optimizer.
        
        Parameters:
        -----------
        enable_parallel : bool, default=True
            Whether to enable parallel computation
        memory_safety : float, default=0.8
            Safety factor for memory utilization (0.0-1.0)
        verbose : bool, default=True
            Whether to print optimization information
        min_threads : int, default=2
            Minimum number of threads to use even with low memory
        force_threads : int, optional
            If provided, forces the use of this many threads regardless of auto-detection
        thread_aggressiveness : float, default=0.7
            How aggressively to allocate threads (0.0-1.0)
            Higher values use more threads but risk memory/resource contention
        suppress_numpy_info : bool, default=True
            Whether to suppress NumPy's detailed build/compiler information
        separate_shap_threads : bool, default=True
            Whether to allocate separate (higher) thread counts for SHAP calculations
            since they occur after training and don't compete for memory
        """
        self.enable_parallel = enable_parallel
        self.memory_safety = memory_safety
        self.verbose = verbose
        self.min_threads = min_threads
        self.force_threads = force_threads
        self.thread_aggressiveness = thread_aggressiveness
        self.suppress_numpy_info = suppress_numpy_info
        self.separate_shap_threads = separate_shap_threads
        
        # If suppressing NumPy info, redirect stdout temporarily
        if self.suppress_numpy_info:
            self._suppress_output()
            
        # Detect resources and configure settings
        self.system_info = self._detect_resources()
        self._configure_global_settings()
        
        # Restore stdout if it was redirected
        if self.suppress_numpy_info:
            self._restore_output()
        
        if self.verbose:
            self._print_system_info()
    
    def _suppress_output(self):
        """Temporarily suppress stdout to hide NumPy build information"""
        import os
        import sys
        import tempfile
        
        # Create a temporary file to redirect stdout
        self._null_file = tempfile.TemporaryFile(mode='w')
        self._old_stdout = sys.stdout
        sys.stdout = self._null_file
    
    def _restore_output(self):
        """Restore stdout after suppressing NumPy build information"""
        import sys
        
        # Restore original stdout
        sys.stdout = self._old_stdout
        self._null_file.close()
    
    def _detect_resources(self):
        """Detect system resources and determine optimal settings."""
        import os
        import multiprocessing
        import platform
        
        # Initialize default conservative values
        system_info = {
            'physical_cores': 2,
            'logical_cores': 4,
            'available_memory_gb': 4.0,
            'total_memory_gb': 8.0,
            'parallel_enabled': self.enable_parallel,
            'resources_detected': False,
            'cpu_info': {},
            'memory_info': {},
            'system_type': platform.system()
        }
        
        # Try to detect actual resources
        try:
            # Try to import psutil for better resource detection
            try:
                import psutil
                # Detect cores
                system_info['physical_cores'] = psutil.cpu_count(logical=False) or 2
                system_info['logical_cores'] = psutil.cpu_count(logical=True) or 4
                
                # Detect memory
                system_info['available_memory_gb'] = psutil.virtual_memory().available / (1024**3)
                system_info['total_memory_gb'] = psutil.virtual_memory().total / (1024**3)
                system_info['memory_percent_available'] = psutil.virtual_memory().available / psutil.virtual_memory().total * 100
                
                # Get CPU frequency
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    system_info['cpu_info']['current_freq_mhz'] = cpu_freq.current
                    if hasattr(cpu_freq, 'max') and cpu_freq.max:
                        system_info['cpu_info']['max_freq_mhz'] = cpu_freq.max
                
                # Get CPU load
                system_info['cpu_load_percent'] = psutil.cpu_percent(interval=0.1)
                
                # Get detailed memory info
                memory = psutil.virtual_memory()
                system_info['memory_info']['total'] = memory.total / (1024**3)
                system_info['memory_info']['available'] = memory.available / (1024**3)
                system_info['memory_info']['used'] = memory.used / (1024**3)
                system_info['memory_info']['percent'] = memory.percent
                
                system_info['resources_detected'] = True
                
            except ImportError:
                # Fallback to multiprocessing if psutil not available
                system_info['logical_cores'] = multiprocessing.cpu_count()
                system_info['physical_cores'] = max(1, int(system_info['logical_cores'] * 0.75))
                system_info['available_memory_gb'] = 8.0  # Conservative assumption
                system_info['total_memory_gb'] = 16.0     # Conservative assumption
            
            # Try to get more detailed CPU info
            try:
                # On Linux, try to get CPU information from /proc/cpuinfo
                if platform.system() == 'Linux':
                    with open('/proc/cpuinfo', 'r') as f:
                        cpu_info = f.readlines()
                    
                    for line in cpu_info:
                        if 'model name' in line:
                            system_info['cpu_info']['model'] = line.split(':')[1].strip()
                            break
                
                # On Windows, try to get CPU information from WMI
                elif platform.system() == 'Windows':
                    try:
                        import wmi
                        w = wmi.WMI()
                        for processor in w.Win32_Processor():
                            system_info['cpu_info']['model'] = processor.Name
                            break
                    except ImportError:
                        # Fall back to platform module
                        system_info['cpu_info']['model'] = platform.processor()
                
                # On macOS
                elif platform.system() == 'Darwin':
                    # Try to get CPU model from sysctl
                    import subprocess
                    try:
                        output = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode('utf-8').strip()
                        system_info['cpu_info']['model'] = output
                    except:
                        # Try alternative Mac CPU model detection
                        try:
                            output = subprocess.check_output(['sysctl', '-n', 'hw.model']).decode('utf-8').strip()
                            system_info['cpu_info']['model'] = output
                        except:
                            system_info['cpu_info']['model'] = "Apple Silicon"
            except:
                # If all else fails, use platform module
                system_info['cpu_info']['model'] = platform.processor() or "Unknown"
                
        except Exception as e:
            if self.verbose:
                print(f"Error detecting system resources: {str(e)}. Using conservative defaults.")
        
        # Calculate optimal thread counts based on resources
        if self.force_threads is not None:
            # Use forced thread count if specified
            system_info['sklearn_threads'] = self.force_threads
            system_info['training_threads'] = self.force_threads
            system_info['hyperopt_workers'] = max(1, min(4, self.force_threads))
            
            # For SHAP, use the same forced thread count
            system_info['shap_threads'] = self.force_threads
        
        elif self.enable_parallel:
            # Calculate threads for training (memory-aware)
            training_threads = self._calculate_adaptive_threads(system_info)
            
            system_info['sklearn_threads'] = training_threads
            system_info['training_threads'] = training_threads
            
            # For hyperopt, use a reduced number to avoid too much parallelism
            system_info['hyperopt_workers'] = max(1, min(4, int(training_threads * 0.6)))
            
            if self.separate_shap_threads:
                # For SHAP, use more threads since it runs separately after training
                # and doesn't compete for memory with the model training process
                system_info['shap_threads'] = max(
                    training_threads,  # At least as many as training
                    min(
                        system_info['physical_cores'],  # But no more than physical cores
                        int(system_info['physical_cores'] * 0.9)  # Use up to 90% of cores
                    )
                )
            else:
                # If not using separate SHAP thread strategy, use same as training
                system_info['shap_threads'] = training_threads
        else:
            # If parallel disabled, use single thread for everything
            system_info['sklearn_threads'] = 1
            system_info['training_threads'] = 1
            system_info['shap_threads'] = 1
            system_info['hyperopt_workers'] = 1
        
        # Hyperopt parallel settings
        system_info['hyperopt_parallel'] = False  # Will be set up on-demand if needed
        system_info['mongodb_process'] = None
        
        return system_info
    
    def _calculate_adaptive_threads(self, system_info):
        """
        Calculate optimal thread count adaptively based on multiple hardware factors.
        
        Parameters:
        -----------
        system_info : dict
            Dictionary containing system resource information
            
        Returns:
        --------
        int
            Optimal number of threads to use
        """
        # Get CPU and memory details
        physical_cores = system_info['physical_cores']
        logical_cores = system_info['logical_cores']
        available_memory_gb = system_info['available_memory_gb']
        total_memory_gb = system_info['total_memory_gb']
        
        # Calculate memory factor - more sophisticated than the original formula
        # Scale with total memory size - larger systems need less conservative scaling
        memory_threshold = max(8, min(32, total_memory_gb / 4))
        memory_factor = min(1.0, available_memory_gb / memory_threshold)
        
        # Adjust memory factor based on the percentage of memory available
        memory_percent = system_info.get('memory_percent_available', 50)
        percent_factor = max(0.5, min(1.0, memory_percent / 50))
        
        # Combine the two memory factors with a bias toward absolute available memory
        combined_memory_factor = (memory_factor * 0.7) + (percent_factor * 0.3)
        
        # Adjust by memory safety parameter
        memory_factor_adjusted = max(0.25, combined_memory_factor * self.memory_safety)
        
        # Calculate thread factor based on physical vs logical cores
        # If we have many more logical than physical cores (hyperthreading), be more conservative
        thread_ratio = physical_cores / logical_cores if logical_cores > 0 else 0.5
        thread_factor = max(0.5, thread_ratio)
        
        # Calculate CPU architecture factor if we can determine it
        arch_factor = 1.0
        cpu_model = system_info.get('cpu_info', {}).get('model', '').lower()
        
        # Adjust for known CPU architectures
        if 'intel' in cpu_model:
            if 'i9' in cpu_model or 'xeon' in cpu_model:
                arch_factor = 1.1  # High-end Intel CPUs handle threading well
            elif 'i7' in cpu_model:
                arch_factor = 1.0  # Standard factor for i7
            elif 'i5' in cpu_model:
                arch_factor = 0.9  # Slightly reduced for i5
            elif 'i3' in cpu_model:
                arch_factor = 0.8  # More reduced for i3
        elif 'amd' in cpu_model:
            if 'ryzen' in cpu_model and ('9' in cpu_model or 'threadripper' in cpu_model):
                arch_factor = 1.2  # High core count AMD CPUs benefit from more threads
            elif 'ryzen' in cpu_model and '7' in cpu_model:
                arch_factor = 1.1  # Ryzen 7 handles threading well
            elif 'ryzen' in cpu_model and '5' in cpu_model:
                arch_factor = 1.0  # Standard factor for Ryzen 5
            elif 'ryzen' in cpu_model and '3' in cpu_model:
                arch_factor = 0.9  # Slightly reduced for Ryzen 3
        elif 'apple' in cpu_model or 'm1' in cpu_model or 'm2' in cpu_model or 'mac' in cpu_model:
            arch_factor = 1.1  # Apple Silicon has good threading performance
        
        # Calculate combined factor
        combined_factor = memory_factor_adjusted * thread_factor * arch_factor * self.thread_aggressiveness
        
        # Calculate raw thread count based on all factors
        raw_threads = int(physical_cores * combined_factor)
        
        # Ensure we stay within reasonable bounds
        threads = max(self.min_threads, min(physical_cores, raw_threads))
        
        return threads
    
    def _configure_global_settings(self):
        """Configure global environment settings for optimal performance."""
        import os
        
        if self.enable_parallel:
            # Set global thread limits for various libraries
            os.environ["OMP_NUM_THREADS"] = str(self.system_info['training_threads'])
            os.environ["MKL_NUM_THREADS"] = str(self.system_info['training_threads'])
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.system_info['training_threads'])
            
            # Configure library-specific threading
            self._configure_libraries()
    
    def _configure_libraries(self):
        """Configure threading for specific ML libraries"""
        try:
            # Configure XGBoost globally if available
            import xgboost as xgb
            xgb.config_context(verbosity=0, nthread=self.system_info['training_threads'])
        except ImportError:
            pass
        
        try:
            # Configure LightGBM verbosity
            import lightgbm as lgb
            try:
                # Try the newer API
                lgb.set_verbosity(0)
            except AttributeError:
                # Older versions might not have set_verbosity
                pass
        except ImportError:
            pass
        
        try:
            # Configure NumPy if available
            import numpy as np
            # Check if we can set threading for numpy
            try:
                # For MKL-based numpy
                from numpy import __config__
                np_info = str(__config__.show())
            except:
                pass
        except ImportError:
            pass
        
        try:
            # Configure scikit-learn if available
            from sklearn.utils import parallel_backend
            # Note: This doesn't actually set anything globally, 
            # but we check if it's available for future use
        except ImportError:
            pass
    
    def _print_system_info(self):
        """Print system information and optimization settings."""
        print("=" * 50)
        print("System Resource Optimization")
        print("=" * 50)
        
        print(f"CPU Information:")
        print(f"  - Physical cores: {self.system_info['physical_cores']}")
        print(f"  - Logical cores: {self.system_info['logical_cores']}")
        if 'cpu_info' in self.system_info and 'model' in self.system_info['cpu_info']:
            print(f"  - CPU model: {self.system_info['cpu_info']['model']}")
        
        print(f"Memory Information:")
        if 'total_memory_gb' in self.system_info:
            print(f"  - Total memory: {self.system_info['total_memory_gb']:.1f} GB")
        print(f"  - Available memory: {self.system_info['available_memory_gb']:.1f} GB")
        if 'memory_percent_available' in self.system_info:
            print(f"  - Memory available: {self.system_info['memory_percent_available']:.1f}%")
        
        print(f"Optimization Settings:")
        print(f"  - Parallel enabled: {self.enable_parallel}")
        print(f"  - Training threads: {self.system_info['training_threads']}")
        print(f"  - SHAP threads: {self.system_info['shap_threads']}")
        print(f"  - Hyperopt workers: {self.system_info['hyperopt_workers']}")
        print("=" * 50)
    
    def setup_parallel_hyperopt(self, exp_key="hyperopt_job"):
        """
        Set up parallel hyperopt with MongoDB if possible.
        
        Parameters:
        -----------
        exp_key : str, default="hyperopt_job"
            Experiment key for hyperopt
            
        Returns:
        --------
        hyperopt.Trials or hyperopt.mongoexp.MongoTrials
            Trials object for hyperopt
        """
        from hyperopt import Trials
        
        # IMPORTANT: Because we've had issues with MongoDB
        # For this version, we'll just use sequential optimization to avoid errors
        if self.verbose:
            print("Using sequential hyperopt optimization for reliability")
        return Trials()
    
    def get_optimized_parameters(self, algorithm):
        """
        Get optimized parameters for a specific algorithm.
        
        Parameters:
        -----------
        algorithm : str
            Algorithm name: 'randomforest', 'xgboost', 'lightgbm', 'catboost'
            
        Returns:
        --------
        dict
            Dictionary of optimized parameters
        """
        algorithm = algorithm.lower()
        params = {'n_jobs': self.system_info['training_threads']}
        
        if algorithm == 'randomforest':
            return {
                'n_jobs': self.system_info['training_threads'],
                'verbose': 0
            }
        elif algorithm == 'xgboost':
            return {
                'n_jobs': self.system_info['training_threads'],
                'verbosity': 0
            }
        elif algorithm == 'lightgbm':
            return {
                'n_jobs': self.system_info['training_threads'],
                'verbose': -1
            }
        elif algorithm == 'catboost':
            return {
                'thread_count': self.system_info['training_threads'],
                'verbose': False
            }
        else:
            return params
    
    def optimize_shap_computation(self, shap_function):
        """
        Decorator to optimize SHAP computation.
        
        Parameters:
        -----------
        shap_function : function
            Function for SHAP computation
            
        Returns:
        --------
        function
            Optimized SHAP function
        """
        def optimized_shap(model, X, algorithm_type, X_train=None, sample_size=None, **kwargs):
            # Set optimal thread count for SHAP - using separate SHAP thread count which is higher
            import os
            old_threads = os.environ.get("OMP_NUM_THREADS", None)
            os.environ["OMP_NUM_THREADS"] = str(self.system_info['shap_threads'])
            
            # Determine optimal sample size based on memory and dataset size
            if sample_size is None:
                available_mem_mb = self.system_info['available_memory_gb'] * 1024
                # Heuristic: 1MB per sample per feature for SHAP computation
                if hasattr(X, 'shape'):
                    n_features = X.shape[1] if len(X.shape) > 1 else 1
                    safe_sample_size = min(len(X), int(available_mem_mb / (n_features * 1.0)))
                    sample_size = max(100, min(1000, safe_sample_size))
                else:
                    sample_size = 200  # Conservative default
            
            try:
                # Call the original SHAP function with optimized parameters
                result = shap_function(
                    model=model,
                    X=X,
                    algorithm_type=algorithm_type,
                    X_train=X_train,
                    sample_size=sample_size,
                    **kwargs
                )
                return result
            finally:
                # Restore original thread settings
                if old_threads is not None:
                    os.environ["OMP_NUM_THREADS"] = old_threads
        
        return optimized_shap
    
    def cleanup(self):
        """Clean up resources when done."""
        if self.system_info.get('mongodb_process'):
            try:
                self.system_info['mongodb_process'].terminate()
                if self.verbose:
                    print("Temporary MongoDB instance stopped")
            except:
                pass
                
