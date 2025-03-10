"""
System Optimizer - Central manager for system resource detection and optimization
"""

class SystemOptimizer:
    """
    Central manager for system resource detection and optimization across the workflow.
    This class optimizes and coordinates resource utilization for model training and
    hyperparameter optimization.
    """
    
    def __init__(self, enable_parallel=True, memory_safety=0.8, verbose=True):
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
        """
        self.enable_parallel = enable_parallel
        self.memory_safety = memory_safety
        self.verbose = verbose
        self.system_info = self._detect_resources()
        self._configure_global_settings()
        
        if self.verbose:
            self._print_system_info()
    
    def _detect_resources(self):
        """Detect system resources and determine optimal settings."""
        import os
        import multiprocessing
        
        # Initialize default conservative values
        system_info = {
            'physical_cores': 2,
            'logical_cores': 4,
            'available_memory_gb': 4.0,
            'parallel_enabled': self.enable_parallel,
            'resources_detected': False
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
                system_info['resources_detected'] = True
            except ImportError:
                # Fallback to multiprocessing if psutil not available
                system_info['logical_cores'] = multiprocessing.cpu_count()
                system_info['physical_cores'] = max(1, int(system_info['logical_cores'] * 0.75))
                system_info['available_memory_gb'] = 8.0  # Conservative assumption
                system_info['total_memory_gb'] = 16.0    # Conservative assumption
        except Exception as e:
            if self.verbose:
                print(f"Error detecting system resources: {str(e)}. Using conservative defaults.")
        
        # Calculate optimal thread counts based on resources
        # Memory-based scaling factor
        mem_scale = min(1.0, system_info['available_memory_gb'] / 16.0)  # Scale by available memory
        mem_scale = max(0.25, mem_scale * self.memory_safety)  # Apply safety factor, minimum 0.25
        
        # Get the total available cores
        total_cores = system_info['logical_cores']
        
        # Calculate optimal threads - dynamically adjust based on system size
        if total_cores >= 8:
            # For large systems, use 75% of logical cores to leave headroom for OS
            optimal_threads = max(1, int(total_cores * 0.75 * mem_scale))
        elif total_cores >= 4:
            # For medium systems, use 80% of logical cores
            optimal_threads = max(1, int(total_cores * 0.8 * mem_scale))
        else:
            # For small systems, use all cores
            optimal_threads = total_cores
        
        # Set uniform thread allocation for all operations
        if self.enable_parallel:
            # Use optimal thread count for everything
            system_info['training_threads'] = optimal_threads
            system_info['sklearn_threads'] = optimal_threads
            system_info['hyperopt_workers'] = min(4, optimal_threads)  # Hyperopt can be unstable with too many workers
        else:
            # If parallel disabled, use single thread for everything
            system_info['training_threads'] = 1
            system_info['sklearn_threads'] = 1
            system_info['hyperopt_workers'] = 1
        
        # Hyperopt parallel settings
        system_info['hyperopt_parallel'] = False  # Will be set up on-demand if needed
        system_info['mongodb_process'] = None
        
        return system_info
    
    def _configure_global_settings(self):
        """Configure global environment settings for optimal performance."""
        import os
        
        if self.enable_parallel:
            # Set global thread limits for various libraries
            os.environ["OMP_NUM_THREADS"] = str(self.system_info['training_threads'])
            os.environ["MKL_NUM_THREADS"] = str(self.system_info['training_threads'])
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.system_info['training_threads'])
            
            try:
                # Configure XGBoost globally if available
                import xgboost as xgb
                xgb.config_context(verbosity=0, nthread=self.system_info['training_threads'])
            except ImportError:
                pass
            
            try:
                # Configure LightGBM verbosity
                import lightgbm as lgb
                # Different versions of LightGBM have different logger APIs
                try:
                    # Try the newer API
                    lgb.set_verbosity(0)
                except AttributeError:
                    # Older versions might not have set_verbosity
                    pass
            except ImportError:
                pass
    
    def _print_system_info(self):
        """Print system information and optimization settings."""
        print("=" * 50)
        print("System Resource Optimization")
        print("=" * 50)
        
        print(f"CPU Cores:")
        print(f"  - Physical cores: {self.system_info['physical_cores']}")
        print(f"  - Logical cores: {self.system_info['logical_cores']}")
        
        print(f"Memory:")
        if 'total_memory_gb' in self.system_info:
            print(f"  - Total memory: {self.system_info['total_memory_gb']:.1f} GB")
        print(f"  - Available memory: {self.system_info['available_memory_gb']:.1f} GB")
        
        print(f"Optimization Settings:")
        print(f"  - Parallel enabled: {self.enable_parallel}")
        print(f"  - Training threads: {self.system_info['training_threads']}")
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
    
    def cleanup(self):
        """Clean up resources when done."""
        if self.system_info.get('mongodb_process'):
            try:
                self.system_info['mongodb_process'].terminate()
                if self.verbose:
                    print("Temporary MongoDB instance stopped")
            except:
                pass