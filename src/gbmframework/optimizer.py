"""
System Optimizer - Central manager for system resource detection and optimization
"""

class SystemOptimizer:
    """
    Central manager for system resource detection and optimization across the workflow.
    This class optimizes and coordinates resource utilization for model training, 
    hyperparameter optimization, and SHAP calculations.
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
        # For ML tasks, sometimes using all cores can cause memory pressure,
        # especially with multiple parallel estimators
        
        # Memory-based scaling factor
        mem_scale = min(1.0, system_info['available_memory_gb'] / 16.0)  # Scale by available memory
        mem_scale = max(0.25, mem_scale * self.memory_safety)  # Apply safety factor, minimum 0.25
        
        # Set optimal thread counts
        if self.enable_parallel:
            system_info['sklearn_threads'] = max(1, int(system_info['physical_cores'] * mem_scale))
            system_info['training_threads'] = max(1, int(system_info['physical_cores'] * mem_scale))
            system_info['shap_threads'] = max(1, int(system_info['physical_cores'] * mem_scale))
            system_info['hyperopt_workers'] = max(1, min(4, int(system_info['physical_cores'] * 0.75)))
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
                # Configure LightGBM verbosity - FIXED
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
            # Set optimal thread count for SHAP
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