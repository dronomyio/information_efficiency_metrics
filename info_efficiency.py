# python/info_efficiency.py
"""
Information Efficiency Python Wrapper
Direct interface to C++ CUDA/SIMD implementations using ctypes
"""

import ctypes
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LibraryLoader:
    """Helper class to load shared libraries"""
    
    @staticmethod
    def find_library_path():
        """Find the build directory containing .so files"""
        possible_paths = [
            Path(__file__).parent.parent / "build",
            Path(__file__).parent.parent / "lib",
            Path("/usr/local/lib"),
            Path.cwd() / "build"
        ]
        
        for path in possible_paths:
            if path.exists() and any(path.glob("*.so")):
                logger.info(f"Found libraries in: {path}")
                return path
        
        raise RuntimeError("Could not find compiled libraries. Please run 'make' first.")
    
    @staticmethod
    def load_libraries():
        """Load all required shared libraries"""
        lib_path = LibraryLoader.find_library_path()
        
        libraries = {}
        required_libs = [
            "libinfo_efficiency_cuda.so",
            "libinfo_efficiency_simd.so", 
            "libinfo_efficiency_core.so",
            "info_efficiency_cpp.so"
        ]
        
        for lib_name in required_libs:
            lib_file = lib_path / lib_name
            if not lib_file.exists():
                logger.warning(f"Library {lib_name} not found, some features may be unavailable")
                continue
                
            try:
                if lib_name == "info_efficiency_cpp.so":
                    libraries['cpp'] = ctypes.CDLL(str(lib_file))
                else:
                    # Load other libraries to resolve dependencies
                    ctypes.CDLL(str(lib_file))
                logger.info(f"Loaded {lib_name}")
            except Exception as e:
                logger.error(f"Failed to load {lib_name}: {e}")
        
        return libraries.get('cpp')

# Load the main library
try:
    cpp_lib = LibraryLoader.load_libraries()
except Exception as e:
    logger.error(f"Failed to load libraries: {e}")
    cpp_lib = None

# Define C function signatures if library loaded successfully
if cpp_lib:
    # Variance Ratio functions
    cpp_lib.cpp_create_vr_calculator.restype = ctypes.c_void_p
    cpp_lib.cpp_destroy_vr_calculator.argtypes = [ctypes.c_void_p]
    cpp_lib.cpp_compute_variance_ratios.argtypes = [
        ctypes.c_void_p,  # calculator
        ctypes.POINTER(ctypes.c_double),  # returns
        ctypes.c_int,  # n_returns
        ctypes.POINTER(ctypes.c_int),  # horizons
        ctypes.c_int,  # n_horizons
        ctypes.POINTER(ctypes.c_double)  # results
    ]

    # Autocorrelation functions
    cpp_lib.cpp_create_autocorr_processor.restype = ctypes.c_void_p
    cpp_lib.cpp_create_autocorr_processor.argtypes = [ctypes.c_int]
    cpp_lib.cpp_destroy_autocorr_processor.argtypes = [ctypes.c_void_p]
    cpp_lib.cpp_compute_autocorrelations.argtypes = [
        ctypes.c_void_p,  # processor
        ctypes.POINTER(ctypes.c_double),  # returns
        ctypes.c_size_t,  # n_returns
        ctypes.c_size_t,  # max_lag
        ctypes.POINTER(ctypes.c_double)  # results
    ]

    # Polygon reader functions
    cpp_lib.cpp_create_polygon_reader.restype = ctypes.c_void_p
    cpp_lib.cpp_create_polygon_reader.argtypes = [ctypes.c_char_p]
    cpp_lib.cpp_destroy_polygon_reader.argtypes = [ctypes.c_void_p]
    cpp_lib.cpp_compute_returns_from_trades.argtypes = [
        ctypes.c_void_p,  # reader
        ctypes.c_char_p,  # symbol
        ctypes.c_char_p,  # date
        ctypes.c_int64,  # interval_ns
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # returns
        ctypes.POINTER(ctypes.c_size_t)  # count
    ]


class VarianceRatioCalculator:
    """
    Multi-GPU Variance Ratio Calculator
    Computes variance ratios for multiple horizons using CUDA
    """
    
    def __init__(self):
        """Initialize the variance ratio calculator"""
        if not cpp_lib:
            raise RuntimeError("C++ library not loaded")
        
        self.calc = cpp_lib.cpp_create_vr_calculator()
        if not self.calc:
            raise RuntimeError("Failed to create variance ratio calculator")
        
        logger.info("Variance Ratio Calculator initialized (Multi-GPU CUDA)")
    
    def compute(self, returns: np.ndarray, horizons: List[int]) -> Dict[int, float]:
        """
        Compute variance ratios for multiple horizons
        
        Args:
            returns: Array of returns
            horizons: List of horizons to compute VR for
            
        Returns:
            Dictionary mapping horizon to variance ratio
        """
        returns = np.asarray(returns, dtype=np.float64)
        horizons_array = np.asarray(horizons, dtype=np.int32)
        results = np.zeros(len(horizons), dtype=np.float64)
        
        cpp_lib.cpp_compute_variance_ratios(
            self.calc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(returns),
            horizons_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(horizons),
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        return dict(zip(horizons, results))
    
    def compute_batch(self, batch_returns: List[np.ndarray], horizons: List[int]) -> List[Dict[int, float]]:
        """
        Compute variance ratios for multiple time series
        
        Args:
            batch_returns: List of return arrays
            horizons: List of horizons
            
        Returns:
            List of VR results for each time series
        """
        results = []
        for returns in batch_returns:
            results.append(self.compute(returns, horizons))
        return results
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'calc') and self.calc and cpp_lib:
            cpp_lib.cpp_destroy_vr_calculator(self.calc)
            logger.debug("Variance Ratio Calculator destroyed")


class AutocorrelationProcessor:
    """
    SIMD-Accelerated Autocorrelation Processor
    Computes ACF using AVX-512/AVX2 instructions
    """
    
    def __init__(self, num_threads: int = 4):
        """
        Initialize the autocorrelation processor
        
        Args:
            num_threads: Number of OpenMP threads to use
        """
        if not cpp_lib:
            raise RuntimeError("C++ library not loaded")
        
        self.proc = cpp_lib.cpp_create_autocorr_processor(num_threads)
        if not self.proc:
            raise RuntimeError("Failed to create autocorrelation processor")
        
        logger.info(f"Autocorrelation Processor initialized (SIMD with {num_threads} threads)")
    
    def compute(self, returns: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Compute autocorrelations up to max_lag
        
        Args:
            returns: Array of returns
            max_lag: Maximum lag to compute
            
        Returns:
            Array of autocorrelations from lag 0 to max_lag
        """
        returns = np.asarray(returns, dtype=np.float64)
        results = np.zeros(max_lag + 1, dtype=np.float64)
        
        cpp_lib.cpp_compute_autocorrelations(
            self.proc,
            returns.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(returns),
            max_lag,
            results.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        return results
    
    def compute_decay_parameters(self, acf: np.ndarray) -> Tuple[float, float]:
        """
        Estimate exponential decay parameters from ACF
        
        Args:
            acf: Autocorrelation function values
            
        Returns:
            Tuple of (phi, half_life)
        """
        if len(acf) < 2:
            return 1.0, np.inf
        
        # Fit exponential decay to log(|ACF|)
        lags = np.arange(1, len(acf))
        log_acf = np.log(np.abs(acf[1:]) + 1e-10)
        
        # Linear regression
        coeffs = np.polyfit(lags, log_acf, 1)
        phi = np.exp(coeffs[0])
        half_life = np.log(0.5) / coeffs[0] if coeffs[0] != 0 else np.inf
        
        return phi, half_life
    
    def compute_partial_acf(self, returns: np.ndarray, max_lag: int) -> np.ndarray:
        """
        Compute partial autocorrelation function
        
        Args:
            returns: Array of returns
            max_lag: Maximum lag
            
        Returns:
            Array of partial autocorrelations
        """
        # First compute regular ACF
        acf = self.compute(returns, max_lag)
        
        # Use Yule-Walker equations to get PACF
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0
        
        if max_lag > 0:
            pacf[1] = acf[1]
            
            for k in range(2, min(max_lag + 1, len(acf))):
                # Solve for k-th partial autocorrelation
                r = acf[1:k]
                R = np.array([[acf[abs(i-j)] for j in range(k-1)] for i in range(k-1)])
                
                try:
                    phi = np.linalg.solve(R, r)
                    pacf[k] = acf[k] - np.dot(phi, acf[k-1:0:-1])
                except:
                    pacf[k] = 0.0
        
        return pacf
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'proc') and self.proc and cpp_lib:
            cpp_lib.cpp_destroy_autocorr_processor(self.proc)
            logger.debug("Autocorrelation Processor destroyed")


class PolygonDataReader:
    """
    High-frequency data reader for Polygon.io flat files
    Handles nanosecond precision timestamps
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the Polygon data reader
        
        Args:
            api_key: Polygon.io API key
        """
        if not cpp_lib:
            raise RuntimeError("C++ library not loaded")
        
        self.reader = cpp_lib.cpp_create_polygon_reader(api_key.encode('utf-8'))
        if not self.reader:
            raise RuntimeError("Failed to create Polygon reader")
        
        logger.info("Polygon Data Reader initialized")
    
    def compute_returns(self, symbol: str, date: str, interval_ns: int = 1_000_000_000) -> np.ndarray:
        """
        Compute returns from trades data
        
        Args:
            symbol: Stock symbol
            date: Date in YYYY-MM-DD format
            interval_ns: Interval in nanoseconds (default 1 second)
            
        Returns:
            Array of returns
        """
        returns_ptr = ctypes.POINTER(ctypes.c_double)()
        count = ctypes.c_size_t()
        
        cpp_lib.cpp_compute_returns_from_trades(
            self.reader,
            symbol.encode('utf-8'),
            date.encode('utf-8'),
            interval_ns,
            ctypes.byref(returns_ptr),
            ctypes.byref(count)
        )
        
        # Convert to numpy array
        if count.value > 0:
            returns = np.ctypeslib.as_array(returns_ptr, shape=(count.value,)).copy()
            # Free the C memory
            ctypes.CDLL('libc.so.6').free(returns_ptr)
            return returns
        else:
            return np.array([])
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'reader') and self.reader and cpp_lib:
            cpp_lib.cpp_destroy_polygon_reader(self.reader)
            logger.debug("Polygon Data Reader destroyed")


class MicrostructureAnalyzer:
    """
    Combined analyzer for all microstructure metrics
    """
    
    def __init__(self, polygon_api_key: Optional[str] = None, num_threads: int = 4):
        """
        Initialize all components
        
        Args:
            polygon_api_key: Optional Polygon.io API key
            num_threads: Number of threads for SIMD operations
        """
        self.vr_calculator = VarianceRatioCalculator()
        self.acf_processor = AutocorrelationProcessor(num_threads)
        self.polygon_reader = PolygonDataReader(polygon_api_key) if polygon_api_key else None
        
        logger.info("Microstructure Analyzer initialized with all components")
    
    def analyze(self, returns: np.ndarray, 
                horizons: List[int] = [2, 5, 10, 20, 50, 100],
                max_lag: int = 100) -> Dict:
        """
        Perform complete microstructure analysis
        
        Args:
            returns: Array of returns
            horizons: VR horizons to compute
            max_lag: Maximum autocorrelation lag
            
        Returns:
            Dictionary with all metrics
        """
        results = {}
        
        # Compute variance ratios
        vr_results = self.vr_calculator.compute(returns, horizons)
        results['variance_ratios'] = vr_results
        
        # Compute autocorrelations
        acf = self.acf_processor.compute(returns, max_lag)
        results['autocorrelation'] = acf.tolist()
        
        # Compute decay parameters
        phi, half_life = self.acf_processor.compute_decay_parameters(acf)
        results['decay_phi'] = phi
        results['decay_half_life'] = half_life
        
        # Compute partial ACF
        pacf = self.acf_processor.compute_partial_acf(returns, min(20, max_lag))
        results['partial_acf'] = pacf.tolist()
        
        # Summary statistics
        results['summary'] = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'skewness': self._compute_skewness(returns),
            'kurtosis': self._compute_kurtosis(returns),
            'num_observations': len(returns)
        }
        
        return results
    
    def _compute_skewness(self, returns: np.ndarray) -> float:
        """Compute skewness of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 3)
    
    def _compute_kurtosis(self, returns: np.ndarray) -> float:
        """Compute excess kurtosis of returns"""
        mean = np.mean(returns)
        std = np.std(returns)
        if std == 0:
            return 0.0
        return np.mean(((returns - mean) / std) ** 4) - 3.0


# Convenience functions
def test_installation():
    """Test if the C++ libraries are properly installed"""
    try:
        logger.info("Testing Information Efficiency installation...")
        
        # Generate test data
        np.random.seed(42)
        test_returns = np.random.randn(1000) * 0.01
        
        # Test variance ratio
        vr = VarianceRatioCalculator()
        vr_results = vr.compute(test_returns, [2, 5, 10])
        logger.info(f"Variance Ratios: {vr_results}")
        
        # Test autocorrelation
        acf = AutocorrelationProcessor()
        acf_results = acf.compute(test_returns, 10)
        logger.info(f"ACF (first 5 lags): {acf_results[:5]}")
        
        logger.info("✓ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_installation()
