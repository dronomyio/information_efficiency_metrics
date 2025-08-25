#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

// External C functions from our libraries
extern void* create_vr_calculator();
extern void destroy_vr_calculator(void* calculator);
extern void compute_variance_ratios(void* calculator, const double* returns,
                                   int n_returns, const int* horizons,
                                   int n_horizons, double* results);

extern void* create_autocorr_processor(int num_threads);
extern void destroy_autocorr_processor(void* processor);
extern void compute_autocorrelations(void* processor, const double* returns,
                                    size_t n_returns, size_t max_lag, double* results);

// Python wrapper for create_vr_calculator
static PyObject* py_create_vr_calculator(PyObject* self, PyObject* args) {
    void* calc = create_vr_calculator();
    return PyLong_FromVoidPtr(calc);
}

// Python wrapper for destroy_vr_calculator
static PyObject* py_destroy_vr_calculator(PyObject* self, PyObject* args) {
    PyObject* calc_obj;
    if (!PyArg_ParseTuple(args, "O", &calc_obj))
        return NULL;
    
    void* calc = PyLong_AsVoidPtr(calc_obj);
    destroy_vr_calculator(calc);
    
    Py_RETURN_NONE;
}

// Python wrapper for compute_variance_ratios
static PyObject* py_compute_variance_ratios(PyObject* self, PyObject* args) {
    PyObject* calc_obj;
    PyArrayObject* returns_array;
    PyArrayObject* horizons_array;
    
    if (!PyArg_ParseTuple(args, "OOO", &calc_obj, &returns_array, &horizons_array))
        return NULL;
    
    void* calc = PyLong_AsVoidPtr(calc_obj);
    
    // Get data pointers
    double* returns = (double*)PyArray_DATA(returns_array);
    int* horizons = (int*)PyArray_DATA(horizons_array);
    int n_returns = PyArray_SIZE(returns_array);
    int n_horizons = PyArray_SIZE(horizons_array);
    
    // Create output array
    npy_intp dims[1] = {n_horizons};
    PyArrayObject* results = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* results_data = (double*)PyArray_DATA(results);
    
    // Call C function
    compute_variance_ratios(calc, returns, n_returns, horizons, n_horizons, results_data);
    
    return (PyObject*)results;
}

// Python wrapper for create_autocorr_processor
static PyObject* py_create_autocorr_processor(PyObject* self, PyObject* args) {
    int num_threads = 4;
    if (!PyArg_ParseTuple(args, "|i", &num_threads))
        return NULL;
    
    void* proc = create_autocorr_processor(num_threads);
    return PyLong_FromVoidPtr(proc);
}

// Python wrapper for compute_autocorrelations
static PyObject* py_compute_autocorrelations(PyObject* self, PyObject* args) {
    PyObject* proc_obj;
    PyArrayObject* returns_array;
    int max_lag;
    
    if (!PyArg_ParseTuple(args, "OOi", &proc_obj, &returns_array, &max_lag))
        return NULL;
    
    void* proc = PyLong_AsVoidPtr(proc_obj);
    double* returns = (double*)PyArray_DATA(returns_array);
    size_t n_returns = PyArray_SIZE(returns_array);
    
    // Create output array
    npy_intp dims[1] = {max_lag + 1};
    PyArrayObject* results = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    double* results_data = (double*)PyArray_DATA(results);
    
    // Call C function
    compute_autocorrelations(proc, returns, n_returns, max_lag, results_data);
    
    return (PyObject*)results;
}

// Method definitions
static PyMethodDef module_methods[] = {
    {"create_vr_calculator", py_create_vr_calculator, METH_NOARGS, 
     "Create a variance ratio calculator"},
    {"destroy_vr_calculator", py_destroy_vr_calculator, METH_VARARGS,
     "Destroy a variance ratio calculator"},
    {"compute_variance_ratios", py_compute_variance_ratios, METH_VARARGS,
     "Compute variance ratios"},
    {"create_autocorr_processor", py_create_autocorr_processor, METH_VARARGS,
     "Create autocorrelation processor"},
    {"compute_autocorrelations", py_compute_autocorrelations, METH_VARARGS,
     "Compute autocorrelations"},
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "info_efficiency_cpp",
    "Information Efficiency C++ Extensions",
    -1,
    module_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_info_efficiency_cpp(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&module_def);
}
