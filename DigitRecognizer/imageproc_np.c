/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

/*  wrapped cosine function */
static PyObject* imageproc_func_np(PyObject* self, PyObject* args)
{

    PyArrayObject *in_array;
    PyObject      *out_array;

    /*  parse single numpy array argument */
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array))
        return NULL;

    /*  construct the output array, like the input array */
    out_array = PyArray_NewLikeArray(in_array, NPY_ANYORDER, NULL, 0);
    if (out_array == NULL)
        return NULL;

    printf("nd = %d, flag = 0x%x\n", in_array->nd, in_array->flags);
    printf("0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x\n", 
	NPY_ARRAY_C_CONTIGUOUS, 
	NPY_ARRAY_F_CONTIGUOUS,
	NPY_ARRAY_ALIGNED,
	NPY_ARRAY_WRITEABLE,
	NPY_ARRAY_ENSURECOPY,
	NPY_ARRAY_ENSUREARRAY, 
	NPY_ARRAY_FORCECAST, 
	NPY_ARRAY_UPDATEIFCOPY
	);

    Py_INCREF(out_array);
    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

/*  define functions in module */
static PyMethodDef ImageProcMethods[] =
{
     {"imageproc_func_np", imageproc_func_np, METH_VARARGS,
         "Processing images in a 2D numpy array"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC
initimageproc_np(void)
{
     (void) Py_InitModule("imageproc_np", ImageProcMethods);
     /* IMPORTANT: this must be called */
     import_array();
}

