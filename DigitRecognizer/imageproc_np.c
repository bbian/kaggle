/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define WIDTH 28
#define HEIGHT 28

static unsigned long long result;
static unsigned char *data;
static unsigned char pixelVal(int x, int y)
{
	return *(data + y * WIDTH + x);
}

/*  wrapped cosine function */
static PyObject* imageproc_func_np(PyObject* self, PyObject* args)
{

	PyArrayObject *in_array;
	PyObject      *out_array;
	npy_intp size = 1;

	/*  parse single numpy array argument */
	if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &in_array))
		return NULL;


/**********************  DEBUG PRINT, COMMENTING OUT ***********************
	printf("nd = %d, dimensions[0] = %d, flag = 0x%x\n", in_array->nd, in_array->dimensions[0], in_array->flags);
	
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
	printf("type: %d\n", in_array->descr->type_num);
***************************************************************************/

	// Example of accessing image array element
	data = in_array->data;

/*************************************************************************
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
			printf("%d ", pixelVal(i, j));
		}
	}
	printf("\n");
*************************************************************************/

	// Example of accessing image label. 
	// If image label is not from 0 to 9, that means it's a test image
	unsigned char label = pixelVal(0, HEIGHT);
/***********************************************************************
	if (label == 255)
		printf("Image is from test set\n");
	else
		printf("Image label = %d\n", label);
***********************************************************************/

	// Return competition test image result
	// Use the following hard-coded 5 as an example
	// Note - if image already has a label, it means that
	// it is from training set, and subsequently the return 
	// value will be ignored

	result = 5;
	out_array = PyArray_SimpleNewFromData(1, &size, NPY_LONGLONG, &result);
	if (out_array == NULL)
		return NULL;

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

