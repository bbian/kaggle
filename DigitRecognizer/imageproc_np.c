/*  Example of wrapping the cos function from math.h using the Numpy-C-API. */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 28
#define HEIGHT 28

static unsigned long long result;
static unsigned char *data;
static unsigned char pixelVal(int x, int y)
{
	return *(data + y * WIDTH + x);
}

static unsigned char height,width,yi[64],xj[64],givenclass,class;
static unsigned char    numeralImageR[64][64],numeralImageG[64][64],numeralImageB[64][64],bmpdata[3*64*64];
static unsigned char binarydata[64][64];
static unsigned short   endingnumber,endingY[16],endingX[16],bifnumber;
static unsigned char endingtracenumber[12],endingI[12],endingJ[12],endingtraceI[12][64],endingtraceJ[12][64],kend,endingconnectivity[12];
static unsigned char holenumber,holeymax[5],holeymin[5],holexmax[5],holexmin[5];
static unsigned char    imagedata[64][64];


static unsigned char  connectivity,trace_number,trace_y[784],trace_x[784],bifY[8],bifX[8],bifconnectivity[8][5];
static signed short blockpixelnum,blockmaxi,blockmaxj,blockmini,blockminj,inflexiondir1[8][8],inflexiondir2[8][8];
static unsigned char  biftracenumber[8][5],branchnumber[8],biftraceI[8][5][64],biftraceJ[8][5][64];
static unsigned char  inflexionnumber,inflexionPos[8][8],inflexionendingnum[8];



static void writeBMPImage(unsigned char bitcount,char tmpfile[])// for RGB image ==24, for grayimage = 8; for binaryimage  = 1


{
    FILE *fp;
    int  filesize,imagesize,k;
    short y,x;
    int n;
    short padding,datashiftvalue;
    //    printf(" in writeimage height-width %d %d \n",height,width);
    if(bitcount == 24)
    {
        padding=(3*width)%4;
        filesize=(int)height*(3*width+(4-padding))+54;  // +54 ?
        imagesize=(int)height*(3*width+(4-padding));
        bmpdata[10]=54;   // [13]-[10] is Offset to get to pixel info     (bytes)
        bmpdata[11]=0;
        bmpdata[12]=0;
        bmpdata[13]=0;
        datashiftvalue=54; // = bmpdata[11]*256+bmp[10]=54
    }
    
    if(bitcount <= 8) // including bitcount = 1
    {
        padding=width%4;
        
        //        printf("padding %d \n",padding);
        filesize=(int) height*(width+(4-padding))+1078;
        imagesize=(int) height*(width+4-padding);
        bmpdata[10]=54;   // [13]-[10] is Offset to get to pixel info   (datashiftvalue)
        bmpdata[11]=4;    // 4*256=1024;
        bmpdata[12]=0;
        bmpdata[13]=0;
        datashiftvalue=1078; // = 4*256+54(bmp[10])=1078
    }
    bmpdata[0]=66;  // it is 0x42
    bmpdata[1]=77;  // it is 0x4D // for windows  bmpdata[1]=77  (ascii BM)
    //    bmpdata[1]=65;  // it is 0x41 // for os2  bmpdata[1]=65  (ascii BA)
    //  bmpdata[2] to bmpdata[8] used to define bmp filesize
    bmpdata[8]=filesize/16777216;
    bmpdata[4]=(filesize-bmpdata[8]*16677216)/65536;
    bmpdata[3]=(filesize-bmpdata[8]*16677216-bmpdata[4]*65536)/256;
    bmpdata[2]=filesize-bmpdata[8]*16677216-bmpdata[4]*65536-bmpdata[3]*256;
    
    bmpdata[6]=0;      // reserved1;
    bmpdata[7]=0;
    bmpdata[8]=0;      // reserved2;
    bmpdata[9]=0;
    
    //    printf("datashiftvalue %d \n",datashiftvalue);
    //    printf("chk filesize by bmpdata %d \n",bmpdata[8]*16677216+bmpdata[4]*65536+bmpdata[3]*256+bmpdata[2]);
    // BITMAPINFOHEADER 结构大小 [14]-[17]
    bmpdata[14]=40; //  BITMAPINFOHEADER 结构大小，bmpdata[17]*16677216+bmpdata[16]*65536+bmpdata[15]*256+bmpdata[14])
    //（ here is 40 bytes)
    bmpdata[15]=0;
    bmpdata[16]=0;
    bmpdata[17]=0;
    // [21]-[18] is width
    
    bmpdata[21]=0; // 有符号整数
    bmpdata[20]=0;
    bmpdata[19]=width/256;
    bmpdata[18]=width-256*bmpdata[19];
    //    printf("bmpdata 18-19 %d %d \n",bmpdata[18],bmpdata[19]);
    //    printf("Width of the bitmap in pixels %d \n",bmpdata[18]+bmpdata[19]*256);
    
    // [25]-[22] is height
    
    bmpdata[25]=0; // 有符号整数
    bmpdata[24]=0;
    bmpdata[23]=height/256;
    bmpdata[22]=height-256*bmpdata[23];
    //    printf("bmpdata 22-23 %d %d \n",bmpdata[22],bmpdata[23]);
    //    printf("height of the bitmap in pixels %d \n",bmpdata[22]+bmpdata[23]*256);
    //    printf("filesize %lu \n",filesize);
    
    bmpdata[26]=1; //  色彩平面数，只有 1 为有效值
    bmpdata[27]=0; //
    
    //    bmpdata[28]=8; // for grayscale =8;   每个象素的位数。典型数为 1，4，8，16，24，和 32
    bmpdata[28]=bitcount; // for RGB 3*8=24bit;  // for binary image bitcount needs to be equal to 8
    if(bmpdata[28] <= 8) bmpdata[28]=8;
    bmpdata[29]=0;
    
    bmpdata[30]=0; //  0000 no pixel array compression used
    bmpdata[31]=0; //
    bmpdata[32]=0; //
    bmpdata[33]=0; //
    
    //    printf("imagesize %lu \n",imagesize);
    bmpdata[37]=imagesize/16777216; // //  图像大小。指原始位图数据的大小，与文件大小不是同一个概念。
    bmpdata[36]=(imagesize-16777216*bmpdata[37])/65536; // 65536=256*256
    bmpdata[35]=(imagesize-16777216*bmpdata[37]-65536*bmpdata[36])/256;
    bmpdata[34]=imagesize-16777216*bmpdata[37]-65536*bmpdata[36]-256*bmpdata[35];
    //    printf("imagesize from bmpdata %d %d %d %d\n",bmpdata[34],bmpdata[35],bmpdata[36],bmpdata[37]);
    
    bmpdata[38]=18; //  水平分辩率，用像素／米 表示
    bmpdata[39]=11; //  =11*256+18  referenece by photoshop
    bmpdata[40]=0; //
    bmpdata[41]=0; //
    
    bmpdata[42]=18; //  垂直分辩率，用像素／米 表示
    bmpdata[43]=11; //  =11*256+18  referenece by photoshop
    bmpdata[44]=0; //
    bmpdata[45]=0; //
    
    bmpdata[46]=0; //   位图实际使用的彩色表中的颜色索引数 （设为0的话，则使用所有调色板)
    
    bmpdata[47]=0; //
    bmpdata[48]=0; //
    bmpdata[49]=0; //
    
    bmpdata[50]=0; //   Àµ√˜∂‘ÕºœÛœ‘ æ”–÷ÿ“™”∞œÏµƒ—’…´À˜“˝µƒ ˝ƒø£¨»Áπ˚ «0£¨±Ì æ∂º÷ÿ“™°£
    bmpdata[51]=0; //
    bmpdata[52]=0; //
    bmpdata[53]=0; //
    
    if(bitcount <= 8)  // 调色板   for bitcount = 1,8
    {
        for(n=0; n < 256; n++)
        {
            for(k = 0; k < 3; k++)
                bmpdata[54+n*4+k]=n;
            bmpdata[54+n*4+3]=0;
        }
        
        if(bitcount == 8)  // for displaying black white image or psuedocolor image,if data has value > 1
        {
            bmpdata[54] = 255;  // for data = 0
            bmpdata[55] = 255;
            bmpdata[56] = 255;
            //                bmpdata[57] = 0;  // has been assigned above also for bmpdata[61] etc.
            bmpdata[58] = 1;    // for data = 1
            bmpdata[59] = 1;
            bmpdata[60] = 1;
            
            bmpdata[62] = 255;  // for data = 2
            bmpdata[63] = 120;
            bmpdata[64] = 120;
            
            bmpdata[66] = 180;  // for data = 3
            bmpdata[67] = 32;
            bmpdata[68] = 180;
            
            bmpdata[70] = 200;  // for data = 4
            bmpdata[71] = 64;
            bmpdata[72] = 80;
            
            bmpdata[74] = 64;  // for data = 5
            bmpdata[75] = 64;
            bmpdata[76] = 200;
            
            bmpdata[78] = 64;  // for data = 6
            bmpdata[79] = 32;
            bmpdata[80] = 128;
            
            bmpdata[81] = 128;  // for data = 7
            bmpdata[82] = 128;
            bmpdata[83] = 16;
            
            bmpdata[85] = 128;  // for data = 8
            bmpdata[86] = 128;
            bmpdata[87] = 128;
            
            bmpdata[89] = 96;  // for data = 9
            bmpdata[90] = 32;
            bmpdata[91] = 180;
            
            bmpdata[93] = 0;  // for data = 10
            bmpdata[94] = 0;
            bmpdata[95] = 254;
            
            bmpdata[97] = 48;  // for data = 11
            bmpdata[98] = 76;
            bmpdata[99] = 32;
            //  if necessary, can add for data = 12 ...
        }
        datashiftvalue=1078;         //54+256*4
    }
    
    
    // imagesize = (BitsPerPixel*width+31)/32 取整 *4, width 以象素为单位。
    n=datashiftvalue-1;
    //    n=datashiftvalue;
    if(bitcount == 24)
    {
        n=53;
        for(y=height-1; y >= 0; y--)
        {
            for(x=0; x < width; x++)
            {
                n++;
                bmpdata[n]=numeralImageB[y][x];
                n++;
                bmpdata[n]=numeralImageG[y][x];
                n++;
                bmpdata[n]=numeralImageR[y][x];
            }
            if(padding > 0 && padding < 4)
            {
                for(k=1; k <= 4-padding; k++)
                {
                    n++;
                    bmpdata[n]=0;
                }
            }        }
        
    }
    if(bitcount <= 8)
    {
        //        printf("height-width %d %d\n",height,width);
        n=1077;
        for(y=height-1; y >=0; y--)
        {
            for(x=0; x < width; x++)
            {
                n++;
                //                printf("n-y-x %d %d %d\n",n,y,x);
                bmpdata[n]=numeralImageR[y][x];
            }
            if(padding > 0)
                for(k=1; k <= 4-padding; k++)
                {
                    n++;
                    bmpdata[n]=0;
                }
        }
        
    }
    //    for(k=638850; k<= 639104; k++)
    //      printf("k-bmpdata %lu %lu \n",k,bmpdata[k]);
    //    printf(" end of  write_bmpImage %s \n", tmpfile);
    if((fp = fopen(tmpfile,"w")) == NULL)
    {
        printf("cannot open file for tmpfile \n");
        return ;
    }
    //    printf("final filesize n %d\n",n);
    fwrite(bmpdata, n+1, sizeof(unsigned char), fp);
    fclose(fp);
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
    FILE *fp_out;
    char tmpfile[64],tmp1file[64];
    char name[6];
    static int num = 0,k;
/*******************************************************
	for (int i = 0; i < HEIGHT; i++) {
		for (int j = 0; j < WIDTH; j++) {
			printf("%d ", pixelVal(i, j));
		}
	}
	printf("\n");
*******************************************************/
    //strcpy(tmpfile, ");
    num++;
    //把整数123 打印成一个字符串保存在s 中。
    //sprintf(name, "%d", num); //产生"123"
//    可以指定宽度，不足的左边补空格：
//    sprintf(s, "%8d%8d", 123, 4567); //产生：" 123 4567"
//    当然也可以左对齐：
//    sprintf(s, "%-8d%8d", 123, 4567); //产生："123 4567"
    //printf("name %s \n", name);
    //strcat(tmpfile, name);
    //strcat(tmpfile, ".raw");
    sprintf(tmpfile, "/users/jbian/imagedataForNumerals/inputimage%d.raw", num);
    printf("tmpfile %s \n", tmpfile);
    if ((fp_out = fopen(tmpfile, "wb")) == NULL) {
        printf("cannot open output file\n");
        return -1;
    }
    
    fwrite(data, 28*28, sizeof(unsigned char), fp_out);
    fclose(fp_out);
//    for (int i= 0; i < 28; i++)
//        for (int j = 0; j < 28; j++)
//            numeralImageR[i][j] = pixelVal(i, j);
//    writeBMPImage(1,"/Users/jbian/pixelVal.bmp"); // bitcount = 1,4,8,24
//    numeralProcessing(temp_file,pixelVal, 28, 28);

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

