#include <iostream>
#include <boost/python/numpy.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#include "bsifmodule.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

int main(int argc, char **argv){
	if (argc < 2) {
		std::cout << "Usage: test <image> " << std::endl;
	}

	Py_Initialize();
	np::initialize();

	// cv::Mat image = cv::imread("06117d321_imno.bmp", 0);
	cv::Mat image = cv::imread(argv[1], 0);

	std::cout << "Image loaded: " << image.rows << "x" << image.cols << std::endl;

	// create a numpy array with known shape and values
	// uchar sample[image.rows][image.cols];
	// memcpy(&sample, image.data, image.cols*image.rows);
	np::dtype dt = np::dtype::get_builtin<uchar>();
	np::dtype dtr = np::dtype::get_builtin<float>();
	np::ndarray myarray = np::from_data(image.data, dt,
										p::make_tuple(image.rows,image.cols),
										p::make_tuple(image.cols,1),
										p::object());

	uint8_t dims[] = {7,7,7};
	np::ndarray mydims = np::from_data(dims, dt,
										p::make_tuple(1,3),
										p::make_tuple(1,1),
										p::object());

	// allocate zeros in the return variable
	np::ndarray result = np::zeros(p::make_tuple(image.rows, image.cols), dtr);

	std::cout << "About to call BSIF module..." << std::endl;
	// call bsif module
	bsif_extract(myarray, result, mydims);
	std::cout << "Just after calling BSIF module..." << std::endl;

	cv::Mat imresult(image.rows, image.cols, CV_32FC1);
	std::cout << "Return variable declared..." << std::endl;

	const Py_intptr_t *arrsize = result.get_shape();
	std::cout << "Return variable size: " << (int)arrsize[0] << "x" << \
																				   (int)arrsize[1] << std::endl;

	int length = result.get_nd();
	std::cout << "Return variable length: " << length << std::endl;

	np::dtype rdt = result.get_dtype();
	std::cout << "Size of data item: " << rdt.get_itemsize() << std::endl;

	float * data = (float *)result.get_data();
	std::cout << "Uchar array pointing to result.get_data(): " << *data << std::endl;

	memcpy(imresult.data, data, image.rows*image.cols*sizeof(float));
	std::cout << "memcpy..." << std::endl;

	// char fname[50];
	// std::ofstream outfile;
	// sprintf(fname, "./output/cppoutput2.csv");
	// outfile.open(fname);
	// outfile << cv::format(imresult, cv::Formatter::FMT_CSV) << std::endl;
	// outfile.close();

	// cv::namedWindow("output", cv::WINDOW_AUTOSIZE);
	// cv::imshow("output", imresult);
	// cv::waitKey(0);
	cv::imwrite("./output/cpp.png",imresult);

	std::cout << "The end." << std::endl;
}
