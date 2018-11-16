#include <opencv2/core/core.hpp>
#include "bsifmodule.h"
#include "bsif.h"

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;

const uint FILTER_DIMS_SIZE = 3;

np::ndarray bsif_extract(np::ndarray src, np::ndarray& dst, np::ndarray filterdims){

  // convert src into a cv::Mat
  const Py_intptr_t *s = src.get_shape();

  int rows = s[0];
  int cols = s[1];
  int imsize = rows*cols;

  // cout << "Filter dimensions =" << endl;
  int vdims[FILTER_DIMS_SIZE];
  uchar * p_fsizes = (uchar *)filterdims.get_data();
  for (uint i=0; i<FILTER_DIMS_SIZE; i++){
	  vdims[i] = (int)p_fsizes[i];
	  // cout << " " << vdims[i];
  }
  // cout << endl;

  cv::Mat img(rows, cols, CV_8UC1);
  cv::Mat imres(rows, cols, CV_32FC1);

  uchar * data = (uchar *)src.get_data();
  memcpy(img.data, data, imsize);

  // call bsif extraction
  bsif(img, imres, vdims);

  // cv::Mat i3;
	// imres.convertTo(i3, CV_8UC1);
	// cv::namedWindow("bsif", cv::WINDOW_AUTOSIZE);
	// cv::imshow("bsif", i3);
	// cv::waitKey(0);

  // return the result
  np::dtype dt = np::dtype::get_builtin<float>();
  np::ndarray bsiftmp = np::from_data(imres.data, dt,
								   p::make_tuple(rows, cols),
								   p::make_tuple(sizeof(float)*cols, sizeof(float)),
								   p::object());
  dst = bsiftmp.copy();

  return dst;
}

BOOST_PYTHON_MODULE(bsif) {
    Py_Initialize();
    np::initialize();
    def("extract",bsif_extract);
}
