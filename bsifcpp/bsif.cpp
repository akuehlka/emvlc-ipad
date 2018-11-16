#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <fstream>
#include "bsif.h"
#include "filtermap.h"

using namespace std;

// convert linear indexing to subscript indexing (3D)
int s2i3d(int* dims, int i, int j, int k){
  // C++ and python use row-major order, so the last dimension is contiguous
  // in doubt, refer to https://en.wikipedia.org/wiki/Row-_and_column-major_order#Column-major_order
	return k + dims[2]*(j+dims[1]*i);
}

void bsif(cv::Mat src, cv::Mat& dst, int dims[3]){

	int numScl = (int) dims[2];
	int r = floor(dims[0]/2);

	//initializing matrix of 1s
	cv::Mat codeImg = cv::Mat::ones(src.rows, src.cols, CV_64FC1);

	// creates the border around the image - it is wrapping
	int border = r;
	cv::Mat imgWrap = src;
	cv::copyMakeBorder(src, imgWrap, border, border, border, border, cv::BORDER_WRAP);

	// cv::imwrite("cppwrap.png",imgWrap);

	// load the hard-coded filters
	t_filtermap filters = build_filter_map();
	// here we retrieve a filter from the map
	char buf[50];
	sprintf(buf, "filter_%d_%d_%d", dims[0], dims[1], dims[2]);
	string filtername(buf);

	double* myFilter;
	myFilter = filters[filtername];
	if (myFilter == NULL){
		cout << "No such filter." << endl;
		return;
	}

	// Loop over scales
	cv::Mat ci; // the textured image after filter
	double tmp[dims[0]*dims[1]];
	int itr = 0;

	// pull the data from the matfile into an array
	// the matlab file is in one long single array
	// we need to start w/ the last filter and work our way forward
	for (int i = numScl - 1; i >= 0; i--){

		int r = 0;
		for (int j=0; j<dims[0]; j++){
			for (int k=0; k<dims[1]; k++){
				tmp[r+(k*dims[0])] = myFilter[s2i3d(dims, k, j, i)];
			}
			r += 1;
		}
		//convert the array into matlab object to use w/ filter
		cv::Mat tmpcv2 = cv::Mat(dims[0], dims[1], CV_64FC1, &tmp);

		// running the filter on the image w/ BORDER WRAP - equivalent to filter2 in matlab
		// filter2d will incidentally create another border - we do not want this extra border
		cv::filter2D(imgWrap, ci, CV_64FC1, tmpcv2, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

		cv::Mat subci = ci(cv::Range(border, ci.rows-border),
												cv::Range(border, ci.cols-border));
		cv::Mat mask = cv::Mat::zeros(subci.size(), CV_64FC1);
		mask.setTo(1, subci > 0);

		mask = mask * pow(2,itr);
		// cout << "Masked value: " << pow(2,itr) << endl;

		codeImg = codeImg + mask;

		// char fname[50];
		// ofstream outfile;
		// sprintf(fname, "scale%d.csv", itr);
		// outfile.open(fname);
		// outfile << cv::format(codeImg, cv::Formatter::FMT_CSV) << endl;
		// outfile.close();

		itr++;
	}

	// int histogram[histsize];
	// memset(histogram, 0, histsize*sizeof(int));
	// for (int j = 0; j < src.rows; j++){
	// 	for (int k = 0; k < src.cols; k++){
	// 		histogram[(int)codeImg[j][k]]++;
	// 	}
	// }

	// // Outputting to a CSV file
	// ofstream histfile;
	// histfile.open("histogram.csv", ios::out | ios::trunc);
	// histfile << "Hist = [";
	// for (int i = 0; i < histsize; i++)
	// 	histfile << histogram[i] << ", ";
	// histfile << "];";
	// histfile.close();

	// convert the calculated image to return it
	// cv::Mat tmpim = cv::Mat(src.rows, src.cols, CV_64FC1, &codeImg);
	codeImg.convertTo(dst, CV_32FC1);

	// char fname[50];
	// ofstream outfile;
	// sprintf(fname, "cppoutput.csv", itr);
	// outfile.open(fname);
	// outfile << cv::format(dst, cv::Formatter::FMT_CSV) << endl;
	// outfile.close();


	// Creating the histogram
	int nbins = pow(2, dims[2]);
	float range[] = {0, 256};
	const float* histRange = { range };
	cv::Mat histogram;
	cv::calcHist(&dst, 1, 0, // channel 0
		cv::Mat(),                 // no mask
		histogram, 1, 						 // dimensions
		&nbins,
		&histRange,
		true, 										 // uniform
		false);										 // accumulate

	// cv::Mat i3;
	// dst.convertTo(i3, CV_8UC1);
	// cv::imwrite("output/raw.png",i3);
	// cv::namedWindow("bsif", cv::WINDOW_AUTOSIZE);
	// cv::imshow("bsif", i3);
	// cv::waitKey(0);


}

int main(int argc, char *argv[]) {
	if (argc != 5) {
		cout << "Usage: bsifcpp <image> <filter height> <filter width> <filter depth>" << endl;
		cout << "Example: bsifcpp myimage.png 3 3 8" << endl;
	}

	// reading in the image with open cv - returns array of doubles
	cv::Mat image = cv::imread(argv[1], 0);

	// when reading the filter, we have to know its dimensions
	int dims[3];
	dims[0] = atoi(argv[2]);
	dims[1] = atoi(argv[3]);
	dims[2] = atoi(argv[4]);

	cv::Mat imout;
	bsif(image, imout, &dims[0]);

	cv::Mat im2 = cv::Mat(image.rows, image.cols, CV_8UC1);
	cv::normalize(imout, im2, 255, 0, cv::NORM_MINMAX);
	imwrite("new_output.png", im2);

	return 0;


}
