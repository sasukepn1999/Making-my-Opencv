#ifndef BlobDetector_HEADER
#define BlobDetector_HEADER
#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;

class BlobDetector
{
private:
	Mat grayImg;
	Mat scales;  // scale selection.
	vector<Mat> scaleImg;  // space-scale.
	float _sigma = 1.6f;

public:
	BlobDetector() {}
	~BlobDetector() {}

public:
	Mat getResponse() { return scales; }
	float getSigma() { return _sigma; }

public:
	Mat detectBlob(Mat img);
	Mat detectDOG(Mat img);

private:
	// LoG
	float laplacianOfGaussian(long x, long y, float sigma);
	vector<Mat> generateScaleLOG();
	Mat scaleSelectionLOG();
	Mat filter(int size, float sigma);
	// My filter and convolution (not use cz slow).
	vector<vector<float>> filterLoG(int size, float sigma);
	Mat convolutionWithLoG(vector<vector<float>> filter);

	// DoG
	vector<Mat> generateScaleDOG();
	Mat scaleSelectionDOG();

	// Ultilites
	Mat convertRbgToGrayscale(Mat img);
	Mat updating(Mat img, Mat r);
};

#endif