#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>

using namespace std;
using namespace cv;

class HarrisDetector
{
private:
	Mat grayImg;
	Mat response;
	Mat_<float> Ixx;
	Mat_<float> Iyy;
	Mat_<float> Ixy;

public:
	HarrisDetector() {}
	~HarrisDetector() {}

public:
	Mat getResponse() { return response; };

public:
	Mat detectHarris(Mat img);

private:
	Mat convertRbgToGrayscale(Mat img);
	Mat computeHarrisReponse(float& rmax);
	Mat suppression(float rmax, Mat response);
	Mat updating(Mat img, Mat response);
	void gradientXY();
	void reduceNoise();
};