#pragma once
#define _USE_MATH_DEFINES

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <math.h>
#include "HarrisDetector.h"
#include "BlobDetector.h"

using namespace std;
using namespace cv;

struct KP
{
	long x;
	long y;
	float s;
	vector<float> orient;
};

struct localizeKeypoint
{
	Mat offset;
	Mat J;
	Mat H;
};

class SIFTDetector
{
private:
	Mat grayImg1, grayImg2;
	vector<Mat> scaleImg;
	vector<Mat> DOG;
	vector<KP> keypoint1;
	vector<KP> keypoint2;
	float _sigma = 0;

public:
	SIFTDetector();
	~SIFTDetector();

public:
	double matchBySIFT(Mat img1, Mat img2, int detector);
	float getDOG(Mat grayImg, long x, long y, float sigma);
	Mat getResult(Mat img, int detector);
	void thresholdingKP1(Mat& response);
	void thresholdingKP2(Mat& response);
	localizeKeypoint localizeKP(Mat img, long x, long y, float s);
	void Orientation(Mat grayImg, vector<KP>& keypoint);
	double matchImgToImg(vector<KP> keypoint1, vector<KP> keypoint2);
	void setKeypoint(Mat res, vector<KP>& keypoint);

	// Ultilites
	Mat convertRbgToGrayscale(Mat img);
	Mat updating(Mat img, Mat r);
	int quantizeBin(float angle);
	double L2(vector<float> v1, vector<float> v2);
	bool showFinalMatch(Mat img1, Mat img2);
};

