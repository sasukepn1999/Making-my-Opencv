#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2\opencv.hpp>

#define Pi 3.14159265

using namespace std;
using namespace cv;

class SobelDetector
{
public:
	SobelDetector() {}
	~SobelDetector() {}

public:
	Mat grayImg, sobelImg, sobelX, sobelY;

private:
	long height;
	long width;

public:
	bool LoadImage(const char* filename);
	bool ShowImage();
	void detectBySobel();

private:
	int Convolution(int a[3][3], int b[3][3]);
	void GaussianBlured(float sigma);
};