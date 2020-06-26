#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;

class LaplacianDetector
{
public:
	LaplacianDetector() {}
	~LaplacianDetector() {}

public:
	Mat grayImg;

private:
	long height;
	long width;

public:
	bool LoadImage(const char* filename);
	bool ShowImage(const char* nameWindow);
	void detectByLaplace();

private:
	
};