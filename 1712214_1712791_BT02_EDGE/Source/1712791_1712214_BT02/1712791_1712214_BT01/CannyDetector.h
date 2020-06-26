#pragma once
#include <iostream>
#include <vector>
#include <math.h> 
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include<opencv2\opencv.hpp>

using namespace std;
using namespace cv;

#define Pi 3.14159265

class CannyDetector
{
private:
	Mat originalImg;
	Mat grayImg;
	int height = 0;
	int width = 0;

public:
	CannyDetector(){}
	~CannyDetector(){}

public:
	bool LoadImage(const char* filename);
	void detectByCany(float sigma, float minVal, float maxVal);
	bool ShowImage(const char* nameWindow);

private:
	void GaussianBlured(float sigma);
	void Gradient(vector<vector<float>>& gradientXY, vector<vector<float>>& angleXY);
	void NonMaximumSuppression(vector<vector<float>>& gradientXY, vector<vector<float>>& angleXY);
	void Hysteresis(float lowThresh, float highThresh);
	void HysteresisRecursion(int x, int y, float lowThresh);
};