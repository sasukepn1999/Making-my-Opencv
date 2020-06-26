#pragma once
#include "LaplacianDetector.h"

bool LaplacianDetector::LoadImage(const char* filename)
{
	grayImg = imread(filename, IMREAD_GRAYSCALE);

	if (!grayImg.data)
	{
		cout << "Error 1: Cannot open image!" << endl;
		return false;
	}

	// Get height and width of image.
	height = grayImg.rows;
	width = grayImg.cols;

	return true;
}

void LaplacianDetector::detectByLaplace()
{
	vector<vector<int>> LaplacianFilter{ {-1, -1, -1},
										 {-1, 8, -1},
										 {-1, -1, -1} };

	Mat tmpImg = grayImg.clone();
	//GaussianBlur(grayImg, grayImg, Size(5, 5), 0.8, 0.8);

	for (long x = 1; x < height - 1; x++)
	{
		for (long y = 1; y < width - 1; y++)
		{
			int sum = 0;
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					sum += tmpImg.at<uchar>(x + i, y + j)*LaplacianFilter[1 + i][1 + j];
				}
			}
			grayImg.at<uchar>(x, y) = saturate_cast<uchar>(sum);
		}
	}
}

bool LaplacianDetector::ShowImage(const char* nameWindow)
{
	if (!grayImg.data)
	{
		cout << "Error 2: Can't show image!" << endl;
		return false;
	}
	imshow(nameWindow, grayImg);
	return true;
}