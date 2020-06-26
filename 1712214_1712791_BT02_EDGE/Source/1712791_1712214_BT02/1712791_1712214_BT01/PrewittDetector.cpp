#pragma once
#include "PrewittDetector.h"

bool PrewittDetector::LoadImage(const char* filename)
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

void PrewittDetector::GaussianBlured(float sigma)
{
	int mask_size = 5; // Size of Gauss matrix.
	float sum = 0; // Sum up all elements of Gauss matrix.

	// Creating matrix gaussian with "0" default values.
	vector<vector<float>> gauss(mask_size, vector<float>(mask_size, 0));

	Mat tmpImg = grayImg.clone();

	/* The formula of Gaussian function in 2D:
		G(x, y) = (1/(2*Pi*sigma^2))*e^(-(x^2+y^2)/(2*sigma^2)).
	*/

	for (int i = -mask_size / 2; i <= mask_size / 2; i++)
		for (int j = -mask_size / 2; j <= mask_size / 2; j++)
		{
			gauss[i + mask_size / 2][j + mask_size / 2] += (float)(1 / (2 * Pi*sigma*sigma))*exp(-(i*i + j * j) / (2 * (sigma*sigma)));
			sum += gauss[i + mask_size / 2][j + mask_size / 2];
		}
	// Ignoring bounding pixel values.
	for (int x = mask_size / 2; x < height - mask_size / 2; x++)
	{
		for (int y = mask_size / 2; y < width - mask_size / 2; y++)
		{
			float new_pixel = 0;
			for (int i = -mask_size / 2; i <= mask_size / 2; i++)
			{
				for (int j = -mask_size / 2; j <= mask_size / 2; j++)
				{
					new_pixel += tmpImg.at<uchar>(x + i, y + j)*gauss[mask_size / 2 - i][mask_size / 2 - j];
				}
			}
			grayImg.at<uchar>(x, y) = saturate_cast<uchar>(new_pixel / sum);
		}
	}
}

int PrewittDetector::Convolution(int a[3][3], int b[3][3])
{
	int res = 0;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++) {
			res = res + a[i][j] * b[2 - i][2 - j];
		}
	return res;
}

void PrewittDetector::detectByPrewitt()
{
	int Gx[3][3] = { {1, 0, -1}, {1, 0, -1}, {1, 0, -1} };
	int Gy[3][3] = { {1, 1, 1}, {0, 0, 0}, {-1, -1, -1} };

	//GaussianBlur(grayImg, grayImg, Size(5, 5), 0);
	GaussianBlured(0.8);
	prewittImg = Mat::zeros(grayImg.size(), grayImg.type());
	prewittX = Mat::zeros(grayImg.size(), grayImg.type());
	prewittY = Mat::zeros(grayImg.size(), grayImg.type());

	for (int x = 1; x < height - 1; x++) {
		for (int y = 1; y < width - 1; y++) {
			int A[3][3] = {
				{grayImg.at<uchar>(x - 1, y - 1), grayImg.at<uchar>(x - 1, y), grayImg.at<uchar>(x - 1, y + 1)},
				{grayImg.at<uchar>(x, y - 1), grayImg.at<uchar>(x, y), grayImg.at<uchar>(x, y + 1)},
				{grayImg.at<uchar>(x + 1, y - 1), grayImg.at<uchar>(x + 1, y), grayImg.at<uchar>(x + 1, y + 1)}
			};

			int X = Convolution(Gx, A);
			int Y = Convolution(Gy, A);

			prewittImg.at<uchar>(x, y) = saturate_cast<uchar>(sqrt(X * X + Y * Y));
			prewittX.at<uchar>(x, y) = saturate_cast<uchar>(X);
			prewittY.at<uchar>(x, y) = saturate_cast<uchar>(Y);
		}
	}
}

bool PrewittDetector::ShowImage()
{
	if (!grayImg.data)
	{
		cout << "Error 2: Can't show image!" << endl;
		return false;
	}
	imshow("My Prewitt", prewittImg);
	imshow("My Prewitt X", prewittX);
	imshow("My Prewitt Y", prewittY);
	return true;
}