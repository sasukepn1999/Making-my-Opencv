#include "HarrisDetector.h"

Mat HarrisDetector::detectHarris(Mat img)
{
	// Processing Image.
	grayImg = convertRbgToGrayscale(img);

	// Step 1: computing gradient magnitude.
	gradientXY();

	// Step 2: reducing noise.
	reduceNoise();

	// Step 3: computing Harris Respone: r(x, y) = det(M) - k*trace(M) (k in [0.04, 0.06]).
	float rmax = 0; // Choosing value max of reponse matrix for threshold (threshold = 0.01*max).
	response = computeHarrisReponse(rmax);

	// Step 4: Non-maxima suppression
	response = suppression(rmax, response);
	
	// Step 5: Updating original image with corners (red colors).
	Mat result;
	result = updating(img, response);

	return result;
}

Mat HarrisDetector::convertRbgToGrayscale(Mat img)
{
	// Convert RBG image to grayscalse image.
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	return gray;
	

	/*Mat gray(img.rows, img.cols, CV_32F);

	for (long x = 0; x < img.rows; x++)
	{
		for (long y = 0; y < img.cols; y++)
		{
			gray.at<float>(x, y) = 
				0.2126 * img.at<cv::Vec3b>(x, y)[0] +
				0.7152 * img.at<cv::Vec3b>(x, y)[1] +
				0.0722 * img.at<cv::Vec3b>(x, y)[2];
		}
	}

	return gray;*/
}

void HarrisDetector::gradientXY()
{
	Mat Ix, Iy;
	Sobel(grayImg, Ix, CV_32F, 1, 0);
	Sobel(grayImg, Iy, CV_32F, 0, 1);

	Ixx = Ix.mul(Ix);
	Iyy = Iy.mul(Iy);
	Ixy = Ix.mul(Iy);	
}

void HarrisDetector::reduceNoise()
{
	GaussianBlur(Ixx, Ixx, Size(3, 3), 0, 0);
	GaussianBlur(Iyy, Iyy, Size(3, 3), 0, 0);
	GaussianBlur(Ixy, Ixy, Size(3, 3), 0, 0);
}

Mat HarrisDetector::computeHarrisReponse(float& rmax)
{
	// Harris reponse functions matrix.
	Mat r(grayImg.rows, grayImg.cols, CV_32F);

	for (long x = 0; x < grayImg.rows; x++)
	{
		for (long y = 0; y < grayImg.cols; y++)
		{
			float a00 = Ixx.at<float>(x, y),
				  a01 = Ixy.at<float>(x, y),
				  a10 = Ixy.at<float>(x, y),
				  a11 = Iyy.at<float>(x, y);

			float det = a00 * a11 - a01 * a10;
			float trace = a00 + a11;
			float k = 0.05f;

			r.at<float>(x, y) = (det - k*trace*trace);

			if (r.at<float>(x, y) > rmax)
				rmax = r.at<float>(x, y);
		}
	}

	return r;
}

Mat HarrisDetector::suppression(float rmax, Mat r)
{
	Mat result(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));

	// Choosing threshold.
	float threshold = 0.01f*rmax;

	for (long x = 1; x < grayImg.rows - 1; x++)
	{
		for (long y = 1; y < grayImg.cols - 1; y++)
		{
			if (r.at<float>(x, y) > threshold &&
				r.at<float>(x, y) > r.at<float>(x - 1, y - 1) &&
				r.at<float>(x, y) > r.at<float>(x - 1, y) &&
				r.at<float>(x, y) > r.at<float>(x - 1, y + 1) &&
				r.at<float>(x, y) > r.at<float>(x, y - 1) &&
				r.at<float>(x, y) > r.at<float>(x, y + 1) &&
				r.at<float>(x, y) > r.at<float>(x + 1, y - 1) &&
				r.at<float>(x, y) > r.at<float>(x + 1, y) &&
				r.at<float>(x, y) > r.at<float>(x + 1, y + 1))

				result.at<float>(x, y) = r.at<float>(x, y);
		}
	}

	return result;
}

Mat HarrisDetector::updating(Mat img, Mat r)
{
	Mat result = img.clone();

	for (long x = 0; x < grayImg.rows; x++)
	{
		for (long y = 0; y < grayImg.cols; y++)
		{
			if (r.at<float>(x, y) > 0.0)
				//result.at<Vec3b>(x, y) = Vec3b(0, 0, 255);
				circle(result, Point(y, x), 2, Scalar(0, 0, 255), 2, 8, 0);
		}
	}

	return result;
}