#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include "CannyDetector.h"
#include "LaplacianDetector.h"
#include "PrewittDetector.h"
#include "SobelDetector.h"

using namespace cv;
using namespace std;

#define NUMBER_CHANNEL_BGR 3

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		cout << "Error : Wrong commandline!" << endl;
		return 0;
	}
	int malenh = stoi(argv[2]);
	switch (malenh)
	{
	case 1:
	{
		SobelDetector *sobel = new SobelDetector;
		sobel->LoadImage(argv[1]);
		sobel->detectBySobel();
		sobel->ShowImage();
		break;
	}
	case 2:
	{
		PrewittDetector *prewitt = new PrewittDetector;
		prewitt->LoadImage(argv[1]);
		prewitt->detectByPrewitt();
		prewitt->ShowImage();
		break;
	}
	case 3:
	{
		LaplacianDetector *laplacian = new LaplacianDetector();
		laplacian->LoadImage(argv[1]);
		laplacian->detectByLaplace();
		laplacian->ShowImage("My Laplacian");
		break;
	}
	case 4:
	{
		double sigma = stod(argv[3]);
		double lowThreshold = stod(argv[4]);
		double highThreshold = stod(argv[5]);
		CannyDetector *canny = new CannyDetector();
		canny->LoadImage(argv[1]);
		canny->detectByCany(sigma, lowThreshold, highThreshold); // sigma = 0.8; low threshold = 30; high threshold = 80.
		break;
	}
	}

	//CannyDetector *canny = new CannyDetector();
	//canny->LoadImage("D:\\HCMUS\\Computer Vision\\1712791_1712214_BT01\\lena_grayscale.png");
	//canny->detectByCany(0.8, 30, 80); // sigma = 0.8; low threshold = 30; high threshold = 80.

	//Mat img = imread("D:\\HCMUS\\Computer Vision\\1712791_1712214_BT01\\lena_grayscale.png", IMREAD_GRAYSCALE);
	//GaussianBlur(img, img, Size(5, 5), 0.8, 0.8);
	//imshow("OpenCV - Smoothing", img);
	//Canny(img, img, 30, 80, 3, true);
	//imshow("OpenCV", img);

	//LaplacianDetector *laplacian = new LaplacianDetector();
	//laplacian->LoadImage("D:\\HCMUS\\Computer Vision\\1712791_1712214_BT01\\lena_grayscale.png");
	//laplacian->detectByLaplace();
	//laplacian->ShowImage("My Laplacian");

	//Mat img1 = imread("D:\\HCMUS\\Computer Vision\\1712791_1712214_BT01\\lena_grayscale.png", IMREAD_GRAYSCALE);
	//Mat img2;
	//Laplacian(img1, img2, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(img2, img1);
	//imshow("Laplacian OpenCV", img1);


	waitKey(0);
	return 0;
}