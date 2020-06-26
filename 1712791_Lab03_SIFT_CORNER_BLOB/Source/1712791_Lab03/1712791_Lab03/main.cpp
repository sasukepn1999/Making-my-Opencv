#include "HarrisDetector.h"
#include "BlobDetector.h"
#include "SIFTDetector.h"

int main(int argc, char** argv)
{
	/*Mat img = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\chess.jpg");
	Mat res;
	HarrisDetector *h = new HarrisDetector();
	res = h->detectHarris(img);
	imshow("Harris Corners Detector", res);*/

	//Mat src, gray;
	//// Load source image and convert it to gray
	//src = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\TestImages\\a.png", 1);
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	//Mat dst, dst_norm, dst_norm_scaled;
	//dst = Mat::zeros(src.size(), CV_32FC1);

	//// Detecting corners
	//cornerHarris(gray, dst, 2, 3, 0.05, BORDER_DEFAULT);

	//// Normalizing
	//normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	//convertScaleAbs(dst_norm, dst_norm_scaled);

	//// Drawing a circle around corners
	//for (int j = 0; j < dst_norm.rows; j++)
	//{
	//	for (int i = 0; i < dst_norm.cols; i++)
	//	{
	//		if ((int)dst_norm.at<float>(j, i) > 100)
	//		{
	//			circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
	//		}
	//	}
	//}


	//// Showing the result
	//imshow("corners_window", dst_norm_scaled);
	

	/*Mat img = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\TestImages\\f.png");
	Mat res;
	BlobDetector *h = new BlobDetector();
	res = h->detectBlob(img);
	imshow("Blob Detector", res);*/

	if (argc < 3)
	{
		cout << "Error : Wrong commandline!" << endl;
		return 0;
	}
	int malenh = stoi(argv[1]);
	switch (malenh)
	{
	case 1:
	{
		Mat img = imread(argv[2]);
		Mat res;
		HarrisDetector *h = new HarrisDetector();
		res = h->detectHarris(img);
		imshow("Harris Corners Detector", res);
		break;
	}
	case 2:
	{
		Mat img = imread(argv[2]);
		Mat res;
		BlobDetector *h = new BlobDetector();
		res = h->detectBlob(img);
		imshow("Blob Detector - LOG", res);
		break;
	}
	case 3:
	{
		Mat img = imread(argv[2]);
		Mat res;
		BlobDetector *h = new BlobDetector();
		res = h->detectDOG(img);
		imshow("Blob Detector - DOG", res);
		break;
	}
	case 4:
	{
		Mat img1 = imread(argv[2]);
		Mat img2 = imread(argv[3]);
		int detector = stoi(argv[4]);
		SIFTDetector *s = new SIFTDetector();
		s->matchBySIFT(img1, img2, detector);
		break;
	}
	}

	/*Mat img = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\TestImages\\f.png");
	Mat res;
	BlobDetector *h = new BlobDetector();
	res = h->detectDOG(img);
	imshow("Blob Detector", res);*/

	//Mat img1 = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\TestImages\\01.jpg");
	//Mat img2 = imread("D:\\HCMUS\\Computer Vision\\1712791_Lab03\\training_images\\01_1.jpg");
	//Mat res1;
	//SIFTDetector *s = new SIFTDetector();
	//cout << s->matchBySIFT(img1, img2, 1);
	
	waitKey();
	destroyAllWindows();

	return 0;
}