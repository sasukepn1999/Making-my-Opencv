#include "SIFTDetector.h"



SIFTDetector::SIFTDetector()
{
}


SIFTDetector::~SIFTDetector()
{
}

double SIFTDetector::matchBySIFT(Mat img1, Mat img2, int detector)
{
	// Converting RBG to GrayScale.
	grayImg1 = convertRbgToGrayscale(img1);
	grayImg2 = convertRbgToGrayscale(img2);

	// Getting the response.
	Mat result1 = getResult(img1, detector);
	Mat result2 = getResult(img2, detector);

	// Keypoint Localization.
	if (detector == 2 || detector == 3)
	{
		thresholdingKP1(result1);
		thresholdingKP2(result2);
		//cout << keypoint2.size();
	}
	else if (detector == 1)
	{
		setKeypoint(result1, keypoint1);
		setKeypoint(result2, keypoint2);
	}
	else
	{
		cout << "Usage: detector: 1-3." << endl;
		return 0;
	}

	
	// Keypoint Descriptor.
	Orientation(grayImg1, keypoint1);
	Orientation(grayImg2, keypoint2);

	// Getting max distance.
	//double res = matchImgToImg(keypoint1, keypoint2);

	result1 = updating(img1, result1);
	result2 = updating(img2, result2);

	// Showing matched image.
	showFinalMatch(img1, img2);

	return 0;
}

bool SIFTDetector::showFinalMatch(Mat img1, Mat img2)
{
	Mat des1 = Mat::zeros(keypoint1.size(), 128, CV_32FC1);;
	Mat des2 = Mat::zeros(keypoint2.size(), 128, CV_32FC1);

	vector<KeyPoint> kp1;
	vector<KeyPoint> kp2;

	for (long i = 0; i < keypoint1.size(); i++)
	{
		long x = keypoint1[i].x,
			y = keypoint1[i].y;
		float sigma = keypoint1[i].s;
		vector<float> orient(keypoint1[i].orient);

		kp1.push_back(KeyPoint(y, x, sigma));
		for (int j = 0; j < orient.size(); j++)
		{
			des1.at<float>(i, j) = orient[j];
		}
	}

	for (long i = 0; i < keypoint2.size(); i++)
	{
		long x = keypoint2[i].x,
			y = keypoint2[i].y;
		float sigma = keypoint2[i].s;
		vector<float> orient(keypoint2[i].orient);

		kp2.push_back(KeyPoint(y, x, sigma));
		for (int j = 0; j < orient.size(); j++)
		{
			des2.at<float>(i, j) = orient[j];
		}
	}

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-L1");
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
	matcher->knnMatch(des1, des2, matches, 2);

	for (int i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() > 1)
		{
			const float ratio = 0.8; // As in Lowe's paper; can be tuned
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		else if (matches[i].size() == 1)
		{
			good_matches.push_back(matches[i][0]);
		}
	}

	Mat matchedImg;
	drawMatches(img1, kp1, img2, kp2, good_matches, matchedImg, Scalar_<double>::all(-1), Scalar_<double>::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	Mat finalMatch;
	resize(matchedImg, finalMatch, Size(matchedImg.cols / 1.8, matchedImg.rows / 1.8));

	if (!finalMatch.data)
	{
		cout << "Error: Can't show image!";
		return false;
	}

	imshow("SIFT", finalMatch);
	return true;
}

Mat SIFTDetector::convertRbgToGrayscale(Mat img)
{
	// Convert RBG image to grayscalse image.
	Mat gray, res32f;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	gray.convertTo(res32f, CV_32F);
	return res32f;
}

float SIFTDetector::getDOG(Mat grayImg, long x, long y, float sigma)
{
	int index = int(2 * log2(sigma / _sigma));

	if (index + 1 >= DOG.size())
	{
		for (int i = DOG.size(); i <= index + 1; i++)
		{
			Mat tmp;
			float sigmaX = (float)pow(sqrt(2), i)*_sigma;
			float sigmaY = 0;

			GaussianBlur(grayImg, tmp, Size(5, 5), sigmaX, sigmaY);
			DOG.push_back(tmp);
		}
	}
	if (index == 0)
		return DOG[index].at<float>(x, y);

	return DOG[index + 1].at<float>(x, y) - DOG[index].at<float>(x, y);
}

Mat SIFTDetector::getResult(Mat img, int detector)
{
	if (detector == 1)
	{
		HarrisDetector *h = new HarrisDetector();
		h->detectHarris(img);

		return h->getResponse();
	}
	else if (detector == 2)
	{
		BlobDetector *b = new BlobDetector();
		b->detectBlob(img);
		_sigma = b->getSigma();

		return b->getResponse();
	}
	else if (detector == 3)
	{
		BlobDetector *b = new BlobDetector();
		b->detectDOG(img);
		_sigma = b->getSigma();

		return b->getResponse();
	}
	else
		return {};
}

void SIFTDetector::setKeypoint(Mat response, vector<KP>& keypoint)
{
	for (long x = 0; x < response.rows; x++)
	{
		for (long y = 0; y < response.cols; y++)
		{
			if (response.at<float>(x, y) > 0)
			{
				float sigma = 0;
				KP k = { x, y, sigma, vector<float>(128, 0) };
				keypoint.push_back(k);
			}
		}
	}
}

void SIFTDetector::thresholdingKP1(Mat& response)
{
	for (long x = 0; x < response.rows; x++)
	{
		for (long y = 0; y < response.cols; y++)
		{
			float sigma = response.at<float>(x, y);
			Mat offset, J, H;

			if (sigma < sqrt(2))
				continue;

			localizeKeypoint l = localizeKP(grayImg1, x, y, sigma);
			offset = l.offset;
			J = l.J;
			H = l.H;

			// Discarding low-constrast keypoints.
			float threshold_c = 0.03f;
			float constrast = (float)(getDOG(grayImg1, x, y, sigma) + 0.5f * J.t().dot(offset));

			if (fabs(constrast) < threshold_c)
			{
				response.at<float>(x, y) = 0;
				continue;
			}

			// Discarding keypoints along edge.
			float r = 10;
			float threshold_r = (r + 1)*(r + 1) / r;

			double det = determinant(H);
			Scalar sca = trace(H);
			double trace = sca.val[0];
			double ratio = trace * trace / det;

			if (ratio > threshold_r)
			{
				response.at<float>(x, y) = 0;
				continue;
			}

			// Adding keypoint.
			KP k = { x, y, sigma, vector<float>(128, 0) };
			keypoint1.push_back(k);
		}
	}
}

void SIFTDetector::thresholdingKP2(Mat& response)
{
	for (long x = 0; x < response.rows; x++)
	{
		for (long y = 0; y < response.cols; y++)
		{
			float sigma = response.at<float>(x, y);
			Mat offset, J, H;

			if (sigma < sqrt(2))
				continue;
			
			localizeKeypoint l = localizeKP(grayImg2, x, y, sigma);
			offset = l.offset;
			J = l.J;
			H = l.H;
			
			// Discarding low-constrast keypoints.
			float threshold_c = 0.03f;
			float constrast = (float)(getDOG(grayImg2, x, y, sigma) + 0.5f * J.t().dot(offset));

			if (fabs(constrast) < threshold_c)
			{
				response.at<float>(x, y) = 0;
				continue;
			}

			// Discarding keypoints along edge.
			float r = 10;
			float threshold_r = (r + 1)*(r + 1) / r;

			double det = determinant(H);
			Scalar sca = trace(H);
			double trace = sca.val[0];
			double ratio = trace * trace / det;

			if (ratio > threshold_r)
			{
				response.at<float>(x, y) = 0;
				continue;
			}

			// Adding keypoint.
			KP k = { x, y, sigma };
			keypoint2.push_back(k);
		}
	}
}

localizeKeypoint SIFTDetector::localizeKP(Mat img, long x, long y, float s)
{
	float Dx = (getDOG(img, x + 1, y, s) - getDOG(img, x - 1, y, s)) / 2.0f;
	float Dy = (getDOG(img, x, y + 1, s) - getDOG(img, x, y - 1, s)) / 2.0f;
	float Ds = (getDOG(img, x, y, s + 1) - getDOG(img, x, y, s - 1)) / 2.0f;

	float Dxx = getDOG(img, x + 1, y, s) - 2 * getDOG(img, x, y, s) + getDOG(img, x - 1, y, s);
	float Dxy = ((getDOG(img, x + 1, y + 1, s) - getDOG(img, x - 1, y + 1, s)) - (getDOG(img, x + 1, y - 1, s) - getDOG(img, x - 1, y - 1, s))) / 4.0f;
	float Dxs = ((getDOG(img, x + 1, y, s + 1) - getDOG(img, x - 1, y, s + 1)) - (getDOG(img, x + 1, y, s - 1) - getDOG(img, x - 1, y, s - 1))) / 4.0f;
	float Dyy = getDOG(img, x, y + 1, s) - 2 * getDOG(img, x, y, s) + getDOG(img, x, y - 1, s);
	float Dys = ((getDOG(img, x, y + 1, s + 1) - getDOG(img, x, y - 1, s + 1)) - (getDOG(img, x, y + 1, s - 1) - getDOG(img, x, y - 1, s - 1))) / 4.0f;
	float Dss = getDOG(img, x, y, s + 1) - 2 * getDOG(img, x, y, s) + getDOG(img, x, y, s - 1);

	float dJ[] = { Dx, Dy, Ds };
	float dH[] = {
		Dxx, Dxy, Dxs,
		Dxy, Dyy, Dys,
		Dxs, Dys, Dss
	};

	Mat J(3, 1, CV_32F, dJ);
	Mat H(3, 3, CV_32F, dH);

	Mat offset = -H.inv(DECOMP_SVD)*J;
	offset = offset.t();

	localizeKeypoint res = {
		offset,
		J,
		H
	};

	return res;
}

void SIFTDetector::Orientation(Mat grayImg, vector<KP>& keypoint)
{
	if (keypoint.size() == 0)
		return ;

	for (long i = 0; i < keypoint.size(); i++)
	{
		long x = keypoint[i].x,
			 y = keypoint[i].y;
		float s = keypoint[i].s;

		vector<float> orient(128, 0);
		Mat L;
		GaussianBlur(grayImg, L, Size(5, 5), s, 0);

		long startX = x - 8, endX = x + 7,
			 startY = y - 8, endY = y + 7;

		/*
		Normalizing startX, startY, endX, endY.
		Ex: (startX, endX) = (-2, 2) => (1, 5).
		*/
		if (startX <= 0)
		{
			endX -= startX - 1;
			startX = 1;
		}

		if (startY <= 0)
		{
			endY -= startY - 1;
			startY = 1;
		}

		if (endX >= grayImg.rows - 1)
		{
			startX -= (endX - grayImg.rows + 2);
			endX = grayImg.rows - 2;
		}

		if (endY >= grayImg.cols - 1)
		{
			startY -= (endY - grayImg.cols + 2);
			endY = grayImg.cols - 2;
		}

		for (long row = startX; row <= endX; row++)
		{
			for (long col = startY; col <= endY; col++)
			{
				long index_sub = 4 * ((row - startX) / 4) + (col - startY) / 4;
				float Lx = L.at<float>(row + 1, col) - L.at<float>(row - 1, col);
				float Ly = L.at<float>(row, col + 1) - L.at<float>(row, col - 1);
				float m = sqrt(Lx*Lx + Ly*Ly);
				float theta = atan2f(Ly, Lx);
				float angle = (theta > 0 ? theta : (2 * M_PI + theta)) * 360 / (2 * M_PI);
				int bin = quantizeBin(angle);
				orient[8 * index_sub + bin] += m;
			}
		}
		keypoint[i].orient = orient;
	}
}

int SIFTDetector::quantizeBin(float angle)
{
	return int(ceil(angle / 45)) - 1;
}

double SIFTDetector::matchImgToImg(vector<KP> keypoint1, vector<KP> keypoint2)
{
	double max = 0;
	for (long i = 0; i < keypoint1.size(); i++)
	{
		for (long j = 0; j < keypoint2.size(); j++)
		{
			double distance = L2(keypoint1[i].orient, keypoint2[j].orient);
			if (max < distance)
				max = distance;
		}
	}
	return max;
}

double SIFTDetector::L2(vector<float> v1, vector<float> v2)
{
	if (v1.size() != v2.size())
		return -1;
	
	double res = 0;
	for (int i = 0; i < v1.size(); i++)
	{
		res += (v1[i] - v2[i])*(v1[i] - v2[i]);
	}
	res = sqrt(res);

	return res;
}

Mat SIFTDetector::updating(Mat img, Mat r)
{
	Mat result = img.clone();
	for (long x = 0; x < img.rows; x++)
	{
		for (long y = 0; y < img.cols; y++)
		{
			if (r.at<float>(x, y) > 0)
				circle(result, Point(y, x), 1, Scalar(0, 0, 255), 2, 8, 0);
		}
	}

	return result;
}