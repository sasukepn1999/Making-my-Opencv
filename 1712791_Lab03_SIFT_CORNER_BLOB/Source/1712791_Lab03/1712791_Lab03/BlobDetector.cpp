#include "BlobDetector.h"

Mat BlobDetector::detectDOG(Mat img)
{
	Mat result;

	// Converting RBG to Grayscale.
	grayImg = convertRbgToGrayscale(img);
	
	// Generating Scale-space.
	scaleImg = generateScaleDOG();

	// Finding local extrema.
	scales = scaleSelectionDOG();

	// Drawing blobs.
	result = updating(img, scales);

	return result;
}

vector<Mat> BlobDetector::generateScaleDOG()
{
	// Generating scale-space. (with 10 scales (1->10))
	int num_of_scale = 10;

	vector<Mat> res;
	for (int scale = 1; scale <= num_of_scale; scale++)
	{
		Mat tmp;
		float sigma = (float)pow(sqrt(2), scale) * _sigma;  // sigma = k*sigma_1 (k = 2^{1/s}).

		// Kernel size (size of kernel = [sigma*6] + (0 or 1 to ensure size to be always odd number)).
		int ksize = (int)ceil(sigma * 6);
		ksize = ksize / 2 * 2 + 1;

		GaussianBlur(grayImg, tmp, Size(ksize, ksize), 0, 0);

		// Constructing DoG.
		if (scale > 1)
		{
			/*
			tmp = G(x, y, k*sigma)*I(x, y).
			res[scale - 1] = G(x, y, sigma)*I(x, y).
			=> res[scale - 1] = G(x, y, k*sigma)*I(x, y) - G(x, y, sigma)*I(x, y).
			*/

			res[scale - 2] = tmp - res[scale - 2];
		}

		if (scale < num_of_scale)
		{
			res.push_back(tmp);
		}
	}

	return res;
}

Mat BlobDetector::scaleSelectionDOG()
{
	if (scaleImg.size() == 0)
		return Mat(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));
	else if (scaleImg.size() == 1)
		return Mat(grayImg.rows, grayImg.cols, CV_32F, Scalar(1));

	/*
		maxima = dark blobs on light background
		minima = light blobs on dark background
		=> Solution: Finding local maxima of square reponses.
	*/
	Mat res(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));

	float max = 0;  // Threshold = 0.3*max.
	for (int idx_scale = 0; idx_scale < scaleImg.size(); idx_scale++)
	{
		for (long x = 0; x < grayImg.rows; x++)
		{
			for (long y = 0; y < grayImg.cols; y++)
			{
				if (max < scaleImg[idx_scale].at<float>(x, y))
					max = scaleImg[idx_scale].at<float>(x, y);
			}
		}
	}

	// Comparing its to the next scales.
	for (long x = 1; x < grayImg.rows - 1; x++)
	{
		for (long y = 1; y < grayImg.cols - 1; y++)
		{
			bool isMin = true, isMax = true;

			// Considering its neighbors.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					if ((i == 0 && j == 0))
						continue;

					// Finding maxima.
					if (isMax  && scaleImg[0].at<float>(x, y) <= scaleImg[0].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			// Considering the next scale.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					// Finding maxima.
					if (isMax && scaleImg[0].at<float>(x, y) <= scaleImg[1].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			if (isMax && scaleImg[0].at<float>(x, y) >= 0.3*max)
				res.at<float>(x, y) = 2 * _sigma;
		}
	}


	// Comparing its to both the previous and next scales.
	for (int idx_scale = 1; idx_scale < scaleImg.size() - 1; idx_scale++)
	{
		for (long x = 1; x < grayImg.rows - 1; x++)
		{
			for (long y = 1; y < grayImg.cols - 1; y++)
			{
				if (res.at<float>(x, y) != 0)
					continue;

				bool isMax = true;

				// Considering its neighbors.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						if ((i == 0 && j == 0))
							continue;
						
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				// Considering the next scale.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale + 1].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				// Considering the previous scale.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale - 1].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				if (isMax && scaleImg[idx_scale].at<float>(x, y) >= 0.3*max)
					res.at<float>(x, y) = (float)pow(sqrt(2), idx_scale + 2) * _sigma;
			}
		}
	}


	// Comparing its to the previous scale.
	int last_scale = scaleImg.size() - 1;
	for (long x = 1; x < grayImg.rows - 1; x++)
	{
		for (long y = 1; y < grayImg.cols - 1; y++)
		{
			if (res.at<float>(x, y) != 0)
				continue;

			bool isMax = true;

			// Considering its neighbors.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					if ((i == 0 && j == 0))
						continue;

					// Finding maxima.
					if (isMax && scaleImg[last_scale].at<float>(x, y) <= scaleImg[last_scale].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			// Considering the previous scale.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					// Finding maxima.
					if (isMax && scaleImg[last_scale].at<float>(x, y) <= scaleImg[last_scale - 1].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			if (isMax && scaleImg[last_scale].at<float>(x, y) >= 0.3*max)
				res.at<float>(x, y) = (float)pow(sqrt(2), last_scale + 2) * _sigma;
		}
	}

	return res;
}

Mat BlobDetector::detectBlob(Mat img)
{
	Mat result;

	// Converting RBG to Grayscale.
	grayImg = convertRbgToGrayscale(img);
	
	// Generating Scale-space.
	scaleImg = generateScaleLOG();
	
	// Finding local extremas.
	scales = scaleSelectionLOG();

	// Drawing blobs.
	result = updating(img, scales);

	return result;
}

Mat BlobDetector::convertRbgToGrayscale(Mat img)
{
	// Convert RBG image to grayscalse image.
	Mat gray, res32f;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	gray.convertTo(res32f, CV_32F);
	return res32f;
}

float BlobDetector::laplacianOfGaussian(long x, long y, float sigma)
{
	// Laplacian of Gauss: stackoverflow.com/questions/2545323/laplacian-of-gaussian-filter-use
	return float((-1.0f / (M_PI*(sigma * sigma))) * (1.0f - ((x*x + y*y) * 1.0f / (2.0f * sigma * sigma))) * exp(-(x * x + y * y) * 1.0f / (2.0f * sigma * sigma)));
}

vector<vector<float>> BlobDetector::filterLoG(int size, float sigma)
{
	int halfSize = size / 2;
	vector<vector<float>> LoG(halfSize * 2 + 1, vector<float>(halfSize * 2 + 1, 0));
	float sum = 0;

	for (int x = -halfSize; x <= halfSize; x++)
	{
		for (int y = -halfSize; y <= halfSize; y++)
		{
			LoG[x + halfSize][y + halfSize] = laplacianOfGaussian(x, y, sigma);
		}
	}
	
	return LoG;
}

Mat BlobDetector::filter(int size, float sigma)
{
	int halfSize = size / 2;
	Mat LoG(halfSize * 2 + 1, halfSize * 2 + 1, CV_32F, Scalar(0));

	for (int x = -halfSize; x <= halfSize; x++)
	{
		for (int y = -halfSize; y <= halfSize; y++)
		{
			LoG.at<float>(x + halfSize, y + halfSize) = laplacianOfGaussian(x, y, sigma);
		}
	}

	return LoG;
}

vector<Mat> BlobDetector::generateScaleLOG()
{
	// Generating scale-space. (with 10 scales (1->10) and size = 3x3)
	int num_of_scale = 10;

	vector<Mat> res;
	for (int scale = 1; scale <= num_of_scale; scale++)
	{
		Mat tmp;
		float sigma = (float)pow(sqrt(2), scale) * _sigma;
		int ksize = (int)ceil(sigma * 6);	// Kernel size.

		Mat LoG_filter = filter(ksize, sigma);
		filter2D(grayImg, tmp, CV_32F, LoG_filter);

		tmp = tmp.mul(tmp);
		res.push_back(tmp);
	}

	return res;
}

Mat BlobDetector::convolutionWithLoG(vector<vector<float>> filter)
{
	Mat res(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));

	int halfSize = filter.size() / 2;
	for (long x = halfSize; x < grayImg.rows - halfSize; x++)
	{
		for (long y = halfSize; y < grayImg.cols - halfSize; y++)
		{
			float sum = 0;
			for (int i = -halfSize; i <= halfSize; i++)
			{
				for (int j = -halfSize; j <= halfSize; j++)
				{
					sum += grayImg.at<float>(x + i, y + j) * filter[halfSize - i][halfSize - j];
				}
			}
			res.at<float>(x, y) = (sum*sum);
		}
	}
	return res;
}

Mat BlobDetector::scaleSelectionLOG()
{
	if (scaleImg.size() == 0)
		return Mat(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));
	else if (scaleImg.size() == 1)
		return Mat(grayImg.rows, grayImg.cols, CV_32F, Scalar(1));

	/* 
		maxima = dark blobs on light background
		minima = light blobs on dark background
		=> Solution: Finding local maxima of square reponses.
	*/
	Mat res(grayImg.rows, grayImg.cols, CV_32F, Scalar(0));

	float max = 0;  // Threshold = 0.3*max.
	for (int idx_scale = 0; idx_scale < scaleImg.size(); idx_scale++)
	{
		for (long x = 0; x < grayImg.rows; x++)
		{
			for (long y = 0; y < grayImg.cols; y++)
			{
				if (max < scaleImg[idx_scale].at<float>(x, y))
					max = scaleImg[idx_scale].at<float>(x, y);
			}
		}
	}

	// Comparing its to the next scales.
	for (long x = 1; x < grayImg.rows - 1; x++)
	{
		for (long y = 1; y < grayImg.cols - 1; y++)
		{
			bool isMax = true;

			// Considering its neighbors.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					if ((i == 0 && j == 0))
						continue;

					// Finding maxima.
					if (isMax  && scaleImg[0].at<float>(x, y) <= scaleImg[0].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			// Considering the next scale.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					// Finding maxima.
					if (isMax && scaleImg[0].at<float>(x, y) <= scaleImg[1].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			if (isMax && scaleImg[0].at<float>(x, y) >= 0.3*max)
				res.at<float>(x, y) = (float)sqrt(2) * _sigma;
		}
	}


	// Comparing its to both the previous and next scales.
	for (int idx_scale = 1; idx_scale < scaleImg.size() - 1; idx_scale++)
	{
		for (long x = 1; x < grayImg.rows - 1; x++)
		{
			for (long y = 1; y < grayImg.cols - 1; y++)
			{
				if (res.at<float>(x, y) != 0)
					continue;

				bool isMax = true;

				// Considering its neighbors.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						if ((i == 0 && j == 0))
							continue;
						
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				// Considering the next scale.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale + 1].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				// Considering the previous scale.
				for (int i = -1; i <= 1; i++)
				{
					for (int j = -1; j <= 1; j++)
					{
						// Finding maxima.
						if (isMax && scaleImg[idx_scale].at<float>(x, y) <= scaleImg[idx_scale - 1].at<float>(x + i, y + j))
						{
							isMax = false;
						}
					}
				}

				if (isMax && scaleImg[idx_scale].at<float>(x, y) >= 0.3*max)
					res.at<float>(x, y) = (float)pow(sqrt(2), idx_scale + 1) * _sigma;
			}
		}
	}
	

	// Comparing its to the previous scale.
	int last_scale = scaleImg.size() - 1;
	for (long x = 1; x < grayImg.rows - 1; x++)
	{
		for (long y = 1; y < grayImg.cols - 1; y++)
		{
			if (res.at<float>(x, y) != 0)
				continue;

			bool isMax = true;

			// Considering its neighbors.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					if ((i == 0 && j == 0))
						continue;

					// Finding maxima.
					if (isMax && scaleImg[last_scale].at<float>(x, y) <= scaleImg[last_scale].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			// Considering the previous scale.
			for (int i = -1; i <= 1; i++)
			{
				for (int j = -1; j <= 1; j++)
				{
					// Finding maxima.
					if (isMax && scaleImg[last_scale].at<float>(x, y) <= scaleImg[last_scale - 1].at<float>(x + i, y + j))
					{
						isMax = false;
					}
				}
			}

			if (isMax && scaleImg[last_scale].at<float>(x, y) >= 0.3*max)
				res.at<float>(x, y) = (float)pow(sqrt(2), last_scale + 1) * _sigma;
		}
	}

	return res;
}

Mat BlobDetector::updating(Mat img, Mat r)
{
	Mat result = img.clone();
	for (long x = 0; x < grayImg.rows; x++)
	{
		for (long y = 0; y < grayImg.cols; y++)
		{
			if (r.at<float>(x, y) != 0)
				circle(result, Point(y, x), (int)r.at<float>(x, y)*sqrt(2), Scalar(0, 0, 255), 2, 8, 0);
				//circle(result, Point(y, x), 2, Scalar(0, 0, 255), 2, 8, 0);
		}
	}

	return result;
}