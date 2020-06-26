#include "CannyDetector.h"

bool CannyDetector::LoadImage(const char* filename)
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

bool CannyDetector::ShowImage(const char* nameWindow)
{
	if (!grayImg.data)
	{
		cout << "Error 2: Can't show image!" << endl;
		return false;
	}
	imshow(nameWindow, grayImg);
	return true;
}

void CannyDetector::detectByCany(float sigma, float lowThresh, float highThresh)
{
	// Noise Redution - Gaussian Filter.
	GaussianBlured(sigma);

	// Gradien magnitue - Sobel Filter.
	// Creating gradient matrix and angel matrix.
	vector<vector<float>> gradientXY(height, vector<float>(width, 0));
	vector<vector<float>> angleXY(height, vector<float>(width, 0));
	Gradient(gradientXY, angleXY);

	// Non-Max Suppression
	NonMaximumSuppression(gradientXY,  angleXY);

	// Hysteresis
	Hysteresis(lowThresh, highThresh);

}

void CannyDetector::GaussianBlured(float sigma)
{
	int mask_size = 5; // Size of Gauss matrix.
	float sum = 0; // Sum up all elements of Gauss matrix.

	// Creating matrix gaussian with "0" default values.
	vector<vector<float>> gauss(mask_size, vector<float>(mask_size, 0));

	Mat tmpImg = grayImg.clone();

	/* The formula of Gaussian function in 2D:
		G(x, y) = (1/(2*Pi*sigma^2))*e^(-(x^2+y^2)/(2*sigma^2)).
	*/

	for (int i = -mask_size /2; i <= mask_size /2; i++)
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
			grayImg.at<uchar>(x, y) = saturate_cast<uchar>(new_pixel/sum);
		}
	}

	//imshow("Noise Redution - Gaussian filter", grayImg);
	ShowImage("Noise Redution - Gaussian filter");
}

void CannyDetector::Gradient(vector<vector<float>>& gradientXY, vector<vector<float>>& angleXY)
{
	// Creating sobelX and sobelY.
	vector<vector<float>> sobelX{ {-1, 0, 1},
								  {-2, 0, 2},
								  {-1, 0, 1} };
	vector<vector<float>> sobelY{ {-1, -2, -1},
								  {0, 0, 0},
								  {1, 2, 1} };
	int mask_size = 3; // Size of sobel matrix.
	float angle; // Caculating angle.
	// Ignoring bounding values.
	for (int x = mask_size / 2; x < height - mask_size / 2; x++)
	{
		for (int y = mask_size / 2; y < width - mask_size / 2; y++)
		{
			float Gx = 0; // Calculating gradient along vertical direction.
			float Gy = 0; // Calculating gradient along horizontal direction. 
			for (int i = -mask_size / 2; i <= mask_size / 2; i++) 
			{
				for (int j = -mask_size / 2; j <= mask_size / 2; j++) 
				{
					Gx += grayImg.at<uchar>(x + i, y + j)*sobelX[mask_size / 2 - i][mask_size / 2 - j];
					Gy += grayImg.at<uchar>(x + i, y + j)*sobelY[mask_size / 2 - i][mask_size / 2 - j];
				}
			}
			gradientXY[x][y] += sqrt(Gx*Gx + Gy * Gy);  // G = sqrlt(Gx^2 + Gy^2).

			// Angel Calculation
			if (Gx != 0 || Gy != 0)
			{
				angle = fabs(float(atan2(Gy, Gx) * 180 / Pi));
			}
			else
			{
				angle = 0;
			}
			if (157.5 < angle || angle <= 22.5)
			{
				angleXY[x][y] = 0;
			}
			else if (22.5 < angle && angle <= 67.5)
			{
				angleXY[x][y] = 45;
			}
			else if (67.5 < angle && angle <= 112.5)
			{
				angleXY[x][y] = 90;
			}
			else if (112.5 < angle && angle <= 157.5)
			{
				angleXY[x][y] = 135;
			}
		}
	}

	// Updating img
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			grayImg.at<uchar>(x, y) = saturate_cast<uchar>(gradientXY[x][y]);
		}
	}
	//imshow("Gradient magnitude - Sobel filter", grayImg);
	ShowImage("Gradient magnitude - Sobel filter");
}

void CannyDetector::NonMaximumSuppression(vector<vector<float>>& gradientXY, vector<vector<float>>& angleXY)
{
	// Comparing pixel value to its neighbors.
	float pixel1; // neighbor 1.
	float pixel2; // neighbor 2.
	float pixel; // current pixel.
	for (int x = 1; x < height - 1; x++)
	{
		for (int y = 1; y < width - 1; y++)
		{
			switch (uchar(angleXY[x][y]))
			{
			case 0:
				pixel1 = gradientXY[x][y - 1];
				pixel2 = gradientXY[x][y + 1];
				break;
			case 45:
				pixel1 = gradientXY[x - 1][y + 1];
				pixel2 = gradientXY[x + 1][y - 1];
				break;
			case 90:
				pixel1 = gradientXY[x - 1][y];
				pixel2 = gradientXY[x + 1][y];
				break;
			case 135:
				pixel1 = gradientXY[x - 1][y - 1];
				pixel2 = gradientXY[x + 1][y + 1];
				break;
			default:
				break;
			}
			pixel = gradientXY[x][y];

			/* Comparing.
			- If it is greater than all its neighbors then not changing.
			- else set pixel value to 0.
			*/
			if (pixel1 > pixel || pixel < pixel2)
			{
				grayImg.at<uchar>(x, y) = 0;
			}
			
		}
	}

	//imshow("Non-Maximum Suppression", grayImg);
	ShowImage("Non-Maximum Suppression");
}

void CannyDetector::Hysteresis(float lowThresh, float highThresh)
{
	// If pixel value >= high threshold, it will be edge, then considering its neighbors.	
	for (int x = 0; x < height; x++)
	{
		for (int y = 0;  y < width;  y++)
		{
			if (grayImg.at<uchar>(x, y) >= highThresh)
			{
				grayImg.at<uchar>(x, y) = 255;
				HysteresisRecursion(x, y, lowThresh);
			}
		}
	}

	// Setting remaining pixel value to {0, 255}.
	for (int x = 0; x < height; x++)
	{
		for (int y = 0; y < width; y++)
		{
			if (grayImg.at<uchar>(x, y) != 255)
			{
				grayImg.at<uchar>(x, y) = 0;
			}
		}
	}

	//imshow("Hysteresis", grayImg);
	ShowImage("Hysteresis");
}

void CannyDetector::HysteresisRecursion(int x, int y, float lowThresh)
{
	/* Considering pixels that:
	- its value is greater equal than low threshold and connects to current pixel being edge to be added edge.
	- else set value to 0.
	To determine either pixel connects edge or not: considering 3x3 matrix around current pixel being edge.
	*/
	uint8_t value = 0;
	for (int x1 = x - 1; x1 <= x + 1; x1++) 
	{
		for (int y1 = y - 1; y1 <= y + 1; y1++)
		{
			if ((0 <= x1 && x1 < height) && (0 <= y1 && y1 < width) && (x1 != x || y1 != y))
			{
				value = grayImg.at<uchar>(x1, y1);  // Setting current pixel value.
				
				if (value != 255)
				{
					if (value >= lowThresh)
					{
						grayImg.at<uchar>(x1, y1) = 255;
						HysteresisRecursion(x1, y1, lowThresh);  // Considering (x1, y1)'s neighbors because it's edge now.
					}
					else
					{
						grayImg.at<uchar>(x1, y1) = 0;
					}
				}
			}
		}
	}
}
