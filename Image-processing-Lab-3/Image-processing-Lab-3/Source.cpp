#define _USE_MATH_DEFINES

#include <stdio.h>
#include <string>
#include <map>
#include <iostream>

#include <opencv2/highgui/highgui_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "hough.h"
#include "hough_circle.h"



int Clamp(int n)
{
	n = n > 255 ? 255 : n;
	return n < 0 ? 0 : n;
}
cv::Mat grayScale(const cv::Mat& pic) {
	int width = pic.cols;
	int height = pic.rows;
	cv::Mat result = cv::Mat(height, width, CV_8UC1);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			int b = pic.at<cv::Vec3b>(i, j)[0];
			int g = pic.at<cv::Vec3b>(i, j)[1];
			int r = pic.at<cv::Vec3b>(i, j)[2];

			double newValue = (r * 0.2126 + g * 0.7152 + b * 0.0722);
			result.at<uchar>(i, j) = newValue;

		}
	return result;
}
cv::Mat medianFilter(const cv::Mat& pic)
{
	cv::Mat result = pic.clone();
	int width = pic.cols;
	int height = pic.rows;
	uint8_t pix[9];

	for (int i = 1; i < height - 1; ++i)
	{
		for (int j = 1; j < width - 1; ++j)
		{
			pix[0] = pic.at<uint8_t>(i - 1, j - 1);
			pix[1] = pic.at<uint8_t>(i, j - 1);
			pix[2] = pic.at<uint8_t>(i + 1, j - 1);
			pix[3] = pic.at<uint8_t>(i - 1, j);
			pix[4] = pic.at<uint8_t>(i, j);
			pix[5] = pic.at<uint8_t>(i + 1, j);
			pix[6] = pic.at<uint8_t>(i - 1, j + 1);
			pix[7] = pic.at<uint8_t>(i, j + 1);
			pix[8] = pic.at<uint8_t>(i + 1, j + 1);

			std::sort(pix, pix + 9, [](const uint8_t& v1, const uint8_t& v2) -> bool
				{
					return v1 > v2;
				});
			result.at<uint8_t>(i, j) = pix[4];
		}
	}
	return result;
}
cv::Mat operatorSobel(const cv::Mat& pic, cv::Mat& anglesVectors) {

	double x1[] = { -1.0, 0, 1.0 };
	double x2[] = { -2.0, 0, 2.0 };
	double x3[] = { -1.0, 0, 1.0 };

	std::vector<std::vector<double>> xFilter(3);

	xFilter[0].assign(x1, x1 + 3);
	xFilter[1].assign(x2, x2 + 3);
	xFilter[2].assign(x3, x3 + 3);


	double y1[] = { 1.0, 2.0, 1.0 };
	double y2[] = { 0, 0, 0 };
	double y3[] = { -1.0, -2.0, -1.0 };

	std::vector<std::vector<double>> yFilter(3);

	yFilter[0].assign(y1, y1 + 3);
	yFilter[1].assign(y2, y2 + 3);
	yFilter[2].assign(y3, y3 + 3);

	int size = (int)xFilter.size() / 2;

	cv::Mat result = cv::Mat(pic.rows - 2 * size, pic.cols - 2 * size, CV_8UC1);
	anglesVectors = cv::Mat(pic.rows - 2 * size, pic.cols - 2 * size, CV_32FC1);

	for (int i = size; i < pic.rows - size; i++)
	{
		for (int j = size; j < pic.cols - size; j++)
		{
			double sumx = 0;
			double sumy = 0;

			for (int x = 0; x < xFilter.size(); x++)
				for (int y = 0; y < xFilter.size(); y++)
				{
					sumx += xFilter[x][y] * (double)(pic.at<uchar>(i + x - size, j + y - size));
					sumy += yFilter[x][y] * (double)(pic.at<uchar>(i + x - size, j + y - size));
				}
			double sumxsq = sumx * sumx;
			double sumysq = sumy * sumy;

			double sq2 = sqrt(sumxsq + sumysq);

			if (sq2 > 255)
				sq2 = 255;
			result.at<uchar>(i - size, j - size) = sq2;

			if (sumx == 0)
				anglesVectors.at<float>(i - size, j - size) = 90;
			else
				anglesVectors.at<float>(i - size, j - size) = atan(sumy / sumx);
		}
	}
	return result;
}
cv::Mat nonMax(const cv::Mat& pic, cv::Mat& anglesVectors) {

	cv::Mat result = cv::Mat(pic.rows - 2, pic.cols - 2, CV_8UC1);

	for (int i = 1; i < pic.rows - 1; i++) {
		for (int j = 1; j < pic.cols - 1; j++) {
			float Tangent = anglesVectors.at<float>(i, j);

			result.at<uchar>(i - 1, j - 1) = pic.at<uchar>(i, j);
			//Horizontal Edge
			if (((-22.5 < Tangent) && (Tangent <= 22.5)) || ((157.5 < Tangent) && (Tangent <= -157.5)))
			{
				if ((pic.at<uchar>(i, j) < pic.at<uchar>(i, j + 1)) || (pic.at<uchar>(i, j) < pic.at<uchar>(i, j - 1)))
					result.at<uchar>(i - 1, j - 1) = 0;
			}
			//Vertical Edge
			if (((-112.5 < Tangent) && (Tangent <= -67.5)) || ((67.5 < Tangent) && (Tangent <= 112.5)))
			{
				if ((pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j)) || (pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j)))
					result.at<uchar>(i - 1, j - 1) = 0;
			}

			//-45 Degree Edge
			if (((-67.5 < Tangent) && (Tangent <= -22.5)) || ((112.5 < Tangent) && (Tangent <= 157.5)))
			{
				if ((pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j + 1)) || (pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j - 1)))
					result.at<uchar>(i - 1, j - 1) = 0;
			}

			//45 Degree Edge
			if (((-157.5 < Tangent) && (Tangent <= -112.5)) || ((22.5 < Tangent) && (Tangent <= 67.5)))
			{
				if ((pic.at<uchar>(i, j) < pic.at<uchar>(i + 1, j + 1)) || (pic.at<uchar>(i, j) < pic.at<uchar>(i - 1, j - 1)))
					result.at<uchar>(i - 1, j - 1) = 0;
			}
		}
	}
	return result;
}
cv::Mat doubleThresholdAndTrace(const cv::Mat& pic, uint8_t min, uint8_t max) {
	if (min > 255)
		min = 255;
	if (max > 255)
		max = 255;
	cv::Mat result = pic.clone();
	int width = pic.cols;
	int height = pic.rows;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			result.at<uchar>(i, j) = pic.at<uchar>(i, j);
			if (result.at<uchar>(i, j) > max)
				result.at<uchar>(i, j) = 255;
			else if (result.at<uchar>(i, j) < min)
				result.at<uchar>(i, j) = 0;
			else
			{
				bool anyHigh = false;
				bool anyBetween = false;
				for (int x = i - 1; x < i + 2; x++)
				{
					for (int y = j - 1; y < j + 2; y++)
					{
						if (x <= 0 || y <= 0 || height || y > width)
							continue;
						else
						{
							if (result.at<uchar>(x, y) > max)
							{
								result.at<uchar>(i, j) = 255;
								anyHigh = true;
								break;
							}
							else if (result.at<uchar>(x, y) <= max && result.at<uchar>(x, y) >= min)
								anyBetween = true;
						}
					}
					if (anyHigh)
						break;
				}
				if (!anyHigh && anyBetween)
					for (int x = i - 2; x < i + 3; x++)
					{
						for (int y = j - 1; y < j + 3; y++)
						{
							if (x < 0 || y < 0 || x > height || y > width)
								continue;
							else
							{
								if (result.at<uchar>(x, y) > max)
								{
									result.at<uchar>(i, j) = 255;
									anyHigh = true;
									break;
								}
							}
						}
						if (anyHigh)
							break;
					}
				if (!anyHigh)
					result.at<uchar>(i, j) = 0;
			}
		}
	}
	return result;
}
cv::Mat algorithmCanny(const cv::Mat& pic, cv::Mat& anglesVectors)
{
	cv::Mat result = pic.clone();
	//-->Gray
	result = grayScale(result);

	//-->Smoothing
	result = medianFilter(result);

	//-->Search for gradients
	result = operatorSobel(result, anglesVectors);

	//-->Suppression of non-maxima
	result = nonMax(result, anglesVectors);

	//-->Double threshold filtering and Trace
	result = doubleThresholdAndTrace(result, 200, 210);

	return result;
}

struct SortCirclesDistance
{
	bool operator()(const std::pair< std::pair<int, int>, int>& a, const std::pair< std::pair<int, int>, int>& b)
	{
		int d = sqrt(pow(abs(a.first.first - b.first.first), 2) + pow(abs(a.first.second - b.first.second), 2));
		if (d <= a.second + b.second)
		{
			return a.second > b.second;
		}
		return false;
	}

};

void HoughMethodLine(const cv::Mat src, const cv::Mat Canny)
{
	int w = Canny.cols;
	int h = Canny.rows;

	//Transform
	keymolen::Hough hough;
	hough.Transform(Canny.data, w, h);

	int threshold = w > h ? w / 4 : h / 4;

	while (1)
	{
		cv::Mat img_res = src.clone();

		//Search the accumulator
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > > lines = hough.GetLines(threshold);

		//Draw the results
		std::vector< std::pair< std::pair<int, int>, std::pair<int, int> > >::iterator it;
		for (it = lines.begin(); it != lines.end(); it++)
		{
			cv::line(img_res, cv::Point(it->first.first, it->first.second), cv::Point(it->second.first, it->second.second), cv::Scalar(0, 0, 255), 2, 8);
		}

		//Visualize all
		int aw, ah, maxa;
		aw = ah = maxa = 0;
		const unsigned int* accu = hough.GetAccu(&aw, &ah);

		for (int p = 0; p < (ah * aw); p++)
		{
			if ((int)accu[p] > maxa)
				maxa = accu[p];
		}
		double contrast = 1.0;
		double coef = 255.0 / (double)maxa * contrast;

		cv::Mat img_accu(ah, aw, CV_8UC3);
		for (int p = 0; p < (ah * aw); p++)
		{
			unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
			img_accu.data[(p * 3) + 0] = 255;
			img_accu.data[(p * 3) + 1] = 255 - c;
			img_accu.data[(p * 3) + 2] = 255 - c;
		}

		const char* CW_IMG_ORIGINAL = "Result";
		const char* CW_ACCUMULATOR = "Accumulator";

		cv::imshow(CW_IMG_ORIGINAL, img_res);
		cv::imshow(CW_ACCUMULATOR, img_accu);

		char c = cv::waitKey(360000);
		if (c == 'p')
			threshold += 5;
		if (c == 'm')
			threshold -= 5;
		if (c == 27)
			break;
	}
}
void HoughMethodCircle(const cv::Mat src, const cv::Mat Canny)
{
	int w = Canny.cols;
	int h = Canny.rows;

	const char* CW_IMG_ORIGINAL = "Result";
	const char* CW_ACCUMULATOR = "Accumulator";

	keymolen::HoughCircle hough;

	std::vector< std::pair< std::pair<int, int>, int> > circles;
	cv::Mat img_accu;
	for (int r = 19; r < h/2; r = r + 1)
	{
		hough.Transform(Canny.data, w, h, r);

		std::cout << r << " / " << h / 2;

		int	threshold = 0.95 * (2.0 * (double)r * M_PI);

		{
			hough.GetCircles(threshold, circles);

			int aw, ah, maxa;
			aw = ah = maxa = 0;
			const unsigned int* accu = hough.GetAccu(&aw, &ah);

			for (int p = 0; p < (ah * aw); p++)
			{
				if ((int)accu[p] > maxa)
					maxa = accu[p];
			}
			double contrast = 1.0;
			double coef = 255.0 / (double)maxa * contrast;
			img_accu = cv::Mat(ah, aw, CV_8UC3);
			for (int p = 0; p < (ah * aw); p++)
			{
				unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
				img_accu.data[(p * 3) + 0] = 255;
				img_accu.data[(p * 3) + 1] = 255 - c;
				img_accu.data[(p * 3) + 2] = 255 - c;
			}
		}
	}

	std::sort(circles.begin(), circles.end(), SortCirclesDistance());
	int a, b, r;
	a = b = r = 0;
	std::vector< std::pair< std::pair<int, int>, int> > result;
	std::vector< std::pair< std::pair<int, int>, int> >::iterator it;
	for (it = circles.begin(); it != circles.end(); it++)
	{
		int d = sqrt(pow(abs(it->first.first - a), 2) + pow(abs(it->first.second - b), 2));
		if (d > it->second + r)
		{
			result.push_back(*it);
			a = it->first.first;
			b = it->first.second;
			r = it->second;
		}
	}

	cv::Mat img_res = src.clone();
	for (it = result.begin(); it != result.end(); it++)
	{
		std::cout << it->first.first << ", " << it->first.second << std::endl;
		cv::circle(img_res, cv::Point(it->first.first, it->first.second), it->second, cv::Scalar(0, 0, 255), 2, 8);
	}
	cv::imshow(CW_IMG_ORIGINAL, img_res);
	cv::imshow(CW_ACCUMULATOR, img_accu);
	cv::waitKey(1);
}

int main()
{
	//--->Algorithm Canny
	cv::Mat anglesVectors;
	cv::Mat pic = cv::imread("line.jpg");
	cv::imshow("Original image", pic);
	unsigned int start_time1 = clock();
	cv::Mat result = algorithmCanny(pic, anglesVectors);
	unsigned int end_time1 = clock();
	unsigned int search_time1 = end_time1 - start_time1;
	cv::imshow("Algorithm Canny", result);
	std::cout << "Working time of our algorithmCanny: " << search_time1 << std::endl;

	//--->Comparison
	cv::Mat openCVresult;
	unsigned int start_time2 = clock();
	cv::Canny(pic, openCVresult, 100, 130, 3);
	unsigned int end_time2 = clock();
	unsigned int search_time2 = end_time2 - start_time2;
	cv::imshow("Algorithm Canny OpenCV", openCVresult);
	std::cout << "Working time of OpenCV algorithmCanny: " << search_time2 << std::endl;

	//-->Hough's Method
	HoughMethodLine(pic, result);
	//HoughMethodCircle(pic, result);

	cv::waitKey(0);
	return 0;
}
