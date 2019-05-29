#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;

const int BLOCK_SIZE = 20;
const int SEARCH_RANGE = 16;

float MAD(Mat src, Mat target, int i, int j, int ti, int tj) {
	float sum = 0;
	int N = 0;
	for (int dy = -BLOCK_SIZE / 2; dy < BLOCK_SIZE / 2; dy++) {
		for (int dx = -BLOCK_SIZE / 2; dx < BLOCK_SIZE / 2; dx++) {
			if (i + dy < 0 || i + dy >= src.rows || j + dx < 0 || j + dx >= src.cols ||
				i + dy + ti < 0 || i + dy + ti >= src.rows || j + dx + tj < 0 || j + dx + tj >= src.cols) return -1;

			float p = src.at<float>(i + dy, j + dx);
			float p2 = target.at<float>(i + dy + ti, j + dx + tj);
			sum += abs(p - p2);
			N++;
		}
	}
	if (N == 0) return INFINITY;
	sum /= N;
	return sum;
}
/*
	@parm: Mat src : source image
	@parm: Mat dst : traget image
	@return: disparity in terms of x
*/
Mat MOF(Mat src, Mat target) {
	Mat dst = Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int max_j = 0;
			float min_MAD = INFINITY;
			// i,j src
			for (int tj = 0; tj < SEARCH_RANGE; tj++) {
				float mad = MAD(src, target, i, j, 0, tj);
				if (min_MAD > mad) {
					min_MAD = mad;
					max_j = tj;
				}
			}
			dst.at<float>(i, j) = max_j;
		}
	}
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			dst.at<float>(i, j) = dst.at<float>(i, j) / 16;
		}
	}
	return dst;
}