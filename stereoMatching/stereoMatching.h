
#ifndef STEREO_MATCHING
#define STEREO_MATCHING
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;

float MAD(Mat src, Mat target, int i, int j, int ti, int tj);
Mat MOF(Mat src, Mat target);
#endif