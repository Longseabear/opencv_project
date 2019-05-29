#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <math.h>

#include "stereoMatching.h"
using namespace std;
using namespace cv;
int main() {
	string li_name("./tsukuba/scene1.row3.col2.png");
	string ri_name("./tsukuba/scene1.row3.col3.png");
	Mat ri, li, ri_gray, li_gray;
	li = imread(li_name.c_str(), IMREAD_COLOR);
	ri = imread(ri_name.c_str(), IMREAD_COLOR);
	if (li.empty() || ri.empty()) {
		printf("file read error");
		return 0;
	}
	cvtColor(li, li_gray, COLOR_BGR2GRAY);
	cvtColor(ri, ri_gray, COLOR_BGR2GRAY);

	li_gray.convertTo(li_gray, CV_32FC1);
	ri_gray.convertTo(ri_gray, CV_32FC1);

	Mat res = MOF(ri_gray, li_gray);

	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("left", li_gray/255);
	imshow("right", ri_gray/255);
	imshow("res", res);
	waitKey(0);
}