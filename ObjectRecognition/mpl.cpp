#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "opencv2/opencv.hpp"
#include <iostream>
#include <sstream>
#include <istream>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;
vector<string> classes;
void parser(const string & name) {
	ifstream f(name);
	if (f.fail()) {
		exit(1);
	}

	while (f.good()) {
		string buffer;
		getline(f, buffer);
		classes.push_back(buffer);
	}
}
int main() {
	VideoCapture cap(0);

	if (!cap.isOpened()) return -1;
	string model = "./bvlc_googlenet.caffemodel";
	string config = "./bvlc_googlenet.prototxt";
	string label = "./classification_classes_ILSVRC2012.txt";

	parser(label);
	Net net = readNet(model, config);

	Mat edges, dst;
	namedWindow("edges", 1);
	for (;;) {
		Mat frame, blob;
		cap >> frame;

		blob = blobFromImage(frame, 1.0, Size(224, 224), Scalar(106,116,124));
		net.setInput(blob);
		Mat prob = net.forward();
		
		Point classIdPoint;
		double confidence;
		minMaxLoc(prob.reshape(1, 1), 0, &confidence,0, &classIdPoint);
		int classId = classIdPoint.x;

		label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
			classes[classId].c_str()),
			confidence);
		putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow("edges",frame);

		if (waitKey(30) >= 0) break;
	}
	return 0;
}