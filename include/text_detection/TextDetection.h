#pragma once
#include <vector>



// OpenCV includes
#include <opencv2\opencv.hpp>
#include <opencv2\dnn\dnn.hpp>


using namespace cv;
using namespace dnn;


class TextDetection
{
public:
	static std::vector<std::vector<Point>> detectTextWithEAST(Mat& orig, const Mat& img);

private:
};