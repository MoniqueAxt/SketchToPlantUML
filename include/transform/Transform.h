#pragma once

//OpenCV includes
#include <opencv2\opencv.hpp>

// Includes
#include "classify\Classsify.h"
#include "util\Util.h"

using namespace cv;



class Transform
{
public:

	static void transform(const Mat& src, int i, const association_set& associationRelationships,
		const inheritance_set& baseInheritedRelationships, const std::vector<std::vector<Point>>& quadrilateralsContours);

private:
	static std::uint32_t fnvHash(const std::vector<cv::Point>& input);
	static std::string toShortHashString(std::uint32_t value);
	static std::string generateShortHash(const std::vector<cv::Point>& input);
};