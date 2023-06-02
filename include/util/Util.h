#pragma once
#include <filesystem>
#include <vector>

// OpenCV includes
#include <opencv2\opencv.hpp>
#include <opencv2\core\ocl.hpp>
#include <opencv2\dnn\dnn.hpp>


// Includes
#include "Color.h"

using namespace cv;
using namespace dnn;
namespace fs = std::filesystem;

class Util
{
public:
	// ===========================================================
	// File management
	// ===========================================================   
	static std::vector<Rect> loadRectangles(const std::string& filename);
	static std::vector<RotatedRect> loadRotatedRectangles(const std::string& filename);
	static void saveRectsToFile(const std::string& filename, const std::vector<Rect>& rects);
	static void saveRotatedRectsToFile(const std::string& filename, const std::vector<RotatedRect>& rects);
	static void loadImages(const std::string& dirPath, std::vector<Mat>& images);
	static void saveImg(const std::string& filename, const Mat& img);	/* Save an image with a leading index number that increments */

	// ===========================================================
	// Image processing utility
	// =========================================================== 	
	static void drawBorder(Mat& image, int borderSize, const Scalar& borderColor = Color::black);
	static void removeNoiseConnectedComp(Mat& img, int threshold);
	static Mat convertGrayToColor(const Mat& src);
	static Mat fillBlackOutsideOfContours(const Mat& src, const std::vector<Point2f>& corners);
	static Mat resizeImage32(const Mat& image);


	// ===========================================================
	// Drawing
	// ===========================================================
	static void drawRect(Mat& outputImg, const std::vector<Point>& rect, const std::string& rectName, const Scalar& color);
	// SINGLE LINE and points
	static void drawLineAndPoints(Mat& outputImg, Point p1, Point p2, bool coords = true,
	                              const Scalar& pointColor = Scalar(203, 192, 255), const Scalar& lineColor = Scalar(50, 205, 154));
	// MANY LINES and points
	static void drawLinesAndPoints(Mat& outputImg, const std::vector<Point>& points, bool coords = true,
	                               const Scalar& pointColor = Scalar(203, 192, 255), const Scalar& lineColor = Scalar(50, 205, 154));

	static void drawRelationship(const std::vector<Point>& rect1, const std::string& rect1Name, std::pair<Point, Point> connectingLine, const std
	                             ::vector<Point>& rect2, const std::string& rect2Name, Mat& outputImg, const Scalar& rectColor, const Scalar& pointColor, bool
	                             parentChildRel);
	// draw watershed markers
	static Mat drawWatershedMarkers(Mat markers, Mat output, bool drawMarkers = true, bool drawBackground = true, int backgroundLabel = 1);

	// ===========================================================
	// Object detection - template matching
	// =========================================================== 
	static std::vector<Rect> performTemplateMatch(const Mat& src, std::vector<Mat>& templates);

	// ===========================================================
	// Point distance
	// =========================================================== 
	static std::pair<Point, Point> getFarthestPoints(const std::vector<Point2f>& points2F);	
	static std::pair<Point, Point> getFarthestPoints(std::vector<Point> points);
	static bool compareDistances(const Point2f& p1, const Point2f& p2, const Point2f& ref);
	static std::pair<Point, Point> getClosestPairOfPoints(std::vector<Point> points, int left, int right);
	static double distance(Point p1, Point p2);
	static bool comparePointsX(Point p1, Point p2);
	static bool comparePointsY(Point p1, Point p2);

	/* Use convolution to get points that are loose ends. This method expects a thinned/skeletonized image as input */
	static std::vector<Point> getLooseEndPoints(Mat& img);
	/* Overload */
	static std::vector<Point> getLooseEndPoints(Size size, const std::vector<Point>& contours);

	/* Closes gaps by drawing a line between points that are within a certain distance of each other */
	static Mat closeLooseEndGaps(const Mat& img, const std::vector<Point>& looseEndsList, int threshold = 50);

private:

};
