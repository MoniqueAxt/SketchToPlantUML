#pragma once

// OpenCV includes
#include <opencv2\opencv.hpp>

// Includes
#include <util\Util.h>


using namespace cv;


/* Preprocess and segment the image. Isolate quadrilaterals representing classes, and
 * lines and arrows representing relationships */
class Segment
{
public:
	static void segment(const Mat& src, int imageIndex,
		std::vector<std::vector<Point>>& allLineContours,			// return line contours
		std::vector<std::pair<Point, Point>>& allArrowEndpoints,	// return arrow tip and shaft end-points
		std::vector<std::vector<Point>>& quadrilateralsContours,	// return quadrilaterals
		const std::vector<Rect>& arrowROIs,
		const std::vector<Rect>& starROIS,
		const std::vector<RotatedRect>& textROIs);

private:

	// ######################################################
	// LINE ENDPOINTS
	// ######################################################
	static Mat preprocessLineImage(const std::vector<Rect>& arrowROIs,
		const std::vector<RotatedRect>& textROIs, Mat& linesImage);


	/* Get all line contours  */
	static std::vector<std::vector<Point>> getAllLines(
		Mat& isolatedArrowsLines,
		const std::vector<Rect>& arrowROIs,
		const std::vector<RotatedRect>& textROIs);


	// ######################################################
	// ARROW endpoints
	// ######################################################
	/* Get all arrow endpoints in the form of tip-point and shaft end-point */
	static std::vector<std::pair<Point, Point>> getAllArrowEndPoints(
		const Mat &isolatedArrowsLines, const std::vector<Rect>& arrowROIs);

	/* Get the tip point and shaft end - points of the arrow.
		* The arrow tip will always be the first of the pair.	*/
	static std::pair<Point, Point> getArrowEndpoints(Mat& individualArrow);


	// ######################################################
	// Preprocessing
	// ######################################################
	static Mat performThresholding(const Mat& src, int imageIndex);
	static Mat preprocessing(const Mat& bin);

	/* Thinning algorithms */
	// Source: https://github.com/bsdnoobz/zhang-suen-thinning
	static void thinningIteration(Mat& img, int iter);
	static void thinning(const Mat& src, Mat& dst);

	/* Create two images: one with filled contours, one with thinned outlines */
	static void fillQuads(Mat& cont, Mat& removeTextImg, Mat& filledShapes, const std::vector<Rect>& arrowROIs);
	
	// Light correction methods
	static Mat removeLightWithPattern(const Mat& gray);
	static Mat calculateLightPattern(const Mat& img);
	static Mat removeLight(const Mat& img, const Mat& pattern, int method);


	// ######################################################
	// Segmentation
	// ######################################################
	/* Use distance transform to get markers for watershed */
	static Mat getWatershedMarkers(const Mat& filledThinned);

	/* Watershed segmentation */
	static void performWatershed(Mat& markers, const Mat& watershedInput);

	/* Isolated classes */
	static void isolateQuadrilaterals(const Mat& markers, Mat& isolatedQuads,
		std::vector<std::vector<Point>>& contours, const std::vector<Rect> & arrowRoIs);

	/* Isolate relationships */
	static Mat isolateLinesArrows(const Mat & isolatedQuads, const Mat & removeTextImg);

	/* Remove arrows from an image */
	static void removeArrows(const Mat& image, const std::vector<Rect> & arrowROIs);


};