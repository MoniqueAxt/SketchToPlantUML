#include <windows.h>
#include <iostream>
#include <string>

// OpenCV includes
#include <opencv2\opencv.hpp>
#include <opencv2\core\utility.hpp> 
#include <opencv2\imgproc.hpp>
#include <opencv2\core\ocl.hpp>

// Includes
#include "Segment.h"
#include "util\Util.h"
#include "TextDetection.h"
#include "classify\Classsify.h"
#include "transform\Transform.h"


/* Main entry point of the program  */
int main()
{
	ocl::setUseOpenCL(true);	// Enable OpenCL acceleration
	setUseOptimized(true);		// Enable multithreading

	// #####################################################################
	// Load images and templates
	// #####################################################################

	//Load dataset images
	std::vector<Mat> inputImages{};
	std::string dirPath = "Resources/dataset";
	Util::loadImages(dirPath, inputImages);
	// Load arrow templates
	std::vector<Mat> arrowTemplates;
	dirPath = "Resources/dataset/arrow_templates";
	Util::loadImages(dirPath, arrowTemplates);
	// Load asteriks templates
	std::vector<Mat> starTemplates;
	dirPath = "Resources/dataset/star_templates";
	Util::loadImages(dirPath, starTemplates);


	// #####################################################################
	// Main loop
	// #####################################################################
	std::cout << std::endl;

	// Iterate through each image
	for (auto i = 0; i < inputImages.size(); i++)
	{
		Mat src = inputImages[i];

		if (src.empty()) {
			std::cerr << "Error loading image " << std::endl;
			continue;
		}
		std::cout << "Image " + std::to_string(i) << std::endl;

		// #####################################################################
		// Detect text regions
		// #####################################################################
		Mat resized;					// input for the EAST model
		Mat srcCopy = src.clone();		// to get the adjusted ROIs based on the original image's size
		resized = Util::resizeImage32(srcCopy);
		std::vector<std::vector<Point>> textDetections = TextDetection::detectTextWithEAST(srcCopy, resized);

		std::vector<RotatedRect> textROIs;
		for (const auto& points : textDetections)
		{
			RotatedRect roi = minAreaRect(points);
			textROIs.push_back(roi);
		}

		// Uncomment to save text ROIs to file		
		// Util::saveRotatedRectsToFile(std::to_string(i) + ". text_rois.yml", textROIs);

		// Uncomment to load pre-saved text regions
		/* const std::vector<RotatedRect> textRoisLoaded = Util::loadRotatedRectangles(
			"resources/templateMatch/" + std::to_string(i) + ". text_rois.yml");

		if (textRoisLoaded.empty())
		{
			std::cout << "\nText ROIs not loaded. Exiting." << std::endl;
			return -1;
		}
		*/


		// #####################################################################
		// Template match arrows and asteriks
		// #####################################################################
		std::vector<Rect> arrowROIs = Util::performTemplateMatch(src, arrowTemplates);
		std::cout << i << ". arrow_rois : " << arrowROIs.size() << std::endl;
		std::vector<Rect> starROIs = Util::performTemplateMatch(src, starTemplates);
		std::cout << i << ". star_rois : " << starROIs.size() << std::endl;

		// Uncomment to save rect ROIs to file
		//Util::saveRectsToFile(std::to_string(i) + ". arrow_rois.yml", arrowROIs);
		//Util::saveRectsToFile(std::to_string(i) + ". star_rois.yml", starROIs);

		// Uncomment to load pre-saved template match regions
		// const std::vector<Rect> arrowRoisLoaded = Util::loadRectangles("resources/templateMatch/" + std::to_string(i) + ". arrow_rois.yml");
		// const std::vector<Rect> starRoisLoaded = Util::loadRectangles("resources/templateMatch/" + std::to_string(i) + ". star_rois.yml");



		// #####################################################################
		// Segment and Classify
		// #####################################################################
		std::vector<std::vector<Point>> allLineContours;		// line contours
		std::vector<std::pair<Point, Point>> allArrowEndpoints;	// arrow tip and shaft end-points
		std::vector<std::vector<Point>> quadrilateralsContours;// quadrilateral contours

		Segment::segment(src, i,
			allLineContours,			// return all lines
			allArrowEndpoints,			// return all aroww endpoints
			quadrilateralsContours,		// return all quads
			arrowROIs,
			starROIs,
			textROIs);


		association_set associationRelationships;
		inheritance_set baseInheritedRelationships;

		Classify::classify(src, allLineContours,
			quadrilateralsContours,
			allArrowEndpoints,
			associationRelationships,
			baseInheritedRelationships // return baseInherited relationships
		);


		// =======================================================================================
		// Transform 
		//========================================================================================

		// PlantUML output is printed to console
		Transform::transform(src, i, associationRelationships, baseInheritedRelationships, quadrilateralsContours);


	}

	return 0;
}
