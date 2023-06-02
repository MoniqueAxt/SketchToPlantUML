#include "text_detection\TextDetection.h"
#include <fstream>

std::vector<std::vector<Point>> TextDetection::detectTextWithEAST(Mat& orig, const Mat& img)
{
	// Model expects image dimensions as multiples of 32
	if (img.size().width % 32 != 0 || img.size().height % 32 != 0) {
		std::cout << "\nWarning: image dimensions are not multiples of 32!" << std::endl;
	}

	// Note: this must not be a binary/greyscale image
	Mat frame = img.clone();

	// Load the network in the memory.
	// Passing an EAST detector as an argument which will automatically detect the framework based on file type.
	// For .pb files, the Tensorflow network will automatically be loaded.

	// Specify the path to the model file
	std::string modelPath = "Resources/frozen_east_text_detection.pb";

	// Check if the file exists
	std::ifstream modelFile(modelPath);
	if (!modelFile.is_open())
	{
		std::cerr << "Error: Model file '" << modelPath << "' not found" << std::endl;
		return std::vector<std::vector<Point>> {};	// empty vector
	}
	modelFile.close();

	// Set the parameters
	TextDetectionModel_EAST model(modelPath);
	float confThreshold = 0.1f;              			// reject detections below the confidence threshold
	float nmsThreshold = 0.1f;               			// Non-Max Suppression (NMS) - reduce overlapping bounding boxes
	model.setConfidenceThreshold(confThreshold)
		.setNMSThreshold(nmsThreshold);

	double detScale = 1.0;						// scaling factor that applied to the input image 
	Size detInputSize = Size(img.size().width, img.size().height);	// size of the input image
	Scalar detMean = Scalar(123.68, 116.78, 103.94);		// subtract the mean value from image data
	// Swap the Red and Blue channels of the input image
	// (required since OpenCV uses BGR format and Tensorflow uses RGB format)
	bool swapRB = true;

	model.setInputParams(detScale, detInputSize, detMean, swapRB);
	model.setPreferableBackend(DNN_BACKEND_OPENCV);
	model.setPreferableTarget(DNN_TARGET_CPU);

	// Detect text
	std::vector<std::vector<Point>> detectionResults;
	model.detect(frame, detectionResults);

	// ADJUST: Get the corresponding text ROIs on the original image (not resized)
	// Calculate the scaling factors along the x and y dimensions
	double scaleX = static_cast<double>(orig.cols) / img.cols;
	double scaleY = static_cast<double>(orig.rows) / img.rows;

	// Iterate over each detected region and adjust the coordinates
	std::vector<std::vector<Point>> adjustedResults;
	for (const auto& region : detectionResults)
	{
		std::vector<Point> adjustedRegion;
		// Iterate over each point in the region and adjust the coordinates
		for (const auto& point : region)
		{
			// Scale the point coordinates by the scaling factors
			Point adjustedPoint(point.x * scaleX, point.y * scaleY);
			adjustedRegion.push_back(adjustedPoint);
		}
		adjustedResults.push_back(adjustedRegion);
	}

	// Draw on original image
	// polylines(orig, adjustedResults, true, Scalar(0, 255, 0), 1);
	// Util::save_img("TEXT_ROIs", orig);

	return adjustedResults;
}
