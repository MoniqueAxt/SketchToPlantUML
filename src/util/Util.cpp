#include <random>
#include <util\Util.h>

#include <utility>


// ===========================================================
// File management
// ===========================================================   
// Load vector of rectangles from file
std::vector<Rect> Util::loadRectangles(const std::string& filename)
{
	std::vector<Rect> rectangles;
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Failed to open the file." << std::endl;
		return rectangles;
	}

	FileNode node = fs["rects"];
	if (node.empty() || !node.isSeq())
	{
		return rectangles;
	}
	FileNodeIterator it = node.begin(), it_end = node.end();
	for (; it != it_end; ++it)
	{
		Rect rect;
		(*it)["x"] >> rect.x;
		(*it)["y"] >> rect.y;
		(*it)["width"] >> rect.width;
		(*it)["height"] >> rect.height;
		rectangles.push_back(rect);
	}
	fs.release();

	return rectangles;
}
// Load vector of rotated rectangles from file
std::vector<RotatedRect> Util::loadRotatedRectangles(const std::string& filename)
{
	std::vector<RotatedRect> rectangles;
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened()) {
		std::cerr << "Failed to open the file." << std::endl;
		return rectangles;
	}

	FileNode node = fs["rects"];
	if (node.empty() || !node.isSeq())
	{
		return rectangles;
	}
	FileNodeIterator it = node.begin(), it_end = node.end();
	for (; it != it_end; ++it)
	{
		RotatedRect rect;
		(*it)["center_x"] >> rect.center.x;
		(*it)["center_y"] >> rect.center.y;
		(*it)["width"] >> rect.size.width;
		(*it)["height"] >> rect.size.height;
		(*it)["angle"] >> rect.angle;
		rectangles.push_back(rect);
	}
	fs.release();

	return rectangles;
}

// Save vector of rectangles to file
void Util::saveRectsToFile(const std::string& filename, const std::vector<Rect>& rects)
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "rects" << "[";
	for (const auto& rect : rects) {
		fs << "{"
			<< "x" << rect.x
			<< "y" << rect.y
			<< "width" << rect.width
			<< "height" << rect.height
			<< "}";
	}
	fs << "]";
}

// Save vector of rotated rectangles to file
void Util::saveRotatedRectsToFile(const std::string& filename, const std::vector<RotatedRect>& rects)
{
	FileStorage fs(filename, FileStorage::WRITE);
	fs << "rects" << "[";
	for (const auto& rect : rects) {
		fs << "{"
			<< "center_x" << rect.center.x
			<< "center_y" << rect.center.y
			<< "width" << rect.size.width
			<< "height" << rect.size.height
			<< "angle" << rect.angle
			<< "}";
	}
	fs << "]";
}

// Load dataset input images
void Util::loadImages(const std::string& dirPath, std::vector<Mat>& images)
{
	// Load input images
	for (const auto& entry : fs::directory_iterator(dirPath)) {
		if (entry.is_regular_file()) {
			Mat image = imread(entry.path().string(), IMREAD_COLOR);
			// Check if image was loaded successfully
			if (!image.empty()) {
				// Add image 				
				images.push_back(image);
			}
		}
	}
}

void Util::saveImg(const std::string& filename, const Mat& img)
{
#ifdef DEBUG_BUILD
	if (img.empty()) {
		return;
	}

	static int i = 1;

	std::string ext = ".png";
	std::string name = std::to_string(i++) + ". " + filename + ext;

	imwrite(name, img);
#endif
}

// ===========================================================
// Image processing utility
// =========================================================== 
void Util::drawBorder(Mat& image, const int borderSize, const Scalar& borderColor)
{
	// Draw a rectangle on the image boundary
	rectangle(image, Point(0, 0), Point(image.cols - 1, image.rows - 1), borderColor, borderSize);
}

void Util::removeNoiseConnectedComp(Mat& img, const int threshold)
{
	Mat labels, stats, centroids;

	const int numComponents = connectedComponentsWithStats(img, labels, stats, centroids);
	for (int i = 1; i < numComponents; i++) { // start from 1 to skip background component 
		const int area = stats.at<int>(i, CC_STAT_AREA);
		if (area < threshold) {				// threshold is the minimum component size to keep
			// set all pixels in this component to 0
			Mat componentMask = (labels == i);
			img.setTo(0, componentMask);
		}
	}
}

Mat Util::convertGrayToColor(const Mat& src)
{
	Mat converted(src.size(), CV_8UC3);			// create 8-bit color img
	cvtColor(src, converted, COLOR_GRAY2RGB);	// convert to color
	return converted;
}

/* Remove noise around ROI contours  */
Mat Util::fillBlackOutsideOfContours(const Mat& src, const std::vector<Point2f>& corners) {
	const Mat img = src.clone();

	// convert Point2f to Point
	std::vector<Point> pts{};
	for (const auto corner : corners)
	{
		pts.emplace_back(corner.x, corner.y);
	}

	Mat stencil = Mat::zeros(img.size(), img.type());	 // black stencil as a mask		
	fillPoly(stencil, pts, Color::white); 		 		// draw the poly in white
	Mat result = Mat::zeros(img.size(), img.type()); 	// result img with black background
	bitwise_and(img, stencil, result);	 				// bitwise to keep only the ROI

	return result;
}

Mat Util::resizeImage32(const Mat& image)
{
	// get the dimensions of the image
	const int height = image.size().height;
	const int width = image.size().width;

	const int scaleWidth = width / 320;
	const int scaleHeight = height / 320;
	const int smallestDivisor = std::min(scaleWidth, scaleHeight);

	int newWidth = width / smallestDivisor;
	int newHeight = height / smallestDivisor;
	// Get the nearest value that's a multiple of 32
	newWidth = newWidth / 32 * 32;
	newHeight = newHeight / 32 * 32;

	// resize the image
	Mat resizeLarger;
	resize(image, resizeLarger, Size(newWidth * 2, newHeight * 2), 0, 0, INTER_LINEAR);
	Mat resizeSmaller;
	resize(resizeLarger, resizeSmaller, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);

	return resizeSmaller;
}


// ===========================================================
// Drawing
// ===========================================================

void Util::drawLineAndPoints(Mat& outputImg, const Point p1, const Point p2, const bool coords,
	const Scalar& pointColor, const Scalar& lineColor)
{
	//draw line
	line(outputImg, p1, p2, lineColor, 2);

	// draw circle at points
	circle(outputImg, p1, 6, pointColor, FILLED);
	circle(outputImg, p2, 6, pointColor, FILLED);

	if (coords) {
		// draw point coordinates on img 
		std::string textX = std::to_string((int)p1.x);
		std::string textY = std::to_string((int)p1.y);
		putText(outputImg, textX, { static_cast<int>(p1.x), static_cast<int>(p1.y) - 40 }, FONT_HERSHEY_PLAIN, 2, Color::orange, 1);
		putText(outputImg, textY, { static_cast<int>(p1.x), static_cast<int>(p1.y) - 10 }, FONT_HERSHEY_PLAIN, 2, Color::red, 1);
		textX = std::to_string((int)p2.x);
		textY = std::to_string((int)p2.y);
		putText(outputImg, textX, { static_cast<int>(p2.x), static_cast<int>(p2.y) - 40 }, FONT_HERSHEY_PLAIN, 2, Color::orange, 1);
		putText(outputImg, textY, { static_cast<int>(p2.x), static_cast<int>(p2.y) - 10 }, FONT_HERSHEY_PLAIN, 2, Color::red, 1);
	}
}

/* Draw points and connect lines */
void Util::drawRect(Mat& outputImg, const std::vector<Point>& rect,
	const std::string& rectName, const Scalar& color)
{
	const Rect boundingBox = boundingRect(rect);
	rectangle(outputImg, boundingBox, color, FILLED);

	// Draw the class names on the rectangles
	constexpr int thickness = 2;
	const Size textSize = getTextSize(rectName, FONT_HERSHEY_SIMPLEX, 1, thickness, 0);

	// Draw rect 1 name
	const Moments m = moments(rect);
	const Point center(m.m10 / m.m00, m.m01 / m.m00);
	const Point textPos(center.x - textSize.width / 2, center.y + textSize.height / 2);
	putText(outputImg, rectName, textPos, FONT_HERSHEY_SIMPLEX, 1, Color::white, thickness);
}

void Util::drawLinesAndPoints(Mat& outputImg, const std::vector<Point>& points, const bool coords,
	const Scalar& pointColor, const Scalar& lineColor) {

	for (size_t i = 0; i < points.size(); i++) {
		const Point p1 = points[i];
		const Point p2 = points[(i + 1) % points.size()];
		drawLineAndPoints(outputImg, p1, p2, coords, pointColor, lineColor);
	}
}


void Util::drawRelationship(
	const std::vector<Point>& rect1,
	const std::string& rect1Name,
	const std::pair<Point, Point> connectingLine,
	const std::vector<Point>& rect2,
	const std::string& rect2Name,
	Mat& outputImg,
	const Scalar& rectColor,
	const Scalar& pointColor,
	bool parentChildRel)
{
	// Draw the rectangles
	Rect boundingRect1 = boundingRect(rect1);
	Rect boundingRect2 = boundingRect(rect2);

	if (parentChildRel)
	{
		constexpr int padding = 20;  // Adjust this value to control the smaller rectangle size
		boundingRect1 = Rect(
			boundingRect1.x + padding,
			boundingRect1.y + padding,
			boundingRect1.width - 2 * padding,
			boundingRect1.height - 2 * padding
		);
		boundingRect2 = Rect(
			boundingRect2.x + padding,
			boundingRect2.y + padding,
			boundingRect2.width - 2 * padding,
			boundingRect2.height - 2 * padding
		);
	}

	rectangle(outputImg, boundingRect1, rectColor, FILLED);
	rectangle(outputImg, boundingRect2, rectColor, FILLED);


	// Draw the line/arrow connecting the rects
	line(outputImg, connectingLine.first, connectingLine.second, rectColor, 3);
	circle(outputImg, connectingLine.first, 7, pointColor, FILLED);
	circle(outputImg, connectingLine.second, 7, pointColor, FILLED);


	// Draw the class names on the rectangles
	constexpr int thickness = 2;
	const Size textSize = getTextSize(rect1Name, FONT_HERSHEY_SIMPLEX, 1, thickness, 0);

	// Draw rect 1 name
	Moments m = moments(rect1);
	Point center(m.m10 / m.m00, m.m01 / m.m00);
	Point textPos(center.x - textSize.width / 2, center.y + textSize.height / 2);
	putText(outputImg, rect1Name, textPos, FONT_HERSHEY_SIMPLEX, 1, Color::white, thickness);

	// Draw rect 2 name
	m = moments(rect2);
	center = Point(m.m10 / m.m00, m.m01 / m.m00);
	textPos = Point(center.x - textSize.width / 2, center.y + textSize.height / 2);
	putText(outputImg, rect2Name, textPos, FONT_HERSHEY_SIMPLEX, 1, Color::white, thickness);


}

Mat Util::drawWatershedMarkers(Mat markers, Mat output, const bool drawMarkers, const bool drawBackground, const int backgroundLabel)
{
	// Range for each color channel
	std::uniform_int_distribution dis(0, 255);

	// Seed the random number generator with the current time
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 gen(seed);

	for (int i = 0; i < markers.rows; i++) {
		for (int j = 0; j < markers.cols; j++) {
			const int label = markers.at<int>(i, j);

			if (label == 0) {
				output.at<Vec3b>(i, j) = Vec3b(128, 128, 128);	// Gray for unknown
			}
			else if (label == -1) {	// boundary
				output.at<Vec3b>(i, j) = Vec3b(0, 255, 255); // Yellow for boundaries
			}
			// background
			else if (label == backgroundLabel && drawBackground) {
				output.at<Vec3b>(i, j) = Vec3b(0, 0, 0); // Black for background

			}
			else if (drawMarkers) {	// Positive int labels >= 2
				// Generate random values for each channel
				const int blue = dis(gen);
				const int green = dis(gen);
				const int red = dis(gen);

				Scalar color(blue, green, red);
				output.at<Vec3b>(i, j) = Vec3b(color[0], color[1], color[2]); // Random colors for markers
			}
		}
	}

	return output;
}

// ===========================================================
// Object detection - template matching
// =========================================================== 
std::vector<Rect> Util::performTemplateMatch(const Mat& src, std::vector<Mat>& templates)
{
	struct RectCompare {
		bool operator()(const Rect& a, const Rect& b) const {
			if (a.x < b.x) {
				return true;
			}
			else if (a.x == b.x && a.y < b.y) {
				return true;
			}
			else {
				return false;
			}
		}
	};

	// Load input image
	Mat inputImage = src.clone();

	// Loop over templates and perform template matching
	std::set<Rect, RectCompare> allDetections;	// vector to store all matching locations
	std::vector<std::thread> threads;			// store all threads

	// Loop over templates
	for (auto& templateImg : templates) {
		double thresholdValue = 0.8;
		// Ensure the template image is not larger than the input image
		if (templateImg.rows > inputImage.rows || templateImg.cols > inputImage.cols)
			continue;

		// Declare a mutex
		std::mutex mutexAllDetections;

		// Start a new thread
		threads.emplace_back([&inputImage, &templateImg, &allDetections, &mutexAllDetections, thresholdValue]()
			{
				std::vector<Rect> detections;// vector to store all matching locations 
				// Perform template matching
				Mat result;
				matchTemplate(inputImage, templateImg, result, TM_CCOEFF_NORMED);

				// Find locations of matches above the threshold
				std::vector<Point> locations;
				threshold(result, result, thresholdValue, 1.0, THRESH_TOZERO);	// threshold the output matrix
				//extract the coordinates of the non-zero values (locations in the input img where the template was found)
				findNonZero(result, locations);	

				Rect bestMatchRect;

				// Apply non-maximum suppression to remove overlapping detections		
				for (const auto& loc : locations)
				{
					Rect detection(loc.x, loc.y, templateImg.cols, templateImg.rows);

					bool overlap = false;
					for (auto& existing : allDetections)
					{
						const float overlapRatio = (detection & existing).area() / static_cast<float>(detection.area());
						if (overlapRatio > 0.2) {
							overlap = true;
							break;
						}
					}
					if (!overlap) {
						// Find best match location
						Mat resultROI = result(detection);
						double minVal, maxVal;
						Point minLoc, maxLoc;
						minMaxLoc(resultROI, &minVal, &maxVal, &minLoc, &maxLoc);
						maxLoc.x += detection.x;
						maxLoc.y += detection.y;

						// Add best match detection to detections vector						
						// lock the mutex before accessing/modifying the set
						std::lock_guard<std::mutex> lock(mutexAllDetections);
						allDetections.insert(Rect(maxLoc.x, maxLoc.y, templateImg.cols, templateImg.rows));
					}
				}
			});
	}
	// Wait for all threads to finish
	for (auto& thread : threads) {
		thread.join();
	}

	// Convert the detections set to a vector before returning
	std::vector detectionsVec(allDetections.begin(), allDetections.end());

	return detectionsVec;
}


// ===========================================================
// Point distance
// =========================================================== 
/*  calculates the Euclidean distance between two points and compares them based on their distance from the reference point */
bool Util::compareDistances(const Point2f& p1, const Point2f& p2, const Point2f& ref) {
	const double dist1 = norm(p1 - ref);
	const double dist2 = norm(p2 - ref);
	return dist1 < dist2;
}

/* Sort a list of contour points, return the extreme points */
std::pair<Point, Point> Util::getFarthestPoints(std::vector<Point> points) {
	// Find the first farthest point
	Point2f refPoint = points[0]; // a reference point for distance comparison 
	auto farthestIt = std::max_element(points.begin(), points.end(),
		[&refPoint](const Point2f& p1, const Point2f& p2) {
			return compareDistances(p1, p2, refPoint);
		}
	);
	Point2f farthestPoint1 = *farthestIt;
	double maxDistance = norm(farthestPoint1 - refPoint);

	// Find the second farthest point
	refPoint = farthestPoint1; // use the first farthest point as a reference point 
	farthestIt = std::max_element(points.begin(), points.end(),
		[&refPoint](const Point2f& p1, const Point2f& p2) {
			return compareDistances(p1, p2, refPoint);
		}
	);
	Point2f farthestPoint2 = *farthestIt;
	maxDistance += norm(farthestPoint2 - refPoint);

	return std::make_pair(farthestPoint1, farthestPoint2);
}
/* Overload */
std::pair<Point, Point> Util::getFarthestPoints(const std::vector<Point2f>& points2F)
{
	std::vector<Point> points;
	for (auto& point2F : points2F)
	{
		points.emplace_back(point2F);
	}

	return getFarthestPoints(points);
}

std::pair<Point, Point> Util::getClosestPairOfPoints(std::vector<Point> points, const int left, const int right)
{
	// sort the list of points (avoid the same point being returned as both in the pair)
	std::sort(points.begin(), points.end(), comparePointsY);

	// Check if there is only one point
	if (left == right)
	{
		return std::make_pair(points[left], points[right]);
	}
	// Check if there are only two points
	else if (left + 1 == right)
	{
		return std::make_pair(points[left], points[right]);
	}
	else {
		// Divide the list of points into two sections 
		// Recursively find the closest pair in each section
		const int m = (left + right) / 2;
		const std::pair<Point, Point> leftPair = getClosestPairOfPoints(points, left, m);
		//std::cout << "Left pair: " << left_pair.first << left_pair.second << std::endl;
		const std::pair<Point, Point> rightPair = getClosestPairOfPoints(points, m + 1, right);
		//std::cout << "Right pair: " << right_pair.first << right_pair.second << std::endl;

		// Compute the distance between the closest pairs in each half
		const double d1 = distance(leftPair.first, leftPair.second);
		const double d2 = distance(rightPair.first, rightPair.second);
		const double d = min(d1, d2);

		// Find pairs with distances less than the minimum (d)
		// Combine the closest pairs in each half 			
		std::vector<Point> strip;
		const double x = points[m].x;
		for (int i = left; i <= right; i++) {
			if (fabs(points[i].x - x) < d) {
				strip.push_back(points[i]);
			}
		}
		// Sort the points in the strip by their y-coordinate
		sort(strip.begin(), strip.end(), [](const Point point1, const Point point2)
			{
				// return comparePointsY(point1,point2)
				return point1.y < point2.y;
			});

		// Find the closest pair among the points in the strip
		std::pair<Point, Point> closestStripPair;
		double closestStripDistance = d;
		for (size_t i = 0; i < strip.size(); i++)
		{
			for (size_t j = i + 1; j < strip.size() && strip[j].y - strip[i].y < d; j++)
			{
				const double dij = distance(strip[i], strip[j]);
				if (dij < closestStripDistance)
				{
					closestStripDistance = dij;
					closestStripPair = std::make_pair(strip[i], strip[j]);
				}
			}
		}

		// Return the closest pair of points from the pairs found in each half and in the strip
		if (closestStripDistance < d)
		{
			return closestStripPair;
		}
		else
		{
			return (d1 < d2) ? leftPair : rightPair;
		}
	}
}

/* Calculate the simple Euclidean distance between two points */
double Util::distance(const Point p1, const Point p2)
{
	const double dx = p1.x - p2.x;
	const double dy = p1.y - p2.y;
	return sqrt(dx * dx + dy * dy);
}
/* Compare points based on x-axis */
bool Util::comparePointsX(const Point p1, const Point p2) {
	// define a comparison function for Point objects
	if (p1.x != p2.x) {
		return p1.x < p2.x; // sort by x-coordinate
	}
	return p1.y < p2.y; // if x-coordinate is the same, sort by y-coordinate
}

/* Compare points based on y-axis */
bool Util::comparePointsY(const Point p1, const Point p2) {
	// define a comparison function for Point objects
	if (p1.y != p2.y) {
		return p1.y < p2.y; // sort by y-coordinate
	}
	return p1.x < p2.x; // if y-coordinate is the same, sort by x-coordinate
}

/* Closes gaps by drawing a line between points that are closest to each other */
Mat Util::closeLooseEndGaps(const Mat& img, const std::vector<Point>& looseEndsList, const int threshold)
{
	// Close gaps
	Mat finalImg = img.clone();
	//std::vector<std::pair<Point, Point>> lines; 

	// iterate every point in the list and draw a line between nearest point
	for (size_t i = 0; i < looseEndsList.size(); i++) {
		int minDist = std::max(img.rows, img.cols);

		Point pt2;
		for (size_t j = 0; j < looseEndsList.size(); j++) {
			if (i == j)		// points are the same
			{
				continue;
			}
			const double dist = norm(looseEndsList[i] - looseEndsList[j]);
			if (dist < minDist) {
				minDist = dist;
				pt2 = looseEndsList[j];
			}
		}
		// Draw lines between points if they're at least min_dist apart
		if (minDist <= threshold) {
			line(finalImg, looseEndsList[i], pt2, Color::white, 1);
			//std::pair<Point, Point> line = { looseEndsList[i], pt2 };
			//lines.push_back(line);
		}
	}

	return finalImg;
}

/* Use convolution to get points that are loose ends.
	This method expects a thinned/skeletonized image as input */
std::vector<Point> Util::getLooseEndPoints(Mat& img)
{
	// Adapted from: https://stackoverflow.com/a/72502985

	// Kernels for each of the 8 variations
	Mat k1 = (Mat_<char>(3, 3) <<
		0, 0, 0,
		-1, 1, -1,
		-1, -1, -1);
	Mat k2 = (Mat_<char>(3, 3) <<
		0, -1, -1,
		0, 1, -1,
		0, -1, -1);
	Mat k3 = (Mat_<char>(3, 3) <<
		-1, -1, 0,
		-1, 1, 0,
		-1, -1, 0);
	Mat k4 = (Mat_<char>(3, 3) <<
		-1, -1, -1,
		-1, 1, -1,
		0, 0, 0);

	Mat k5 = (Mat_<char>(3, 3) <<
		-1, -1, -1, -1, 1, -1, 0, -1, -1);
	Mat k6 = (Mat_<char>(3, 3) <<
		-1, -1, -1,
		-1, 1, -1,
		-1, -1, 0);
	Mat k7 = (Mat_<char>(3, 3) <<
		-1, -1, 0,
		-1, 1, -1,
		-1, -1, -1);
	Mat k8 = (Mat_<char>(3, 3) <<
		0, -1, -1,
		-1, 1, -1,
		-1, -1, -1);

	// hit-or-miss transform
	Mat o1, o2, o3, o4, o5, o6, o7, o8, out1, out2, out;
	morphologyEx(img, o1, MORPH_HITMISS, k1);
	morphologyEx(img, o2, MORPH_HITMISS, k2);
	morphologyEx(img, o3, MORPH_HITMISS, k3);
	morphologyEx(img, o4, MORPH_HITMISS, k4);
	out1 = o1 + o2 + o3 + o4;

	morphologyEx(img, o5, MORPH_HITMISS, k5);
	morphologyEx(img, o6, MORPH_HITMISS, k6);
	morphologyEx(img, o7, MORPH_HITMISS, k7);
	morphologyEx(img, o8, MORPH_HITMISS, k8);
	out2 = o5 + o6 + o7 + o8;

	// contains all the loose end points
	add(out1, out2, out);

	// store the loose end points in a Mat
	Mat pts;
	findNonZero(out == 255, pts);

	// Put points in a vector
	std::vector<Point> looseEndsList{};
	for (int i = 0; i < pts.rows; i++)
	{
		looseEndsList.emplace_back(pts.at<Point>(i).x, pts.at<Point>(i).y);
	}

	return looseEndsList;
}

/*Overload*/
std::vector<Point> Util::getLooseEndPoints(const Size size, const std::vector<Point>& contours)
{
	Mat img = Mat::zeros(size, CV_8UC1);
	drawContours(img, std::vector<std::vector<Point>>{contours}, -1, Color::white, 1);

	return getLooseEndPoints(img);
}