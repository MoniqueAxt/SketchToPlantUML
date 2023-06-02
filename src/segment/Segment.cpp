
#include "segment\Segment.h"


void Segment::segment(const Mat& src, const int imageIndex,
	std::vector<std::vector<Point>>& allLineContours,		// return line contours
	std::vector<std::pair<Point, Point>>& allArrowEndpoints,	// return arrow tip and shaft end-points
	std::vector<std::vector<Point>>& quadrilateralsContours,
	const std::vector<Rect>& arrowROIs,
	const std::vector<Rect>& starROIS,
	const std::vector<RotatedRect>& textROIs) {


	// Binarize
	Mat bin = performThresholding(src, imageIndex);

	// ===========================================================
	// PREPROCESSING
	// ===========================================================   
	// Remove astericks
	if (!starROIS.empty()) {
		for (const auto& rect : starROIS)
		{
			Mat roiMat = bin(rect);		// extract the ROI as a separate matrix
			roiMat.setTo(0);		// set all pixel values within the ROI to 0 (black)

		}
		//Util::saveImg("Remove_stars", bin);
	}

	// Preprocess
	Mat close2 = preprocessing(bin);

	// Get arrows tip- and shaft-endpoint
	allArrowEndpoints = getAllArrowEndPoints(close2, arrowROIs);


	// ===========================================================
	//  ISOLATE QUADS
	//  Distance transform and watershed
	// =========================================================== 
	Mat cont = close2.clone();
	Mat filledShapes = Mat::zeros(cont.size(), CV_8UC1);
	Mat removeTextImg = Mat::zeros(cont.size(), CV_8UC1);
	fillQuads(cont, removeTextImg, filledShapes, arrowROIs);

	Mat cleanOrig = removeTextImg.clone();	    	// non-filled image without text
	Mat filledThinned = filledShapes;	        // filled shapes

	// Perform watershed
	Mat markers = getWatershedMarkers(filledShapes);
	performWatershed(markers, removeTextImg);

	// Create an image with the quadrilaterals isolated
	Mat isolatedQuads = Mat::zeros(src.size(), CV_8UC1);
	isolateQuadrilaterals(markers, isolatedQuads, quadrilateralsContours, arrowROIs);

	// ===========================================================
	//  ISOLATE lines and/or arrows		
	//  Get the endpoints
	// =========================================================== 
	Mat isolatedArrowsLines = isolateLinesArrows(isolatedQuads, removeTextImg);

	// Lines ----------
	allLineContours = getAllLines(isolatedArrowsLines, arrowROIs, textROIs);

	Mat isolatedArrowsLines_copy = Mat::zeros(isolatedArrowsLines.size(), CV_8UC3);


	// Line and arrow endpoints are returned by reference

}


// ######################################################
// LINE ENDPOINTS
// ######################################################
Mat Segment::preprocessLineImage(const std::vector<Rect>& arrowROIs,
	const std::vector<RotatedRect>& textROIs, Mat& linesImage)
{
	// Remove arrow(s) from image
	if (!arrowROIs.empty())
	{
		for (const auto& rect : arrowROIs)
		{
			Mat roiMat = linesImage(rect); 	// extract the ROI as a separate matrix
			roiMat.setTo(0); 		// set all pixel values within the ROI to 0 (black)

		}
	}

	// Remove text region(s) from image		
	if (!textROIs.empty())
	{
		Mat mask(linesImage.size(), linesImage.type(), Color::white);

		for (const auto& roi : textROIs)
		{
			Point2f vertices[4];
			roi.points(vertices);
			std::vector<Point> pointVec = { vertices[0], vertices[1], vertices[2], vertices[3], vertices[0] };
			fillConvexPoly(mask, pointVec, Color::black);

		}
		bitwise_and(linesImage, mask, linesImage);
	}

	// Denoise
	std::vector<Point> looseEndPoints{};
	int repeat = 3;
	do {
		Util::removeNoiseConnectedComp(linesImage, 80);				// remove small components
		Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(linesImage, linesImage, MORPH_CLOSE, kernel);		// close small gaps before filling

		std::vector<std::vector<Point>> contours;
		contours.clear();
		findContours(linesImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(linesImage, contours, -1, Color::white, FILLED);		// fill closed contours 
		Mat openingMask;
		morphologyEx(linesImage, openingMask, MORPH_OPEN, kernel);		// create a mask of filled contours 
		dilate(openingMask, openingMask, kernel);				// dilate
		bitwise_and(linesImage, ~openingMask, linesImage);			// remove small filled contours
		Util::removeNoiseConnectedComp(linesImage, 80);				// remove small components
		thinning(linesImage, linesImage);
		looseEndPoints.clear();
		looseEndPoints = Util::getLooseEndPoints(linesImage);
		linesImage = Util::closeLooseEndGaps(linesImage, looseEndPoints, 50);	// close gaps in lines
		thinning(linesImage, linesImage);
	} while (--repeat > 0);


	return linesImage;

}

/* Get all line contours and loose-endpoints */
std::vector<std::vector<Point>> Segment::getAllLines(
	Mat& isolatedArrowsLines,
	const std::vector<Rect>& arrowROIs,
	const std::vector<RotatedRect>& textROIs)
{
	// Process the image to remove residual noise
	isolatedArrowsLines = preprocessLineImage(arrowROIs, textROIs, isolatedArrowsLines);
	const Mat linesImage = isolatedArrowsLines.clone();

	std::vector<std::vector<Point>> allLineContours{};

	// Process individual line
	std::vector<std::vector<Point>> contours;
	findContours(linesImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < contours.size(); i++)
	{
		Mat individualLine = Mat::zeros(linesImage.size(), linesImage.type());
		drawContours(individualLine, contours, i, Color::white);

		// Store line's contours
		allLineContours.push_back(contours[i]);
	}

	return allLineContours;
}


// ######################################################
// ARROW endpoints
// ######################################################
/* Get all arrow endpoints in the form of tip-point and shaft end-point */
std::vector<std::pair<Point, Point>> Segment::getAllArrowEndPoints(const Mat& isolatedArrowsLines,
	const std::vector<Rect>& arrowROIs)
{
	std::vector<std::pair<Point, Point>> allArrowEndpoints{};

	if (!arrowROIs.empty()) {
		// Create an image containing only arrows (no lines)
		Mat arrowMask = Mat::zeros(isolatedArrowsLines.size(), CV_8UC1);
		for (const auto& arrow : arrowROIs) {
			rectangle(arrowMask, arrow, Color::white, FILLED);
		}
		Mat arrowImage;
		isolatedArrowsLines.copyTo(arrowImage, arrowMask); 	// exclude everything except the arrows

		// Process each arrow individually to get its tip and shaft-end points
		for (const auto& arrowRoi : arrowROIs)
		{
			std::vector<Point2f> corners{ 4 };
			corners[0] = Point(arrowRoi.x, arrowRoi.y);
			corners[1] = Point(arrowRoi.x + arrowRoi.width, arrowRoi.y);
			corners[2] = Point(arrowRoi.x + arrowRoi.width, arrowRoi.y + arrowRoi.height);
			corners[3] = Point(arrowRoi.x, arrowRoi.y + arrowRoi.height);

			Mat individualArrow = 
				Util::fillBlackOutsideOfContours(arrowImage, corners);

			// get the points representing the TIP and ENDPOINT of the arrow
			std::pair<Point, Point> arrowEndPoints = getArrowEndpoints(individualArrow);
			allArrowEndpoints.push_back(arrowEndPoints);
		}
	}
	return allArrowEndpoints;
}

/* Get the tip point and shaft end - points of the arrow.
* The arrow tip will always be the first of the pair.	*/
std::pair<Point, Point> Segment::getArrowEndpoints(Mat& individualArrow)
{
	// Close any gaps in the arrow
	const std::vector<Point> looseEnds = Util::getLooseEndPoints(individualArrow);
	individualArrow = Util::closeLooseEndGaps(individualArrow, looseEnds);

	// Fill the arrow head
	std::vector<std::vector<Point>> contours;
	findContours(individualArrow, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	drawContours(individualArrow, contours, -1, Color::white, FILLED);

	// Isolate the arrow head
	Mat isolatedArrowHead;
	const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(individualArrow, isolatedArrowHead, MORPH_OPEN, kernel);
	dilate(isolatedArrowHead, isolatedArrowHead, kernel, Point(-1, -1), 3);

	// Get all points of interest - HEAD
	std::vector<Point2f> keypointsHead;
	goodFeaturesToTrack(isolatedArrowHead, keypointsHead, 20, 0.15, 5);

	// Get all points of interest - SHAFT
	Mat isolatedShaft;
	bitwise_and(~isolatedArrowHead, individualArrow, isolatedShaft);
	
	// Close small gaps
	dilate(isolatedShaft, isolatedShaft, kernel);
	// remove small noise from shaft image
	Util::removeNoiseConnectedComp(isolatedShaft, 5);
	std::vector<Point2f> keypointsShaft;
	goodFeaturesToTrack(isolatedShaft, keypointsShaft, 20, 0.05, 5);


	// Get the two points in the arrow-head and -shaft that are furtherest away	from each other
	std::pair<Point, Point> endpoints;
	double maxDistance = -1;

	auto distanceFn = [](const Point& a, const Point& b) { return Util::distance(a, b); };

	for (auto& itHead : keypointsHead)
	{
		auto itShaft = std::max_element(keypointsShaft.begin(), keypointsShaft.end(),
			[=](const Point& a, const Point& b) { return distanceFn(itHead, a) < distanceFn(itHead, b); });

		const double distance = distanceFn(itHead, *itShaft);
		if (distance > maxDistance)
		{
			maxDistance = distance;
			endpoints = std::make_pair(itHead, *itShaft);
		}
	}

	return endpoints;
}



// ######################################################
// Preprocessing
// ######################################################
Mat Segment::performThresholding(const Mat& src, const int imageIndex)
{
	// Grayscale
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	
	// ===========================================================
	// Thresholding (with or without illumination correction)
	// ===========================================================
	// images that don't require illumination correction
	std::vector<int> ostuImages = {
		15,
		23,24,25,26,27,28,29,30,
		34,35,
		41,43,
		57,58,59
	};

	Mat bin;
	bool useOtsu = std::binary_search(ostuImages.begin(), ostuImages.end(), imageIndex);
	if (useOtsu)	// use Otsu thresholding
	{
		// Blur the image to reduce small noise
		Mat gaus_blur;
		GaussianBlur(gray, gaus_blur, Size(3, 3), 0);
		threshold(gaus_blur, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	}
	else		// use illumination correction
	{
		// Bilateral blur to make contours slightly sharper
		Mat bilateralBlur;
		constexpr int pixelMaxDistance = 5;
		constexpr double sigmaColor = 25;
		constexpr double sigmaSpace = 25;
		bilateralFilter(gray, bilateralBlur, pixelMaxDistance, sigmaColor, sigmaSpace);
		bin = removeLightWithPattern(bilateralBlur);
	}
	
	return bin;
}

Mat Segment::preprocessing(const Mat& bin)
{
	// Thin 1
	Mat skeletonImg;
	thinning(bin, skeletonImg);

	// Dilate the skeleton to join pixels
	int morphologySize = 1;
	int width = 2 * morphologySize + 1;
	int height = width;
	Mat kernel = getStructuringElement(MORPH_RECT, Size(width, height));
	Mat dilateSkel;
	dilate(skeletonImg, dilateSkel, kernel, Point(-1, -1), 1);

	// Thin 2 (remove small isolated pixels)
	Mat thinDilate;
	thinning(dilateSkel, thinDilate);

	// Morphological closing after thinning
	Mat closing;
	morphologyEx(thinDilate, closing, MORPH_CLOSE, kernel, Point(-1, -1), 1);

	// Close any larger gaps in contours
	Mat closedGaps = Util::closeLooseEndGaps(closing, Util::getLooseEndPoints(closing));
	closedGaps = Util::closeLooseEndGaps(closedGaps, Util::getLooseEndPoints(closedGaps));

	// Morph close
	Mat closing2;
	morphologyEx(closedGaps, closing2, MORPH_CLOSE, kernel, Point(-1, -1), 1);

	// Remove small noise
	Util::removeNoiseConnectedComp(closing2, 80);

	return closing2;
}

/* Thinning */
void Segment::thinningIteration(Mat& img, const int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	Mat marker = Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar* pAbove;
	uchar* pCurr;
	uchar* pBelow;
	uchar* nw, * no, * ne;      // north (pAbove)
	uchar* we, * me, * ea;
	uchar* sw, * so, * se;      // south (pBelow)

	uchar* pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}
void Segment::thinning(const Mat& src, Mat& dst)
{
	dst = src.clone();
	dst /= 255;             // convert to binary image

	Mat prev = Mat::zeros(dst.size(), CV_8UC1);
	Mat diff;

	do {
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (countNonZero(diff) > 0);

	dst *= 255;
}

/* Create two images: one with filled contours, one with thinned outlines */
void Segment::fillQuads(Mat& cont, Mat& removeTextImg, Mat& filledShapes, const std::vector<Rect>& arrowROIs)
{
	// Make a gap in the arrow(s) to reduce unwanted filled contours
	if (!arrowROIs.empty()) {
		for (const auto& rect : arrowROIs)
		{
			const Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
			const int len = std::min(rect.width, rect.height);
			line(cont, Point(rect.x, center.y), Point(rect.x + len, center.y), Color::black, 2);
		}
	}

	// Fill shapes with black 
	std::vector<std::vector<Point>> contoursDraw;
	findContours(cont, contoursDraw, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// NON-filled shapes (clean outlines)
	for (int i = 0; i < contoursDraw.size(); i++)
	{
		drawContours(removeTextImg, contoursDraw, i, Scalar(255, 255, 255), 1);	 // for Watershed input (thinned external)
	}

	// Close gaps in image without text
	const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	morphologyEx(removeTextImg, removeTextImg, MORPH_CLOSE, kernel, Point(-1, -1), 2);
	removeTextImg = Util::closeLooseEndGaps(removeTextImg, Util::getLooseEndPoints(removeTextImg));

	// Draw a border in order to fill cut-off quadrilaterals
	Util::drawBorder(cont, 1, Color::white);

	// FILL shapes (gaps have been closed)	
	contoursDraw.clear();
	Util::removeNoiseConnectedComp(cont, 80);		// remove noise before filling
	findContours(cont, contoursDraw, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < contoursDraw.size(); i++)
	{
		const Rect bb = boundingRect(contoursDraw[i]);
		const int contourSize = bb.width * bb.height;
		const int imgSize = cont.rows * cont.cols;

		if (abs(contourSize - imgSize) <= 3) {	// ignore the image boundary drawn
			continue;
		}

		// Fill the contours
		drawContours(filledShapes, contoursDraw, i, Scalar(255, 255, 255), -1);
	}

	// remove small noise and draw thin border
	Util::removeNoiseConnectedComp(removeTextImg, 80);
	Util::removeNoiseConnectedComp(filledShapes, 80);
	Util::drawBorder(removeTextImg, 2);
	Util::drawBorder(filledShapes, 2);
}

// ---------------------------------
// LIGHT PATTERN methods
// --------------------------------
Mat Segment::removeLightWithPattern(const Mat& gray)
{
	// light removal method
	const int DIFF = 0;
	const int DIVISION = 1;

	const Mat& img = gray;
	// Remove noise	
	Mat imgNoise;
	medianBlur(img, imgNoise, 3);

	// Calculate light pattern
	Mat lightPattern = calculateLightPattern(img);
	medianBlur(lightPattern, lightPattern, 3);

	//Apply the light pattern - remove light
	Mat imgNoLightDiv;
	imgNoise.copyTo(imgNoLightDiv);
	imgNoLightDiv = removeLight(imgNoise, lightPattern, DIVISION);

	// Binarize image for segment
	Mat binDiv;
	threshold(imgNoLightDiv, binDiv, 30, 255, THRESH_BINARY);

	return binDiv;
}
Mat Segment::calculateLightPattern(const Mat& img)
{
	Mat pattern;
	// Basic way to calculate the light pattern 
	blur(img, pattern, Size(img.cols / 3, img.cols / 3));
	return pattern;
}
Mat Segment::removeLight(const Mat& img, const Mat& pattern, const int method)
{
	Mat output;

	if (method == 1)
	{
		// Require change our image to 32 float for division
		Mat img32, pattern32;
		img.convertTo(img32, CV_32F);
		pattern.convertTo(pattern32, CV_32F);
		// Divide the imabe by the pattern
		output = 1 - (img32 / pattern32);
		// Convert 8 bits format
		output.convertTo(output, CV_8U, 255);
	}
	else {
		output = pattern - img;
	}
	//equalizeHist( output, output );
	return output;
}


// ######################################################
// Segmentation
// ######################################################
/* Use distance transform to get markers for watershed */
Mat Segment::getWatershedMarkers(const Mat& filledThinned)
{
	// Get the distance transform
	Mat distTrans = Mat::zeros(filledThinned.size(), CV_8UC1);
	// Calculates the distance to the closest zero pixel for each pixel of the source image
	distanceTransform(~filledThinned, distTrans, DIST_L2, DIST_MASK_PRECISE);

	// Convert the image to a single-channel floating-point format
	distTrans.convertTo(distTrans, CV_32FC1);
	double maxVal;
	//Finds the minimum and maximum values
	minMaxLoc(distTrans, nullptr, &maxVal);

	// Dilate the distance transform	
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat threshDist = filledThinned.clone();
	dilate(threshDist, threshDist, kernel, Point(-1, -1), 2);

	// Get sure background by dilating
	Mat sureBg;
	dilate(threshDist, sureBg, kernel, Point(-1, -1), 2);

	// Get sure foreground: 1. get distance transform
	Mat distTransForFg;
	distanceTransform(threshDist, distTransForFg, DIST_L2, DIST_MASK_PRECISE);

	// Get sure foreground: 2. threshold distance transform
	Mat sureFg;
	maxVal = 0.0;
	minMaxLoc(distTransForFg, nullptr, &maxVal);
	threshold(distTransForFg, sureFg, 0.2 * maxVal, 255, THRESH_BINARY);

	// Finding unknown region (by getting the difference between bg and fg)
	Mat sureFg8U;
	sureFg.convertTo(sureFg8U, CV_8U);
	Mat unknown;
	subtract(sureBg, sureFg8U, unknown);

	sureFg = sureFg8U.clone();

	// --------- Watershed ------------------------------
	// This is more precise than morph opening in terms of isolating arrows (no/less cut-off)
	// Marker labelling
	Mat markers;
	int numLabels = connectedComponents(sureFg, markers);
	// Add one to all labels so that sure_background is 1 (not 0)
	markers = markers + 1;
	// Mark the region of unknown with zero
	markers.setTo(0, unknown == 255);

	return markers;
}

void Segment::performWatershed(Mat& markers, const Mat& watershedInput)
{
	// Perform watershed		
	Mat watershedImg = Util::convertGrayToColor(watershedInput.clone());
	Util::drawBorder(watershedImg, 1, Color::white);
	
	watershed(watershedImg, markers);
	// Boundaries marked with -1
	watershedImg.setTo(Scalar(0, 0, 255), markers == -1);
}

void Segment::isolateQuadrilaterals(const Mat& markers, Mat& isolatedQuads,
	std::vector<std::vector<Point>>& contours, const std::vector<Rect>& arrowRoIs)
{
	// Isolate quads (create a mask with only filled quads)

	// Find contours from the watershed markers		
	std::vector<std::vector<Point>> contoursMarkers;
	findContours(markers, contoursMarkers, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	// Find the index of the contour with the largest area
	const auto maxAreaContour = std::max_element(
		contoursMarkers.begin(), contoursMarkers.end(),
		[](const auto& c1, const auto& c2)
		{ return contourArea(c1) < contourArea(c2);  });

	const auto maxAreaIndex = maxAreaContour - contoursMarkers.begin();

	// Draw the contours of the quads
	for (int i = 0; i < contoursMarkers.size(); i++) {
		if (i == maxAreaIndex) {	// WD includes img boundary -ignore 
			continue;
		}

		drawContours(isolatedQuads, contoursMarkers, i, Color::black, 2, LINE_AA);
		// Isolate filled shapes
		fillPoly(isolatedQuads, contoursMarkers[i], Scalar(255, 255, 255), LINE_AA, 0);
	}

	// Remove possible filled arrow-heads from quads
	removeArrows(isolatedQuads, arrowRoIs);
	// Save contours
	findContours(isolatedQuads, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	const Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(isolatedQuads, isolatedQuads, kernel);
}

Mat Segment::isolateLinesArrows(const Mat& isolatedQuads, const Mat& removeTextImg)
{
	Mat mask;
	bitwise_not(isolatedQuads, mask);

	Mat isolated_arrows_lines;
	bitwise_and(mask, removeTextImg, isolated_arrows_lines);

	// Denoise
	Util::removeNoiseConnectedComp(isolated_arrows_lines, 50);
	Util::drawBorder(isolated_arrows_lines, 180);
	
	return isolated_arrows_lines;
}

/* Remove arrows from an image */
void Segment::removeArrows(const Mat& image, const std::vector<Rect>& arrowROIs)
{
	for (const auto& rect : arrowROIs)
	{
		Mat roiMat = image(rect); 	// extract the ROI 
		roiMat.setTo(0); 			// set all pixel values within the ROI to 0 (black)
	}

}





