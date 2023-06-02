#include "classify\Classsify.h"


void Classify::classify(
	const Mat& src,
	std::vector<std::vector<Point>>& allLinesConts,
	const std::vector<std::vector<Point>>& quads,
	std::vector<std::pair<Point, Point>>& arrows,
	association_set& associationRelationships,
	inheritance_set& baseInheritedRelationships) {
	

	// Uncomment to reduce the contour points of the the quadrilaterals using approxPolyP
	// Faster comparison but potentially less accurate
	/*
		std::vector<std::vector<Point>> approxQuads(quads.size());
		for (int index = 0; index < quads.size(); index++) {
			const double peri = arcLength(quads[index], true);
			approxPolyDP(quads[index], approxQuads[index], 0.25 * peri, true);
		}*/


	// Sort the lines by size
	std::sort(allLinesConts.begin(), allLinesConts.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
		return a.size() > b.size();
		});


	// Classify parent-multiple-children relationships first
	if (!arrows.empty())
	{		
		for (const auto& arrow: arrows)
		{
			auto shaftPoint = arrow.second;
			bool arrowPairedWithLine = false;

			auto itLine = allLinesConts.begin();	// line iterator
			while (itLine != allLinesConts.end())
			{
				arrowPairedWithLine = false;
				auto& singleLineConts = *itLine;

				double minDist = std::numeric_limits<double>::max();
				// Check whether each point in the line is close to the arrow's shaft endpoint
				for (auto& linePoint : singleLineConts)
				{
					double distance = norm(shaftPoint - linePoint); // Calculate Euclidean distance
					if (distance < minDist) minDist = distance;

					if (distance <= 50)				// If the line is close to the shaft endpoint
					{
						arrowPairedWithLine = true;		// The classes connected to this arrow/line are children
						break;
					}
				}

				// Indentify and classify the inheritance relationship
				if (arrowPairedWithLine)
				{
					// The classes near this line's loose-endpoints are children
					auto lineLooseEnds = Util::getLooseEndPoints(src.size(), singleLineConts);

					// Get the parent
					std::vector<Point> parent{};
					if (auto parentIdx = getClosestQuad(quads, arrow.first, 100); parentIdx != std::nullopt)
						parent = quads[parentIdx.value()];

					// Get the child(ren)
					std::vector<std::vector<Point>> children;
					for (auto& pt : lineLooseEnds)
					{
						double distance = norm(shaftPoint - pt);	// Calculate Euclidean distance
						if (distance <= 15)				// If the line loose end is the shaft point...
						{
							continue;				// ...ignore it
						}

						// Point must be relatively near a class to be paired
						if (auto idxChild = getClosestQuad(quads, pt, 50); idxChild != std::nullopt)	
						{
							// Classes cannot be the parent and child in the same relationship
							if (auto& child = quads[idxChild.value()]; child != parent)				
								children.push_back(child);
						}
					}
					// Create the inheritance relationship 
					if (!parent.empty() && !children.empty()) {
						auto tuple = std::make_tuple(parent, arrow, children);	
						baseInheritedRelationships.insert(tuple);
					}

					// Remove this line once classified
					itLine = allLinesConts.erase(itLine);
				}
				else
				{					
					++itLine;					
				}
			}

			// If this arrow was not paired with a line, create an inheritance relationship
			if (!arrowPairedWithLine)
			{				
				std::vector<Point> parent{};
				if (auto parentIdx = getClosestQuad(quads, arrow.first); parentIdx != std::nullopt)
					parent = quads[parentIdx.value()];	// get the parent

				std::vector<Point> child{};
				if (auto childIdx = getClosestQuad(quads, arrow.second); childIdx != std::nullopt)
					child = quads[childIdx.value()];	// get the child

				std::vector<std::vector<Point>> children{ child };
				auto tuple = std::make_tuple(parent, arrow, children);
				baseInheritedRelationships.insert(tuple);
			}
		}
	}
	
	// Only association lines to classify
	if (!allLinesConts.empty())
	{
		for (auto& singleLineConts : allLinesConts)
		{
			auto lineLooseEnds = Util::getLooseEndPoints(src.size(), singleLineConts);
			if (auto tuple = createAssociationRelationship(lineLooseEnds, quads, 200); tuple.has_value())
			{
				associationRelationships.insert(tuple.value());
			}
		}
	}
}


/* Create an association relationship */
std::optional<std::tuple<quad_class, connecting_line, quad_class>> Classify::createAssociationRelationship
(const std::vector<Point>& lineLooseEnds, const std::vector<std::vector<Point>>& quads, const int threshold)
{
	// Create ASSOCIATION relationship
	std::pair<Point, Point> line = Util::getFarthestPoints(lineLooseEnds);	// Create a line from the 2 furtherest loose ends

	// If line is within range of two classes, pair it
	const auto idx1 = getClosestQuad(quads, line.first, threshold);
	const auto idx2 = getClosestQuad(quads, line.second, threshold);

	if (idx1 == std::nullopt || idx2 == std::nullopt)	// This line is likely noise so ..
		return std::nullopt;				// .. ignore this line

	auto& class1 = quads[idx1.value()];			// Get the closest class to line.first
	auto& class2 = quads[idx2.value()];			// Get the closest class to line.second

	return std::make_tuple(class1, line, class2);		// Create association relationship
}

/* Find the class closest to the given point */
std::optional<int> Classify::getClosestQuad(const std::vector<std::vector<Point>>& contours, const Point& endPoint,
	const int thresholdMinDistance) {
	std::vector<Point> closestRect;
	double minDistance = std::numeric_limits<double>::max();

	size_t quadIndex = -1;

	for (size_t i = 0; i < contours.size(); i++)
	{
		const double distance = pointPolygonTest(contours[i], endPoint, true);

		if (std::abs(distance) < minDistance) {
			quadIndex = i;
			minDistance = std::abs(distance);
		}
	}
	// Check that min distance is above the threshold and the index is within bounds
	if (minDistance > thresholdMinDistance || quadIndex > contours.size())
		return std::nullopt;

	return quadIndex;

}
