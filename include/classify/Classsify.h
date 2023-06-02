#pragma once
#include <optional>

//OpenCV includes
#include <opencv2\opencv.hpp>

// Includes
#include "util\Util.h"


using namespace cv;

typedef std::vector<Point> quad_class;
typedef std::vector<std::vector<Point>> children_classes;
typedef std::pair<Point, Point> arrow_pair;
typedef std::pair<Point, Point> connecting_line;


/* Compare two association tuples */
struct AssociationTupleComparator {
	bool operator()(const std::tuple<quad_class, connecting_line, quad_class>& tuple1,
		const std::tuple<quad_class, connecting_line, quad_class>& tuple2) const
	{
		// Compare the elements of tuple1 and tuple2
		const auto& [quad1, line1, quad2] = tuple1;
		const auto& [quad3, line2, quad4] = tuple2;

		// If the lines the same, do not insert into the set
		if ((line1.first == line2.first && line1.second == line2.second) ||
			(line1.first == line2.second && line1.second == line2.first))		
		{
			return false;
		}		

		// The same quads can be paired multiple times, but with different lines
		if ((quad1 == quad3 && quad2 == quad4) || (quad1 == quad4 && quad2 == quad3))
		{
			// A duplicate, do not insert
			return false;
		}

		// If the quads are different, sort by their sizes
		if (quad1 != quad3) {
			if (quad1.size() != quad3.size()) {
				return quad1.size() < quad3.size();
			}
			// If the quads have the same size, sort by their x-coords
			else {
				for (size_t i = 0; i < quad1.size(); i++)
				{
					if (quad1[i].x != quad2[i].x)
					{
						return quad1[i].x < quad2[i].x;
					}
				}
			}
		}

		if (quad2 != quad4) {
			if (quad2.size() != quad4.size()) {
				return quad2.size() < quad4.size();
			}
			else {
				for (size_t i = 0; i < quad2.size(); i++)
				{
					if (quad2[i].x != quad4[i].x)
					{
						return quad2[i].x < quad4[i].x;
					}
				}
			}
		}


		return false;
	}

};

/* Compare two baseInherited tuples */
struct BaseInheritedTupleComparator {
	bool operator()(const std::tuple <quad_class, arrow_pair, children_classes>& tuple1,
		const std::tuple < quad_class, arrow_pair, children_classes>& tuple2) const
	{

		// Compare the elements of tuple1 and tuple2
		const auto& [quadClass1, arrow1, children1] = tuple1;
		const auto& [quadClass3, arrow2, children2] = tuple2;

		// If the arrows are different, sort the tuple by the arrow tip's y-coord
		if ( arrow1.first.y != arrow2.first.y)
		{
			return arrow1.first.y < arrow2.first.y;
		}

		return false;

	}
};

// alias for the association relationships 
typedef std::set<std::tuple<quad_class, connecting_line, quad_class>, AssociationTupleComparator> association_set;
// alias for the inheritance relationships 
typedef std::set<std::tuple <quad_class, arrow_pair, children_classes>, BaseInheritedTupleComparator> inheritance_set;


/* Classify the relationships */
class Classify
{
public:
	static std::optional<std::tuple<quad_class, connecting_line, quad_class>> createAssociationRelationship
	(const std::vector<Point>& lineLooseEnds, const std::vector<std::vector<Point>>& quads, int threshold = 200);

	/* Identify and classify each relationship */
	static void classify(
		const Mat& src,
		std::vector<std::vector<Point>>& allLinesConts,
		const std::vector<std::vector<Point>>& quads,
		std::vector<std::pair<Point, Point>>& arrows,
		association_set& associationRelationships,
		inheritance_set& baseInheritedRelationships);



private:
	/* Get the class that is closest to the given point */
	static std::optional<int> getClosestQuad(const std::vector<std::vector<Point>>& contours, const Point& endPoint,
	                                         int thresholdMinDistance = 50);


};


