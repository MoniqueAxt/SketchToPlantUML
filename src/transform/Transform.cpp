#include "transform\Transform.h"

void Transform::transform(const Mat& src, const int i, const association_set& associationRelationships,
	const inheritance_set& baseInheritedRelationships,
	const std::vector<std::vector<Point>>& quadrilateralsContours)
{
	//Mat relationshipImg = src.clone();		// draw the relationships
	std::stringstream ss;
	ss << "@startuml\n";

	// Single class(es) only
	if (baseInheritedRelationships.empty() && associationRelationships.empty())
	{
		for (const auto& quad : quadrilateralsContours)
		{
			std::string rectHash = generateShortHash(quad);
			ss << "class Class_" << rectHash << "\n";

			//Util::drawRect(relationshipImg, quad, rectHash, Color::darkblue);
		}
	}

	// Association relationships
	for (const auto& trio : associationRelationships)
	{
		std::string associationSymbol = " - ";
		auto& [rect1, connectingLine, rect2] = trio;			// get the elements in this relationship

		std::string rect1Hash = generateShortHash(rect1);
		std::string rect2Hash = generateShortHash(rect2);
		// Pair classes that are associated
		ss << "Class_" << rect1Hash
			<< associationSymbol
			<< "Class_" << rect2Hash
			<< "\n";

		/*Util::drawRelationship(
			rect1, { "Class_" + rect1Hash + "_a_ :" + std::to_string(rect1.size()) },
			connectingLine,
			rect2, { "Class_" + rect2Hash + "_a :" + std::to_string(rect2.size()) },
			relationshipImg, Color::red, Color::purple, false);*/
	}

	// Base/inherited relationships
	for (const auto& trio : baseInheritedRelationships)
	{
		std::string relationsSymbol = " <|-- ";
		auto& [parentClass, connectingArrow, children] = trio;		// get the elements in this relationship from the tuple

		std::string parentHash = generateShortHash(parentClass);
		// Pair each child with the parent
		for (const auto& child : children)
		{
			std::string childHash = generateShortHash(child);

			ss << "Class_" << parentHash
				<< relationsSymbol
				<< "Class_" << childHash
				<< "\n";

			/*Util::drawRelationship(
				parentClass, { "Class_" + parentHash + "_p :" + std::to_string(parentClass.size()) },
				connectingArrow,
				child, { "Class_" + childHash + "_c :" + std::to_string(child.size()) },
				relationshipImg, Color::green, Color::yellow, true);*/
		}

	}

	ss << "@enduml\n";
	std::cout << ss.str();

	//imwrite(std::to_string(i) + ". Relationships.png", relationshipImg);

	std::cout << std::endl;
}

std::uint32_t Transform::fnvHash(const std::vector<cv::Point>& input) {
	constexpr std::uint32_t fnvOffsetBasis = 2166136261u;
	constexpr std::uint32_t fnvPrime = 16777619u;
	std::uint32_t hash = fnvOffsetBasis;

	for (const auto& p : input) {
		const std::uint32_t val = static_cast<std::uint32_t>(p.x) << 16 | p.y;
		hash ^= val;
		hash *= fnvPrime;
	}

	return hash;
}
// Convert a 32-bit integer to a 5-character hex string
std::string Transform::toShortHashString(std::uint32_t value) {
	static constexpr char hex_digits[] = "0123456789abcdef";
	char buffer[6] = { 0 };
	for (int i = 4; i >= 0; i--) {
		buffer[i] = hex_digits[value & 0xF];
		value >>= 4;
	}
	return buffer;
}
// Generate a short hash string from a vector of points
std::string Transform::generateShortHash(const std::vector<cv::Point>& input) {
	const std::uint32_t hash = fnvHash(input);
	return toShortHashString(hash);
}
