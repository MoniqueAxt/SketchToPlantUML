# Sketch to PlantUML

Transformation of a sketchy UML model into a formal PlantUML model.

## Overview

This C++ project was created as part of a BSc Computer Engineering thesis. The goal of the tool is to evaluate the possibility of identifying, classifying and transforming sketched UML Class Diagrams into formal PlantUML models, using image processing techniques in the OpenCV library.

## Install

The project is built using CMake GUI and OpenCV. OpenCV *source* files can be downloaded from **Sources** from their [website](https://opencv.org/releases/). In CMake GUI, specify the OpenCV source file location and the desired location of the final build. After selecting specific flags to install, configuring and generating, open the project and run `ALL_BUILD` for debug and/or release mode(s) followed by `INSTALL` to build the OpenCV library. Note: the first build/install takes a long time.

## How to use
The program can be run as-is from the main entry point located in `SketchToPlantUML.cpp`.

Images are stored in the `resouces` directory: input images are in the sub-directory `/dataset` and template images for template matching are stored in sub-directories `/arrow_templates` and `/star_templates`. These file paths can be modified in `SketchToPlantUML.cpp`. Note that if any of the directory or file names are changed, the `CMakeLists.txt` file should be modified accordingly.

Modifications can be made to different parts of the program. The program consists of four steps: text detection, segmentation, classification and transformation; the directory structure is separated based on these steps, as well as a Utility directory, all located in the `src` and `ìnclude` directories. The entry point file calls each of these steps in order.


### Text detection

[TextDetection class](TextDetection.h)

While text detection is implemented, the program treats text as noise and detection is used only for noise removal. Detection is performed using the `EAST` implementation of OpenCV's `TextDetection` model, with the pre-trained  `frozen_east_text_detection.pb` data included in the repository. Note that the models expects image dimensions to be in multiples of 32. To change the model data used for detection, modify the `modelPath` variable in the `TextDetection::detectTextWithEAST()` method. The following parameters for the model can also be modified here: confidence and Non-Maximum Suppression thresholds, input image scaling factor and input image size.

### Segmentation

[Segmentation class](Segment.h)

All preprocessing and segmentation is done here. The `segment()` method is separated as follows: thresholding -> preprocessing -> arrow identification -> isolation of quadrilaterals (UML classes) ->
isolation of relationship lines. As these steps are separated into different methods, changes can be made to individual methods. However, each step relies on the processed image of the previous step. The output of the segmentation is the contours of the quadrilaterals (classes), arrow tip- and shaft-endpoints, and contours of the lines representing association relationships.

### Classification

[Classification class](Classify.h)

The classification process takes the output of the segmentation step and creates tuples representing UML relationships: two classes and the relationship. The two classified relationships are association and inheritance. Custom comparators are used to compare the tuples. The classification is based on the location of elements to other elements, and the thresholds for accepted range can be modified. Note that `typedef`s are used for the sets containing the tuples.

### Transformation

[Transformation class](Transform.h)

The two sets (association-relationships and inheritance-relationships) are used to create the PlantUML syntax output, which is printed to the console. Short hashes are created for each of the classes to distinguish different classes. This can be removed if text recognition is implemented and used instead. The PlantUML syntax for the different relationships is as follows:

```java
// Association
@startuml
Class_1 - Class_2   
@enduml

// Inheritance 
@startuml
Class_3 <|-- Class_4   
@enduml
```

### Utility

[Utility class](Util.h)

This class contains a collection of methods that are generalised helper methods for file management, drawing functions for visualisation, and calculations on `cv::Point`s. The method for template matching is also contained here solely to simplify the main entry-point file.
