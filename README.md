# Parcel Detection and Localization with CAMShift using RealSense D435

This software module enables the detection and localization of a parcel, such as those used by DHL Express, suitable for robotic manipulator grasping in delivery applications. The CAMShift algorithm (Continuously Adaptive Mean-Shift) is employed for object tracking based on HSV (Hue-Saturation-Value) information and depth data from a RealSense D435 camera. The program computes the 3D position of the parcel's corners relative to the camera axes (Z: optical axis, Y: vertical axis downwards, X: horizontal axis right). Two modes are supported: manual bounding box selection for color distribution analysis using a graphical interface and automatic parcel detection using parameters from a configuration file.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Program](#running-the-program)
  - [Adjusting Parameters](#adjusting-parameters)
- [Functions Overview](#functions)
  - [Core Functions](#core-functions)
    - [processImage](#processimage)
    - [getBestPolygon](#getbestpolygon)
    - [markVertexDistances](#markvertexdistances)
    - [vertexOrthoedro](#vertexorthoedro)
    - [calcBoxOrientation](#calcboxorientation)
    - [drawVertices](#drawvertices)
  - [Utility Functions](#utility-functions)
    - [isValidPoint](#isvalidpoint)
    - [loadConfigFromXML and saveConfigToXML](#loadconfigfromxml-and-saveconfigtoxml)
    - [adjustROI](#adjustroi)
    - [calculateLongestVector](#calculatelongestvector)
    - [separateVertices](#separatevertices)
- [Main Function Structure](#main-function-structure)

## Prerequisites
- [OpenCV](https://opencv.org/)
- [librealsense](https://github.com/IntelRealSense/librealsense)
- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html)
- [tinyxml2](https://github.com/leethomason/tinyxml2)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/JorgePogue37/RealSenseBoxDetection
    cd RealSenseBoxDetection
    ```

2. Build the project:
    ```bash
    mkdir build
    cd build
    cmake ..
    make

## Usage
### Running the Program

To run the program, execute the following command from the `build` directory:
```bash
./object_detection
```

### Adjusting Parameters

You can adjust the following parameters using the trackbars in the GUI:

- `Vmin` and `Vmax`: Minimum and maximum values for the V channel in HSV color space.
- `Smin` and `Smax`: Minimum and maximum values for the S channel in HSV color space.
- `Huemin` and `Huemax`: Minimum and maximum values for the H channel in HSV color space.
- `minContour`: Minimum contour area to consider for object detection.

## Functions Overview

### Core Functions

#### processImage

- **Purpose:** Processes the input image to track the parcel using CAMShift algorithm.
- **Parameters:**
  - `color`: BGR image for processing.
  - `hsv`: Output HSV image.
  - `hue`, `mask`, `backproj`, `edges`: Output matrices for intermediate image processing steps.
  - `trackWindow`: Bounding box for tracking.
  - `hist`: Histogram of hue distribution.
  - `hsize`, `phranges`: Parameters for histogram calculations.
  - `mode`: Operating mode ('manual' or 'automatic').
  - `camshiftbox`: Mask image used in CAMShift processing.
- **Description:** Converts the input image to HSV format, applies color segmentation, tracks the parcel using CAMShift, and computes its spatial orientation.

#### getBestPolygon

- **Purpose:** Extracts the best polygon approximation from edge-detected contours.
- **Parameters:**
  - `edges`: Binary edge-detected image.
- **Description:** Finds and returns the polygon with the smallest area from the detected contours.

#### markVertexDistances

- **Purpose:** Computes 3D spatial points corresponding to the vertices of the parcel.
- **Parameters:**
  - `image`: BGR image for displaying depth information.
  - `bestPolygon`: Best polygon approximating the parcel.
  - `depth_frame`, `intr`, `color_stream`, `depth_stream`: RealSense camera parameters.
- **Description:** Uses depth information to compute and mark spatial coordinates of parcel vertices on the image.

#### vertexOrthoedro

- **Purpose:** Computes vertices and dimensions of an orthoedron.
- **Parameters:**
  - `box`: Structure containing measured points, rotation matrix, and dimensions.
- **Description:** Calculates vertices, dimensions, and center of the parcel in 3D space.

#### calcBoxOrientation

- **Purpose:** Calculates the orientation of the parcel.
- **Parameters:**
  - `box`: Structure containing measured points and rotation matrix.
- **Description:** Determines the rotational matrix for aligning the parcel with the camera's coordinate system.

#### drawVertices

- **Purpose:** Draws projected vertices and reference points on the image.
- **Parameters:**
  - `image`: Output image with projected vertices.
  - `box`: Structure containing vertices and rotation matrix.
  - `intr`: RealSense camera intrinsics.
- **Description:** Projects and visualizes parcel vertices and reference points on the image.

### Utility Functions

#### isValidPoint

- **Purpose:** Checks if a given point is valid within the frame dimensions.
- **Parameters:**
  - `pt`: Point to be validated.
  - `depth_frame`: Depth frame from RealSense camera.
- **Description:** Verifies if the point is within the valid frame dimensions.

#### loadConfigFromXML and saveConfigToXML

- **Purpose:** Load and save configuration parameters from/to an XML file.
- **Parameters:**
  - `filename`: Path to the XML configuration file.
- **Description:** Loads and saves HSV and contour parameters from/to an XML configuration file.

#### adjustROI

- **Purpose:** Adjusts the region of interest (ROI) to ensure it remains within the image bounds.
- **Parameters:**
  - `roi`: Current region of interest.
  - `imgSize`: Size of the input image.
- **Description:** Ensures the ROI stays within valid image dimensions.

#### calculateLongestVector

- **Purpose:** Finds the longest vector sum among sequential triplets in a given set of 3D vectors.
- **Parameters:**
  - `vertices`: A vector containing 3D points (`Eigen::Vector3d`) representing vertices.
- **Returns:** The longest vector sum found among the sequential triplets.
- **Description:** This function iterates through the list of vertices, computes the sum of vectors for each sequential triplet, and returns the vector sum with the longest magnitude. It is used in `calcBoxOrientation` to determine the primary vector direction for orientation calculation.

#### separateVertices

- **Purpose:** Separates vertices into two groups based on their relative position to an average vector.
- **Parameters:**
  - `vertices`: A vector containing 3D points (`Eigen::Vector3d`) representing vertices.
  - `avgVector`: Average vector direction used for separation.
  - `sideA`, `sideB`: Output vectors to store vertices separated into two groups.
  - `centroid`: Centroid point around which separation is performed.
- **Description:** This function categorizes vertices into two groups (`sideA` and `sideB`) based on their dot product with `avgVector`. Vertices with a positive dot product are added to `sideA`, while those with a non-positive dot product are added to `sideB`. It assists in organizing vertices for further processing in `calcBoxOrientation`.

  

## Main Function Structure

```cpp
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include "tinyxml2.h"

using namespace cv;
using namespace std;
using namespace Eigen;

// Global variables and constants

// Function prototypes

int main(int argc, char** argv) {
    // Initialize RealSense camera and other necessary components

    // Main loop for capturing frames
    
    // Image processing and parcel detection using CAMshift (processImage)

    // Get the best adjusted polygon to the contour of the parcel (getBestPolygon)

    // Calculate the distances to the vertex of this polygon which are the corners of the box (markVertexDistances)

    // Calculate parcel Orientation based on the vectors to the points calculated on the previous function (calcBoxOrientation)

    // Calculate the vectors to the corners of the parcel using the information obtained from the previous functions. It also calculate the size of the parcel, grasping points on the center of the sides of the parcel and it's center (vertexOrthoedro)
    
    // Display results and user interface for manual selection
    
    // Clean up and release resources
    
    return 0;
}
```
