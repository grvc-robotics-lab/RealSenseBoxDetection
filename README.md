# RealSenseBoxDetection
Box identification and detection of it's orientation and vertex with Intel RealSense camera using Camshift, Opencv, Eigen and librealsense2.


# Real-time Object Detection and Measurement using OpenCV and RealSense

This repository contains a program for real-time object detection and measurement using OpenCV, Intel RealSense, and Eigen. The program tracks objects, detects their edges, and calculates the dimensions of detected orthoedros (rectangular prisms) in 3D space.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Program](#running-the-program)
  - [Mouse Interaction](#mouse-interaction)
  - [Adjusting Parameters](#adjusting-parameters)
- [Structure](#structure)
- [Functions](#functions)
  - [Main Functions](#main-functions)
  - [Utility Functions](#utility-functions)
  - [3D Calculation and Drawing Functions](#3d-calculation-and-drawing-functions)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- [OpenCV](https://opencv.org/)
- [librealsense](https://github.com/IntelRealSense/librealsense)
- [Eigen](https://eigen.tuxfamily.org/dox/GettingStarted.html)

Ensure you have the required dependencies installed. You can use package managers like `apt` for Linux or `vcpkg` for Windows.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/real-time-object-detection.git
    cd real-time-object-detection
    ```

2. Build the project:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

## Usage
### Running the Program

To run the program, execute the following command from the `build` directory:
```bash
./object_detection
```
### Mouse Interaction

- **Left Mouse Button Down:** Start selecting the region of interest (ROI).
- **Left Mouse Button Up:** Finish selecting the ROI. The program will begin tracking the selected object.


### Adjusting Parameters

You can adjust the following parameters using the trackbars in the GUI:

- `Vmin` and `Vmax`: Minimum and maximum values for the V channel in HSV color space.
- `Smin` and `Smax`: Minimum and maximum values for the S channel in HSV color space.
- `Huemin` and `Huemax`: Minimum and maximum values for the H channel in HSV color space.
- `minContour`: Minimum contour area to consider for object detection.


### Section 5: Structure

## Structure
```markdown

- **Main Program:** Contains the main function which initializes the camera, sets up the GUI, and processes the video stream.
- **Image Processing Functions:** Functions for converting images to HSV, creating masks, and tracking objects.
- **3D Calculation Functions:** Functions for deprojecting pixel coordinates to 3D space and calculating the dimensions and orientation of detected orthoedros.
```

## Functions

### Main Functions


    int main(): The entry point of the program.
    void onMouse(int event, int x, int y, int, void*): Handles mouse events for selecting the ROI.
    void processImage(Mat& color, Mat& hsv, Mat& hue, Mat& mask, Rect& trackWindow, int& trackObject, Mat& hist, Mat& backproj, Mat& edges, int hsize, const float* phranges): Processes the image to detect and track objects.

### Utility Functions


    bool isValidPoint(const Point& pt, const rs2::depth_frame& depth_frame): Checks if a point is valid within the depth frame.
    Rect adjustROI(Rect roi, Size imgSize): Adjusts the ROI to ensure it is within the image boundaries.
    Eigen::Vector3d deproject_pixel_to_point(const rs2_intrinsics& intrinsics, const rs2::depth_frame& depth_frame, int x, int y): Converts pixel coordinates to 3D coordinates.
    vector<Point> getBestPolygon(const Mat& edges): Finds the best polygon (approximating a rectangle) from edge contours.

### 3D Calculation and Drawing Functions


    vector<Eigen::Vector3d> markVertexDistances(Mat& image, const vector<Point>& bestPolygon, const rs2::depth_frame& depth_frame, const rs2_intrinsics& intr, const rs2::video_stream_profile& color_stream, const rs2::video_stream_profile& depth_stream): Marks distances to vertices and converts them to 3D points.
    void vertexOrthoedro(orthoedro& box): Calculates the vertices of the orthoedro.
    void calcBoxOrientation(orthoedro& box): Calculates the orientation of the orthoedro.
    void drawVertices(cv::Mat& image, const orthoedro& box, const rs2_intrinsics& intr): Draws the vertices of the orthoedro on the image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
