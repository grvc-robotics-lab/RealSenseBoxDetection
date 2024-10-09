#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include "tinyxml2.h"
#include <open3d/Open3D.h>
#include <sys/time.h>
#include <deque>
#include <fstream>

using namespace cv;
using namespace std;
using namespace Eigen;

#define MAX_POINTS 100000

Mat image;
Rect selection;
bool selectObject = false;
int trackObject = 0;
Point origin;

string xmlFileName = "../Config.xml";

int vmin = 50, vmax = 256, smin = 150, smax = 256, huemin = 20, huemax = 30, minContour = 5000;

struct orthoedro{

    vector<Vector3d> vertex;
    Vector3d center;
    Vector3d refPointRightside;
    Vector3d refPointLeftside;
    Matrix3d rotation;
    double width;
    double length;
    double height;
    std::shared_ptr<open3d::geometry::PointCloud> filteredCloud;
    std::shared_ptr<open3d::geometry::PointCloud> measuredPoints;
    std::shared_ptr<open3d::geometry::PointCloud> remaining_cloud;
    std::shared_ptr<open3d::geometry::PointCloud> mainPlane; 
    open3d::geometry::OrientedBoundingBox obb;

    // Constructor to initialize the shared pointer
    orthoedro() 
        : filteredCloud(std::make_shared<open3d::geometry::PointCloud>()),
          measuredPoints(std::make_shared<open3d::geometry::PointCloud>()),
          remaining_cloud(std::make_shared<open3d::geometry::PointCloud>()),
          mainPlane(std::make_shared<open3d::geometry::PointCloud>()) {}
};

// Function to get the current time in milliseconds
long long getCurrentTimeInMilliseconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec) * 1000LL+ (tv.tv_usec / 1000);
}


// Verifies if the point is within the valid frame dimensions.
bool isValidPoint(const Point& pt, const rs2::depth_frame& depth_frame) {
    return pt.x >= 0 && pt.x < depth_frame.get_width() && pt.y >= 0 && pt.y < depth_frame.get_height();
}

// Ensures the ROI stays within valid image dimensions.
Rect adjustROI(Rect roi, Size imgSize) {
    roi.x = max(0, roi.x);
    roi.y = max(0, roi.y);
    roi.width = min(imgSize.width - roi.x, roi.width);
    roi.height = min(imgSize.height - roi.y, roi.height);
    return roi;
}

void onVminChange(int, void*) { vmin = getTrackbarPos("Vmin", "Options"); }
void onVmaxChange(int, void*) { vmax = getTrackbarPos("Vmax", "Options"); }
void onSminChange(int, void*) { smin = getTrackbarPos("Smin", "Options"); }
void onSmaxChange(int, void*) { smax = getTrackbarPos("Smax", "Options"); }
void onHueminChange(int, void*) { huemin = getTrackbarPos("Huemin", "Options"); }
void onHuemaxChange(int, void*) { huemax = getTrackbarPos("Huemax", "Options"); }
void onminContourChange(int, void*) { minContour = getTrackbarPos("minContour", "Options"); }

// Loads HSV and contour parameters from/to an XML configuration file.
void loadConfigFromXML(const string& filename) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(filename.c_str()) != tinyxml2::XML_SUCCESS) {
        cerr << "Error loading XML file: " << filename << endl;
        return;
    }

    tinyxml2::XMLElement* root = doc.FirstChildElement("Config");
    if (!root) {
        cerr << "No 'Config' element in XML file: " << filename << endl;
        return;
    }

    root->FirstChildElement("vmin")->QueryIntText(&vmin);
    root->FirstChildElement("vmax")->QueryIntText(&vmax);
    root->FirstChildElement("smin")->QueryIntText(&smin);
    root->FirstChildElement("smax")->QueryIntText(&smax);
    root->FirstChildElement("huemin")->QueryIntText(&huemin);
    root->FirstChildElement("huemax")->QueryIntText(&huemax);
    root->FirstChildElement("minContour")->QueryIntText(&minContour);

    cout << "Config loaded: "
         << "vmin=" << vmin << ", vmax=" << vmax << ", "
         << "smin=" << smin << ", smax=" << smax << ", "
         << "huemin=" << huemin << ", huemax=" << huemax << ", "
         << "minContour=" << minContour << endl;
}

// Saves HSV and contour parameters from/to an XML configuration file.
void saveConfigToXML(const string& filename) {
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLElement* root = doc.NewElement("Config");
    doc.InsertFirstChild(root);

    tinyxml2::XMLElement* vminElement = doc.NewElement("vmin");
    vminElement->SetText(vmin);
    root->InsertEndChild(vminElement);

    tinyxml2::XMLElement* vmaxElement = doc.NewElement("vmax");
    vmaxElement->SetText(vmax);
    root->InsertEndChild(vmaxElement);

    tinyxml2::XMLElement* sminElement = doc.NewElement("smin");
    sminElement->SetText(smin);
    root->InsertEndChild(sminElement);

    tinyxml2::XMLElement* smaxElement = doc.NewElement("smax");
    smaxElement->SetText(smax);
    root->InsertEndChild(smaxElement);

    tinyxml2::XMLElement* hueminElement = doc.NewElement("huemin");
    hueminElement->SetText(huemin);
    root->InsertEndChild(hueminElement);

    tinyxml2::XMLElement* huemaxElement = doc.NewElement("huemax");
    huemaxElement->SetText(huemax);
    root->InsertEndChild(huemaxElement);

    tinyxml2::XMLElement* minContourElement = doc.NewElement("minContour");
    minContourElement->SetText(minContour);
    root->InsertEndChild(minContourElement);

    tinyxml2::XMLError eResult = doc.SaveFile(filename.c_str());
    if (eResult != tinyxml2::XML_SUCCESS) {
        cerr << "Error saving XML file: " << filename << endl;
    }
}


//Converts the input image to HSV format, applies color segmentation, tracks the parcel using CAMShift, and uses opencv image processing functions to obtain a mask that isolates the parcel.
void processImage(Mat& color, Mat& hsv, Mat& hue, Mat& mask, Rect& trackWindow, int& trackObject, Mat& hist, Mat& backproj, Mat& edges, int hsize, const float* phranges, string& mode, Mat& camshiftbox) {
    cvtColor(color, hsv, COLOR_BGR2HSV);

    if (trackObject) {
        inRange(hsv, Scalar(huemin, smin, vmin),
                Scalar(huemax, smax, vmax), mask);

        if (mask.empty()) {
            cerr << "Error: mask is emptyDownloads after inRange." << endl;
            return;
        }

        Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
        dilate(mask, camshiftbox, element);

        vector<vector<Point>> contoursMask;
        findContours(camshiftbox, contoursMask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        for (const auto& contour : contoursMask) {
            Rect rect = boundingRect(contour);
            double area = rect.area();
            if (area > maxArea) {
                maxArea = area;
                selection = Rect (rect.x + rect.width/4,rect.y + rect.height/4,rect.width/2, rect.height/2);
            }
        }

        trackObject = -1;

        int ch[] = {0, 0};
        hue.create(hsv.size(), hsv.depth());
        mixChannels(&hsv, 1, &hue, 1, ch, 1);

        hue.setTo(Scalar(0), ~mask);

        if (trackObject < 0) {
            Mat roi(hue, selection), maskroi(mask, selection);
            if (roi.empty() || maskroi.empty()) {
                cerr << "Error: roi or maskroi is empty in initialization." << endl;
                return;
            }
            calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
            normalize(hist, hist, 0, 255, NORM_MINMAX);

            trackWindow = selection;
            trackObject = 1;
        }

        if (!hist.empty()) {
            calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
            //imshow("backproj",backproj);
            backproj &= mask;

            if (backproj.empty()) {
                cerr << "Error: backproj is empty after calcBackProject." << endl;
                return;
            }

            vector<vector<Point>> contours;
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                drawContours(backproj, contours, (int)i, Scalar(255), FILLED);
            }

            Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
            dilate(backproj, backproj, element);
            erode(backproj, backproj, element);
            morphologyEx(backproj, backproj, MORPH_CLOSE, element);
            cv::threshold(backproj, backproj, 20, 255, cv::THRESH_BINARY);

            contours.clear();
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) < minContour) {
                    drawContours(backproj, contours, (int)i, Scalar(0), FILLED);
                }
            }

            contours.clear();
            findContours(backproj, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i < contours.size(); i++) {
                if (contourArea(contours[i]) < minContour) {
                    drawContours(backproj, contours, (int)i, Scalar(255), FILLED);
                }
            }
            RotatedRect trackBox;
                        trackBox = CamShift(backproj, trackWindow,
                            TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));

            int r = 80;

            Rect roiRect = Rect(trackBox.boundingRect().x - r, trackBox.boundingRect().y - r,
                                    trackBox.boundingRect().width + 2*r, trackBox.boundingRect().height + 2*r);


            roiRect = adjustROI(roiRect, backproj.size());
            int cols = backproj.cols, rows = backproj.rows;
            roiRect = roiRect & Rect(0, 0, cols, rows);
            //Mat roi(backproj, roiRect);
            backproj.copyTo(edges);

            vector<vector<Point>> edgeContours;
            findContours(edges, edgeContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            drawContours(edges, edgeContours, -1, Scalar(255), 1);
        }
    }
}

// Uses depth information to compute and mark spatial coordinates of parcel corners on the image.
void getCloud(Mat& image, Mat& edges, const rs2::depth_frame& depth_frame, const rs2_intrinsics& intr, const rs2::video_stream_profile& color_stream, const rs2::video_stream_profile& depth_stream, orthoedro& box) {
    rs2_extrinsics extrinsics = depth_stream.get_extrinsics_to(color_stream);
    rs2_intrinsics color_intrinsics = color_stream.get_intrinsics();
  
    // Iterate through each point in the depth frame area
    for (int y = 0; y < depth_frame.get_height(); ++y) {
        for (int x = 0; x < depth_frame.get_width(); ++x) {
            if (edges.at<uchar>(y, x) == 255) {
                // Get the depth at the point (x, y)
                float depth = depth_frame.get_distance(x, y);

                // If the depth is valid
                if (depth > 0) {
                    // Convert from 2D coordinates to 3D using rs2_deproject_pixel_to_point
                    float point[3]; // To store the 3D coordinates
                    float pixel[2] = { static_cast<float>(x), static_cast<float>(y) };
                    rs2_deproject_pixel_to_point(point, &color_intrinsics, pixel, depth);

                    // Add the point to the Open3D cloud
                    box.measuredPoints->points_.emplace_back(point[0], point[1], point[2]);
                }
            }
        }
    }
}


//This function iterates through the list of vertex, computes the sum of vectors for each sequential triplet, and returns the vector sum with the longest magnitude. It is used in `calcBoxOrientation` to determine the primary vector direction for orientation calculation.
Vector3d calculateLongestVector(const vector<Vector3d>& vertex) {
    Vector3d sum;
    Vector3d sol(0, 0, 0);
    double length = 0;
    for (int i = 0; i < (vertex.size()); i++){
        int j = i;
        int k = i + 1;
        int l = i + 2;
        if (k >= vertex.size()){
            k = k - vertex.size();
        }
        if (l >= vertex.size()){
            l = l - vertex.size();
        }
        sum = vertex[j]+vertex[k]+vertex[l];
        if (sum.norm() > length){
            length = sum.norm();
            sol = sum;
        }
    }
    return sol;
}

void drawVertex(cv::Mat& image, const orthoedro& box, const rs2_intrinsics& intr) {

    for (size_t i = 0; i < box.vertex.size(); ++i) {
        const auto& vertex = box.vertex[i];


        float point3D[3] = { static_cast<float>(vertex.x()), static_cast<float>(vertex.y()), static_cast<float>(vertex.z()) };
        float pixel[2];
        rs2_project_point_to_pixel(pixel, &intr, point3D);

        Scalar color;
        if (i == 9) {
            color = Scalar(255, 0, 0); 
        } else if (i == 9) {
            color = Scalar(0, 255, 0); 
        } else if (i == 9) {
            color = Scalar(0, 0, 255); 
        } else if (i == 9) {
            color = Scalar(0, 0, 0); 
        } else {
            color = Scalar(255, 255, 255); 
        }

        Point vertexPoint(static_cast<int>(pixel[0]), static_cast<int>(pixel[1]));
        circle(image, vertexPoint, 5, color, FILLED);

        stringstream ss;
        ss << fixed << setprecision(2);
        ss << "(" << vertex.x() << "," << vertex.y() << "," << vertex.z() << ")";

        putText(image, ss.str(), vertexPoint + Point(10, 0), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }

    auto drawPoint = [&](const Eigen::Vector3d& point, const cv::Scalar& color) {
        float point3D[3] = { (float)point.x(), (float)point.y(), (float)point.z() };
        float pixel[2];
        rs2_project_point_to_pixel(pixel, &intr, point3D);
        cv::Point refPoint((int)pixel[0], (int)pixel[1]);
        cv::circle(image, refPoint, 5, color, cv::FILLED);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << "(" << point.x() << "," << point.y() << "," << point.z() << ")";
        cv::putText(image, oss.str(), refPoint + cv::Point(10, 0), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    };

    // Draw ref points in different colors
    drawPoint(box.refPointRightside, cv::Scalar(255, 0, 128)); // refPointRightSide
    drawPoint(box.refPointLeftside, cv::Scalar(203, 192, 255));  // refPointLeftSide
    drawPoint(box.center, cv::Scalar(0, 255, 255));          // center
}

// Rotates a point cloud based on a given plane model and target axis.
// The function computes the rotation axis from the plane normal and target axis,
// calculates the rotation angle, and applies the rotation to the point cloud.
// If `inverse` is true, the rotation will be applied in the opposite direction.
void rotateCloud(std::shared_ptr<open3d::geometry::PointCloud>& cloud, const Eigen::Vector4d& plane_model, bool inverse, const Eigen::Vector3d& target_axis = Eigen::Vector3d(0, 0, 1)) {
    Eigen::Vector3d plane_normal(plane_model[0], plane_model[1], plane_model[2]);
    // Calculate the rotation axis as the cross product between the plane normal and the target axis
    Eigen::Vector3d rotation_axis = plane_normal.cross(target_axis);
    
    double angle = 0.0;
    if (plane_normal.norm() > 1e-6 && target_axis.norm() > 1e-6) {
        // Calculate the rotation angle between the plane and the target axis (e.g., z-axis)
        angle = std::acos(plane_normal.dot(target_axis) / (plane_normal.norm() * target_axis.norm()));
    }

    if (inverse == true) angle = -angle;

    // If there is a valid rotation axis
    if (rotation_axis.norm() > 1e-6) {
        rotation_axis.normalize();
        Eigen::AngleAxisd rotation_vector(angle, rotation_axis);
        Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();

        // Rotate the point cloud using the calculated rotation matrix
        cloud->Rotate(rotation_matrix, Eigen::Vector3d(0, 0, 0));
    }
}


// Filters a point cloud by downsampling and removing outliers.
// It segments the main plane from the measured points, rotates the clouds,
// and constructs a filtered cloud by adding points from the detected planes.
void filterPointCloud(orthoedro& box) {
    box.measuredPoints = box.measuredPoints->VoxelDownSample(0.002);
    std::vector<size_t> outlier_indices;
    //tie(box.measuredPoints, outlier_indices) = box.measuredPoints->RemoveRadiusOutliers(60, 0.01);

    if (box.measuredPoints->points_.size() == 0) return;

    // Parameters for plane segmentation
    double distance_threshold = 0.002;  // Distance threshold to consider a point on the plane
    int ransac_n = 3;                   // Number of points for each plane model estimation
    int num_iterations = 1000;          // Number of RANSAC iterations

    Eigen::Vector4d plane_model;
    std::vector<size_t> inliers;

    if (box.measuredPoints->points_.size() < ransac_n) return;

    std::tie(plane_model, inliers) = box.measuredPoints->SegmentPlane(distance_threshold, ransac_n, num_iterations);

    rotateCloud(box.measuredPoints, plane_model, false);

    box.remaining_cloud = box.measuredPoints->SelectByIndex(inliers, true);
    box.mainPlane = box.measuredPoints->SelectByIndex(inliers);

    outlier_indices.clear();
    tie(box.mainPlane, outlier_indices) = box.mainPlane->RemoveRadiusOutliers(35, 0.01);

    if (box.remaining_cloud->points_.size() == 0) return;

    int neighbors = 40;
    outlier_indices.clear();
    if (box.remaining_cloud->points_.size() > neighbors) {
        tie(box.remaining_cloud, outlier_indices) = box.remaining_cloud->RemoveRadiusOutliers(neighbors, 0.01);
    }

    if (box.remaining_cloud->points_.size() == 0) return;

    std::vector<size_t> auxInliers;
    Eigen::Vector4d auxPlane_model;

    if (box.remaining_cloud->points_.size() < ransac_n) return;
    std::tie(auxPlane_model, auxInliers) = box.remaining_cloud->SegmentPlane(distance_threshold, ransac_n, num_iterations);
    
    std::shared_ptr<open3d::geometry::PointCloud> planeCloud = std::make_shared<open3d::geometry::PointCloud>();

    planeCloud = box.remaining_cloud->SelectByIndex(auxInliers);
    
    outlier_indices.clear();
    tie(planeCloud, outlier_indices) = planeCloud->RemoveRadiusOutliers(35, 0.01);
    
    *box.filteredCloud += *planeCloud;

    if (planeCloud->points_.size() == 0) return;
    
    double max_z = planeCloud->GetMaxBound().z();
    double min_z = planeCloud->GetMinBound().z();

    //tie(box.mainPlane, outlier_indices) = box.mainPlane->RemoveRadiusOutliers(10, 0.01);

    for (auto& point : box.mainPlane->points_) {
        box.filteredCloud->points_.push_back(point);
        box.filteredCloud->colors_.push_back(Eigen::Vector3d(0.0, 0.0, 0.0)); // Black for the upper plane points
        double x = point.x();
        double y = point.y();
        box.filteredCloud->points_.push_back(Vector3d(x, y, min_z));
        box.filteredCloud->colors_.push_back(Eigen::Vector3d(1.0, 0.0, 0.0)); // Red for the upper plane points
        box.filteredCloud->points_.push_back(Vector3d(x, y, max_z));
        box.filteredCloud->colors_.push_back(Eigen::Vector3d(0.0, 1.0, 0.0)); // Green for the lower plane points
    }

    rotateCloud(box.measuredPoints, plane_model, true);
    rotateCloud(box.filteredCloud, plane_model, true);
}


void calculateBoxParameters(orthoedro& box){
    box.obb = box.filteredCloud->GetMinimalOrientedBoundingBox();
    box.vertex = box.obb.GetBoxPoints();
    box.center = box.obb.GetCenter();
    double a = (box.vertex[0]-box.vertex[1]).norm();
    double b = (box.vertex[0]-box.vertex[2]).norm();
    double c = (box.vertex[0]-box.vertex[3]).norm();
    vector<double> Size = {a, b, c};
    std::sort(Size.begin(), Size.end());
    box.height = Size[0];
    box.width = Size[1];
    box.length = Size [2];

    Vector3d ref1;
    Vector3d ref2;
    Vector3d j = (box.vertex[1]-box.vertex[0] + box.vertex[2]-box.vertex[0])/2;
    Vector3d k = (box.vertex[2]-box.vertex[0] + box.vertex[3]-box.vertex[0])/2;
    Vector3d l = (box.vertex[3]-box.vertex[0] + box.vertex[1]-box.vertex[0])/2;
    
    vector<double> refPointvector = {j.norm(), k.norm(), l.norm()};
    std::sort(refPointvector.begin(), refPointvector.end());
    
    if (refPointvector[0] == j.norm()) ref1 = box.vertex[0] + j;
    if (refPointvector[0] == k.norm()) ref1 = box.vertex[0] + k;
    if (refPointvector[0] == l.norm()) ref1 = box.vertex[0] + l;

    j = (box.vertex[5]-box.vertex[4] + box.vertex[6]-box.vertex[4])/2;
    k = (box.vertex[6]-box.vertex[4] + box.vertex[7]-box.vertex[4])/2;
    l = (box.vertex[7]-box.vertex[4] + box.vertex[5]-box.vertex[4])/2;
    
    refPointvector = {j.norm(), k.norm(), l.norm()};
    std::sort(refPointvector.begin(), refPointvector.end());
    
    if (refPointvector[0] == j.norm()) ref2 = box.vertex[4] + j;
    if (refPointvector[0] == k.norm()) ref2 = box.vertex[4] + k;
    if (refPointvector[0] == l.norm()) ref2 = box.vertex[4] + l;

    if (ref1.x() > ref2.x()) {
        box.refPointLeftside = ref1;
        box.refPointRightside = ref2;
    } else {
        box.refPointLeftside = ref2;
        box.refPointRightside = ref1;
        
    }
}


// Visualize the point cloud
void visualizePointCloud(orthoedro& box) {

    // Create a vector of geometry objects for visualization
    std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries;

    box.obb.color_ = Eigen::Vector3d(0.0, 0.0, 0.0); // Black for the bounding box
    
    geometries.push_back(std::make_shared<open3d::geometry::OrientedBoundingBox>(box.obb));

    // Add the measured point cloud (original)
    //geometries.push_back(box.measuredPoints);
    geometries.push_back(box.filteredCloud);
    geometries.push_back(box.remaining_cloud);

    // Visualize all geometries
    open3d::visualization::DrawGeometries(geometries, "Point Cloud Visualization with Segmented Plane");
}


// Function to apply a moving average filter
Eigen::Vector3d movingAverageFilter(const std::deque<Eigen::Vector3d>& points) {
    Eigen::Vector3d addition(0, 0, 0);

    int window = 10;
    int count = 0;
    for (int i = std::max(0, static_cast<int>(points.size()) - window); i < points.size(); ++i) {
        addition += points[i];
        count++;
    }

    return addition / count;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <mode>" << endl;
        cout << "mode: auto or manual" << endl;
        return -1;
    }

    string mode(argv[1]);

    if (mode != "auto" && mode != "manual") {
        cerr << "Invalid mode. Use 'auto' or 'manual'." << endl;
        return -1;
    }

    loadConfigFromXML(xmlFileName);

    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);
    if (mode=="manual"){
 
        namedWindow("View", WINDOW_AUTOSIZE);
        moveWindow("View", 80, 0);

        namedWindow("Options", WINDOW_AUTOSIZE);

    
        createTrackbar("Vmin", "Options", NULL, 256, onVminChange);
        createTrackbar("Vmax", "Options", NULL, 256, onVmaxChange);
        createTrackbar("Smin", "Options", NULL, 256, onSminChange);
        createTrackbar("Smax", "Options", NULL, 256, onSmaxChange);
        createTrackbar("Huemin", "Options", NULL, 256, onHueminChange);
        createTrackbar("Huemax", "Options", NULL, 256, onHuemaxChange);
        createTrackbar("minContour", "Options", NULL, 20000, onminContourChange);

        moveWindow("Options", 1300, 0);

    }

    trackObject = -1;
    
    Mat hsv, hue, mask, hist, backproj, camshiftbox;
    Rect trackWindow;
    int hsize = 16;
    float phranges[] = {0, 180};

    std::ofstream logFile;

    // Define a deque to store the points and their times
    std::deque<Eigen::Vector3d> pointsDequeL;
    std::deque<Eigen::Vector3d> pointsDequeR;
    std::deque<long long> timeDeque;

    std::deque<Eigen::Vector3d> filteredPointL;
    std::deque<Eigen::Vector3d> filteredPointR;

    std::deque<Eigen::Vector3d> refilteredPointL;
    std::deque<Eigen::Vector3d> refilteredPointR;

	
	// Timing variables
	struct timeval tini;
	struct timeval tend;
	double delta_t = 0;
	

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);

        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            cerr << "Error: failed to get frames." << endl;
            continue;
        }

        image = Mat(Size(color_frame.get_width(), color_frame.get_height()), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        Mat depth_image(Size(depth_frame.get_width(), depth_frame.get_height()), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        rs2_intrinsics intr = color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
        Mat edges = Mat::zeros(image.size(), CV_8UC1);

        Mat vertexImage = Mat::zeros(image.size(), CV_8UC3);
        image.copyTo(vertexImage);

        if (image.empty()) {
            cerr << "Error: image is empty." << endl;
            continue;
        }

        if (depth_image.empty()) {
            cerr << "Error: depth image is empty." << endl;
            continue;
        }

        processImage(image, hsv, hue, mask, trackWindow, trackObject, hist, backproj, edges, hsize, phranges, mode, camshiftbox);



        if (!hist.empty()) {
            orthoedro box;
            try {
                getCloud(image, edges, depth_frame, color_frame.get_profile().as<rs2::video_stream_profile>().get_intrinsics(), color_frame.get_profile().as<rs2::video_stream_profile>(), depth_frame.get_profile().as<rs2::video_stream_profile>(), box);
                if (!box.measuredPoints->points_.empty()) {
                    filterPointCloud(box);
                    if (box.filteredCloud->points_.size() > 0) {
                        calculateBoxParameters(box);
                        drawVertex(vertexImage, box, intr);
                        cout << "length:" << abs(box.length) << endl;
                        cout << "width:" << abs(box.width) << endl;
                        cout << "height:" << abs(box.height) << endl;
                        if (box.length*box.height*box.width > 0.35*0.1*0.18) visualizePointCloud(box);
                        //visualizePointCloud(box);


                        long long now_ms = getCurrentTimeInMilliseconds();
                        timeDeque.push_back(now_ms);
                        pointsDequeL.push_back(box.refPointLeftside);
                        pointsDequeR.push_back(box.refPointRightside);

                        if (timeDeque.size() > MAX_POINTS) {
                            pointsDequeL.pop_front();
                            pointsDequeR.pop_front();
                            timeDeque.pop_front();
                            filteredPointL.pop_front();
                            filteredPointR.pop_front();
                        }

                        if (!filteredPointL.empty() && !pointsDequeL.empty()) {
                            if ((movingAverageFilter(pointsDequeL)-pointsDequeL[pointsDequeL.size()-1]).norm()<0.1){
                                filteredPointL.push_back(movingAverageFilter(pointsDequeL));
                                filteredPointR.push_back(movingAverageFilter(pointsDequeR));
                            }
                            else {
                                filteredPointL.push_back(filteredPointL[filteredPointL.size()-1]);
                                filteredPointR.push_back(filteredPointR[filteredPointR.size()-1]);
                            }
                        }
                        else {
                            filteredPointL.push_back(movingAverageFilter(pointsDequeL));
                            filteredPointR.push_back(movingAverageFilter(pointsDequeR));
                        }

                        refilteredPointL.push_back(movingAverageFilter(filteredPointL));
                        refilteredPointR.push_back(movingAverageFilter(filteredPointR));

                        // The calculated points are saved on a csv file named points_log up to MAX_POINTS, when more points are calculated the oldest points of the list are deleted      
                        logFile.open("../points_log.csv", std::ios::out | std::ios::trunc);
                        logFile << "Time,Left_X,Left_Y,Left_Z,Right_X,Right_Y,Right_Z,F_Left_X,F_Left_Y,F_Left_Z,F_Right_X,F_Right_Y,F_Right_Z\n";
                        for (int j = 0; j < timeDeque.size(); j++) {
                            logFile << (timeDeque[j]-timeDeque[0]) << "," << pointsDequeL[j][0] << "," << pointsDequeL[j][1] << "," << pointsDequeL[j][2] << "," << pointsDequeR[j][0] << "," << pointsDequeR[j][1] << "," << pointsDequeR[j][2] << "," << refilteredPointL[j].x() << "," << refilteredPointL[j].y() << "," << refilteredPointL[j].z() << "," << refilteredPointR[j].x() << "," << refilteredPointR[j].y() << "," << refilteredPointR[j].z() << "\n";
                        }
                        logFile.close();
                    }
                }
            } catch (const std::exception& e) {
                    std::cerr << "Error calculating orthoedro's center: " << e.what() << std::endl;
            }
        }


        char c = (char)waitKey(10);
        if (c == 27) break;
        if (c == 's') { // 's' to save the configuration
            saveConfigToXML(xmlFileName);
            cout << "Configuración guardada en " << xmlFileName << endl;
        }


        Mat depth_image_8bit;
        depth_image.convertTo(depth_image_8bit, CV_8U, 255.0 / 1000); // Adjust scaling factor if needed
        cv::applyColorMap(depth_image_8bit, depth_image_8bit, COLORMAP_JET); // Apply color map to visualize depth data better

        if (!depth_image_8bit.empty() && mode == "manual") {
            cv::imshow("Depth Image", depth_image_8bit);
        } 
        
        if (selectObject && selection.width > 0 && selection.height > 0) {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }
        
        if (mode == "manual"){
            cv::Mat view1, view2, view;
            cv::cvtColor(edges, edges, cv::COLOR_GRAY2BGR);
            cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
            cv::cvtColor(camshiftbox, camshiftbox, cv::COLOR_GRAY2BGR);

            Scalar color(0, 0, 255);

            rectangle(camshiftbox, selection, color, 2);

            cv::vconcat(image, camshiftbox, view1);
            cv::vconcat(edges, vertexImage, view2);
            cv::hconcat(view1, view2, view);

            cv::Size newSize(1220, 915);
            cv::resize(view, view, newSize, cv::INTER_NEAREST);
            cv::resize(mask, mask, newSize/2, cv::INTER_NEAREST);

            cv::imshow("View", view);
            cv::imshow("Options", mask);
        }
    }

    return 0;
}
