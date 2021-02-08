#ifndef __STEREO_CALIBRATE_HPP__
#define __STEREO_CALIBRATE_HPP__


#include "single_calibrate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;
 
class DoubleCalibrate
{
public:
	DoubleCalibrate(std::string leftImagePath, std::string rightImagePath, int cornerNumInWidth, int cornerNumInHeight, int lengthOfSquare, std::string paramPath);
    
	SingleCalibrate leftCamera;
	SingleCalibrate rightCamera;
	void ProcessCalibrate(void); // 双目标定计算
	void CalcTransformMap(void);
	void ShowRectified(std::string leftImage, std::string rightImage);
	void ShowRectified(cv::Mat leftImage, cv::Mat rightImage, cv::Mat &frame_left_rect, cv::Mat &frame_right_rect);
	cv::Mat rectifyImage(cv::Mat inputImage, cv::Mat map1, cv::Mat map2);

private:
	Mat map_left1, map_left2, map_right1, map_right2; //pixel maps for rectification
	Size imageSize; 
	std::string paramPath_;
 
	//Mat l_cameraMatrix, l_distCoeffs;
	//Mat r_cameraMatrix, r_distCoeffs;

	//vector<vector<Point2f> > l_image_points, r_image_points; 	
	//vector<vector<Point3f> > object_points; 
	Mat R, T, E, F; // 立体相机参数
};
#endif