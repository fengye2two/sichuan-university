#ifndef __SINGLE_CALIBRATE_HPP__
#define __SINGLE_CALIBRATE_HPP__

#include <opencv2/imgproc/types_c.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <vector>

class SingleCalibrate
{
public:
	SingleCalibrate(std::string inputImagePath, int cornerNumInWidth, int cornerNumInHeight, int lengthOfSquare):
		inputImagePath_(inputImagePath),
		cornerNumInWidth_(cornerNumInWidth),
		cornerNumInHeight_(cornerNumInHeight),
		lengthOfSquare_(lengthOfSquare)
	{
	}

	void GetImages(void);

	void ExtractImageCorners2D(void);

	void CalcRealCorners3D(void);

	void SingleCalibrateCamera(void);

	void Estimate(void);

	void ProcessCalibrate(void);

	cv::Size GetImageSize(void)
	{
		return cv::Size(imageWidth_, imageHeight_);
	}

	std::vector<std::vector<cv::Point2f>> GetImageCorners2D(void)
	{
		return imageCorners2D_;
	}

	std::vector<std::vector<cv::Point3f>> GetRealCorners3D(void)
	{
		return realCorners3D_;
	}

	cv::Mat GetCameraMatrix(void)
	{
		return cameraMatrix_;
	}

	cv::Mat GetDistCoeffs(void)
	{
		return distCoeffs_;
	}

private:
	std::string inputImagePath_;
	int imageWidth_;
	int imageHeight_;
	int cornerNumInWidth_;
	int cornerNumInHeight_;
	int lengthOfSquare_;
	std::vector<cv::Mat> rgbImages_;
	std::vector<cv::Mat> grayImages_;
	int imageCount_;

	std::vector<std::vector<cv::Point2f>> imageCorners2D_;
	std::vector<std::vector<cv::Point3f>> realCorners3D_;

	cv::Mat cameraMatrix_;
	cv::Mat distCoeffs_;
	std::vector<cv::Mat> tvecsMat_;
	std::vector<cv::Mat> rvecsMat_;
};

#endif