
#include "single_calibrate.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// 获取彩色和灰度图像
void SingleCalibrate::GetImages(void)
{
	/* 标定所用图像文件的路径 */
	ifstream fin(inputImagePath_);
	string filename;

	while (getline(fin, filename))
	{
		cout << filename << endl;
		cv::Mat rgbImage = cv::imread(filename);
		if (!rgbImage.empty()) {
			// 获取彩色RGB图像
			rgbImages_.push_back(rgbImage);

			// 转换为灰度图像
			cv::Mat grayImage;
			cv::cvtColor(rgbImage, grayImage, CV_RGB2GRAY);
			grayImages_.push_back(grayImage);
		}
	}

	// 图像数目
	imageCount_ = rgbImages_.size();

	// 所有图像具有相同的宽度和高度
	if (imageCount_ > 0) {
		imageWidth_ = rgbImages_[0].cols;
		imageHeight_ = rgbImages_[0].rows;
	}
}

// 提取图像中的二维角点，并保存在imageCorners2D_中
void SingleCalibrate::ExtractImageCorners2D(void)
{
	std::vector<cv::Point2f> tempImageCorners2D;
	for (int i = 0; i < imageCount_; ++i) {
		// 初步提取像素级精度的角点
		if (0 == findChessboardCorners(rgbImages_[i], cv::Size(cornerNumInWidth_, cornerNumInHeight_), tempImageCorners2D))
		{
			cout << "cannot get image corners 2D: " << i << endl;
			continue;
		}

		// 提取精度更高的角点（亚像素级）
		cv::cornerSubPix(grayImages_[i], tempImageCorners2D, cv::Size(11, 11), cv::Size(-1, -1),
			cv::TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 0.01));

		imageCorners2D_.push_back(tempImageCorners2D);
		// 绘制并显示角点
		drawChessboardCorners(rgbImages_[i], cv::Size(cornerNumInWidth_, cornerNumInHeight_), tempImageCorners2D, true);
		imshow("Camera Calibration", rgbImages_[i]);
		waitKey(10);
	}
}

// 计算角点真实的3维世界坐标（Z轴为0），并保存在realCorners3D_中
void SingleCalibrate::CalcRealCorners3D(void)
{
	for (int n = 0; n < imageCount_; ++n)
	{
		vector<Point3f> tempRealCorners3D;
		for (int h = 0; h < cornerNumInHeight_; ++h)
		{
			for (int w = 0; w < cornerNumInWidth_; ++w)
			{
				Point3f realPoint;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realPoint.y = h * lengthOfSquare_;// lengthOfSquare_为打印出来的纸上，黑白正方形格子的实际宽度
				realPoint.x = w * lengthOfSquare_;
				realPoint.z = 0;
				tempRealCorners3D.push_back(realPoint);
			}
		}
		realCorners3D_.push_back(tempRealCorners3D);
	}
}

// 根据二维和三维角点信息，计算出单目相机的旋转矩阵cameraMatrix_和平移矩阵distCoeffs_
void SingleCalibrate::SingleCalibrateCamera(void)
{
	cameraMatrix_ = cv::Mat(3, 3, CV_32FC1, Scalar::all(0));
	distCoeffs_ = cv::Mat(1, 5, CV_32FC1, Scalar::all(0));
	calibrateCamera(realCorners3D_, imageCorners2D_, cv::Size(imageWidth_, imageHeight_), cameraMatrix_, distCoeffs_, rvecsMat_, tvecsMat_, 0);
}

// 评估单目相机标定的误差（暂不需要详细研究）
void SingleCalibrate::Estimate(void)
{
	std::string outputCalibrateParam = "D:/double_camera/left/caliberation_result.xml";
	FileStorage fs(outputCalibrateParam, FileStorage::WRITE);
	double total_err = 0.0; /* 所有图像的平均误差的总和 */
	double err = 0.0; /* 每幅图像的平均误差 */
	vector<Point2f> image_points2; /* 保存重新计算得到的投影点 */

	for (int i = 0; i < imageCount_; i++)
	{
		vector<Point3f> tempPointSet = realCorners3D_[i];
		/* 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点 */
		projectPoints(tempPointSet, rvecsMat_[i], tvecsMat_[i], cameraMatrix_, distCoeffs_, image_points2);
		/* 计算新的投影点和旧的投影点之间的误差*/
		vector<Point2f> tempImagePoint = imageCorners2D_[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (int j = 0; j < tempImagePoint.size(); j++)
		{
			image_points2Mat.at<Vec2f>(0, j) = Vec2f(image_points2[j].x, image_points2[j].y);
			tempImagePointMat.at<Vec2f>(0, j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
		}
		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= (cornerNumInWidth_ * cornerNumInHeight_);
		std::cout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
	}

	std::cout << "显示矫正结果：" << std::endl;
	for (int k = 0; k < imageCount_; ++k)
	{
		cv::namedWindow("src");
		cv::namedWindow("dst");
		cv::moveWindow("src", 100, 0);
		cv::moveWindow("dst", 800, 0);

		Mat imageInput = rgbImages_[k];
		cv::imshow("src", imageInput);

		cv::Mat imageOutput;
		undistort(imageInput, imageOutput, cameraMatrix_, distCoeffs_);

		cv::imshow("dst", imageOutput);
		cv::waitKey(500);
	}
	std::cout << "总体平均误差：" << total_err / imageCount_ << "像素" << endl;
	std::cout << "评价完成！" << endl;
	//保存定标结果  	
	std::cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	fs << "cameraMatrix" << cameraMatrix_;
	fs << "distCoeffs" << distCoeffs_;
	std::cout << "完成保存" << endl;
}


// 单目相机标定的全流程
void SingleCalibrate::ProcessCalibrate(void)
{
	GetImages();
	ExtractImageCorners2D();
	CalcRealCorners3D();
	SingleCalibrateCamera();
}


//// 测试函数（忽略）
//void new_one()
//{
//	std::string inputImagePath = "D:/double_camera/calib_imgs/calib_data_left.txt";
//	int lengthOfSquare = 10;
//	int cornerNumInWidth = 6;
//	int cornerNumInHeight = 9;
//
//	SingleCalibrate singleCalibrate(inputImagePath, cornerNumInWidth, cornerNumInHeight, lengthOfSquare);
//	singleCalibrate.ProcessCalibrate();
//	singleCalibrate.Estimate();
//}

//void main()
//{
//	//original();
//	new_one();
//}