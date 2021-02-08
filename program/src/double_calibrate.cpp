#include "double_calibrate.h"
 
// 构造函数，初始化左右摄像头：leftCamera和rightCamera
DoubleCalibrate::DoubleCalibrate(string leftImagePath, string rightImagePath, int cornerNumInWidth, int cornerNumInHeight, int lengthOfSquare, std::string paramPath) :
	leftCamera(leftImagePath, cornerNumInWidth, cornerNumInHeight, lengthOfSquare),
	rightCamera(rightImagePath, cornerNumInWidth, cornerNumInHeight, lengthOfSquare),
	paramPath_(paramPath)
{
}
 
// 双目标定处理，并将标定参数保存在paramPath_
void DoubleCalibrate::ProcessCalibrate(void)
{
	// 对左右摄像头，分别进行单目标定
	leftCamera.ProcessCalibrate();
	rightCamera.ProcessCalibrate();
	imageSize = leftCamera.GetImageSize();

	// 获得单目标定结果后，进行双目标定，求得R/T/E/F矩阵
	double rms = stereoCalibrate(leftCamera.GetRealCorners3D(), 
		leftCamera.GetImageCorners2D(), rightCamera.GetImageCorners2D(),
		leftCamera.GetCameraMatrix(), leftCamera.GetDistCoeffs(), 
		rightCamera.GetCameraMatrix(), rightCamera.GetDistCoeffs(), 
		imageSize, R, T, E, F);

	// R– 输出第一和第二相机坐标系之间的旋转矩阵。
	// T– 输出第一和第二相机坐标系之间的旋转矩阵平移向量。 
	// E-包含了两个摄像头相对位置关系的本征矩阵Essential Matrix（3*3）
	//其物理意义是左右图像坐标系相互转换的矩阵
	// F-既包含两个摄像头相对位置关系、也包含摄像头各自内参信息的基础矩阵
	cout << "校正后的立体相机的均方根误差为：" << rms << endl;
	FileStorage fs(paramPath_, FileStorage::WRITE);
	fs << "l_cameraMatrix" << leftCamera.GetCameraMatrix();
	fs << "r_cameraMatrix" << rightCamera.GetCameraMatrix();
	fs << "l_distCoeffs" << leftCamera.GetDistCoeffs();
	fs << "r_distCoeffs" << rightCamera.GetDistCoeffs();
	fs << "R" << R;
	fs << "T" << T;
	fs << "E" << E;
	fs << "F" << F;
	cout << "校准参数保存至：" << paramPath_ << endl;
}
 
// 求得图像矫正的map：map_left1, map_left2, map_right1, map_right2
void DoubleCalibrate::CalcTransformMap(void)
{
	Mat R_left, R_right, P_left, P_right, Q;
	stereoRectify(leftCamera.GetCameraMatrix(), leftCamera.GetDistCoeffs(), rightCamera.GetCameraMatrix(), rightCamera.GetDistCoeffs(),
		imageSize, R, T, R_left, R_right, P_left, P_right, Q);
	//R1– 输出第一个相机的3x3矫正变换(旋转矩阵) .
	//R2– 输出第二个相机的3x3矫正变换(旋转矩阵) .
	//P1–在第一台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
	//P2–在第二台相机的新的坐标系统(矫正过的)输出 3x4 的投影矩阵
	//Q–输出深度视差映射矩阵

	// Calculate pixel maps for efficient rectification of images via lookup tables
	initUndistortRectifyMap(leftCamera.GetCameraMatrix(), leftCamera.GetDistCoeffs(), R_left, P_left, imageSize, CV_16SC2, map_left1, map_left2);
	initUndistortRectifyMap(rightCamera.GetCameraMatrix(), rightCamera.GetDistCoeffs(), R_right, P_right, imageSize, CV_16SC2, map_right1, map_right2);

	FileStorage fs(paramPath_, FileStorage::APPEND);
	fs << "R_left" << R_left;
	fs << "R_right" << R_right;
	fs << "P_left" << P_left;
	fs << "P_right" << P_right;
	fs << "Q" << Q;
	fs << "map_left1" << map_left1;
	fs << "map_left2" << map_left2;
	fs << "map_right1" << map_right1;
	fs << "map_right2" << map_right2;
	fs.release();
}
 
// 输入左右图像，通过map矫正，然后显示
void DoubleCalibrate::ShowRectified(std::string leftImage, std::string rightImage) {
	Mat frame_left, frame_left_rect, frame_right, frame_right_rect;
 
	frame_left = imread(leftImage);
	frame_right = imread(rightImage);
	if (frame_left.empty() || frame_right.empty())
	{
		cout << "读取需要矫正的图像失败！！" << endl;
	}

	// 矫正得到frame_left_rect
	remap(frame_left, frame_left_rect, map_left1, map_left2, INTER_LINEAR);
	// 矫正得到frame_right_rect
	remap(frame_right, frame_right_rect, map_right1, map_right2, INTER_LINEAR);
 
	//将两幅图像拼接在一起
	Mat combo(imageSize.height, 2 * imageSize.width, CV_8UC3);
	Rect rectLeft(0, 0, imageSize.width, imageSize.height); //左半部分
	Rect rectRight(imageSize.width, 0, imageSize.width, imageSize.height);//右半部分 
 
	frame_left_rect.copyTo(combo(rectLeft));// frame_left_rect拷贝至左半部分
	frame_right_rect.copyTo(combo(rectRight));// frame_right_rect拷贝至右半部分
 
	// Draw horizontal red lines in the combo image to make comparison easier
	for (int y = 0; y < combo.rows; y += 20)
	{
		line(combo, Point(0, y), Point(combo.cols, y), Scalar(0, 0, 255));
	}
 
	imshow("left_rect", frame_left_rect);
	imshow("right_rect", frame_right_rect);
	imshow("Combo", combo);
	waitKey(0);
	imwrite("Combo.jpg", combo);
}

// 输入左右图像，通过map矫正，然后显示
void DoubleCalibrate::ShowRectified(cv::Mat leftImage, cv::Mat rightImage, cv::Mat &frame_left_rect, cv::Mat &frame_right_rect) {
	if (leftImage.empty() || rightImage.empty())
	{
		cout << "读取需要矫正的图像失败！！" << endl;
	}

	// 矫正得到frame_left_rect
	remap(leftImage, frame_left_rect, map_left1, map_left2, INTER_LINEAR);
	// 矫正得到frame_right_rect
	remap(rightImage, frame_right_rect, map_right1, map_right2, INTER_LINEAR);

	////将两幅图像拼接在一起
	//Mat combo(imageSize.height, 2 * imageSize.width, CV_8UC3);
	//Rect rectLeft(0, 0, imageSize.width, imageSize.height); //左半部分
	//Rect rectRight(imageSize.width, 0, imageSize.width, imageSize.height);//右半部分 

	//frame_left_rect.copyTo(combo(rectLeft));// frame_left_rect拷贝至左半部分
	//frame_right_rect.copyTo(combo(rectRight));// frame_right_rect拷贝至右半部分

	//// Draw horizontal red lines in the combo image to make comparison easier
	//for (int y = 0; y < combo.rows; y += 20)
	//{
	//	line(combo, Point(0, y), Point(combo.cols, y), Scalar(0, 0, 255));
	//}

	////imshow("left_rect", frame_left_rect);
	////imshow("right_rect", frame_right_rect);

	//resize(combo, combo, cv::Size(combo.cols / 2, combo.rows / 2));
	//imshow("Combo", combo);
	//waitKey(100);
	//imwrite("Combo.jpg", combo);
}

// 输入左右图像，通过map矫正，然后显示
cv::Mat DoubleCalibrate::rectifyImage(cv::Mat inputImage, cv::Mat map1, cv::Mat map2) {	
	if (inputImage.empty())
	{
		cout << "读取需要矫正的图像失败！！" << endl;
	}

	Mat outputImage;

	// 矫正得到frame_left_rect
	remap(inputImage, outputImage, map1, map2, INTER_LINEAR);

	return outputImage;
}