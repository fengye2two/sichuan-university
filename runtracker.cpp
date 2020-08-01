#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "dirent.h"

#include "Longinus/LonginusDetector.hpp"
#include "Excalibur/tensor_operation_cpu.hpp"

#include "detectnet.h"
#include "yolo_v2_class.hpp"

//#define DETAIL

using namespace std;
using namespace cv;
using namespace glasssix;
using namespace glasssix::excalibur;

void show_detection_results(cv::Mat img, std::vector<longinus::FaceRectwithFaceInfo> face_info)
{
	cv::Mat img_copy = img.clone();
	for (int i = 0; i < face_info.size(); i++)
	{
		cv::rectangle(img_copy, cv::Rect(face_info[i].x, face_info[i].y, face_info[i].width, face_info[i].height), cv::Scalar(0, 255, 0), 2);
		for (int j = 0; j < 5; j++)
		{
			cv::circle(img_copy, cv::Point(face_info[i].pts[j].x, face_info[i].pts[j].y), 2, cv::Scalar(0, 0, 255), 2);
		}
	}
	cv::imshow("detection results", img_copy);
	cv::waitKey(100);
}

void show_alignment_results(std::vector<unsigned char> alignedfaces_data, int face_count)
{
	auto face_mats = longinus::encode2mats(alignedfaces_data, face_count);
	for (int i = 0; i < face_count; i++)
	{
		cv::imshow("aligned face", face_mats[i]);
		cv::waitKey(100);
	}
}

void display_video()
{
	VideoCapture cap;
	cap.open("D:/dataset/tracking2.mp4");
	Mat img, gray;

	while (1)
	{
		cap >> img;
		if (img.empty()) break;		

		int channel = img.channels();
		int height = img.rows;
		int width = img.cols;
		imshow("video", img);

		cv::cvtColor(img, gray, CV_BGR2GRAY);
		unsigned char* data = gray.data;

		int device = 0;
		glasssix::longinus::LonginusDetector detector;
		detector.set(longinus::FRONTALVIEW_REINFORCE, device);
		std::vector<longinus::FaceRectwithFaceInfo> face_info;
		std::vector<std::vector<int>> bboxes;
		std::vector<std::vector<int>> landmarks;
		std::vector<unsigned char> alignedfaces_data;

		float factor = 0.709f;
		float threshold[3] = { 0.8f, 0.8f, 0.6f };
		int minSize = 48;

		face_info = detector.detectEx(img.data, channel, height, width, minSize, threshold, factor, 3);
		show_detection_results(img, face_info);

		if (face_info.size() <= 0)
		{
			continue;
		}

		// alignment step
		longinus::extract_biggest_faceinfo(face_info, bboxes, landmarks);
		alignedfaces_data = detector.alignFace(data, 1, 1, height, width, bboxes, landmarks);
		for (int i = 0; i < bboxes.size(); i++)
		{
			std::cout << "bbox:" << bboxes[i][0] << " " << bboxes[i][1] << " " << bboxes[i][2] << " " << bboxes[i][3] << std::endl;
			std::cout << "landmarks:" << landmarks[i][0] << " " << landmarks[i][1] << " " << landmarks[i][2] << " " << landmarks[i][3] << " " << landmarks[i][4] << " "
				<< landmarks[i][5] << " " << landmarks[i][6] << " " << landmarks[i][7] << " " << landmarks[i][8] << " " << landmarks[i][9] << std::endl;
		}

		show_alignment_results(alignedfaces_data, 1);
	}
}


int original_tracking(int argc, char* argv[])
{
	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	display_video();
	system("pause");
	return 0;

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// img readed
	Mat img;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "images.txt";
	listFile.open(fileName);

	// Read groundtruth for the 1st img
	ifstream groundtruthFile;
	string groundtruth = "region.txt";
	groundtruthFile.open(groundtruth);
	string firstLine;
	getline(groundtruthFile, firstLine);
	groundtruthFile.close();

	istringstream ss(firstLine);

	// Read groundtruth like a dumb
	float x1, y1, x2, y2, x3, y3, x4, y4;
	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4;

	// Using min and max of X and Y for groundtruth rectangle
	float xMin = min(x1, min(x2, min(x3, x4)));
	float yMin = min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;


	// Read Images
	ifstream listimgsFile;
	string listimgs = "images.txt";
	listimgsFile.open(listimgs);
	string imgName;


	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// img counter
	int nimgs = 0;


	while (getline(listimgsFile, imgName)) {
		imgName = imgName;

		// Read each img from the list
		img = imread(imgName, CV_LOAD_IMAGE_COLOR);

		// First img, give the groundtruth to the tracker
		if (nimgs == 0) {
			tracker.init(Rect(xMin, yMin, width, height), img);
			cv::rectangle(img, Point(xMin, yMin), Point(xMin + width, yMin + height), Scalar(0, 255, 255), 1, 8);
			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
		}
		// Update
		else {
			result = tracker.update(img);
			cv::rectangle(img, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 255, 255), 1, 8);
			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
		}

		nimgs++;

		if (!SILENT) {
			imshow("Image", img);
			waitKey(1);
		}
	}
	resultsFile.close();

	listFile.close();
}

int face_tracking(int argc, char* argv[])
{
	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	VideoCapture cap;
	cap.open("D:/dataset/tracking2.mp4");
	Mat img, gray;
	bool isFirst = true;

	while (1)
	{
		cap >> img;
		if (img.empty()) break;

		cv::resize(img, img, cv::Size(img.cols / 4, img.rows / 4));
		int channel = img.channels();
		int height = img.rows;
		int width = img.cols;
		imshow("video", img);

		if (isFirst == true) {
			isFirst = false;
			int device = 0;
			glasssix::longinus::LonginusDetector detector;
			detector.set(longinus::FRONTALVIEW_REINFORCE, device);
			std::vector<longinus::FaceRectwithFaceInfo> face_info;
			std::vector<std::vector<int>> bboxes;
			std::vector<std::vector<int>> landmarks;

			float factor = 0.709f;
			float threshold[3] = { 0.8f, 0.8f, 0.6f };
			int minSize = 48;

			face_info = detector.detectEx(img.data, channel, height, width, minSize, threshold, factor, 3);

			if (face_info.size() <= 0)
			{
				continue;
			}

			// alignment step
			longinus::extract_biggest_faceinfo(face_info, bboxes, landmarks);
			for (int i = 0; i < bboxes.size(); i++)
			{
				std::cout << "bbox:" << bboxes[i][0] << " " << bboxes[i][1] << " " << bboxes[i][2] << " " << bboxes[i][3] << std::endl;
			}

			tracker.init(Rect(bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]), img);
			cv::rectangle(img, Point(bboxes[0][0], bboxes[0][1]), Point(bboxes[0][0] + bboxes[0][2], bboxes[0][1] + bboxes[0][3]), Scalar(0, 0, 255), 2, 8);
		}
		else {
			cv::Rect result = tracker.update(img);
			cv::rectangle(img, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 0, 255), 2, 8);
		}

		imshow("result", img);
		waitKey(10);
	}

	system("pause");
	return 0;
}

int car_tracking_yoloV3(int argc, char* argv[])
{
	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	YOLODetect carDetector;

	VideoCapture cap;
	cap.open("D:/dataset/car.mp4");
	Mat img, gray;
	bool isDetect = true;

	int count = 0;
	int interval = 50;
	while (1)
	{
		cap >> img;
		if (img.empty()) break;

		cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));

		if (isDetect == true) {
			vector<vector<float> > targets = carDetector.Detect(img);
			if (targets.size() <= 0) {
				continue;
			}

			isDetect = false;

			for (int i = 0; i < targets.size(); ++i) {
				cv::rectangle(img, Point(targets[i][2], targets[i][4]), Point(targets[i][3], targets[i][5]), Scalar(0, 0, 255), 3, 8, 0);

				//for (int j = 0; j < targets[i].size(); ++j) {
				//	cout << setw(10) << targets[i][j] << ",";
				//}
				//cout << endl;
			}

			tracker.init(Rect(targets[0][2], targets[0][4], targets[0][3] - targets[0][2], targets[0][5] - targets[0][4]), img);
		}
		else {
			cv::Rect result = tracker.update(img);
			cv::rectangle(img, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 0, 255), 2, 8);
		}

		++count;
		if (count % interval >= 0) {
			isDetect = true;
		}

		imshow("result", img);
		waitKey(10);
	}
}

int car_tracking_yoloV4(int argc, char* argv[])
{
	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for (int i = 0; i < argc; i++) {
		if (strcmp(argv[i], "hog") == 0)
			HOG = true;
		if (strcmp(argv[i], "fixed_window") == 0)
			FIXEDWINDOW = true;
		if (strcmp(argv[i], "singlescale") == 0)
			MULTISCALE = false;
		if (strcmp(argv[i], "show") == 0)
			SILENT = false;
		if (strcmp(argv[i], "lab") == 0) {
			LAB = true;
			HOG = true;
		}
		if (strcmp(argv[i], "gray") == 0)
			HOG = false;
	}

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	string cfg = "D:/darknet/cfg/yolov4.cfg";
	string weight = "D:/darknet/cfg/yolov4.weights";
	Detector carDetector = Detector(cfg, weight, 0);

	for (int interval = 1; interval <= 100; ++interval) {
		cout << "interval: " << interval << "\r";
		std::ofstream out;
		out.open("D:/object_tracking/data/interval_" + to_string(interval) + ".txt");

		glasssix::Timer calcTime;	

		VideoCapture cap;
		cap.open("D:/dataset/car.mp4");

		Mat img;
		bool isDetect = true;
		int count = 0;

		calcTime.Start();
		while (1)
		{
			cap >> img;
			if (img.empty()) break;
			++count;
			if (count <= 6 || count == 47 || count == 118 || count == 119 || 
				count == 489 || count == 541 || count == 542 || count == 545 || 
				count == 1597 || count == 1598) {
				continue;
			}

			if (isDetect == true) {
				std::vector<bbox_t> result = carDetector.detect(img);
				if (result.size() <= 0) {
					continue;
				}

				isDetect = false;
				tracker.init(Rect(result[0].x, result[0].y, result[0].w, result[0].h), img);
				out << "init :" << count << ", " << result[0].x << ", " << result[0].y << ", " << result[0].w << ", " << result[0].h << endl;

	#ifdef DETAIL
				for (int i = 0; i < result.size(); ++i) {
					cv::rectangle(img, Point(result[i].x, result[i].y), Point(result[i].x + result[i].w, result[i].y + result[i].h), Scalar(0, 0, 255), 3, 8, 0);
				}
				cout << "init :" << count << ", " << result[0].x << ", " << result[0].y << ", " << result[0].w << ", " << result[0].h << endl;
	#endif // DETAIL	
			}
			else {
				cv::Rect result = tracker.update(img);
				out << "track:" << count << ", " << result.x << ", " << result.y << ", " << result.width << ", " << result.height << endl;

	#ifdef DETAIL
				cout << "track:" << count << ", " << result.x << ", " << result.y << ", " << result.width << ", " << result.height << endl;
				cv::rectangle(img, Point(result.x, result.y), Point(result.x + result.width, result.y + result.height), Scalar(0, 0, 255), 2, 8);
	#endif // DETAIL
			}

			if (interval == 1 || count % interval == 0) {
				isDetect = true;
			}

	#ifdef DETAIL
			imshow("result", img);
			waitKey(10);
	#endif // DETAIL
		}

		calcTime.Stop();
		double elapseTime = calcTime.GetElapsedMilliseconds() / 1000;

	#ifdef DETAIL
		std::cout << "elapseTime:" << elapseTime << std::endl;
	#endif // DETAIL

		out << "elapseTime: " << elapseTime << std::endl;
		out.close();
	}
}

float calcIOU(vector<int> left, vector<int> right)
{
	int maxX = std::max(left[0], right[0]);
	int maxY = std::max(left[1], right[1]);
	int minX = std::min(left[0] + left[2], right[0] + right[2]);
	int minY = std::min(left[1] + left[3], right[1] + right[3]);

	//maxX1 and maxY1 reuse 
	maxX = ((minX - maxX) > 0) ? (minX - maxX) : 0;
	maxY = ((minY - maxY) > 0) ? (minY - maxY) : 0;

	//IOU reuse for the area of two bboxintersection_width
	int intersection_area = maxX * maxY;

	int area_left = left[2] * left[3];
	int area_right = right[2] * right[3];

	float IOU = float(intersection_area) / (area_left + area_right - intersection_area);

	//std::cout << "left:" << std::endl;
	//for (int i = 0; i < left.size(); ++i) {
	//	cout << left[i] << ", ";
	//}
	//std::cout << std::endl;

	//std::cout << "right:" << std::endl;
	//for (int i = 0; i < right.size(); ++i) {
	//	cout << right[i] << ", ";
	//}
	//std::cout << std::endl;
	//std::cout << "area_left:" << area_left << ", area_right:" << area_right << std::endl;
	//std::cout << "maxX:" << maxX << ", maxY:" << maxY << ", intersection_area:" << intersection_area << std::endl;
	//std::cout << "IOU: " << IOU << std::endl;

	return IOU;
}

typedef struct {
	int frameIndex;
	vector<int> coordinates;
} coordinatesInfo;

static std::vector<std::string> split_string(const std::string& s, const std::string& c)
{
	std::vector<std::string> v;
	std::string::size_type pos1, pos2;
	pos2 = s.find(c);
	pos1 = 0;
	while (std::string::npos != pos2)
	{
		v.push_back(s.substr(pos1, pos2 - pos1));

		pos1 = pos2 + c.size();
		pos2 = s.find(c, pos1);
	}
	if (pos1 != s.length())
		v.push_back(s.substr(pos1));
	return v;
}

vector<coordinatesInfo> extract_coordinates_from_file(std::string file_path)
{
	string temp;
	std::ifstream file(file_path);
	vector<coordinatesInfo> positions;
	while (getline(file, temp)) {
		vector<string> split1 = split_string(temp, ", ");
		if (split1.size() != 5) {
			break;
		}

		vector<string> split2 = split_string(split1[0], ":");

		coordinatesInfo temp_position;
		temp_position.frameIndex = std::atoi(split2[1].c_str());
		vector<int> coordinate;
		coordinate.push_back(std::atoi(split1[1].c_str()));
		coordinate.push_back(std::atoi(split1[2].c_str()));
		coordinate.push_back(std::atoi(split1[3].c_str()));
		coordinate.push_back(std::atoi(split1[4].c_str()));
		temp_position.coordinates = coordinate;

		positions.push_back(temp_position);
	}

	return positions;
}

float extract_time_from_file(std::string file_path)
{
	string temp;
	std::ifstream file(file_path);
	while (getline(file, temp)) {
		if (temp.find(",") != std::string::npos) {
			continue;
		}
		
		temp = temp.substr(temp.find(":") + 1);
		return std::atof(temp.c_str());
	}

	return 0;
}


void generateIOU_time()
{
	std::ofstream out("D:/object_tracking/data/1time_IOU.txt");
	std::string benchmark_file = "D:/object_tracking/data/interval_1.txt";
	vector<coordinatesInfo> benchmark_coordinates = extract_coordinates_from_file(benchmark_file);
	float benchmark_time = extract_time_from_file(benchmark_file);

	for (int i = 2; i <= 100; ++i) {
		std::string compare_file = "D:/object_tracking/data/interval_" + to_string(i) + ".txt";
		vector<coordinatesInfo> compare_coordinates = extract_coordinates_from_file(compare_file);

		float sum = 0;
		int totalNum = min(benchmark_coordinates.size(), compare_coordinates.size());
		for (int j = 0; j < totalNum; ++j) {
			float IOU = calcIOU(benchmark_coordinates[j].coordinates, compare_coordinates[j].coordinates);
			sum += IOU;
		}

		float avgIOU = sum / totalNum;
		cout << "avgIOU:" << avgIOU << std::endl;

		float compare_time = extract_time_from_file(compare_file);
		cout << benchmark_time << ", " << compare_time << ", " << benchmark_time / compare_time << std::endl;

		out << i << "," << avgIOU << "," << benchmark_time / compare_time << endl;
	}
}

int main(int argc, char* argv[]){

	//car_tracking_yoloV3(argc, argv);
	generateIOU_time();
	//car_tracking_yoloV4(argc, argv);

	system("pause");
	return 0;
}