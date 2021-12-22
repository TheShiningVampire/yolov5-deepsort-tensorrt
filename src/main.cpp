#include<iostream>
#include "manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <time.h>
using namespace cv;




int main(){
	// calculate every person's (id,(up_num,down_num,average_x,average_y))
	map<int,vector<int>> personstate;
	map<int,int> classidmap;
	bool is_first = true;
	char* yolo_engine = "/home/volta03/Multi_Object_Tracking_Vinit/PERCEPTION_ARTPARK/Object_tracking/yolov5-deepsort-tensorrt/resources/yolov5s_v5.engine";
	char* sort_engine = "/home/volta03/Multi_Object_Tracking_Vinit/PERCEPTION_ARTPARK/Object_tracking/yolov5-deepsort-tensorrt/resources/deepsort.engine";
	float conf_thre = 0.4;

	Trtyolosort yosort(yolo_engine,sort_engine);
	VideoCapture capture;
	cv::Mat frame;
	frame = capture.open("/dev/video0");
	// frame = capture.open("../../../Datasets/test_video.mp4");


	if (!capture.isOpened()){
		std::cout<<"can not open"<<std::endl;
		return -1 ;
	}
	capture.read(frame);
	std::vector<DetectBox> det;

	// std::vector<DetectBox> traj_ends;
	std::vector<std::vector<DetectBox>> traj_ends;

	auto start_draw_time = std::chrono::system_clock::now();
	
	clock_t start_draw,end_draw;
	start_draw = clock();
	int i = 0;

	// // Video save path
	// std::string video_save_path = "/home/volta03/Multi_Object_Tracking_Vinit/PERCEPTION_ARTPARK/Object_tracking/inference/test.avi";
	
	// // Create a video writer object
	// cv::VideoWriter writer(video_save_path,cv::VideoWriter::fourcc('M','J','P','G'),30,cv::Size(frame.cols,frame.rows));

	// // Check if the video writer object is created successfully
	// if (!writer.isOpened()){	
	// 	std::cout << "Could not open the output video for write: " << video_save_path << std::endl;
	// 	return -1;
	// }

	while(capture.read(frame)){
		if (i%3==0){
		//std::cout<<"origin img size:"<<frame.cols<<" "<<frame.rows<<std::endl;
		auto start = std::chrono::system_clock::now();
		yosort.TrtDetect(frame,conf_thre,det,traj_ends);
		auto end = std::chrono::system_clock::now();
		int delay_infer = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout  << "delay_infer:" << delay_infer << "ms" << std::endl;
		
		// Write the frame into the file
		// writer.write(frame);

		
		}
		i++;
	}
	capture.release();
	
	// writer.release();
	return 0;
}
