#include "manager.hpp"
using std::vector;
using namespace cv;
static Logger gLogger;

Trtyolosort::Trtyolosort(char *yolo_engine_path,char *sort_engine_path){
	sort_engine_path_ = sort_engine_path;
	yolo_engine_path_ = yolo_engine_path;
	trt_engine = yolov5_trt_create(yolo_engine_path_);
	printf("create yolov5-trt , instance = %p\n", trt_engine);
	DS = new DeepSort(sort_engine_path_, 128, 256, 0, &gLogger);

}
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes, std::vector<std::vector<DetectBox>>& trajectory_boxes) //, cv::VideoWriter &writer)
{	
    cv::Mat temp = img.clone();
    for (auto box : boxes) {
        cv::Point lt(box.x1, box.y1);
        cv::Point br(box.x2, box.y2);
        cv::rectangle(temp, lt, br, cv::Scalar(255, 0, 0), 1);
        // std::string lbl = cv::format("ID:%d_C:%d_CONF:%.2f", (int)box.trackID, (int)box.classID, box.confidence);
		std::string lbl = cv::format("ID:%d", (int)box.trackID);
		// std::string lbl = cv::format("ID:%d_x:%f_y:%f",(int)box.trackID,(box.x1+box.x2)/2,(box.y1+box.y2)/2);
        cv::putText(temp, lbl, lt, cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255,255,0));
    }
	for (int i = 0; i < trajectory_boxes.size(); i++)
	{	
		int j = 0;
		for (auto box : trajectory_boxes[i])
		{
			// cv::Point lt(box.x1, box.y1);
			// cv::Point br(box.x2, box.y2);
			// cv::rectangle(temp, lt, br, cv::Scalar(0, 0, 255 - 10* i ), 1);
			// Dot at the center of the box
			cv::circle(temp, cv::Point((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2), 2, cv::Scalar(200, 150, 255-20* j), 4);
			j++;
		}
	}
	
    cv::imshow("img", temp);
    cv::waitKey(1);

	// Save the video in a file
	// writer.write(temp);
	// return temp;
}



int Trtyolosort::TrtDetect(cv::Mat &frame,float &conf_thresh,std::vector<DetectBox> &det, std::vector<std::vector<DetectBox>> &traj)
{
	// Video save path
	// std::string video_save_path = "/home/volta03/Multi_Object_Tracking_Vinit/PERCEPTION_ARTPARK/Object_tracking/inference/test_1.avi";
	// Create a video writer object
	// VideoWriter writer(video_save_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 20, cv::Size(frame.cols, frame.rows));
	// yolo detect
	auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh,det);
	DS->sort(frame,det,traj);


	showDetection(frame, det, traj);
	// tempy =  showDetection(frame,det,traj, writer);
	// writer.write(tempy);
	// writer.release();

	return 1;
}
