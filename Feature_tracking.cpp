/*
****************************************
Created on Tu Jun 9  2020		       *
									   *
@author: willy wandji	   *
****************************************

#######################  FEATURES Tracking #################################

Help:
		1-pressed down the left Mouse button and move it to drawing a rectangle
		2-After releasing the left mouse button, press the Button <a>.
		3-press <ESC> to exit

*/


// import von Module
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <opencv2/calib3d.hpp>


#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>



// Namespace
using namespace cv;
using namespace std;
constexpr auto WINDOW_NAME = "Drawing Rectangle";


// Globale variable
Rect gRect;
Rect rectangle_2D;
auto drawingState = false;

// optical flow : Lucas Kanade methode
void track_feature_points(Mat prev_img_gray, Mat new_img_gray, vector<Point2f> prev_pts, vector<Point2f>* ptr_New_pts, vector<float>* ptr_Err, vector<unsigned char>* ptr_Status)
{
	//intern Variable
	vector<unsigned char> status;
	vector<float> err;
	vector<Point2f> new_pts;

	calcOpticalFlowPyrLK(prev_img_gray, new_img_gray, prev_pts, new_pts, status, err);

	*ptr_New_pts = new_pts;
	*ptr_Err = err;
	*ptr_Status = status;
}

// FFunktion Zur Extraction of Feature
void extract_feature(vector<Point2f>* ptr_prev_pts, std::vector<KeyPoint>* ptr_obj_keypoints, Mat* ptr_ObjetGray, Mat objt, Ptr<Feature2D> detector)
{
	// intern Variable
	Mat ObjetGray;
	std::vector<KeyPoint> obj_keypoints;
	cvtColor(objt, ObjetGray, cv::COLOR_BGR2GRAY);
	vector<Point2f> prev_pts;


	Mat mask = Mat::zeros(ObjetGray.size(), CV_8UC1);
	rectangle(mask, gRect, Scalar::all(255), -1);
	//imshow("mask", mask);
	detector->detect(ObjetGray, obj_keypoints, mask);


	for (auto i = 0; i < obj_keypoints.size(); ++i)
	{
		prev_pts.push_back(obj_keypoints[i].pt);
	}

	*ptr_obj_keypoints = obj_keypoints;
	*ptr_ObjetGray = ObjetGray;
	*ptr_prev_pts = prev_pts;
}

//Funktion für das Zeichnen des Rectangles
void DrawRectangle(Mat& img, Rect box)
{
	rectangle(img, box.tl(), box.br(), Scalar(0, 0, 255), 1);
}

// Funktion für das Auswerten von mouse Event
void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
	Mat& image = *(cv::Mat*) param;

	switch (event) 
	{
		case EVENT_MOUSEMOVE: 
		{
			if (drawingState)
			{
				gRect.width = x - gRect.x;
				gRect.height = y - gRect.y;

				std::cout << "gRect x: " << gRect.x << std::endl;
				std::cout << "gRect y: " << gRect.y << std::endl;
				std::cout << "x: " << x << std::endl;
			}
		}
		break;

		case EVENT_LBUTTONDOWN: 
		{
			if (drawingState == false)
			{

				drawingState = true;
				gRect = Rect(x, y, 0, 0);
			}
			else
			{
				drawingState = false;
			}

		}
		break;

		case EVENT_LBUTTONUP:
		{
			drawingState = false;
		}
		break;
	}
}


//###################### HauptProgramm '##################################


int main(int argc, char** argv)
{
	//Deklaration von Variablen
	Mat oldFrame;
	Mat prev_img_gray;
	Mat srcImage;
	Mat tempImage;
	Mat objetGray;
	Mat new_img_gray;
	Mat img;
	vector<unsigned char> status;
	vector<float> err;

	//Videocapture
	VideoCapture video;
	int camera_Id = 0;

	video.open(camera_Id);
	video.set(CAP_PROP_FRAME_WIDTH, 1800);
	video.set(CAP_PROP_FRAME_HEIGHT, 920);

	vector<KeyPoint> keypoints;
	vector<Point2f> prev_pts;
	vector<Point2f> new_pts;

	namedWindow(WINDOW_NAME, WINDOW_KEEPRATIO);
	setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

	video >> oldFrame;
	cvtColor(oldFrame, prev_img_gray, cv::COLOR_BGR2GRAY);

	// Create a mask image for drawing purposes
	Mat mask = Mat::zeros(oldFrame.size(), oldFrame.type());

	while (video.grab())
	{

		video.retrieve(srcImage);
		int keyPressed = waitKey(10);
		if (!(video.isOpened()))
		{
			break;
		}

		cvtColor(srcImage, new_img_gray, cv::COLOR_BGR2GRAY);

		if (1)
		{
			DrawRectangle(srcImage, gRect);
		}

		if ((gRect.area() > 0) && ((keyPressed == 'a')))
		{

			std::vector<KeyPoint> obj_keypoints;
			auto minHessian = 500;
			Ptr<Feature2D> detector = cv::xfeatures2d::SURF::create(minHessian);
			
			// extraction of Feature
			extract_feature(&prev_pts, &obj_keypoints, &objetGray, srcImage, detector);

		}

		if (prev_pts.size() > 0)
		
		{
			track_feature_points(prev_img_gray, new_img_gray, prev_pts, &new_pts, &err, &status);

		}

		vector<Point2f> good_new;
		for (auto i = 0; i < prev_pts.size(); ++i)
		{
			RNG rng;
			Scalar color;
				
			int r = rng.uniform(0, 256);
			int g = rng.uniform(0, 256);
			int b = rng.uniform(0, 256); 
			if (new_pts.size() > 0 && status[i] == 1)
			{
				good_new.push_back(new_pts[i]);
				line(mask, new_pts[i], prev_pts[i], Scalar(r, g, b), 1);
				circle(srcImage, prev_pts[i], 3, Scalar(0, 0, 255),-1);
				
			}

		}

		
		add(srcImage, mask, img);

		resizeWindow(WINDOW_NAME, img.cols,img.rows);
		imshow(WINDOW_NAME, img);
		imwrite("featureTracking.png", img);
		if (waitKey(1) == 27)
			break;

		prev_img_gray = new_img_gray.clone();
		prev_pts = good_new;
	}
	
	//End
	return 0;
}

