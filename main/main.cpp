#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include <iostream>
#include <thread>
#include <sstream>
#include <deque>
#include <queue>
#include <mutex>
#include <condition_variable>

using namespace std;
template <typename T>
class Queue
{
 public:
	T pop() 
	{
		clock_t begin_time = clock();
		std::unique_lock<std::mutex> mlock(mutex_);
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		auto val = queue_.front();
		queue_.pop();
		return val;
	}

	void pop(T& item)
	{
		std::unique_lock<std::mutex> mlock(mutex_);
		
		while (queue_.empty())
		{
			cond_.wait(mlock);
		}
		item = queue_.front();
		queue_.pop();
	}

	void push(const T& item)
	{
		{
			std::unique_lock<std::mutex> mlock(mutex_);
			queue_.push(item);
		}		
		cond_.notify_one();
	}
	
	Queue()=default;
	
	// Disable copying
	Queue(const Queue&) = delete;            
	
	// Disable assignment
	Queue& operator=(const Queue&) = delete; 
	
	int rows;
	int cols;
	
 private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_;
};

void dilate(Queue<unsigned char *> &Q)
{
	int valThreshold = 150;
    double timeSum = 0;
    int frameNumber = 1;
   
    // Read in all the images to the vector
    string filename = "//home//firefly//Desktop//rockChip//resource//keytyping.avi";
    cv::VideoCapture capture(filename);
    vector<cv::Mat> frameColors;
    cv::Mat frameColor;
    int N = 300;
	while(N--)
	{
		capture >> frameColor;
		cvtColor(frameColor, frameColor, CV_BGR2GRAY);	
		frameColors.push_back(frameColor);
	}
	
	while(1)
	{
		clock_t begin_time = clock();
		frameColor  = frameColors[frameNumber];
		
		// Generate the mask and do the dilation
		cv::threshold(frameColor, frameColor, valThreshold, 1, CV_THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE , cv::Size(3,3));
		cv::morphologyEx(frameColor, frameColor, cv::MORPH_OPEN, element,cv::Point(-1,-1),1);
		Q.push( (unsigned char*)(frameColor.data));
		Q.rows = frameColor.rows;
		Q.cols = frameColor.cols;
		
		// To show the result
		timeSum = (float(clock() - begin_time) / CLOCKS_PER_SEC ) ;
		cout << "Total1 :" + to_string(timeSum)<<endl;
		cout << "fps1 = " + to_string(1 / (timeSum)) << endl;
		frameNumber++;
	}
	
}

void getresult(Queue<unsigned char *> &Q)
{
	double timeSum = 0 ;
	int frameNumber = 1;
	int componentsThr = 150;
	double numComponents = 0;
	int numArea;
	cv::Mat frameConnected;
	unsigned char* fb;
	
	while(1)
	{
		clock_t begin_time = clock();
		
		// Get the frameColor from the camera
		Q.pop(fb);
		cv::Mat frameColor(Q.rows, Q.cols,CV_8UC1, fb);
		
		// Do the connection
		cv::connectedComponents(frameColor, frameConnected, 4);
		cv::minMaxLoc(frameConnected, NULL, &numComponents);
		
		// NumArea means how many areas are there
		numArea = (int)numComponents;
		
		// To get the histogram for the connected area 
        int* histoPtr = (int*)malloc(sizeof(int) * (numArea + 1)); 
        memset(histoPtr, 0, sizeof(int) * (numArea + 1)); 
        for (int r = 0; r < frameConnected.rows; r++) 
        { 
            unsigned int* frameConnectedPtr = frameConnected.ptr<unsigned int>(r); 
            for (int c = 0; c < frameConnected.cols; c++) 
            { 
                histoPtr[frameConnectedPtr[c]]++; 
            } 
        } 
        
        // If the histogram is smaller than the threshold, then all the pixel fall in this bin would be erased 
        for (int r = 0; r < frameConnected.rows; r++) 
        { 
            unsigned int* frameConnectedPtr = frameConnected.ptr<unsigned int>(r); 
            unsigned char* frameColorPtr = frameColor.ptr<unsigned char>(r); 
            for (int c = 0; c < frameConnected.cols; c++) 
            { 
                if (histoPtr[frameConnectedPtr[c]] < componentsThr) 
                { 
                    frameColorPtr[c] = 0; 
                } 
            } 
        } 
        
		// Init result vector
		vector<cv::Point2i> pointsPos(numArea + 1);
		for (int i = 0; i < numArea + 1; i++)
		{
			pointsPos.at(i).x = 0;
			pointsPos.at(i).y = 0;
		}

		// Find the sum cooridnates of all points! 
        for (int r = 0; r < frameConnected.rows; r++) 
        { 
            unsigned int* frameConnectedPtr = frameConnected.ptr<unsigned int>(r); 
            unsigned char* frameColorPtr = frameColor.ptr<unsigned char>(r); 
            for (int c = 0; c < frameConnected.cols; c++) 
            { 
                if (frameConnectedPtr[c] && frameColorPtr[c]) 
                { 
                    pointsPos.at(frameConnectedPtr[c]).x += c; 
                    pointsPos.at(frameConnectedPtr[c]).y += r; 
                } 
            } 
        }
        
		// Average the position
		for (int i = 1; i < numArea + 1; i++)
		{
			pointsPos.at(i).x /= histoPtr[i];
			pointsPos.at(i).y /= histoPtr[i];
		}
		free(histoPtr);
		
		// Give out the result
		for (int i = 1; i < numArea + 1; i++)
		{
			if (pointsPos.at(i).x && pointsPos.at(i).y)
			{
				cout <<"There is a point at ("+ to_string(pointsPos.at(i).x) + "," + to_string(pointsPos.at(i).y)+ ")" << endl;
			}
		}
		
		// To show the result
		timeSum = (float(clock() - begin_time) / CLOCKS_PER_SEC ) ;
		cout << "Total2 :" + to_string(timeSum)<<endl;
		cout << "fps2 = " + to_string(1 / (timeSum)) << endl;
		frameNumber++;	
	}
}

int main()
{
	//==================================
	// Multi threading
	//================================== 
	Queue<unsigned char *> Q;	
	std::thread thread1(dilate, std::ref(Q));
	std::thread thread2(getresult, std::ref(Q));
	thread2.join();
	
	return 0;
}

