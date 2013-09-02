#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdio>
#include <string>
#include <ctime>
#include "ContourExtraction.h"
using namespace std;

int main(int argc, char **argv) {
	cv::Mat im = cv::imread("test.png");
	CContourExtractor ce;
	ce.setImg(im);
	bool inner = false;

	cv::Mat cont(im.size().height, im.size().width, CV_8U, cv::Scalar(0));
	int count = 0;
	clock_t t;
	while(true) {
		t = clock();
		std::vector< std::vector<cv::Point2d> > cc = ce.getOrderedContour();


		for(int i = 0;i < cc.size();i++) {
			//check inner loop
			if(cc[i].size()) {
				cv::Point2d v = cc[i][0];
				v.y -= 1;
				if(im.at<cv::Vec3b>(v.y, v.x)[0] + im.at<cv::Vec3b>(v.y, v.x)[1] + im.at<cv::Vec3b>(v.y, v.x)[2]) {
					continue;
				}
			}
 			for(int j = 0;j < cc[i].size();j++) {
				cont.at<uchar>(cc[i][j].y, cc[i][j].x) = count++ % 255;
				
			}
		}

		cv::imshow("contours", cont);
		
		cout<<1.0/((double)(clock() - t) / CLOCKS_PER_SEC)<<endl;
		//imshow("show", show);
		int key = cv::waitKey(100);
		switch(key) {
		case 'a': count++;
		}
	}


	return 0;
}