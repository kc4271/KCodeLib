#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cstdio>
#include <string>


class CContourExtractor {
	cv::Mat srcImg;
	
	int width, height;
	cv::Mat contours;
public:
	void setImg(cv::Mat &img) {
		if(img.channels() != 1) {
			cv::cvtColor(img, srcImg, CV_BGR2GRAY);
		} else {
			srcImg = img.clone();
		}

		assert(srcImg.channels() == 1);
		if(srcImg.depth() != CV_8U) {
			fprintf(stderr, "%s:%d CContourExtractor: img depth is not 8 Bit\n", __FILE__, __LINE__);
			exit(0);
		}

		width = srcImg.size().width;
		height = srcImg.size().height;
	}

	cv::Mat getContourPixels(bool inner = false) {
		contours = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
		
		cv::Mat tcontours = contours.clone();
		cv::vector<cv::Point> points;


		int sX[] = {1,1,0,-1,-1,-1,0,1};
		int sY[] = {0,1,1,1,0,-1,-1,-1};
		int sl = sizeof(sX) / sizeof(int);

		cv::Mat_<uchar> img_ptr(srcImg);
		
		for(int y = 0;y < height;y++) { for(int x = 0;x < width;x++) {
			if(!img_ptr(y, x)) continue;

			for(int z = 0;z < sl;z++) {
				int newX = x + sX[z];
				int newY = y + sY[z];

				if(!(newX >= 0 && newY >= 0 && newX < width && newY < height)) continue;

				if(!img_ptr(newY, newX)) {
					tcontours.at<uchar>(y,x) = 255;
					points.push_back(cv::Point(x, y));
					break;
				}
			}
		}}

		if(inner) {
			contours = tcontours;
		} else {
			for(int i = 0;i < points.size();i++) {
				int x = points[i].x;
				int y = points[i].y;
				
				for(int z = 0;z < sl;z++) {
					int newX = x + sX[z];
					int newY = y + sY[z];

					if(!(newX >= 0 && newY >= 0 && newX < width && newY < height)) continue;

					if(!img_ptr(newY, newX)) {
						contours.at<uchar>(newY,newX) = 255;
					}
				}
			}
		}

		 
		return contours;
	}

	std::vector< std::vector<cv::Point2d> > getOrderedContour() {
		getContourPixels(false);

		std::vector< std::vector<cv::Point2d> > lines;

		cv::Mat visited(height, width, CV_8UC1, cv::Scalar(0));
		
		int sX8[] = {1,1,0,-1,-1,-1,0,1};
		int sY8[] = {0,1,1,1,0,-1,-1,-1};
		int sX[] = {1, 0, -1, 0};
		int sY[] = {0, 1, 0, -1};
		int sl = sizeof(sX) / sizeof(int);
		int sl8 = sizeof(sX8) / sizeof(int);
		cv::Mat_<uchar> contour_ptr(contours);
		cv::Mat_<uchar> visited_ptr(visited);
		
		for(int y = 0;y < height;y++) { for(int x = 0;x < width;x++) {
			if(!contour_ptr(y, x) || visited_ptr(y, x)) continue;

			int curX = x, curY = y;
			int lastDir = 0;
			std::vector<cv::Point2d> points;
			while(true) {
				if(!visited_ptr(curY, curX)) {
					points.push_back(cv::Point2d(curX, curY));
					visited_ptr(curY, curX) = true;
				}
	
				bool move = false;
				bool ret = false;
				for(int z = 0;z < sl;z++) {
					int dir = (lastDir + z) % sl;
					int newX = curX + sX[dir];
					int newY = curY + sY[dir];

					if(!(newX >= 0 && newY >= 0 && newX < width && newY < height)) continue;

					if(newX == x && newY == y) {
						ret = true;
					}

					if(contour_ptr(newY, newX) && !visited_ptr(newY, newX)) {
						move = true;

						curY = newY;
						curX = newX;
						lastDir = dir;
						break;
					}
				}

				if(ret) break;

				if(!move) {
					points.erase(points.end() - 1);

					curX = points[points.size() - 1].x;
					curY = points[points.size() - 1].y;
				}
			}

			for(int i = 0;i < points.size();i++) {
				curX = points[i].x;
				curY = points[i].y;
				
				for(int z = 0;z < sl8;z++) {
					int newX = curX + sX8[z];
					int newY = curY + sY8[z];
					if(!(newX >= 0 && newY >= 0 && newX < width && newY < height)) continue;
					if(contour_ptr(newY, newX)) {
						visited_ptr(newY, newX) = true;	
					}
					
				}
			}

			lines.push_back(points);


		}}
		return lines;
	}
};