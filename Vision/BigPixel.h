#pragma once

#include <opencv2/core/core.hpp>
#include <cstdio>
#include <string>

struct CBigPixel {
	int cellX, cellY;
	int cellW, cellH;
	cv::Mat img;
	
	CBigPixel() {
		setCellSize(1, 1, 1, 1);
	}

	void setCellSize(int cW, int cH, int cX = 20, int cY = 20) {
		cellX = cX;
		cellY = cY;
		cellW = cW;
		cellH = cH;

		img = cv::Mat(cellY * cellH, cellX * cellW, CV_8UC3, cv::Scalar(100, 100, 100));

		clean();
	}

	void clean() {
		for(int y = 0;y < cellH; y++) {
			for(int x = 0;x < cellW; x++) {
				setColor(x, y, cv::Vec3b(0, 0, 0));
			}
		}
	}

	void setColor(int x, int y, cv::Vec3b c) {
		if(x >= 0 && x < cellW && y >= 0 && y < cellH) {
			for(int j = 0;j < cellY - 1;j++) {
				for(int i = 0;i < cellX - 1;i++) {
					img.at<cv::Vec3b>(cellY * y + j, cellX * x + i) = c;
				}
			}
		} else {
			fprintf(stderr, "%s:%d CBigPixel: x or y out of range, x:%d y:%d width:%d height:%d \n", __FILE__, __LINE__, x, y, cellW, cellH);
		}
	}

	void setColor(int x, int y, const char color) {
		cv::Vec3b c(255, 0, 255);
		switch(color) {
		case 'R':
		case 'r': c = cv::Vec3b(0, 0, 255); break;
		case 'G':
		case 'g': c = cv::Vec3b(0, 255, 0); break;
		case 'B':
		case 'b': c = cv::Vec3b(255, 0, 0); break;
		case 'W':
		case 'w': c = cv::Vec3b(255, 255, 255); break;
		case 'Y':
		case 'y': c = cv::Vec3b(0, 216, 255); break;
		}
		
		if(x >= 0 && x < cellW && y >= 0 && y < cellH) {
			for(int j = 0;j < cellY - 1;j++) {
				for(int i = 0;i < cellX - 1;i++) {
					img.at<cv::Vec3b>(cellY * y + j, cellX * x + i) = c;
				}
			}
		} else {
			fprintf(stderr, "%s:%d CBigPixel: x or y out of range, x:%d y:%d width:%d height:%d \n", __FILE__, __LINE__, x, y, cellW, cellH);
		}
	}

	void drawText(int x, int y, const std::string &str, int fontsize = 1) {
		if(x >= 0 && x < cellW && y >= 0 && y < cellH) {
			cv::putText(img, str, cv::Point(x * cellX + 2, y * cellY + 13), cv::FONT_HERSHEY_PLAIN, fontsize, cv::Scalar(0,0,0),1);
		} else {
			fprintf(stderr, "%s:%d CBigPixel: x or y out of range, x:%d y:%d width:%d height:%d \n", __FILE__, __LINE__, x, y, cellW, cellH);
		}
	}

	cv::Mat getImg() {
		return img;
	}

	void show(int wait = 1) {
		cv::imshow("CBigPixel", img);
		cv::waitKey(wait);
	}

};