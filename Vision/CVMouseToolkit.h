#ifndef __CV_MOUSE_TOOLKIT__
#define __CV_MOUSE_TOOLKIT__
#include <opencv2/opencv.hpp>

//select a rect using mouse
class CSelectRect {
private:
	cv::Rect selectedRect;
	bool flag;

public:
	CSelectRect():flag(false) {}

	void ButtonDown(int x, int y) {
		selectedRect.x = x;
		selectedRect.y = y;
		flag = true;
	}

	void ButtonUp(int x, int y) {
		selectedRect.width = x - selectedRect.x;
		selectedRect.height = y - selectedRect.y;
		flag = false;
	}

	void MouseMove(int x, int y) {
		if(!flag) return;
		selectedRect.width = x - selectedRect.x;
		selectedRect.height = y - selectedRect.y;
	}

	cv::Rect GetSelectedRect() {
		cv::Rect tRect = selectedRect;
		if(tRect.width < 0) {
			tRect.x += tRect.width;
			tRect.width = -tRect.width;
		}
		if(tRect.height < 0) {
			tRect.y += tRect.height;
			tRect.height = -tRect.height;
		}
		return tRect;
	}

	bool IsRunning() {
		return flag;
	}
};

//drag a selected rect
class CDragRect {
private:
	cv::Point offset;
	cv::Rect rect;
	bool flag;
	bool inited;

public:
	CDragRect():flag(false),inited(false) {}

	void InitRect(cv::Rect rect_) {
		rect = rect_;
		inited = true;
	}

	void ButtonDown(int x, int y) {
		if(rect.contains(cv::Point(x, y))) {
			flag = true;
			offset.x = rect.x - x;
			offset.y = rect.y - y;
		}
	}

	void ButtonUp(int x, int y) {
		flag = false;
	}

	void MouseMove(int x, int y) {
		if(!flag) return;
		rect.x = x + offset.x;
		rect.y = y + offset.y;
	}

	cv::Rect GetDraggedRect() {
		CV_Assert(true == inited); //rect should be inited!
		return rect;
	}

	bool IsRunning() {
		return flag;
	}

	bool HasInited() {
		return inited;
	}
};

//calculate cross area
//[IN] cv::Rect rect
//[IN] cv::Rect mask
//[OUT] cross area in rect
//[RES] success or fail;
bool CrossRect(cv::Rect rect, cv::Rect mask, cv::Rect &cross);

class CImageDragger {
public:
	static std::string windowName;

	static void setup(const char *windowName_, cv::Mat &img) {
		windowName = windowName_;
		cv::setMouseCallback(windowName, &drag, &img);
	}

	static void drag(int event_, int x, int y, int flags, void* param) {
		cv::Mat img = ((cv::Mat *)param)->clone();
		static CSelectRect selectRect;
		static CDragRect dragRect;
		static cv::Mat imgSelected;
		cv::Rect imgRect(0, 0, img.size().width, img.size().height);

		switch(event_) {
		case CV_EVENT_MOUSEMOVE:
			if(!imgRect.contains(cv::Point(x, y))) break;
			selectRect.MouseMove(x, y);
			if(selectRect.IsRunning()) {
				cv::Rect rect = selectRect.GetSelectedRect();
				cv::rectangle(img, rect, cv::Scalar(255, 0, 0));
				cv::imshow(windowName, img);
			}

			dragRect.MouseMove(x, y);
			if(dragRect.HasInited()) {
				if(dragRect.GetDraggedRect().contains(cv::Point(x,y))) {
					cv::rectangle(imgSelected, cv::Rect(0, 0, imgSelected.size().width, imgSelected.size().height), cv::Scalar(0, 0, 255));
				} else {
					cv::rectangle(imgSelected, cv::Rect(0, 0, imgSelected.size().width, imgSelected.size().height), cv::Scalar(255, 0, 0));
				}

				cv::Rect rect = dragRect.GetDraggedRect();
				cv::Rect subRect;
				CV_Assert(CrossRect(rect,cv::Rect(0,0,img.size().width, img.size().height), subRect));
				cv::Mat imgSubRect = cv::Mat(imgSelected, cv::Rect(subRect.x - rect.x, subRect.y - rect.y, subRect.width, subRect.height));
				cv::Mat area = cv::Mat(img, subRect);
				cv::addWeighted(area, 0.3, imgSubRect, 0.7, 0.0, area);
				cv::imshow(windowName, img);
			}
			break;
		case CV_EVENT_RBUTTONDOWN:
			selectRect.ButtonDown(x, y);
			break;
		case CV_EVENT_RBUTTONUP:
			selectRect.ButtonUp(x, y);
			imgSelected = cv::Mat(img, selectRect.GetSelectedRect()).clone();
			dragRect.InitRect(selectRect.GetSelectedRect());
			break;
		case CV_EVENT_LBUTTONDOWN:
			dragRect.ButtonDown(x, y);
			break;
		case CV_EVENT_LBUTTONUP:
			dragRect.ButtonUp(x, y);
			break;
		}
	}
};

#endif