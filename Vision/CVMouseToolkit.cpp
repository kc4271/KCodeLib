#include "CVMouseToolkit.h"

bool CrossRect(cv::Rect rect, cv::Rect mask, cv::Rect &cross) {
	cv::Rect t;
	
	if(mask.x >= rect.x + rect.width) return false;
	if(mask.y >= rect.y + rect.height) return false;
	if(mask.x + mask.width < rect.x) return false;
	if(mask.y + mask.height < rect.y) return false;

	t.x = std::max(rect.x, mask.x);
	t.y = std::max(rect.y, mask.y);
	t.width = std::min(rect.x + rect.width, mask.x + mask.width) - t.x;
	t.height = std::min(rect.y + rect.height, mask.y + mask.height) - t.y;

	cross = t;
	return true;
}

std::string CImageDragger::windowsName;