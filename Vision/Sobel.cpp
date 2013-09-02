#include "ximage.h"
#include <cmath>

void Sobel(CxImage *img)
{
	static int sobel[8][9]={
		{-1,-2,-1,0,0,0,1,2,1},
		{0,-1,-2,1,0,-1,2,1,0},
		{1,0,-1,2,0,-2,1,0,-1},
		{2,1,0,1,0,-1,0,-1,-2},
		{1,2,1,0,0,0,-1,-2,-1},
		{0,1,2,-1,0,1,-2,-1,0},
		{-1,0,1,-2,0,2,-1,0,1},
		{-2,-1,0,-1,0,1,0,1,2}
	};

	int width = img->GetWidth();
	int height = img->GetHeight();
	CxImage Out(width,height,24);
	BYTE *buf = new BYTE[width * height];
	
	for(int i = 0;i < height;i++)
	{
		for(int j = 0;j < width;j++)
		{
			buf[i * width + j] = img->GetPixelGray(j,i);
		}
	}
	
	int MAXX,d;
	for(int i = 1;i < height - 1;i++)
	{
		for(int j = 1;j < width - 1;j++)
		{
			MAXX = 0;
			for(int z = 0; z < 8;z++)
			{
				d = 0;
				for(int x = -1;x <= 1;x++)
				{
					for(int y = -1;y <= 1;y++)
					{
						int X = j + y;
						int Y = i + x;
						d += sobel[z][(1 + x) * 3 + (1 + y)] * buf[Y * width + X];
					}
				}
				if(d > MAXX)
						MAXX = d;
			}
			if(MAXX > 255) MAXX = 255;
			Out.SetPixelColor(j,i,RGB(MAXX,MAXX,MAXX));
		}
	}
	Out.Save("myResult.bmp",CXIMAGE_FORMAT_BMP);
	delete []buf;
}

int main()
{
	CxImage img("test.png",CXIMAGE_FORMAT_PNG);
	Sobel(&img);


	return 0;
}