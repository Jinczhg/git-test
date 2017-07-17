#ifndef FUNCTION_H
#define FUNCTION_H
#include "opencv2/opencv.hpp"

const double EPSILON = 1e-6;

template<class T>
inline void bilinearInterpolate(float* result, const cv::Point2f& p, const T* img, int width, int height, int channels)
{
	if (!img)
		return;
	int xx = int(p.x), yy = int(p.y), m, n, c, u, v, offset;
	float dx = max(min(p.x - xx, 1), 0), dy = max(min(p.y - yy, 1), 0), s;
	for (c = 0; c<channels; c++)
		result[c] = 0;
	for (m = 0; m <= 1; m++)
	{
		for (n = 0; n <= 1; n++)
		{
			u = min(max(xx + m, 0), width - 1);
			v = min(max(yy + n, 0), height - 1);
			offset = (v*width + u)*channels;
			s = fabs(1 - m - dx)*fabs(1 - n - dy);
			for (c = 0; c<channels; c++)
				result[c] += img[offset + c] * s;
		}
	}
}



#endif