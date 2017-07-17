#ifndef COMFUN_H
#define COMFUN_H

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"  //SurfDescriptorExtractor
#include "opencv2/stitching/stitcher.hpp" //Stitcher

#include <direct.h> //mkdir
#include <windows.h> //<windows.h> confilicts with <limits>
#include <io.h> //access
#include <fstream>
#include <time.h>

using namespace std;
using namespace cv;

const int MaxWidth = 1080;
const int MaxFeaNum = 256;
const int MaxIters = 1000;
const int MinInliers = 9;
const double HomoConf = 0.80;
const double EPSILON = 1e-6;
const double KAPPA = 1e6;

struct ImageShape{
	int w;
	int h;
	int c;
};

struct tracking_header
{
	int first_frame;
	int last_frame;  //last id+1
	int num_features;
};

struct tracking_feature
{
	double x, y;
	int status;
	double data;
};

struct tracking_pnt
{
	double x, y, px, py, pz, pcx, pcy;
	int manual, type3d, ident, hasprev, support;
};

struct tracking_trajectory
{
	int first_frame;
	int last_frame;
	int length;
	vector<Point2f> xy;
	vector<Point2f> change;
	vector<int> feature_idx;
	tracking_trajectory(int frm_id, Point2f pt, int pt_id)
	{
		first_frame = frm_id;
		last_frame = -1;
		length = -1;
		xy.push_back(pt);
		feature_idx.push_back(pt_id);
	}
	~tracking_trajectory(){}
};

struct GMMInfo
{
	EM model;
	Mat labels; //CV_32S
	Mat probs; //CV_64F

	GMMInfo() : model(EM()), labels(Mat()), probs(Mat()){}
	~GMMInfo() {}
};

static bool Video2Images(vector<string>& imagefiles, vector<string>& filenames, const string& videofile, int maxframes=100)
{
	VideoCapture pCapture(videofile);
	if(!pCapture.isOpened())
	{
		printf("Can not open video file %s\n", videofile.c_str());
		return false;
	}
	printf_s("read %s...\n", videofile.c_str());
	
	Mat pFrame, pResize;
	int width = pCapture.get(CV_CAP_PROP_FRAME_WIDTH);
	int height = pCapture.get(CV_CAP_PROP_FRAME_HEIGHT);
	double ratio = double(MaxWidth)/double(min(width, height));
	double fps = pCapture.get(CV_CAP_PROP_FPS);
	int numFrame = min(pCapture.get(CV_CAP_PROP_FRAME_COUNT), maxframes);

	string filepre = videofile;
	filepre.erase(filepre.find_last_of('.'));
	string imagedir = filepre;
	_mkdir(imagedir.c_str());
	filepre = filepre.substr(filepre.find_last_of('/')+1);
	char imgpath[512], imgname[512];

	imagefiles.clear();
	filenames.clear();
	for (int ctFrame = 0; ctFrame < numFrame; ctFrame ++)
	{
		printf_s("    %d of %d\r", ctFrame, numFrame);
		pCapture.read(pFrame);
		if (pFrame.empty())
			break;
		sprintf(imgpath, "%s/%s_%05d.png", imagedir.c_str(), filepre.c_str(), ctFrame);
		if (ratio<1.f)
		{
			resize(pFrame, pResize, Size(0,0), ratio, ratio);
			imwrite(imgpath, pResize);
		}
		else
			imwrite(imgpath, pFrame);
		imagefiles.push_back(string(imgpath));
		sprintf(imgname, "%s_%05d", filepre.c_str(), ctFrame);
		filenames.push_back(imgname);
	}
	return true;
}

static int str_compare(const void *arg1, const void *arg2)
{
	return strcmp((*(std::string*)arg1).c_str(), (*(std::string*)arg2).c_str());
}

template <class T>
inline T EnforceRange(const T& x,const int& MaxValue) 
{
	return min(max(x,0),MaxValue-1);
}

inline Point2f warpPoint(const Point2f& p, const double* h)
{
	double dx = p.x*h[0] + p.y*h[1] + h[2];
	double dy = p.x*h[3] + p.y*h[4] + h[5];
	double dz = p.x*h[6] + p.y*h[7] + h[8];
	if (dz)
		return Point2f(dx/dz, dy/dz);
	else
		return Point2f(dx, dy);
}

template<class T>
inline void bilinearInterpolate(float* result, const Point2f& p, const T* img, int width, int height, int channels)
{
	if (!img)
		return;
	int xx = int(p.x), yy = int(p.y), m, n, c, u, v, offset;
	float dx = max(min(p.x-xx, 1), 0), dy = max(min(p.y-yy, 1), 0), s;
	for ( c = 0; c<channels; c++ )
		result[c] = 0;
	for ( m = 0; m <= 1; m++ )
	{
		for ( n = 0; n <= 1; n++ )
		{
			u = min(max(xx+m, 0), width-1);
			v = min(max(yy+n, 0), height-1);
			offset = (v*width+u)*channels;
			s = fabs(1-m-dx)*fabs(1-n-dy);
			for ( c = 0; c<channels; c++ )
				result[c] += img[offset+c]*s;
		}
	}
}

template <class T1,class T2>
void ComputeMean(int Dim,int NumData,T1* pData,T2* pMean,double* pWeight)
{
	int i,j,k;
	memset(pMean,0,sizeof(T2)*Dim);

	bool IsWeightLoaded=false;
	double Sum;
	if(pWeight!=NULL)
		IsWeightLoaded=true;

	// compute mean first
	Sum=0;
	if(IsWeightLoaded)
		for(i=0;i<NumData;i++)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j]*pWeight[i];
			Sum+=pWeight[i];
		}
	else
	{
		for(i=0;i<NumData;i++)
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j];
		Sum=NumData;
	}
	for(j=0;j<Dim;j++)
		pMean[j]/=Sum;
}

template <class T1,class T2>
void ComputeMeanCovariance(int Dim,int NumData,T1* pData,T2* pMean,T2* pCovariance,double* pWeight)
{
	int i,j,k;
	memset(pMean,0,sizeof(T2)*Dim);
	memset(pCovariance,0,sizeof(T2)*Dim*Dim);

	bool IsWeightLoaded=false;
	double Sum;
	if(pWeight!=NULL)
		IsWeightLoaded=true;

	// compute mean first
	Sum=0;
	if(IsWeightLoaded)
		for(i=0;i<NumData;i++)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j]*pWeight[i];
			Sum+=pWeight[i];
		}
	else
	{
		for(i=0;i<NumData;i++)
			for(j=0;j<Dim;j++)
				pMean[j]+=pData[i*Dim+j];
		Sum=NumData;
	}
	for(j=0;j<Dim;j++)
		pMean[j]/=Sum;

	//compute covariance;
	T2* pTempVector;
	pTempVector=new T2[Dim];

	for(i=0;i<NumData;i++)
	{
		for(j=0;j<Dim;j++)
			pTempVector[j]=pData[i*Dim+j]-pMean[j];
		if(IsWeightLoaded)
		{
			if(pWeight[i]==0)
				continue;
			for(j=0;j<Dim;j++)
				for(k=0;k<=j;k++)
					pCovariance[j*Dim+k]+=pTempVector[j]*pTempVector[k]*pWeight[i];
		}
		else
			for(j=0;j<Dim;j++)
				for(k=0;k<=j;k++)
					pCovariance[j*Dim+k]+=pTempVector[j]*pTempVector[k];
	}
	for(j=0;j<Dim;j++)
		for(k=j+1;k<Dim;k++)
			pCovariance[j*Dim+k]=pCovariance[k*Dim+j];

	for(j=0;j<Dim*Dim;j++)
		pCovariance[j]/=Sum;

	delete []pTempVector;
}

#endif
