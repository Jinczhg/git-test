#ifndef WARPING_H
#define WARPING_H

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"  //SurfDescriptorExtractor
#include "opencv2/stitching/stitcher.hpp" //Stitcher
#include "ComFun.h"
#include <Core>
#include <Dense>
#include <Sparse>

using namespace Eigen;
using namespace cv;

class Warping{
public:
	Warping() = default;
	Warping(Size size1, Size size2, int xquads, int yquads, float lambda);
	~Warping();

	Vec4f calcQuadCoordinates(const Point2f& v00, const Point2f& v01, const Point2f& v10, const Point2f& v11, const Point2f& pt);

	Vec2f calcLocalCoordinates(const Point2f& v1, const Point2f& v2, const Point2f& v3);

	void addSmoothConstraints(MatrixXf& A, int& cnt_eqns, int id1, int id2, int id3, float u, float v, float w, bool clockwise);
	
	void solve(const detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, const Mat& weight = Mat());
	
	void solve(Ptr<detail::RotationWarper> warper, const Mat& K, const Mat& R, const detail::ImageFeatures& feature, const Mat& mask, float smoother);
	
	void warp(Mat& result, const Mat& img);

public:
	Size _src_size, _dst_size;
	int _xquads, _yquads;
	float _quadWidth, _quadHeight, _lambda;
	
	VectorXf __vertices1, __vertices2;
	vector<Mat> __homos;

};

#endif