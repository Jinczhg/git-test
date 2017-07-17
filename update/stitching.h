#ifndef STITCHING_H
#define STITCHING_H

#include "opencv2/opencv.hpp"
#include "opencv2/nonfree/features2d.hpp"  //SurfDescriptorExtractor
#include "opencv2/stitching/stitcher.hpp" //Stitcher
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include <vector>
#include <Core>
#include <Dense>
#include <Sparse>
#include <direct.h> //mkdir
#include <windows.h> //<windows.h> confilicts with <limits>
#include <fstream>
#include <time.h>
#include <string.h> 
//#include <math.h>
#include <iostream>
#include <memory.h> //vector中有memory
#include "Warping.h"
#include "ComFun.h"
#include "graph.h"
//#include "homographyfunction.h"



#define RANSAC_THRE0 3.0 //全局H
#define RANSAC_THRE1 3.0 //局部H

using namespace std;
using namespace cv;
using namespace Eigen;

class Stitching{
public:
	Stitching(string imgname0, string imgname1);
	Stitching(string imgname0, string imgname1, string simgname0, string simgname1, string feaname, string matname);
	Stitching(string imgname0, string imgname1, string feaname, string matname);

	~Stitching();
	void ShowMatches(string outDir, bool show_outlier=false);
	void ShowMatches(string outDir, vector<uchar> select);
	void ShowMatches(string outDir, string outFile, vector<uchar> select);
	void ShowMatches(string outDir, const vector<detail::ImageFeatures> &fea, const vector<detail::MatchesInfo> &mat);
	void ShowFeatures(string outDir, const VectorXf& labels);
	void FeatureDetection();
	void ReadFeaturesMatches(string feaname, string matname);
	bool Stitch(string outDir);
	bool calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2);
	bool calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, vector<uchar> selectMask);
	bool calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, const float rantresh);
	
	void calcDualMatches(detail::MatchesInfo& dm, const detail::MatchesInfo& m);
	bool Composite_Opencv(string outDir);
	inline void calcPoint_after_H(Point2f p_src, Point2f& p_dst, const double* h);
	Mat WarpImg(const Mat img_src, Point2i &TL_rect, Matrix<double, 3, 3, RowMajor> H);
	void Findseam_and_Blend(Mat& result, Mat& result_mask, vector<Mat>& Ims, vector<Mat>& Mks, vector<Point>& corners, int blend_type);
	void Blend(Mat& result, Mat& result_mask, vector<Mat>& Ims, vector<Mat>& Mks, vector<Point>& corners, int blend_type);
	Mat CalLinearBlend(Mat im0, Mat im1, Mat mask0, Mat mask1, Point2i corner0, Point2i corner1);

	//根据拟合误差的方法
	void Select_by_FitErr(string outDir, float threshold);
	void UpdateH_by_SelectMask(vector<uchar> selectMask);
	void calParallax_and_Select(float parallax_thre);
	void ShowParallax(string outDir);
	void calFitErr(string outDir,vector<uchar> selectMask);
	void initInliersMask();
	
	void GetParallaxImg(string outDir);
	void ParallaxPropagation(string outDir);
	void ConstructPatch(Point2f centPnt, int psize, Mat srcImg, float* patchData);
	void PropagateE2X(VectorXf W, VectorXf E, VectorXf& X, Matrix<float, Dynamic, Dynamic, RowMajor> Z,int numPnts,int numEdges);
	
	void anPropagation(Mat& outImg, string &outName, Mat &inImg, Mat &inMask, const detail::MatchesInfo& machesInfo, const detail::ImageFeatures imageFeatures, bool is_queryIdx);
	void xuPropagation(string outName, Mat inImg, const detail::MatchesInfo& machesInfo, const detail::ImageFeatures imageFeatures, bool is_queryIdx);
	float calAffinitybtPatches(float* patchData0, float* patchData1);
	void PatchInitialization_on_Pixel(Point2i pos, Mat img, float* patchdata);

	void FindSeambyGraphcut_Parallax(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1);
	void FindSeambyGraphcut_PatchDifference(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1);
	void liFindSeambyGraphcut(const vector<Mat>& images, const vector<Point>& corners, vector<Mat>& masks, Mat select_mask, int selected);
	bool Composite_Graphcut(string outDir);
	void AlignmentbyH(Matrix<double, 3, 3, RowMajor> H);
	void SetStripeImg(Mat &Im, uchar colBeg);
	void SetPlaidImg(Mat &Im, uchar colBeg, uchar colEnd);
	void assignVector(string &inFile, vector<uchar> &inVec);
	Mat findHomography_by_weight(InputArray _points1, InputArray _points2,
		int method, double ransacReprojThreshold, OutputArray _mask, vector<float> weightVec);
	void calAffinity();
	void ShowAlignQuality(string outDir, const vector<detail::ImageFeatures> &fea, const vector<detail::MatchesInfo> &mat, 
		const Mat& seamMask, const Mat& seamQualityMat, const Mat& canvasIm);
	void generateCanvasMasks(const Mat& im0, const Mat& im1, const Point2i& corner0, const Point2i& corner1, Mat &outIm0, Mat &outImg1);
	void generateCanvasImgs(const Mat& im0, const Mat& im1, const Point2i& corner0, const Point2i& corner1, Mat &outIm0, Mat &outImg1);
	void convert_to_CanvasCoordinate(const vector<Point2i>& corners, Point2f& inPt, Point2f& outPt,bool in_RefImg=true);
	Mat generateSeamMask_on_Canvas(string outDir);


	bool calcHomoFromOutliers(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, vector<uchar> outliersMask);
	void ShowInliersOfOutliers(string outDir, vector<uchar> &outliersMask, vector<uchar> &inliersMaskOfOutliers);
	void initCluster2();
	Rect transCanvasRect0ToOriginalRect0(Rect &inRect, Matrix<double, 3, 3, RowMajor> H);//配准图像上rect区域对应的im0上的rect区域
	Rect transCanvasRect1ToOriginalRect1(Rect &inRect, Matrix<double, 3, 3, RowMajor> H);//配准图像上rect区域对应的im1上的rect区域

	bool initLocalFeasMats(string outDir, vector<Point> inSeamSection, Matrix<double, 3, 3, RowMajor> H, 
		vector<detail::ImageFeatures> &outFeas, vector<detail::MatchesInfo> &outMats);
	bool addLocalFeasMats(string outDir, vector<Point> inSeamSection, Matrix<double, 3, 3, RowMajor> H,
		vector<detail::ImageFeatures> &outFeas, vector<detail::MatchesInfo> &outMats);
	bool calcHomoFromMatches_8PointMethod(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2);
	vector<Point> getSeamSection(string outDir, Mat &seamQualityMap, Mat &seamMask, int selConIdx = 0);
	vector<Point> getEndPoints(vector<Point> &inPoints);
	float calEdgeCostWeight(Point &pt,vector<Point2f> matFeas);
	void FindSeambyGraphcut_WeightedPatchDifference(string outDir, const vector<Mat>& imgs, const vector<Mat>& masks, const vector<Point2i>& corners, 
		vector<detail::ImageFeatures> &feas, vector<detail::MatchesInfo> &mats, const vector<Mat>& masks_avoid);
	void FindSeambyGraphcut_WeightedPatchDifference(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1);
	void FindSeambyGraphcut_WeightedRect(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1);
	Mat convertoColoredEdgeImage(const Mat& inImg);
	void ShowAffinityImg_4ch(string outDir, const vector<uchar> &clusterMask, const Mat& seamMask);
	Mat wellAlignedRegion(string outDir, vector<detail::ImageFeatures> &fea, vector<detail::MatchesInfo> &mat);
	bool isSeamPassAlignedRegion(Mat &seam, Mat &region);
	bool isHomoCorrect(double* h_data);
	Mat getMaskfromPoints(Size inSize, vector<Point> &pts);
	vector<Point> getPointsfromMask(Mat &img);
	void MixFeasMats(vector<detail::ImageFeatures> &inFeas, vector<detail::MatchesInfo> &inMats, vector<detail::ImageFeatures> &addFeas, vector<detail::MatchesInfo> &addMats);
	Mat getSeamQualityMat(const Mat& seamMask, const Mat& overlapRegionMask);
	bool iteration(string outDir, int idx);
	Mat getOverlapRegionMask(const vector<Mat>& masksWarped, const vector<Point>& corners);
	Point canPt2orgPt0(Point canPt, Matrix<double, 3, 3, RowMajor> H, vector<Point> corners);
	Point canPt2orgPt1(Point canPt, vector<Point> corners);
	//局部配准方法的函数
	Mat WarpImg(const Mat imgSrc, Point2i& outCorner, const Warping& meshWarper);
	Mat gridWarpImg(const Warping& meshWarper);
	void AlignmentbyMesh(Warping& meshWarper, Mat prewarpImg, Mat prewarpMask);
	bool addLocalFeasMats(string outDir, vector<Point> inSeamSection, const Warping& meshWarper, vector<detail::ImageFeatures> &features, vector<detail::MatchesInfo> &matches);
	Point canPt2orgPt0(Point canPt, const Warping& meshWarper, vector<Point> &corners);
	Point orgPt02canPt(Point orgPt, const Warping& meshWarper, vector<Point> &corners);
	void ShowAlignQuality(string outDir, const vector<detail::ImageFeatures> &fea, const vector<detail::MatchesInfo> &mat,
		const Mat& seamMask, const Mat& seamQualityMat, const Mat& canvasIm, const Warping& meshWarper);
	Mat wellAlignedRegion(string outDir, vector<detail::ImageFeatures> &fea, vector<detail::MatchesInfo> &mat,
		const Warping& meshWarper);
	Point2f orgPt02warpPt(Point2f orgPt, const double* homo, vector<Point> &corners);
	Point2f warpPt2orgPt0(Point2f warpPt, const double* homo, vector<Point> &corners);
	float getSeamQuality(const Mat& seamMask, const Mat& seamQualityMat);

public:
	int _width, _height, _channels;
	int _canvas_width, _canvas_height;
	int _num_features0, _num_features1, _num_features, _num_matched, _num_edges, _num_selmat;
	int _num_patch0, _num_patch1, _num_patch;
	int _psize,_wsize,_pnum,_pdim;
	Mat __img0, __img1,__result,__simg0,__simg1,__smooth_result;
	//Mat __simg0, __simg1;
	vector<detail::ImageFeatures> __features;
	vector<detail::MatchesInfo> __matches;
	VectorXf __W, __G, __X;
	Matrix<float, Dynamic, Dynamic, RowMajor> __Z;
	Matrix<float, Dynamic, Dynamic, RowMajor> __Patch;
	vector<int> __idx_Patch;//存储特征点(patch中心)标号	
	Matrix<double, 3, 3, RowMajor> _H;

	//根据拟合误差的方法
	vector<uchar> _inliers_mask, _outliers_mask,_select, _cluster1, _cluster2;
	vector<uchar> _inliers_mask_of_outliers;
	vector<float> _weight;
	Matrix<float, Dynamic, Dynamic, RowMajor> _Mat_Parallax;
	Mat _prlx_img0, _prlx_img1;
	vector<Mat> __images_warped, __smooth_warped, __masks_warped;
	vector<Point> __corners;
	vector<Point2f> __fit_err;

	//rolling select方法
	vector<uchar> _num_matches;
	//局部配准方法
	Warping _meshwarper;

};

#endif