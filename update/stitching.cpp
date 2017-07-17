#include "stitching.h"
#include "ComFun.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "Warping.h"
#include <Core>
#include <Dense>
#include <Sparse>
#include <algorithm>

using cv::detail::ImageFeatures;
using cv::detail::MatchesInfo;
using cv::dilate;
using cv::imwrite;

#define sigma_spatial_init 40
#define sigma_color_init 6


float bad_thre = 0.6;
int dis_thre = 10;
float dilate_times = 1.5;
float _sigma_spatial = sigma_spatial_init;
float _sigma_color = sigma_color_init;
int xquads = 4, yquads = 4;
float less_rigid = 0.01, more_rigid = 0.7;

Mat __seam_mask;
vector<Point> _cpw_corners,_prewarp_corners;
Mat mask0_avoid, mask1_avoid;
Mat aligned_region_mask;
Mat __seam_quality_map;
int selcon_idx = 0;
bool H_flag = false;
bool Pass_flag = false;
bool Dilate_flag = false;
vector<ImageFeatures> feas;
vector<MatchesInfo> mats;
vector<Point> seam_section;


Stitching::Stitching(string imgname0, string imgname1)
{
	__img0 = imread(imgname0);
	__img1 = imread(imgname1);
	//__simg0 = imread(simgname0);
	//__simg1 = imread(simgname1);

	_width = __img0.cols;
	_height = __img0.rows;
	_channels = __img0.channels();
	__images_warped.resize(2);
	__masks_warped.resize(2);
	__corners.resize(2);
	_psize = 5, _wsize = _psize * 2 + 1, _pnum = _wsize*_wsize, _pdim = _pnum*_channels;

	FeatureDetection();
}



Stitching::Stitching(string imgname0, string imgname1, string simgname0, string simgname1, string feaname, string matname)
{
	__img0 = imread(imgname0);
	__img1 = imread(imgname1);
	//__simg0 = imread(simgname0);
	//__simg1 = imread(simgname1);

	//resize(__img0, __img0, Size(0.5*__img0.cols, 0.5*__img0.rows));
	//resize(__img1, __img1, Size(0.5*__img1.cols, 0.5*__img1.rows));

	_width = __img0.cols;
	_height = __img0.rows;
	_channels = __img0.channels();
	__images_warped.resize(2);
	__smooth_warped.resize(2);
	__masks_warped.resize(2);
	__corners.resize(2);
	_psize = 5, _wsize = _psize * 2 + 1, _pnum = _wsize*_wsize, _pdim = _pnum*_channels;

	ReadFeaturesMatches(feaname, matname);
}

Stitching::Stitching(string imgname0, string imgname1, string simgname0, string simgname1)
{
	__img0 = imread(imgname0);
	__img1 = imread(imgname1);
	__simg0 = imread(simgname0);
	__simg1 = imread(simgname1);

	_width = __img0.cols;
	_height = __img0.rows;
	_channels = __img0.channels();
	__images_warped.resize(2);
	__smooth_warped.resize(2);
	__masks_warped.resize(2);
	__corners.resize(2);
	_psize = 5, _wsize = _psize * 2 + 1, _pnum = _wsize*_wsize, _pdim = _pnum*_channels;


	FeatureDetection();
}

Stitching::~Stitching()
{

}

void Stitching::ShowMatches(string outDir, bool show_outlier)
{
	_mkdir(outDir.c_str());
	RNG rng = theRNG();
	const vector<DMatch>& valid_matches = __matches[1].matches;
	const vector<uchar>& inliers_mask = __matches[1].inliers_mask;
	int width = _width, height = 2 * _height;
	Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	Mat imgA(imgAB, Rect(0, 0, _width, _height));
	__img0.copyTo(imgA);
	Mat imgB(imgAB, Rect(0, _height, _width, _height));
	__img1.copyTo(imgB);
	const vector<KeyPoint>& points1 = __features[0].keypoints;
	const vector<KeyPoint>& points2 = __features[1].keypoints;


	for (int k = 0; k < _num_matched; k++)
	{
		if (!show_outlier && !inliers_mask[k])
			continue;
		const DMatch& t = valid_matches[k];
		Point2f p1 = points1[t.queryIdx].pt;
		Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
		Scalar newvalue(rng(256), rng(256), rng(256));
		line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
		circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
		circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
	}
	//for (int k = 0; k < _num_features0; k++)
	//{
	//	if (flag1[k])		continue;
	//	Point2f p1 = points1[k].pt;
	//	circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, Scalar(0,255,0), CV_FILLED, CV_AA);
	//}
	//for (int k = 0; k < _num_features1; k++)
	//{
	//	if (flag2[k])		continue;
	//	Point2f p2 = points2[k].pt + Point2f(0, _height);
	//	circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, Scalar(0,255,0), CV_FILLED, CV_AA);
	//}
	if (show_outlier){
		imwrite(outDir + "/matches_all.jpg", imgAB);
	}
	else{
		imwrite(outDir + "/matches_inliers.jpg", imgAB);
	}

	
}

void Stitching::ShowMatches(string outDir, const vector<ImageFeatures> &fea, const vector<MatchesInfo> &mat)
{
	_mkdir(outDir.c_str());
	//调试，显示特征点和inliers匹配点
	const vector<KeyPoint>& fea_points0 = fea[0].keypoints;
	Mat img0;
	__img0.copyTo(img0);
	for (int k = 0; k < fea[0].keypoints.size(); k++)
	{
		Point2f pt = fea_points0[k].pt;
		circle(img0, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/features0.jpg", img0);
	const vector<KeyPoint>& fea_points1 = fea[1].keypoints;
	Mat img1;
	__img1.copyTo(img1);
	for (int k = 0; k < fea[1].keypoints.size(); k++)
	{
		Point2f pt = fea_points1[k].pt;
		circle(img1, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/features1.jpg", img1);

	int width = _width, height = 2 * _height;
	Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	Mat imgA(imgAB, Rect(0, 0, _width, _height));
	__img0.copyTo(imgA);
	Mat imgB(imgAB, Rect(0, _height, _width, _height));
	__img1.copyTo(imgB);

	if (mat.size()==0)
		imwrite(outDir + "/inliers.jpg", imgAB);
	else{
		RNG rng = theRNG();
		const vector<DMatch>& valid_matches = mat[1].matches;
		const vector<uchar>& inliers_mask = mat[1].inliers_mask;
		const vector<KeyPoint>& points1 = fea[0].keypoints;
		const vector<KeyPoint>& points2 = fea[1].keypoints;
		for (int k = 0; k < mat[1].matches.size(); k++)
		{
			if (!inliers_mask[k])
				continue;
			const DMatch& t = valid_matches[k];
			Point2f p1 = points1[t.queryIdx].pt;
			Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
			Scalar newvalue(rng(256), rng(256), rng(256));
			line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
			circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
			circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
		}
		imwrite(outDir + "/inliers.jpg", imgAB);
	}
}

void Stitching::ShowMatches(string outDir, vector<uchar> select)
{
	_mkdir(outDir.c_str());
	RNG rng = theRNG();
	const vector<DMatch>& valid_matches = __matches[1].matches;
	int width = _width, height = 2 * _height;
	Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	Mat imgA(imgAB, Rect(0, 0, _width, _height));
	__img0.copyTo(imgA);
	Mat imgB(imgAB, Rect(0, _height, _width, _height));
	__img1.copyTo(imgB);
	const vector<KeyPoint>& points1 = __features[0].keypoints;
	const vector<KeyPoint>& points2 = __features[1].keypoints;


	for (int k = 0; k < _num_matched; k++)
	{
		if (!select[k])
			continue;
		const DMatch& t = valid_matches[k];
		Point2f p1 = points1[t.queryIdx].pt;
		Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
		Scalar newvalue(rng(256), rng(256), rng(256));
		line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
		circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
		circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
	}
	//for (int k = 0; k < _num_features0; k++)
	//{
	//	if (flag1[k])		continue;
	//	Point2f p1 = points1[k].pt;
	//	circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, Scalar(0, 255, 0), CV_FILLED, CV_AA);
	//}
	//for (int k = 0; k < _num_features1; k++)
	//{
	//	if (flag2[k])		continue;
	//	Point2f p2 = points2[k].pt + Point2f(0, _height);
	//	circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, Scalar(0, 255, 0), CV_FILLED, CV_AA);
	//}
	char fea_name[512];
	sprintf(fea_name, "%s/seleted_matches.jpg", outDir.c_str());
	imwrite(fea_name, imgAB);
}

void Stitching::ShowFeatures(string outDir, const VectorXf& labels)
{
	_mkdir(outDir.c_str());
	vector<string> paths, names;
	int num_frames = 2;

	const vector<KeyPoint>& points0 = __features[0].keypoints;
	Mat img0;
	__img0.copyTo(img0);
	for (int k = 0; k < _num_features0; k++)
	{
		Point2f pt = points0[k].pt;
		float c = min(max(labels[k], 0), 1);
		circle(img0, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
		//circle(img0, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar((1-c)*255,0,c*255), CV_FILLED, CV_AA);
	}
	char fea_name[512];
	sprintf(fea_name, "%s/features0.jpg", outDir.c_str());
	imwrite(fea_name, img0);

	const vector<KeyPoint>& points1 = __features[1].keypoints;
	Mat img1;
	__img1.copyTo(img1);
	for (int k = 0; k < _num_features1; k++)
	{
		Point2f pt = points1[k].pt;
		float c = min(max(labels[k + _num_features0], 0), 1);
		circle(img1, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
		//circle(img1, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar((1-c)*255,0,c*255), CV_FILLED, CV_AA);
	}
	sprintf(fea_name, "%s/features1.jpg", outDir.c_str());
	imwrite(fea_name, img1);
}

void Stitching::FeatureDetection()
{
	cout << "Detecting features" << endl;
	float match_conf = 0.3f;
	__features.resize(2);
	__matches.resize(4);

	//Surf特征点检测
	//// detecting keypoints
	//SurfFeatureDetector detector(2000);
	//vector<KeyPoint> &keypoints1 = __features[0].keypoints, &keypoints2 = __features[1].keypoints;
	//detector.detect(__img0, keypoints1);
	//detector.detect(__img1, keypoints2);
	//// computing descriptors
	//SurfDescriptorExtractor extractor;
	//Mat &descriptors1 = __features[0].descriptors, &descriptors2 = __features[1].descriptors;
	//extractor.compute(__img0, keypoints1, descriptors1);
	//extractor.compute(__img1, keypoints2, descriptors2);
	//// matching descriptors
	//BFMatcher matcher(NORM_L2);
	//vector<DMatch> &matches = __matches[1].matches;
	//matcher.match(descriptors1, descriptors2, matches);

	//SIFT特征点检测
	SIFT sift(2000);
	Mat gray;
	cvtColor(__img0, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), __features[0].keypoints, __features[0].descriptors);
	//sift.operator()(gray, Mat(), __features[0].keypoints, __features[0].descriptors);
	__features[0].img_idx = 0;
	__features[0].img_size = gray.size();
	cvtColor(__img1, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), __features[1].keypoints, __features[1].descriptors);
	__features[1].img_idx = 1;
	__features[1].img_size = gray.size();
	//特征点匹配
	detail::BestOf2NearestMatcher matcher(0, match_conf);
	matcher(__features, __matches);
	matcher.collectGarbage();

	//cout << "inliers数目：" << __matches[1].num_inliers << endl;
	calcHomoFromMatches(__matches[1], __features[0], __features[1]);
	calcDualMatches(__matches[2], __matches[1]);
	//赋值全局变量_H
	double *homography = (double*)__matches[1].H.data;//指向Mat型变量H的data的指针
	for (int j = 0; j < 3; ++j)
	for (int k = 0; k < 3; ++k)
	{
		double tmp = homography[j * 3 + k];
		_H(j, k) = tmp;//用Mat型H的data对Matrix型_H赋值
	}

	_num_matched = __matches[1].matches.size();
	__fit_err.resize(_num_matched, Point2f(0,0));
	//_select.setOnes(_num_matched);
	initInliersMask();
	_select = _inliers_mask;
	_weight.resize(_num_matched, 0.0);
	_num_selmat = _num_matched;
	_num_features0 = __features[0].keypoints.size();
	_num_features1 = __features[1].keypoints.size();
	_num_features = _num_features0 + _num_features1;
	_Mat_Parallax.resize(_num_matched, 2);
	_Mat_Parallax.setOnes();

	_num_matches.clear();

	cout << "    # of features in Img0: " << _num_features0 << endl;
	cout << "    # of features in Img1: " << _num_features1 << endl;
	cout << "    # of Matches: " << _num_matched << endl;
}

void Stitching::ReadFeaturesMatches(string feaname, string matname)
{
	cout << "Reading features and matches" << endl;
	__features.clear();
	__features.resize(2);
	__features[0].img_idx = 0;
	__features[0].img_size = Size(__img0.cols, __img0.rows);
	__features[1].img_idx = 1;
	__features[1].img_size = Size(__img1.cols, __img1.rows);

	ifstream fp_fea(feaname.c_str());
	if (!fp_fea)
	{
		printf_s("can not read %s\n", feaname.c_str());
		return;
	}
	while (!fp_fea.eof())
	{
		Point2f tp0, tp1;
		fp_fea >> tp0.x >> tp0.y >> tp1.x >> tp1.y;
		if (fp_fea.eof())
			break;
		__features[0].keypoints.push_back(KeyPoint(tp0.x, tp0.y, 0));
		__features[1].keypoints.push_back(KeyPoint(tp1.x, tp1.y, 0));
	}
	fp_fea.close();

	__matches.clear();
	__matches.resize(4);
	__matches[1].src_img_idx = 0; __matches[1].dst_img_idx = 1;
	__matches[2].src_img_idx = 1; __matches[2].dst_img_idx = 0;
	ifstream fp_mat(matname.c_str());
	if (!fp_mat)
	{
		printf_s("can not read %s\n", feaname.c_str());
		return;
	}
	while (!fp_mat.eof())
	{
		float tid;
		fp_mat >> tid;
		if (fp_mat.eof())
			break;
		__matches[1].matches.push_back(DMatch(tid, tid, -1, 0));
		__matches[2].matches.push_back(DMatch(tid, tid, -1, 0));
	}
	fp_mat.close();

	calcHomoFromMatches(__matches[1], __features[0], __features[1]);
	calcDualMatches(__matches[2], __matches[1]);
	//赋值全局变量_H
	double *homography = (double*)__matches[1].H.data;//指向Mat型变量H的data的指针
	for (int j = 0; j < 3; ++j)
	for (int k = 0; k < 3; ++k)
	{
		double tmp = homography[j * 3 + k];
		_H(j, k) = tmp;//用Mat型H的data对Matrix型_H赋值
	}

	_num_matched = __matches[1].matches.size();
	_num_features0 = __features[0].keypoints.size();
	_num_features1 = __features[1].keypoints.size();
	_num_features = _num_features0 + _num_features1;

	//_select.setOnes(_num_matched);
	__fit_err.resize(_num_matched, Point2f(0, 0));
	initInliersMask();
	_select = _inliers_mask;
	_weight.resize(_num_matched, 0.0);
	_num_selmat = _num_matched;
	_Mat_Parallax.resize(_num_matched, 2);
	_Mat_Parallax.setOnes();
	cout << "    # of features in Img0: " << _num_features0 << endl;
	cout << "    # of features in Img1: " << _num_features1 << endl;
	cout << "    # of Matches: " << _num_matched << endl;
}

void Stitching::initInliersMask()
{
	int num_mat = __matches[1].matches.size();
	_outliers_mask.clear();
	_inliers_mask.clear();
	_outliers_mask.resize(num_mat, 1);
	_inliers_mask.resize(num_mat, 0);
	int count = 0;
	for (int idx_mat = 0; idx_mat < num_mat; ++idx_mat){
		bool is_inlier = __matches[1].inliers_mask[idx_mat];
		if (is_inlier){
			_inliers_mask[idx_mat] = 1;
			_outliers_mask[idx_mat] = 0;
			++count;
		}
	}
	cout << "全局内点有：" << count << " 个" << endl;
}

bool Stitching::calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < 5)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion
	vector<Point2f> src_points, dst_points;
	for (int j = 0; j < num_matched; j++)
	{
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	}
	m.H = findHomography(src_points, dst_points, CV_RANSAC, RANSAC_THRE0, m.inliers_mask);
	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	// num of inliers
	m.num_inliers = 0;
	for (int j = 0; j < num_matched; j++)
	if (m.inliers_mask[j])
		m.num_inliers++;

	// confidence, copied from matchers.cpp
	m.confidence = m.num_inliers / (8 + 0.3 * num_matched);
	m.confidence = m.confidence > 3. ? 0. : m.confidence;

	// refine the homography matrix
	//if (m.num_inliers < 9 /*|| m.confidence < 0.80*/)
	//{
	//	m = detail::MatchesInfo();
	//	return false;
	//}
	//src_points.clear();
	//dst_points.clear();
	//for (int j = 0; j < num_matched; j++)
	//{
	//	if (!m.inliers_mask[j])
	//		continue;
	//	const DMatch& t = m.matches[j];
	//	src_points.push_back(f1.keypoints[t.queryIdx].pt);
	//	dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	//}
	//m.H = findHomography(src_points, dst_points, CV_RANSAC); // do NOT output inlier mask
	//cout << "计算inliers点的H：" << m.H << endl;
	return true;
}

bool Stitching::calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, const float rantresh)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < 5)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion
	vector<Point2f> src_points, dst_points;
	for (int j = 0; j < num_matched; j++)
	{
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	}
	m.H = findHomography(src_points, dst_points, CV_RANSAC, rantresh, m.inliers_mask);
	//m.H = findHomography(src_points, dst_points, CV_LMEDS);
	//ofstream fea0("output/test3/testH/fea0.txt"), fea1("output/test3/testH/fea1.txt");
	//if (fea0.is_open()){
	//	for (int i = 0; i < num_matched; ++i){
	//		if (!m.inliers_mask[i])
	//			continue;
	//		fea0 << src_points[i].x << " " << src_points[i].y << endl;
	//		fea1 << dst_points[i].x << " " << dst_points[i].y << endl;
	//	}
	//}
	//fea0.close();
	//fea1.close();
	//cout << "H为：" << endl;
	//cout << m.H << endl;
	//ofstream H_file("output/test3/H.txt");
	//if (H_file.is_open()){
	//	H_file << m.H << endl;
	//}
	//H_file.close();

	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	// num of inliers
	m.num_inliers = 0;
	for (int j = 0; j < num_matched; j++)
	if (m.inliers_mask[j])
		m.num_inliers++;

	// confidence, copied from matchers.cpp
	m.confidence = m.num_inliers / (8 + 0.3 * num_matched);
	m.confidence = m.confidence > 3. ? 0. : m.confidence;

	return true;
}

bool Stitching::calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, vector<uchar> select_mask)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < 5)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion
	vector<Point2f> src_points, dst_points;
	for (int j = 0; j < num_matched; j++)
	{
		if (!select_mask[j])
			continue;
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	}
	m.H = findHomography(src_points, dst_points, CV_RANSAC, RANSAC_THRE0);
	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	//更新m成员变量
	//// num of inliers
	//m.num_inliers = 0;
	//for (int j = 0; j < num_matched; j++)
	//if (m.inliers_mask[j])
	//	m.num_inliers++;
	//// confidence, copied from matchers.cpp
	//m.confidence = m.num_inliers / (8 + 0.3 * num_matched);
	//m.confidence = m.confidence > 3. ? 0. : m.confidence;

	//// refine the homography matrix
	//if (m.num_inliers < 9 /*|| m.confidence < 0.80*/)
	//{
	//	m = detail::MatchesInfo();
	//	return false;
	//}
	//src_points.clear();
	//dst_points.clear();
	//for (int j = 0; j < num_matched; j++)
	//{
	//	if (!dist_mask(j))
	//		continue;
	//	if (!m.inliers_mask[j])
	//		continue;
	//	const DMatch& t = m.matches[j];
	//	src_points.push_back(f1.keypoints[t.queryIdx].pt);
	//	dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	//}
	//m.H = findHomography(src_points, dst_points, CV_RANSAC); // do NOT output inlier mask
	////cout << "计算inliers点的H：" << m.H << endl;
	return true;
}


void Stitching::calcDualMatches(detail::MatchesInfo& dm, const detail::MatchesInfo& m)
{
	if (m.matches.size() <= 0)
	{
		dm = detail::MatchesInfo();
		return;
	}
	dm = m;
	swap(dm.src_img_idx, dm.dst_img_idx);
	if (!m.H.empty())
		dm.H = m.H.inv();
	for (int j = 0; j < dm.matches.size(); j++)
		swap(dm.matches[j].queryIdx, dm.matches[j].trainIdx);
}

bool Stitching::Composite_Opencv(string outDir)
{
	_mkdir(outDir.c_str());

	

	Mat result_lin_bld = CalLinearBlend(__images_warped[0], __images_warped[1], __masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);
	imwrite(outDir + "/result_lin_bld.jpg", result_lin_bld);

	cout << "Seam Finding by Opencv" << endl;
	int blend_type = detail::Blender::NO;
	Mat result, result_mask;
	Findseam_and_Blend(result, result_mask, __images_warped, __masks_warped, __corners, blend_type);
	cout << "Blending by Opencv" << endl;

	result.copyTo(__result);

	//imwrite(outDir + "/images_warped0.jpg", __images_warped[0]);
	//imwrite(outDir + "/images_warped1.jpg", __images_warped[1]);
	//imwrite(outDir + "/masks0.jpg", __masks_warped[0]);
	//imwrite(outDir + "/masks1.jpg", __masks_warped[1]);
	imwrite(outDir + "/result.jpg", result);
	return true;

}

bool Stitching::Composite_Graphcut(string outDir)//use color different as cost
{
	_mkdir(outDir.c_str());
	cout << "Composite by Graphcut_color" << endl;

	//配准
	AlignmentbyH(_H);


	//找缝隙
	//FindSeambyGraphcut_PatchDifference(outDir, __images_warped[0], __images_warped[1], __masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);
	Mat mask0_avoid(__img0.size(), CV_8UC1, Scalar(0)), mask1_avoid(__img1.size(), CV_8UC1, Scalar(0));//初始时，不用避开某片区域
	FindSeambyGraphcut_WeightedPatchDifference(outDir, __images_warped[0], __images_warped[1], __masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);
	

	//融合
	//线性融合
	Mat result_lin_bld = CalLinearBlend(__images_warped[0], __images_warped[1], __masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);
	imwrite(outDir + "/result_lin_bld.jpg", result_lin_bld);
	//blender融合
	int blend_type = detail::Blender::NO;
	Mat result, result_mask;
	Blend(result, result_mask, __images_warped, __masks_warped, __corners, blend_type);
	result.copyTo(__result);
	imwrite(outDir + "/result.jpg", result);

	return true;
}

void Stitching::Findseam_and_Blend(Mat& result, Mat& result_mask, vector<Mat>& Ims, vector<Mat>& Mks, vector<Point>& corners, int blend_type)
{
	//int expos_comp_type = detail::ExposureCompensator::GAIN_BLOCKS;
	//Ptr<detail::ExposureCompensator> compensator = detail::ExposureCompensator::createDefault(expos_comp_type);
	////光补偿器compensator
	//compensator->feed(corners, Ims, Mks);


	bool try_gpu = false;
	string seam_find_type = "gc_color";
	//int blend_type = detail::Blender::NO; //MULTI_BAND, FEATHER, NO
	float blend_strength = 5;
	int num_images = Ims.size();
	if (num_images < 2)
	{
		cout << "please load more images" << endl;
		return;
	}


	Ptr<detail::SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type == "gc_color")
		seam_finder = new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR);
	else if (seam_find_type == "gc_colorgrad")
		seam_finder = new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD);
	else if (seam_find_type == "dp_color")
		seam_finder = new detail::DpSeamFinder(detail::DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(detail::DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return;
	}

	//imwrite("Im0.png", Ims[0]);
	//imwrite("Im1.png", Ims[1]);

	vector<Mat> Ims_f(num_images);
	for (int i = 0; i < num_images; ++i)
		Ims[i].convertTo(Ims_f[i], CV_32F);
	vector<Size> sizes(num_images);
	for (int i = 0; i < num_images; i++)
		sizes[i] = Ims[i].size();

	// find seam
	seam_finder->find(Ims_f, corners, Mks);
	////findGraphCutSeams(Ims, corners, Mks, Mat(), -1); // wrong result

	//imwrite("Mk0.png", Mks[0]);
	//imwrite("Mk1.png", Mks[1]);

	Ptr<detail::Blender> blender = detail::Blender::createDefault(blend_type, try_gpu);
	Size dst_sz = detail::resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	if (blend_type == detail::Blender::NO || blend_width < 1.f)
		blender = detail::Blender::createDefault(detail::Blender::NO, try_gpu);
	else if (blend_type == detail::Blender::MULTI_BAND)
	{
		detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
		mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	}
	else if (blend_type == detail::Blender::FEATHER)
	{
		detail::FeatherBlender* fb = dynamic_cast<detail::FeatherBlender*>(static_cast<detail::Blender*>(blender));
		fb->setSharpness(1.f / blend_width);
	}
	blender->prepare(corners, sizes);
	for (int i = 0; i < num_images; i++)
	{

		// Compensate exposure
		//compensator->apply(i, corners[i], Ims[i], Mks[i]);
		////由角点坐标以及经过扭曲的图像和掩模，对第img_idx帧图像进行光度补偿

		Mat img_s;
		Ims[i].convertTo(img_s, CV_16S);
		blender->feed(img_s, Mks[i], corners[i]);
	}

	Mat R, M;
	blender->blend(R, M); // the result is CV_16
	R.convertTo(result, CV_8U);
	M.convertTo(result_mask, CV_8U);

	Vec3b color;
	color[0] = 255;
	color[1] = 255;
	color[2] = 255;
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			uchar value = result_mask.at<uchar>(i, j);
			if (value < 127)
			{
				result.at<Vec3b>(i, j) = color;
			}
		}
	}
}

void Stitching::Blend(Mat& result, Mat& result_mask, vector<Mat>& Ims, vector<Mat>& Mks, vector<Point>& corners, int blend_type)
{
	bool try_gpu = false;
	string seam_find_type = "gc_color";
	//int blend_type = detail::Blender::NO; //MULTI_BAND, FEATHER, NO
	float blend_strength = 5;
	int num_images = Ims.size();
	if (num_images < 2)
	{
		cout << "please load more images" << endl;
		return;
	}

	vector<Size> sizes(num_images);
	for (int i = 0; i < num_images; i++)
		sizes[i] = Ims[i].size();

	Ptr<detail::Blender> blender = detail::Blender::createDefault(blend_type, try_gpu);
	Size dst_sz = detail::resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	if (blend_type == detail::Blender::NO || blend_width < 1.f)
		blender = detail::Blender::createDefault(detail::Blender::NO, try_gpu);
	else if (blend_type == detail::Blender::MULTI_BAND)
	{
		detail::MultiBandBlender* mb = dynamic_cast<detail::MultiBandBlender*>(static_cast<detail::Blender*>(blender));
		mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
	}
	else if (blend_type == detail::Blender::FEATHER)
	{
		detail::FeatherBlender* fb = dynamic_cast<detail::FeatherBlender*>(static_cast<detail::Blender*>(blender));
		fb->setSharpness(1.f / blend_width);
	}
	blender->prepare(corners, sizes);
	for (int i = 0; i < num_images; i++)
	{
		Mat img_s;
		Ims[i].convertTo(img_s, CV_16S);
		blender->feed(img_s, Mks[i], corners[i]);
	}

	Mat R, M;
	blender->blend(R, M); // the result is CV_16
	R.convertTo(result, CV_8U);
	M.convertTo(result_mask, CV_8U);

	Vec3b color;
	color[0] = 255;
	color[1] = 255;
	color[2] = 255;
	for (int i = 0; i < result.rows; ++i)
	{
		for (int j = 0; j < result.cols; ++j)
		{
			uchar value = result_mask.at<uchar>(i, j);
			if (value < 127)
			{
				result.at<Vec3b>(i, j) = color;
			}
		}
	}
}

void Stitching::calcPoint_after_H(Point2f p_src, Point2f& p_dst, const double* h)
//输入p_src，输出p_dst
{
	double xx = p_src.x*h[0] + p_src.y*h[1] + h[2];
	double yy = p_src.x*h[3] + p_src.y*h[4] + h[5];
	double zz = p_src.x*h[6] + p_src.y*h[7] + h[8];
	if (zz){
		p_dst.x = xx / zz;
		p_dst.y = yy / zz;
	}
	else{
		p_dst.x = xx;
		p_dst.y = yy;
	}
	
}

Mat Stitching::WarpImg(const Mat img_src, Point2i &TL_rect, Matrix<double, 3, 3, RowMajor> H)//根据_H计算出img_wraped和TL_rect
{
	//计算四个角点经H变换后的点
	double *homo = H.data();
	Matrix<double, 3, 3, RowMajor> invH = H.inverse();
	double *invhomo = invH.data();

	int width = img_src.cols, height = img_src.rows;
	Point2f TL(0, 0), TR(width - 1, 0), BL(0, height - 1), BR(width - 1, height - 1);
	Point2f TL_dst, TR_dst, BL_dst, BR_dst;
	calcPoint_after_H(TL, TL_dst, homo);
	calcPoint_after_H(TR, TR_dst, homo);
	calcPoint_after_H(BL, BL_dst, homo);
	calcPoint_after_H(BR, BR_dst, homo);
	float x_min, x_max, y_min, y_max;
	x_min = min(TL_dst.x, min(TR_dst.x, min(BL_dst.x, BR_dst.x)));
	y_min = min(TL_dst.y, min(TR_dst.y, min(BL_dst.y, BR_dst.y)));
	x_max = max(TL_dst.x, max(TR_dst.x, max(BL_dst.x, BR_dst.x)));
	y_max = max(TL_dst.y, max(TR_dst.y, max(BL_dst.y, BR_dst.y)));
	//构建包络左上角点
	TL_rect.x = x_min;
	TL_rect.y = y_min;

	int rect_wd = x_max - x_min + 1;
	int rect_ht = y_max - y_min + 1;
	int nchannels = img_src.channels();
	Mat img_wraped;
	if (nchannels == 3)
	{
		img_wraped.create(rect_ht, rect_wd, CV_8UC3);
		img_wraped.setTo(Scalar(0, 0, 0));
	}
	else if (nchannels == 1)
	{
		img_wraped.create(rect_ht, rect_wd, CV_8U);
		img_wraped.setTo(Scalar(0));
	}
	Point2f off(0 - TL_rect.x, 0 - TL_rect.y);//img_src坐标系原点(0,0)相对于img_wraped坐标系原点TL_rect的相对位置
	uchar *im_warp_data = img_wraped.data;
	const uchar *img_src_data = img_src.data;

	//构建img_wraped：将img_wraped的点反映射到img_src上，然后双线性插值
	float *cl = new float[nchannels];//(x,y)插值结果
	for (int y = 0; y < rect_ht; ++y)
	{
		for (int x = 0; x < rect_wd; ++x)//(x,y)为img_wraped上坐标点
		{

			Point2f pt_imdst(x - off.x, y - off.y);//转换到img_src坐标系
			//if ((pt_imdst.x < -215) || pt_imdst.x > -215 || (pt_imdst.y < 700) || pt_imdst.y > 700)
			//	continue;
			Point2f pt_imsrc;//(x,y)反映射到imsrc上的坐标为小数
			calcPoint_after_H(pt_imdst, pt_imsrc, invhomo);
			if (pt_imsrc.x<0 || pt_imsrc.x>width - 1 || pt_imsrc.y<0 || pt_imsrc.y>height - 1)
				continue;//如果超出原图像边界，则(x,y)为全黑

			//双线性插值
			bilinearInterpolate(cl, pt_imsrc, img_src_data, width, height, nchannels);
			for (int c = 0; c < nchannels; ++c)
				im_warp_data[(y*rect_wd + x)*nchannels + c] = uchar(cl[c]);
		}
	}
	delete cl;
	return img_wraped;
}

Mat Stitching::WarpImg(const Mat imgSrc, Point2i& outCorner, const Warping& meshWarper)
{
	VectorXf vertices_dst = meshWarper.__vertices2;
	vector<float> x_set, y_set;
	int yctrls = meshWarper._yquads + 1;//y轴顶点个数
	int xctrls = meshWarper._xquads + 1;//x轴顶点个数
	int width1 = imgSrc.cols, height1 = imgSrc.rows;
	for (int i = 0; i <= meshWarper._yquads; i++) {
		for (int j = 0; j <= meshWarper._xquads; j++) {
			x_set.push_back(vertices_dst[(i*xctrls+j) * 2 + 0]);
			y_set.push_back(vertices_dst[(i*xctrls+j) * 2 + 1]);
		}
	}
	float x_min, x_max, y_min, y_max;
	x_min = *(min_element(x_set.begin(), x_set.end()));
	x_max = *(max_element(x_set.begin(), x_set.end()));
	y_min = *(min_element(y_set.begin(), y_set.end()));
	y_max = *(max_element(y_set.begin(), y_set.end()));
	x_set.clear();
	y_set.clear();
	//构建包络左上角点
	outCorner.x = x_min;
	outCorner.y = y_min;

	Mat imgWraped;
	int canvas_wd = x_max - x_min + 1;
	int canvas_ht = y_max - y_min + 1;
	int nchannels = imgSrc.channels();
	if (nchannels == 3)
	{
		imgWraped.create(canvas_ht, canvas_wd, CV_8UC3);
		imgWraped.setTo(Scalar(0, 0, 0));
	}
	else if (nchannels == 1)
	{
		imgWraped.create(canvas_ht, canvas_wd, CV_8U);
		imgWraped.setTo(Scalar(0));
	}
	Point2f off(0 - outCorner.x, 0 - outCorner.y);//img_src坐标系原点(0,0)相对于img_wraped坐标系原点outCorner的相对位置
	
	Point2f* V2 = (Point2f *)vertices_dst.data();
	uchar* im_data = imgSrc.data;
	uchar* re_data = imgWraped.data;
	float *rgb = new float[nchannels];
	Point2f p1, p2;
	Mat ind(canvas_ht, canvas_wd, CV_8U, Scalar(0, 0, 0));
	uchar* ind_data = ind.data;




	for (int i = 0; i < meshWarper._yquads; i++)
	{
		for (int j = 0; j < meshWarper._xquads; j++)//为逐个quad内像素点赋值
		{
			if (meshWarper.__homos[i*meshWarper._xquads + j].empty())
				continue;
			double *h = (double *)meshWarper.__homos[i*meshWarper._xquads + j].data;
			vector<Point2f> vpts;//该quad四个顶点（float型坐标）
			vpts.push_back(V2[i*xctrls + j]);
			vpts.push_back(V2[i*xctrls + j + 1]);
			vpts.push_back(V2[(i + 1)*xctrls + j + 1]);
			vpts.push_back(V2[(i + 1)*xctrls + j]);
			//求该quad四个顶点的包络（int型坐标）
			int minx = cvFloor(min(min(min(vpts[0].x, vpts[1].x), vpts[2].x), vpts[3].x));
			int miny = cvFloor(min(min(min(vpts[0].y, vpts[1].y), vpts[2].y), vpts[3].y));
			int maxx = cvCeil(max(max(max(vpts[0].x, vpts[1].x), vpts[2].x), vpts[3].x));
			int maxy = cvCeil(max(max(max(vpts[0].y, vpts[1].y), vpts[2].y), vpts[3].y));
			minx = min(max(minx, x_min), x_max);
			miny = min(max(miny, y_min), y_max);
			maxx = min(max(maxx, x_min), x_max);
			maxy = min(max(maxy, y_min), y_max);
			//result图中quad包络内像素点赋值
			for (int y = miny; y <= maxy; y++)//像素点h反变换到img上，双线性插值得到该像素点的RGB值
			{
				for (int x = minx; x <= maxx; x++)
				{
					//点(x,y)从img0坐标系转换到canvas坐标系下
					int x_canvas = min(max(x + off.x, 0), canvas_wd - 1);
					int y_canvas = min(max(y + off.y, 0), canvas_ht - 1);
					if (ind_data[y_canvas*canvas_wd + x_canvas])//是否已经赋值
						continue;

					p2 = Point2f(x, y);
					if (pointPolygonTest(vpts, p2, false)<0)
						continue;
					p1 = warpPoint(p2, h);
					if (p1.x<0 || p1.y<0 || p1.x>width1 - 1 || p1.y>height1 - 1)
						continue;

					ind_data[y_canvas*canvas_wd + x_canvas] = 255;
					bilinearInterpolate(rgb, p1, im_data, width1, height1, nchannels);
					
					for (int c = 0; c < nchannels; c++)
						re_data[(y_canvas*canvas_wd + x_canvas)*nchannels + c] = uchar(min(max(cvRound(rgb[c]), 0), 255));
				}
			}

			//调试
			//Point2f pt = V2[i*xctrls + j];
			//pt = pt + off;
			//circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			//pt = V2[i*xctrls + j + 1];
			//pt = pt + off;
			//circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			//pt = V2[(i + 1)*xctrls + j + 1];
			//pt = pt + off;
			//circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			//pt = V2[(i + 1)*xctrls + j];
			//pt = pt + off;
			//circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
		}
	}

	return imgWraped;
}

Mat Stitching::gridWarpImg(const Warping& meshWarper)
{
	Mat imgWraped;
	VectorXf vertices_dst = meshWarper.__vertices2;
	vector<float> x_set, y_set;
	int yctrls = meshWarper._yquads + 1;//y轴顶点个数
	int xctrls = meshWarper._xquads + 1;//x轴顶点个数
	int width1 = meshWarper._src_size.width, height1 = meshWarper._src_size.height;
	for (int i = 0; i <= meshWarper._yquads; i++) {
		for (int j = 0; j <= meshWarper._xquads; j++) {
			x_set.push_back(vertices_dst[(i*xctrls + j) * 2 + 0]);
			y_set.push_back(vertices_dst[(i*xctrls + j) * 2 + 1]);
		}
	}
	float x_min, x_max, y_min, y_max;
	x_min = *(min_element(x_set.begin(), x_set.end()));
	x_max = *(max_element(x_set.begin(), x_set.end()));
	y_min = *(min_element(y_set.begin(), y_set.end()));
	y_max = *(max_element(y_set.begin(), y_set.end()));
	x_set.clear();
	y_set.clear();
	//构建包络左上角点
	Point outCorner;
	outCorner.x = x_min;
	outCorner.y = y_min;

	int canvas_wd = x_max - x_min + 1;
	int canvas_ht = y_max - y_min + 1;
	imgWraped.create(canvas_ht, canvas_wd, CV_8UC3);
	imgWraped.setTo(Scalar(255, 255, 255));
	
	Point2f off(0 - outCorner.x, 0 - outCorner.y);//img_src坐标系原点(0,0)相对于img_wraped坐标系原点outCorner的相对位置

	Point2f* V2 = (Point2f *)vertices_dst.data();
	Point2f p1, p2;
	for (int i = 0; i < meshWarper._yquads; i++)
	{
		for (int j = 0; j < meshWarper._xquads; j++)
		{
			if (meshWarper.__homos[i*meshWarper._xquads + j].empty())
				continue;
			//调试
			Point2f pt = V2[i*xctrls + j];
			pt = pt + off;
			circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			pt = V2[i*xctrls + j + 1];
			pt = pt + off;
			circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			pt = V2[(i + 1)*xctrls + j + 1];
			pt = pt + off;
			circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
			pt = V2[(i + 1)*xctrls + j];
			pt = pt + off;
			circle(imgWraped, Point(cvRound(pt.x), cvRound(pt.y)), 8, Scalar(255, 0, 0), CV_FILLED, CV_AA);
		}
	}
	return imgWraped;
}

void Stitching::PatchInitialization_on_Pixel(Point2i pos, Mat img, float* patchdata)
{
	const uchar *img_data = img.data;//——————————————————在Img上构建patch
	int width = img.cols,height=img.rows;
	int channels = img.channels();


	for (int p = -_psize; p <= _psize; p++)
	{
		for (int q = -_psize; q <= _psize; q++)//横向扫描，对patch逐点赋颜色值
		{
			Point2i pt_on_patch(q + _psize, p + _psize);// patch内当前点在patch上坐标
			int offset_patch = (pt_on_patch.y*_wsize + pt_on_patch.x)*channels;
			Point2i pt_on_img(pos.x + q, pos.y + p);//patch内当前点在warpImg上坐标
			int offset_img = (pt_on_img.y*width + pt_on_img.x)*channels;
			//如果patch超过了原图像边界，则在图像外的部分像素设为零
			if (pt_on_img.x<0 || pt_on_img.x>width - 1 || pt_on_img.y<0 || pt_on_img.y>height - 1)
			{
				for (int c = 0; c < channels; ++c)
					patchdata[c + offset_patch] = 0;//该像素颜色值设为0
				continue;
			}
			//否则用warpImg上的像素点颜色值赋值给patch内当前的像素点
			for (int c = 0; c < channels; ++c)
				patchdata[c + offset_patch] = img_data[c + offset_img];
		}
	}
}

void Stitching::ConstructPatch(Point2f centPnt, int psize, Mat srcImg, float* patchData)
{
	int wsize = 2 * psize + 1;
	const uchar *imgData = srcImg.data;//图像数据
	//对11*11的patch插值
	for (int p = -psize; p <= psize; p++) {
		for (int q = -psize; q <= psize; q++)
			bilinearInterpolate(patchData + ((p + psize)*wsize + q + psize)*_channels, Point2f(centPnt.x + q, centPnt.y + p), imgData, srcImg.cols, srcImg.rows, _channels);
	}
}

void Stitching::ShowMatches(string outDir, string outFile, vector<uchar> select)
{
	_mkdir(outDir.c_str());
	RNG rng = theRNG();
	const vector<DMatch>& valid_matches = __matches[1].matches;
	const vector<uchar>& inliers_mask = __matches[1].inliers_mask;
	int width = _width, height = 2 * _height;
	Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	Mat imgA(imgAB, Rect(0, 0, _width, _height));
	__img0.copyTo(imgA);
	Mat imgB(imgAB, Rect(0, _height, _width, _height));
	__img1.copyTo(imgB);
	const vector<KeyPoint>& points1 = __features[0].keypoints;
	const vector<KeyPoint>& points2 = __features[1].keypoints;


	for (int k = 0; k < _num_matched; k++)
	{
		if (!select[k])
			continue;
		const DMatch& t = valid_matches[k];
		Point2f p1 = points1[t.queryIdx].pt;
		Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
		Scalar newvalue(rng(256), rng(256), rng(256));
		line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
		circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
		circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);

	}

	char fea_name[512];
	sprintf(fea_name, "%s/%s.jpg", outDir.c_str(), outFile.c_str());
	imwrite(fea_name, imgAB);
}

Mat Stitching::CalLinearBlend(Mat im0, Mat im1, Mat mask0, Mat mask1, Point2i corner0, Point2i corner1)
{
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;
	Mat outImg0 = Mat::zeros(height, width, CV_8UC3);
	Mat outImg0Mask = Mat::zeros(height, width, CV_8UC1);
	Mat outImg1 = Mat::zeros(height, width, CV_8UC3);
	Mat outImg1Mask = Mat::zeros(height, width, CV_8UC1);

	Mat img0(outImg0, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0);
	Mat maskLeft(outImg0Mask, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	mask0.copyTo(maskLeft);

	Mat img1(outImg1, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1);
	Mat maskRight(outImg1Mask, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	mask1.copyTo(maskRight);

	Mat outImg = Mat::zeros(height, width, CV_8UC3);
	for (unsigned idx_row = 0; idx_row < height; idx_row++)	 //set pixel value for outImg
	{
		for (unsigned idx_col = 0; idx_col < width; idx_col++)
		{
			unsigned ptMsk_im0 = outImg0Mask.at<uchar>(idx_row, idx_col);
			unsigned ptMsk_im1 = outImg1Mask.at<uchar>(idx_row, idx_col);
			if (ptMsk_im0 != 0 && ptMsk_im1 != 0)	//overlapping area
			{
				Vec3b pixel;
				pixel[0] = (outImg0.at<Vec3b>(idx_row, idx_col)[0] + outImg1.at<Vec3b>(idx_row, idx_col)[0]) / 2;
				pixel[1] = (outImg0.at<Vec3b>(idx_row, idx_col)[1] + outImg1.at<Vec3b>(idx_row, idx_col)[1]) / 2;
				pixel[2] = (outImg0.at<Vec3b>(idx_row, idx_col)[2] + outImg1.at<Vec3b>(idx_row, idx_col)[2]) / 2;
				outImg.at<Vec3b>(idx_row, idx_col) = pixel;	
			}
			else if (ptMsk_im0 != 0 && ptMsk_im1 == 0)//non-overlapping area on outImg0
			{
				Vec3b pixel = outImg0.at<Vec3b>(idx_row, idx_col);
				outImg.at<Vec3b>(idx_row, idx_col) = pixel;
			}			
			else if (ptMsk_im0 == 0 && ptMsk_im1 != 0)//non-overlapping area on outImg1
			{
				Vec3b pixel = outImg1.at<Vec3b>(idx_row, idx_col);
				outImg.at<Vec3b>(idx_row, idx_col) = pixel;
			}
		}
	}
	return 	outImg;
}

float Stitching::calAffinitybtPatches(float* patchData0, float* patchData1)
{
	float sigma_normalized = 0.10f;
	float sigma_color = sqrt(255 * 255 * _channels)*sigma_normalized;
	float sigma_spatial = sqrt(((_width - 1)*(_width - 1) + (_height - 1)*(_height - 1)))*sigma_normalized;
	float thre_dist2 = -log(0.1f);// exp(-dist2)>=0.1f;

	float affinity;
	float dist_color = 0;
	for (int c = 0; c < _pdim; c++) // 指针操作更快--------------------by Kai Li
	{
		//这里的dstimg上的patch是第idx_mat个patch
		dist_color += (patchData0[c] - patchData1[c])*(patchData0[c] - patchData1[c]);
	}
	if (dist_color> _pnum * 2 * sigma_color*sigma_color*thre_dist2)//颜色距离太远不用计算，注意这里color_d是平方和----------------by Kai Li
	{
		affinity = 0;
	}
	float dist_norm = exp(-(dist_color / (_pnum * 2 * sigma_color*sigma_color)));
	affinity = dist_norm;
	return affinity;
}

void Stitching::FindSeambyGraphcut_Parallax(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1)
{
	_mkdir(outDir.c_str());
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;
	//construct im0_canvas and im1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat mask0_valid(mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	mask0.copyTo(mask0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);
	Mat mask1_valid(mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	mask1.copyTo(mask1_valid);

	char file_im0_canvas[256], file_mask0_canvas[256], file_im1_canvas[256], file_mask1_canvas[256];
	sprintf(file_im0_canvas, "%s/canvas_im0.jpg", outDir.c_str());
	imwrite(file_im0_canvas, im0_canvas);
	sprintf(file_mask0_canvas, "%s/canvas_mask0.jpg", outDir.c_str());
	imwrite(file_mask0_canvas, mask0_canvas);
	sprintf(file_im1_canvas, "%s/canvas_im1.jpg", outDir.c_str());
	imwrite(file_im1_canvas, im1_canvas);
	sprintf(file_mask1_canvas, "%s/canvas_mask1.jpg", outDir.c_str());
	imwrite(file_mask1_canvas, mask1_canvas);

	//construct graph
	typedef Graph<int, int, int> GraphType;
	int num_nodes = width*height, num_edges = num_nodes * 2;
	GraphType *g = new GraphType(/*estimated # of nodes*/num_nodes, /*estimated # of edges*/ num_edges);
	int terminal_cost = INT_MAX;
	g->add_node(num_nodes);

	//construct cost_map
	Mat _prlx_img0_canvas = Mat::zeros(height, width, CV_32FC1);
	Mat _prlx_img0_valid(_prlx_img0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	_prlx_img0.copyTo(_prlx_img0_valid);

	Mat _prlx_img1_canvas = Mat::zeros(height, width, CV_32FC1);
	Mat _prlx_img1_valid(_prlx_img1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	_prlx_img1.copyTo(_prlx_img1_valid);

	Mat cost_img(height, width, _prlx_img0_canvas.type());
	cost_img = _prlx_img0_canvas + _prlx_img1_canvas;
	float *cost_img_data = (float*)cost_img.data;
	//debug
	Mat cost_grey_img(height, width, CV_8UC1);
	uchar *cost_grey_img_data = cost_grey_img.data;
	for (int col = 0; col < width;++col)
	for (int row = 0; row < height; ++row)
		cost_grey_img_data[row*width + col] = cost_img_data[row*width + col] * 0.5 * 255;

	char file_cost_img[256];
	sprintf(file_cost_img, "%s/cost_img.jpg", outDir.c_str());
	imwrite(file_cost_img, cost_grey_img);
	//imwrite("output/test10/DebugStep/cost_img.jpg", cost_img);

	//define egdes(excluding t-links)
	int psize = 0;
	uchar *mask0_canvas_data = mask0_canvas.data;
	uchar *mask1_canvas_data = mask1_canvas.data;
	Mat edge_map(height, width, CV_8UC3, Scalar(0, 0, 0));
	uchar *edge_map_data = edge_map.data;
	for (int col = 0; col < width ; ++col)
	{
		bool right_exist = (col != (width-1));//当col=width-1时，该node没有right_edge
		for (int row = 0; row < height; ++row)
		{
			bool below_exist = (row != (height - 1));//当row=height-1时，该node没有below_edge
			int x = col, y = row;
			int pos = row*width + col;
			bool node_in0 = mask0_canvas_data[pos], node_in1 = mask1_canvas_data[pos];
			bool node_in_ol = node_in0 && node_in1;

			if (right_exist)
			{
				bool node_right_in0 = mask0_canvas_data[pos + 1], node_right_in1 = mask1_canvas_data[pos + 1];
				bool node_right_in_ol = node_right_in0 && node_right_in1;
				if (node_in_ol && node_right_in_ol)//当前点(x,y)与(x+1,y)在重叠区
				{
					//float d1 = 0, d2 = 0;
					//for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					//{
					//	for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
					//	{
					//		d1 += cost_img_data[p*width + q];
					//		d2 += cost_img_data[p*width + q + 1];
					//	}
					//}
					float d1 = cost_img_data[pos], d2 = cost_img_data[pos + 1];
					float dd = (d1 + d2) * 1000;
					g->add_edge(pos, pos + 1, int(dd), int(dd));

					#ifdef _DEBUG
					float norm = (d1 + d2) / 4 * 255;
					edge_map_data[pos * 3 + 2] = int(norm);//(x,y)点的R通道
					#endif // DEBUG
				}
				if (node_in_ol^node_right_in_ol)//点(x, y)与点(x+1, y)有一点在重叠区
				{
					float d1 = cost_img_data[pos], d2 = cost_img_data[pos + 1];
					float dd = node_in_ol ? (2 * d1) : (2 * d2);//edge(pi,pj)=2*cost(pk),pk为pi和pj中位于重叠区的点
					g->add_edge(pos, pos + 1, int(dd), int(dd));

					#ifdef _DEBUG
					edge_map_data[pos * 3 + 1] = 255;//(x,y)点的G通道
					#endif// DEBUG	
				}
			}
			
			if (below_exist)
			{
				bool node_below_in0 = mask0_canvas_data[pos + width], node_below_in1 = mask1_canvas_data[pos + width];
				bool node_below_in_ol = node_below_in0 && node_below_in1;
				if (node_in_ol && node_below_in_ol)//点(x,y)与(x,y+1)在重叠区
				{

					float d1 = cost_img_data[pos], d2 = cost_img_data[pos + width];
					float dd = (d1 + d2) * 1000;
					//cout << "node(" << x << "," << y << ")与node(" << x << "," << y+1 << ")的代价为" << int(dd) << endl;
					g->add_edge(pos, pos + width, int(dd), int(dd));

					#ifdef _DEBUG
					float norm = (d1 + d2) / 4 * 255;
					edge_map_data[pos * 3 + 0] = int(norm);//(x,y)点的B通道
					#endif// DEBUG
					
				}
				if (node_in_ol^node_below_in_ol)//点(x,y)与点(x,y+1)有一点在重叠区
				{
					float d1 = cost_img_data[pos], d2 = cost_img_data[pos + width];
					float dd = node_in_ol ? (2 * d1) : (2 * d2);//edge(pi,pj)=2*cost(pk),pk为pi和pj中位于重叠区的点
					g->add_edge(pos, pos + width, int(dd), int(dd));
					#ifdef _DEBUG
					//edge_map_data[pos * 3 + 1] = 255;//(x,y)点的B通道				
					#endif// DEBUG
					
				}
			}		
		}
	}

	char file_edge_map[256];
	sprintf(file_edge_map, "%s/edge_map.jpg", outDir.c_str());
	imwrite(file_edge_map, edge_map);


	//define t-links
	Mat source_area = Mat::zeros(height, width, CV_8UC1);
	Mat sink_area = Mat::zeros(height, width, CV_8UC1);
	uchar *source_data = source_area.data;
	uchar *sink_data = sink_area.data;
	for (int col = 0; col < width; ++col)
	{
		for (int row = 0; row < height; ++row)
		{
			int pose = row*width + col;
			bool in0 = mask0_canvas_data[pose];
			bool in1 = mask1_canvas_data[pose];
			if (in0 && !in1)
			{
				g->add_tweights(pose,   /* capacities */  terminal_cost, 0);//link to source
				source_data[pose] = 255;
			}
			else if (!in0 && in1)
			{
				g->add_tweights(pose,   /* capacities */  0, terminal_cost);//link to sink
				sink_data[pose] = 255;
			}
			else if (in0 && in1)
			{
			}
			else
			{
				g->add_tweights(pose,   /* capacities */  1, 1);
				source_data[pose] = 120;
				sink_data[pose] = 120;
			}
		}
	}
	char file_source_area[256], file_sink_area[256];
	sprintf(file_source_area, "%s/source_area.jpg", outDir.c_str());
	imwrite(file_source_area, source_area);
	sprintf(file_sink_area, "%s/sink_area.jpg", outDir.c_str());
	imwrite(file_sink_area, sink_area);
	//imwrite("output/test10/DebugStep/source_area.jpg", source_area);
	//imwrite("output/test10/DebugStep/sink_area.jpg", sink_area);

	int flow = g->maxflow();
	Mat result_mask0_canvas = mask0_canvas.clone(), result_mask1_canvas = mask1_canvas.clone();
	uchar *result_mask0_data = result_mask0_canvas.data, *result_mask1_data = result_mask1_canvas.data;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			int pos = row*width + col;
			//source set refers to im0		
			if (g->what_segment(pos) == GraphType::SOURCE)//pixel at pos belongs to source set
			{
				result_mask0_data[pos] = 255;
				result_mask1_data[pos] = 0;
			}
			else
			{
				result_mask0_data[pos] = 0;
				result_mask1_data[pos] = 255;
			}
		}
	}
	delete g;

	result_mask0_canvas = result_mask0_canvas & mask0_canvas;
	result_mask1_canvas = result_mask1_canvas & mask1_canvas;
	Mat result_mask0(result_mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	Mat result_mask1(result_mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	result_mask0.copyTo(mask0);
	result_mask1.copyTo(mask1);

	char file_mask0[256], file_mask1[256];
	sprintf(file_mask0, "%s/result_mask0.jpg", outDir.c_str());
	imwrite(file_mask0, result_mask0);
	sprintf(file_mask1, "%s/result_mask1.jpg", outDir.c_str());
	imwrite(file_mask1, result_mask1);
	//imwrite("output/test10/DebugStep/result_mask0.jpg", mask0);
	//imwrite("output/test10/DebugStep/result_mask1.jpg", mask1);
}

void Stitching::FindSeambyGraphcut_PatchDifference(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1)
{
	_mkdir(outDir.c_str());
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//construct im0_canvas,im1_canvas and mask0_canvas,mask1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat mask0_valid(mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	mask0.copyTo(mask0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);
	Mat mask1_valid(mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	mask1.copyTo(mask1_valid);

	//char file_im0_canvas[256], file_mask0_canvas[256], file_im1_canvas[256], file_mask1_canvas[256];
	//sprintf(file_im0_canvas, "%s/canvas_im0.jpg", outDir.c_str());
	//imwrite(file_im0_canvas, im0_canvas);
	//sprintf(file_mask0_canvas, "%s/canvas_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0_canvas, mask0_canvas);
	//sprintf(file_im1_canvas, "%s/canvas_im1.jpg", outDir.c_str());
	//imwrite(file_im1_canvas, im1_canvas);
	//sprintf(file_mask1_canvas, "%s/canvas_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1_canvas, mask1_canvas);

	//construct graph
	typedef Graph<int, int, int> GraphType;
	int num_nodes = width*height, num_edges = num_nodes * 2;
	GraphType *g = new GraphType(/*estimated # of nodes*/num_nodes, /*estimated # of edges*/ num_edges);
	int terminal_cost = INT_MAX;

	g->add_node(num_nodes);

	//define egdes(excluding t-links)
	int psize = 2;
	uchar *mask0_canvas_data = mask0_canvas.data;
	uchar *mask1_canvas_data = mask1_canvas.data;
	uchar *im0_data = im0_canvas.data;
	uchar *im1_data = im1_canvas.data;
	//-------------------------------------------20170515 用colored_edge设置代价
	//Mat im0_canvas_cedge = convertoColoredEdgeImage(im0_canvas);
	//Mat im1_canvas_cedge = convertoColoredEdgeImage(im1_canvas);
	//uchar *im0_data = im0_canvas_cedge.data;
	//uchar *im1_data = im1_canvas_cedge.data;
	Mat im0_edge = convertoColoredEdgeImage(im0_canvas);
	uchar *im0_edge_data = im0_edge.data;
	Mat im1_edge = convertoColoredEdgeImage(im1_canvas);
	uchar *im1_edge_data = im1_edge.data;
	for (int col = 0; col < width; ++col)
	{
		bool right_exist = (col != (width - 1));//当col=width-1时，该node没有right_edge
		for (int row = 0; row < height; ++row)
		{
			bool below_exist = (row != (height - 1));
			int x = col, y = row;
			int chs = 3;
			int pos = row*width + col;
			bool node_in0 = mask0_canvas_data[pos], node_in1 = mask1_canvas_data[pos];
			bool node_in_ol = node_in0 && node_in1;
			if (right_exist)
			{
				bool node_right_in0 = mask0_canvas_data[pos + 1], node_right_in1 = mask1_canvas_data[pos + 1];
				bool node_right_in_ol = node_right_in0 && node_right_in1;
				if (node_in_ol && node_right_in_ol)//当前点(x,y)与(x+1,y)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[(p*width + q + 1)*chs + c] - im1_data[(p*width + q + 1)*chs + c], 2);
							}
							d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
							d1 += pow(im0_edge_data[p*width + q + 1] - im1_edge_data[p*width + q + 1], 2);
						}
					}

					float dd = (sqrt(d1) + sqrt(d2));
					dd = max(dd, 1.0f);
					//int dd = d1 + d2;
					g->add_edge(pos, pos + 1, int(dd), int(dd));
				}
				if (node_in_ol^node_right_in_ol){//当前点(x,y)与(x+1,y)有一点在重叠区
					g->add_edge(pos, pos + 1, terminal_cost, terminal_cost);
				}
					
			}
			
			if (below_exist)
			{
				bool node_below_in0 = mask0_canvas_data[pos + width], node_below_in1 = mask1_canvas_data[pos + width];
				bool node_below_in_ol = node_below_in0 && node_below_in1;
				if (node_in_ol && node_below_in_ol)//当前点(x,y)与(x,y+1)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[((p + 1)*width + q)*chs + c] - im1_data[((p + 1)*width + q)*chs + c], 2);
							}
							d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
							d1 += pow(im0_edge_data[(p + 1)*width + q] - im1_edge_data[(p + 1)*width + q], 2);
						}
					}
					float dd = (sqrt(d1) + sqrt(d2));
					dd = max(dd, 1.0f);
					//int dd = d1 + d2;
					g->add_edge(pos, pos + width, int(dd), int(dd));
				}
				if (node_in_ol^node_below_in_ol){//当前点(x,y)与(x,y+1)有一点在重叠区
					g->add_edge(pos, pos + width, terminal_cost, terminal_cost);
				}
					
			}
			
		}
	}
	
	//define t-links
	Mat source_area = Mat::zeros(height, width, CV_8UC1);
	Mat sink_area = Mat::zeros(height, width, CV_8UC1);
	uchar *source_data = source_area.data;
	uchar *sink_data = sink_area.data;

	for (int col = 0; col < width; ++col)
	{
		for (int row = 0; row < height; ++row)
		{
			int pose = row*width + col;
			bool in0 = mask0_canvas_data[pose];
			bool in1 = mask1_canvas_data[pose];
			if (in0 && !in1)
			{
				g->add_tweights(pose,   /* capacities */  terminal_cost, 0);//link to source
				source_data[pose] = 255;
			}
			else if (!in0 && in1)
			{
				g->add_tweights(pose,   /* capacities */  0, terminal_cost);//link to sink
				sink_data[pose] = 255;
			}
			else if (in0 && in1)
			{	}
			else 
			{
				g->add_tweights(pose,   /* capacities */  1, 1);
				source_data[pose] = 120;
				sink_data[pose] = 120;
			}
		}
	}

	//imwrite(outDir+"/source_area.jpg", source_area);
	//imwrite(outDir+"/sink_area.jpg", sink_area);


	int flow = g->maxflow();
	Mat result_mask0_canvas = mask0_canvas.clone(), result_mask1_canvas = mask1_canvas.clone();
	uchar *result_mask0_data = result_mask0_canvas.data, *result_mask1_data = result_mask1_canvas.data;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			int pos = row*width + col;
			//source set refers to im0		
			if (g->what_segment(pos) == GraphType::SOURCE)//pixel at pos belongs to source set
			{
				result_mask0_data[pos] = 255;
				result_mask1_data[pos] = 0;
			}
			else
			{
				result_mask0_data[pos] = 0;
				result_mask1_data[pos] = 255;
			}	
		}
	}
	delete g;

	result_mask0_canvas = result_mask0_canvas & mask0_canvas;
	result_mask1_canvas = result_mask1_canvas & mask1_canvas;
	Mat result_mask0(result_mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	Mat result_mask1(result_mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	result_mask0.copyTo(mask0);
	result_mask1.copyTo(mask1);

	//imwrite(outDir + "/result_mask0.jpg", result_mask0_canvas);
	//imwrite(outDir + "/result_mask1.jpg", result_mask1_canvas);

}

void Stitching::liFindSeambyGraphcut(const vector<Mat>& images, const vector<Point>& corners, vector<Mat>& masks, Mat select_mask, int selected)
{
	int terminal_cost = INT_MAX;
	typedef Graph<int, int, int> GraphType;
	int num_images = images.size();
	int chs = images[0].channels();
	Mat result = images[0].clone();
	Point tl = corners[0];
	int psize = 2;

	for (int i = 0; i < num_images - 1; i++)
	{
		int minx = corners[0].x, maxx = minx + images[0].cols, miny = corners[0].y, maxy = miny + images[0].rows;
		//确定画布canvas大小
		for (int j = 1; j <= i + 1; j++)
		{
			minx = min(minx, corners[j].x);
			miny = min(miny, corners[j].y);
			maxx = max(maxx, corners[j].x + images[j].cols);
			maxy = max(maxy, corners[j].y + images[j].rows);
		}
		int width = maxx - minx, height = maxy - miny;
		int num_pixels = width*height;

		GraphType *G = new GraphType(num_pixels, num_pixels * 2);
		G->add_node(num_pixels);

		Mat mask1(height, width, CV_8U, Scalar(0));//mask1:所有图片的mask
		for (int j = 0; j <= i; j++)
		{
			Mat mask1_copy(mask1, Rect(corners[j].x - minx, corners[j].y - miny, masks[j].cols, masks[j].rows));
			mask1_copy |= masks[j];
		}
		Mat mask2(height, width, CV_8U, Scalar(0));//mask2:图片[i + 1]的mask
		Mat mask2_copy(mask2, Rect(corners[i + 1].x - minx, corners[i + 1].y - miny, masks[i + 1].cols, masks[i + 1].rows));
		masks[i + 1].copyTo(mask2_copy);

		Mat mask3(height, width, CV_8U, Scalar(0));//mask3:选中select图片的mask
		if ((selected == i || selected == i + 1) && !select_mask.empty())//当前image[i]作为source图像或者sink图像
		{
			Mat mask3_copy(mask3, Rect(corners[selected].x - minx, corners[selected].y - miny, select_mask.cols, select_mask.rows));
			select_mask.copyTo(mask3_copy);
		}
		//imwrite("mask1.png", mask1);
		//imwrite("mask2.png", mask2);

		Mat img1(height, width, result.type(), Scalar(0, 0, 0));
		Mat img1_copy(img1, Rect(tl.x - minx, tl.y - miny, result.cols, result.rows));
		result.copyTo(img1_copy);
		Mat img2(height, width, images[i + 1].type(), Scalar(0, 0, 0));
		Mat img2_copy(img2, Rect(corners[i + 1].x - minx, corners[i + 1].y - miny, images[i + 1].cols, images[i + 1].rows));
		images[i + 1].copyTo(img2_copy);

		//Mat grad1 = getGradientImg(img1);
		//Mat grad2 = getGradientImg(img2);
		//float *gd1_data = (float *)grad1.data;
		//float *gd2_data = (float *)grad2.data;

		Mat source(height, width, CV_8U, 0);
		Mat sink(height, width, CV_8U, 0);
		uchar *source_data = source.data;
		uchar *sink_data = sink.data;

		uchar* mk1_data = mask1.data;
		uchar* mk2_data = mask2.data;
		uchar* mk3_data = mask3.data;
		uchar* im1_data = img1.data;
		uchar* im2_data = img2.data;

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				int pos = y*width + x;
				bool in1 = mk1_data[pos];
				bool in2 = mk2_data[pos];
				bool in3 = mk3_data[pos];
				if (in3)//当前点(x,y)在~Mks[selected]上，即不在image[selected]上
				{
					if (selected == i)//不在image[i]上
					{
						G->add_tweights(pos, terminal_cost, 0);//当前点(x,y)连到source
						source_data[pos] = 255;
					}
						
					else if (selected == i + 1)//不在image[i+1]上
					{
						G->add_tweights(pos, 0, terminal_cost);//当前点(x,y)连到sink
						sink_data[pos] = 255;
					}
						
				}
				else if (in1 && !in2)//在canvas上，不在image[i+1]上
				{
					G->add_tweights(pos, terminal_cost, 0);
					source_data[pos] = 255;
				}
					
				else if (!in1 && in2)//在image[i+1]上，不在canvas上
				{
					G->add_tweights(pos, 0, terminal_cost);
					sink_data[pos] = 255;
				}
					
				else
				{
					G->add_tweights(pos, 1, 1);
					source_data[pos] = 120;
					sink_data[pos] = 120;
				}
					
			}
		}
		imwrite("output/test10/source.jpg", source);
		imwrite("output/test10/sink.jpg", sink);

		for (int y = 0; y < height - 1; y++)
		{
			for (int x = 0; x < width - 1; x++)
			{
				int pos = y*width + x;
				if (mk1_data[pos] && mk1_data[pos + 1] && mk2_data[pos] && mk2_data[pos + 1])//当前点(x,y)与(x+1,y)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im1_data[(p*width + q)*chs + c] - im2_data[p*width + q*chs + c], 2);
								d2 += pow(im1_data[(p*width + q + 1)*chs + c] - im2_data[(p*width + q + 1)*chs + c], 2);
							}
						}
					}
					float dd = (sqrt(d1) + sqrt(d2));///(gd1_data[pos]+gd1_data[pos+1]+gd2_data[pos]+gd2_data[pos+1]+EPSILON);
					G->add_edge(pos, pos + 1, int(dd), int(dd));
				}
				if (mk1_data[pos] && mk1_data[pos + width] && mk2_data[pos] && mk2_data[pos + width])//当前点(x,y)与(x,y+1)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im1_data[(p*width + q)*chs + c] - im2_data[(p*width + q)*chs + c], 2);
								d2 += pow(im1_data[((p + 1)*width + q)*chs + c] - im2_data[((p + 1)*width + q)*chs + c], 2);
							}
						}
					}
					float dd = (sqrt(d1) + sqrt(d2));///(gd1_data[pos]+gd1_data[pos+width]+gd2_data[pos]+gd2_data[pos+width]+EPSILON);
					G->add_edge(pos, pos + width, int(dd), int(dd));
				}
			}
		}

		// solve the max_flow
		float max_flow = G->maxflow();

		// save the results
		tl = Point(minx, miny);
		result.create(height, width, images[0].type());
		result.setTo(Scalar(0, 0, 0));
		uchar *re_data = result.data;
		for (int k = 0; k < num_pixels; k++)
		{
			if (G->what_segment(k) == GraphType::SOURCE)
			{
				if (mk2_data[k])
					mk2_data[k] = 0;
			}
			else	{
				if (mk1_data[k])
					mk1_data[k] = 0;
			}
			if (mk1_data[k])
			{
				for (int c = 0; c < chs; c++)
					re_data[k*chs + c] = im1_data[k*chs + c];
			}
			else	{
				for (int c = 0; c < chs; c++)
					re_data[k*chs + c] = im2_data[k*chs + c];
			}
		}
		if (i == 0)
		{
			Mat mask1_copy(mask1, Rect(corners[i].x - minx, corners[i].y - miny, masks[i].cols, masks[i].rows));
			mask1_copy.copyTo(masks[i]);
		}
		mask2_copy.copyTo(masks[i + 1]);
		if (selected >= 0)
			select_mask = mask3 & (mask1 | mask2);
		delete G;
	}

	imwrite("output/test10/mask0_Li_result.jpg", masks[0]);
	imwrite("output/test10/mask1_Li_result.jpg", masks[1]);
}

void Stitching::AlignmentbyH(Matrix<double, 3, 3, RowMajor> H)
{
	//构建__images_warped，__masks_warped，__corners
	__images_warped[0] = WarpImg(__img0,__corners[0], H);
	cout << "Img0 Warped" << endl;
	__images_warped[1] = __img1;
	__corners[1] = Point(0, 0);
	Mat AllWhite;
	AllWhite.create(__img0.size(), CV_8U);
	AllWhite.setTo(Scalar::all(255));
	Point2i rect_TL;
	__masks_warped[0] = WarpImg(AllWhite, rect_TL, H);
	cout << "Img1 Warped" << endl;
	__masks_warped[1] = AllWhite;
}

void Stitching::AlignmentbyMesh(Warping& meshWarper, Mat prewarpImg, Mat prewarpMask)
{
	__images_warped[0] = WarpImg(prewarpImg, __corners[0], meshWarper);
	__masks_warped[0] = WarpImg(prewarpMask,  Point(), meshWarper);
	__images_warped[1] = __img1;
	__masks_warped[1] = Mat(__img1.size(), CV_8U, Scalar(255));
	__corners[1] = Point(0, 0);
}

void Stitching::SetStripeImg(Mat &Img, uchar colBeg)
{
	int width = Img.cols, height = Img.rows;
	int stride = 50;
	int num_stride = width / stride;
	//int remain_hori = width % psize;
	//int pnum_vert = height / psize, remain_vert = height % psize;
	//int color_stride = int(float(colEnd - colBeg) / float(num_stride));
	int color_stride = 10;
	//int color_stride_vert = int(float(colEnd - colBeg) / float(pnum_vert));
	uchar *imdata = Img.data;
	int ch = Img.channels();
	//for (int i = 0; i < pnum_hori; ++i)
	//{
	//	for (int j = 0; j < pnum_vert; ++j)
	//	{
	//		
	//		uchar value_hori = colBeg + i * color_stride_hori, value_vert = colBeg + j*color_stride_vert;
	//		for (int p_col = 0; p_col < psize; ++p_col)
	//			for (int p_row = 0; p_row < psize; ++p_row)
	//			{
	//				int x = i*psize + p_col, y = j*psize + p_row;
	//				int pixel_pos = (y*width+x)*ch;
	//				imdata[pixel_pos] = value_hori;
	//				imdata[pixel_pos + 1] = value_hori;
	//				imdata[pixel_pos + 2] = value_hori;
	//			}		
	//	}
	//}
	for (int i = 0; i < num_stride; ++i)
	{
		uchar value = colBeg + i * color_stride;
		for (int p_col = 0; p_col < stride; ++p_col)
		for (int p_row = 0; p_row < height; ++p_row)
		{
			int x = i*stride + p_col, y = p_row;
			int pixel_pos = (y*width + x)*ch;
			imdata[pixel_pos] = value;
			//imdata[pixel_pos + 1] = value;
			imdata[pixel_pos + 2] = 255-value;
		}
	}

}

void Stitching::SetPlaidImg(Mat &Img, uchar colBeg, uchar colEnd)
{
	int width = Img.cols, height = Img.rows;
	int psize = 25;
	//int remain_hori = width % psize;
	int pnum_vert = height / psize, remain_vert = height % psize;
	int pnum_hori = width / psize, remain_hori = width % psize;
	int color_stride_hori = int(float(colEnd - colBeg) / float(pnum_hori));
	int color_stride_vert = int(float(colEnd - colBeg) / float(pnum_vert));
	uchar *imdata = Img.data;
	int ch = Img.channels();
	for (int i = 0; i < pnum_hori; ++i)
	{
		for (int j = 0; j < pnum_vert; ++j)
		{
			
			uchar value_hori = colBeg + i * color_stride_hori, value_vert = colBeg + j*color_stride_vert;
			for (int p_col = 0; p_col < psize; ++p_col)
				for (int p_row = 0; p_row < psize; ++p_row)
				{
					int x = i*psize + p_col, y = j*psize + p_row;
					int pixel_pos = (y*width+x)*ch;
					imdata[pixel_pos] = value_hori;
					imdata[pixel_pos + 1] = 0;
					imdata[pixel_pos + 2] = value_vert;
				}		
		}
	}

}

void Stitching::assignVector(string &inFileName, vector<uchar> &inVec)
{
	inVec=_select;
	ifstream inFile(inFileName);
	if (inFile.is_open()){
		for (int idx = 0; idx < inVec.size(); ++idx){
			if (_select[idx] == 1){
				int elem;
				inFile >> elem;
				inVec[idx] = elem;
			}		
		}
	}
	inFile.close();
}

//Mat Stitching::findHomography_by_weight(InputArray _points1, InputArray _points2,
//	int method, double ransacReprojThreshold, OutputArray _mask, vector<float> weightVec)
//{
//	Mat points1 = _points1.getMat(), points2 = _points2.getMat();
//	int npoints = points1.checkVector(2);
//	CV_Assert(npoints >= 0 && points2.checkVector(2) == npoints &&
//		points1.type() == points2.type());
//
//	Mat H(3, 3, CV_64F);
//	CvMat _pt1 = points1, _pt2 = points2;
//	CvMat matH = H, c_mask, *p_mask = 0;
//	if (_mask.needed())
//	{
//		_mask.create(npoints, 1, CV_8U, -1, true);
//		p_mask = &(c_mask = _mask.getMat());
//	}
//	bool ok = FindHomography(&_pt1, &_pt2, &matH, method, ransacReprojThreshold, p_mask) > 0;
//	if (!ok)
//		H = Scalar(0);
//	return H;
//}

void Stitching::ShowAlignQuality(string outDir, const vector<detail::ImageFeatures> &fea, 
	const vector<MatchesInfo> &mat, const Mat& seamMask, const Mat& seamQualityMat, const Mat& canvasIm)
{
	_mkdir(outDir.c_str());
	int num_mat = mat[1].matches.size();
	double *h_data = (double*)mat[1].H.data;
	Matrix<double, 3, 3, RowMajor> H;
	for (int j = 0; j < 3; ++j)
	for (int k = 0; k < 3; ++k)
	{
		double tmp = h_data[j * 3 + k];
		H(j, k) = tmp;//用Mat型H的data对Matrix型H赋值
	}

	const vector<KeyPoint>& points0 = fea[0].keypoints;
	const vector<KeyPoint>& points1 = fea[1].keypoints;
	Mat canvas_img;
	canvasIm.copyTo(canvas_img);

	//画匹配点
	Mat img0_canvas, img1_canvas;
	generateCanvasImgs(__images_warped[0], __images_warped[1], __corners[0], __corners[1], img0_canvas, img1_canvas);
	for (int idx_mat = 0; idx_mat < num_mat; idx_mat++)
	{
		if (mat[1].inliers_mask[idx_mat] == 0) continue;
		int idx_fea0 = mat[1].matches[idx_mat].queryIdx;
		int idx_fea1 = mat[1].matches[idx_mat].trainIdx;
		Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;
		Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
		calcPoint_after_H(pt0, pt0_warped, H.data());
		pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
		pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
		convert_to_CanvasCoordinate(__corners, pt0_in_warpImg0, pt0_canvas, 0);
		convert_to_CanvasCoordinate(__corners, pt1, pt1_canvas, 1);
		VectorXf Patch0(_pdim);
		float* patchData0 = Patch0.data();
		PatchInitialization_on_Pixel(Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), img0_canvas, patchData0);
		VectorXf Patch_dst(_pdim);
		float* patchData_dst = Patch_dst.data();
		PatchInitialization_on_Pixel(Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), img1_canvas, patchData_dst);
		float aff = calAffinitybtPatches(patchData_dst, patchData0);
		float c = min(max(aff, 0), 1);
		circle(canvas_img, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img0_canvas, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(canvas_img, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img1_canvas, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/img0_canvas.jpg", img0_canvas);
	imwrite(outDir + "/img1_canvas.jpg", img1_canvas);

	//画缝隙
	for (int i = 0; i < canvas_img.rows; ++i){
		for (int j = 0; j < canvas_img.cols; ++j){
			Point2i pt(j,i);
			bool is_seam = seamMask.at<uchar>(i, j);
			if (is_seam){
				float c = seamQualityMat.at<float>(i, j);
				circle(canvas_img, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
			}		
		}
	}
	imwrite(outDir+"/AffinityImg.jpg", canvas_img);

}

void Stitching::ShowAlignQuality(string outDir, const vector<detail::ImageFeatures> &fea,
	const vector<MatchesInfo> &mat, const Mat& seamMask, const Mat& seamQualityMat, const Mat& canvasIm,
	const Warping& meshWarper)
{
	_mkdir(outDir.c_str());
	int num_mat = mat[1].matches.size();
	const vector<KeyPoint>& points0 = fea[0].keypoints;
	const vector<KeyPoint>& points1 = fea[1].keypoints;
	Mat canvas_img;
	canvasIm.copyTo(canvas_img);

	//画匹配点
	Mat img0_canvas, img1_canvas;
	generateCanvasImgs(__images_warped[0], __images_warped[1], __corners[0], __corners[1], img0_canvas, img1_canvas);
	for (int idx_mat = 0; idx_mat < num_mat; idx_mat++)
	{
		if (mat[1].inliers_mask[idx_mat] == 0) continue;
		//计算patch0Data
		int idx_fea0 = mat[1].matches[idx_mat].queryIdx;
		Point2f pt0 = points0[idx_fea0].pt, pt0_canvas;
		pt0_canvas=orgPt02canPt(pt0, meshWarper, __corners);
		pt0_canvas.x = min(max(pt0_canvas.x, 0), canvasIm.cols - 1);
		pt0_canvas.y = min(max(pt0_canvas.y, 0), canvasIm.rows - 1);
		VectorXf Patch0(_pdim);
		float* patch0Data = Patch0.data();
		PatchInitialization_on_Pixel(Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), img0_canvas, patch0Data);
		//计算patch1Data
		int idx_fea1 = mat[1].matches[idx_mat].trainIdx;
		Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
		convert_to_CanvasCoordinate(__corners, pt1, pt1_canvas, 1);
		pt1_canvas.x = min(max(pt1_canvas.x, 0), canvasIm.cols - 1);
		pt1_canvas.y = min(max(pt1_canvas.y, 0), canvasIm.rows - 1);
		VectorXf Patch1(_pdim);
		float* patch1Data = Patch1.data();
		PatchInitialization_on_Pixel(Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), img1_canvas, patch1Data);
		//计算相似度
		float aff = calAffinitybtPatches(patch1Data, patch0Data);
		float c = min(max(aff, 0), 1);
		circle(canvas_img, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img0_canvas, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(canvas_img, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img1_canvas, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/im0_canvas.jpg", img0_canvas);
	imwrite(outDir + "/im1_canvas.jpg", img1_canvas);

	//画缝隙
	for (int i = 0; i < canvas_img.rows; ++i){
		for (int j = 0; j < canvas_img.cols; ++j){
			Point2i pt(j, i);
			bool is_seam = seamMask.at<uchar>(i, j);
			if (is_seam){
				float c = seamQualityMat.at<float>(i, j);
				circle(canvas_img, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
			}
		}
	}
	imwrite(outDir + "/AffinityImg.jpg", canvas_img);

}


void Stitching::generateCanvasMasks(const Mat& im0, const Mat& im1, const Point2i& corner0, const Point2i& corner1, Mat &outIm0, Mat &outIm1)
{
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//construct im0_canvas,im1_canvas and mask0_canvas,mask1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);

	im0_canvas.copyTo(outIm0);
	im1_canvas.copyTo(outIm1);

}
void Stitching::generateCanvasImgs(const Mat& im0, const Mat& im1, const Point2i& corner0, const Point2i& corner1, Mat &outIm0, Mat &outIm1)
{
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//construct im0_canvas,im1_canvas and mask0_canvas,mask1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);

	im0_canvas.copyTo(outIm0);
	im1_canvas.copyTo(outIm1);
}
void Stitching::convert_to_CanvasCoordinate(const vector<Point2i>& corners, Point2f& inPt, Point2f& outPt, bool Pt_in_Img1)
{
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);

	if (Pt_in_Img1){
		outPt.x = inPt.x + corners[1].x - mincol;
		outPt.y = inPt.y + corners[1].y - minrow;
	}
	else
	{
		outPt.x = inPt.x + corners[0].x - mincol;
		outPt.y = inPt.y + corners[0].y - minrow;
	}
	
}
Mat Stitching::generateSeamMask_on_Canvas(string outDir)
{
	_mkdir(outDir.c_str());
	Mat mask0, mask1;
	generateCanvasMasks(__masks_warped[0], __masks_warped[1], __corners[0], __corners[1], mask0, mask1);
	//imwrite(outDir + "/mask0_canvas.jpg", mask0);
	//imwrite(outDir + "/mask1_canvas.jpg", mask1);
	//对mask0, mask1进行膨胀，然后与运算，得到缝隙mask
	Mat mask0_dilated, mask1_dilated;
	dilate(mask0, mask0_dilated, Mat());
	dilate(mask1, mask1_dilated, Mat());
	Mat __seam_mask = mask0_dilated & mask1_dilated;

	imwrite(outDir+"/seam_mask.jpg", __seam_mask);
	return __seam_mask;
}

bool Stitching::calcHomoFromOutliers(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, vector<uchar> outliersMask)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < 5)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion
	vector<Point2f> src_points, dst_points;
	for (int j = 0; j < num_matched; j++)
	{
		if (!outliersMask[j])
			continue;
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	}
	int num_outliers = src_points.size();
	_inliers_mask_of_outliers.resize(num_outliers);
	m.H = findHomography(src_points, dst_points, CV_RANSAC, RANSAC_THRE1, _inliers_mask_of_outliers);
	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	return true;
}

void Stitching::ShowInliersOfOutliers(string outDir, vector<uchar> &outliersMaskOfAll, vector<uchar> &inliersMaskOfOutliers)
{
	_mkdir(outDir.c_str());
	RNG rng = theRNG();
	const vector<DMatch>& valid_matches = __matches[1].matches;
	int width = _width, height = 2 * _height;
	Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	Mat imgA(imgAB, Rect(0, 0, _width, _height));
	__img0.copyTo(imgA);
	Mat imgB(imgAB, Rect(0, _height, _width, _height));
	__img1.copyTo(imgB);
	const vector<KeyPoint>& points1 = __features[0].keypoints;
	const vector<KeyPoint>& points2 = __features[1].keypoints;

	int idx_outliers = 0;
	for (int k = 0; k < _num_matched; k++)
	{
		//if (!outliersMaskOfAll[k])//当前点不是outlier，则跳过
		//	continue;
		//if (!inliersMaskOfOutliers[idx_outliers])//当前点是outlier,但是不是outlier选中的inlier，则跳过
		//{
		//	idx_outliers++;
		//	continue;
		//}		
		if (outliersMaskOfAll[k]){//当前点是outlier
			if (inliersMaskOfOutliers[idx_outliers]){//当前点是outliers选中的inlier
				const DMatch& t = valid_matches[k];
				Point2f p1 = points1[t.queryIdx].pt;
				Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
				Scalar newvalue(rng(256), rng(256), rng(256));
				line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
				circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
				circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
				idx_outliers++;
			}
			else//当前点是outliers没选中的outlier
			{
				idx_outliers++;
			}
		}
		
	}

	imwrite(outDir+"/inliers.jpg", imgAB);
}

void Stitching::initCluster2()
{
	_cluster2.resize(_num_matched);
	int idx_outliers = 0;
	for (int k = 0; k < _num_matched; k++)
	{
		if (_outliers_mask[k]){//当前点是outlier
			if (_inliers_mask_of_outliers[idx_outliers]){//当前点是outliers选中的inlier
				_cluster2[k] = 1;
				idx_outliers++;
			}
			else//当前点是outliers没选中的outlier
			{
				idx_outliers++;
			}
		}

	}
}

Rect Stitching::transCanvasRect0ToOriginalRect0(Rect &inRect, Matrix<double, 3, 3, RowMajor> H)
{
	Rect output;
	//构建corners
	vector<Point> corners(2);
	WarpImg(__img0, corners[0], H);
	corners[1] = Point(0, 0);

	//对输入rect四个顶点变换
	//canvas坐标系
	Point TL(inRect.x, inRect.y), TR(inRect.x + inRect.width - 1, inRect.y), 
		BL(inRect.x, inRect.y + inRect.height - 1), 
		BR(inRect.x + inRect.width - 1, inRect.y + inRect.height - 1);
	//变到warped_im0坐标系
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	Point TL_in_warpim0 = corners[0] - Point(mincol, minrow);
	TL.x = TL.x - TL_in_warpim0.x;
	TL.y = TL.y - TL_in_warpim0.y;
	TR.x = TR.x - TL_in_warpim0.x;
	TR.y = TR.y - TL_in_warpim0.y;
	BL.x = BL.x - TL_in_warpim0.x;
	BL.y = BL.y - TL_in_warpim0.y;
	BR.x = BR.x - TL_in_warpim0.x;
	BR.y = BR.y - TL_in_warpim0.y;
	//变到im1坐标系
	Point TL_in_im0 = corners[0] - corners[1];
	TL.x = TL.x + TL_in_im0.x;
	TL.y = TL.y + TL_in_im0.y;
	TR.x = TR.x + TL_in_im0.x;
	TR.y = TR.y + TL_in_im0.y;
	BL.x = BL.x + TL_in_im0.x;
	BL.y = BL.y + TL_in_im0.y;
	BR.x = BR.x + TL_in_im0.x;
	BR.y = BR.y + TL_in_im0.y;
	//反变换
	Point2f TL_dst, TR_dst, BL_dst, BR_dst;
	Matrix<double, 3, 3, RowMajor> H_inv = H.inverse();
	double *homo = H_inv.data();
	calcPoint_after_H(Point2f(TL), TL_dst, homo);
	calcPoint_after_H(Point2f(TR), TR_dst, homo);
	calcPoint_after_H(Point2f(BL), BL_dst, homo);
	calcPoint_after_H(Point2f(BR), BR_dst, homo);
	//取包络
	float x_min, x_max, y_min, y_max;
	x_min = min(TL_dst.x, min(TR_dst.x, min(BL_dst.x, BR_dst.x)));
	y_min = min(TL_dst.y, min(TR_dst.y, min(BL_dst.y, BR_dst.y)));
	x_max = max(TL_dst.x, max(TR_dst.x, max(BL_dst.x, BR_dst.x)));
	y_max = max(TL_dst.y, max(TR_dst.y, max(BL_dst.y, BR_dst.y)));
	output.x = x_min;
	output.y = y_min;
	output.width = x_max - x_min + 1;
	output.height = y_max - y_min + 1;
	//保证输出矩形有效
	output.x = min(max(0, output.x), __img0.cols - 1);
	output.y = min(max(0, output.y), __img0.cols - 1);
	output.width = min(__img0.cols - 1 - output.x, output.width);
	output.height = min(__img0.rows - 1 - output.y, output.height);

	return output;
}

Rect Stitching::transCanvasRect1ToOriginalRect1(Rect &inRect, Matrix<double, 3, 3, RowMajor> H)
{
	Rect output;
	//构建corners
	vector<Point> corners(2);
	WarpImg(__img0, corners[0], H);
	corners[1] = Point(0, 0);

	//对输入rect四个顶点变换
	//canvas坐标系
	Point TL(inRect.x, inRect.y), TR(inRect.x + inRect.width - 1, inRect.y),
		BL(inRect.x, inRect.y + inRect.height - 1),
		BR(inRect.x + inRect.width - 1, inRect.y + inRect.height - 1);
	//变到warped_im1坐标系
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	Point TL_in_warpim0 = corners[1] - Point(mincol, minrow);
	TL.x = TL.x - TL_in_warpim0.x;
	TL.y = TL.y - TL_in_warpim0.y;
	TR.x = TR.x - TL_in_warpim0.x;
	TR.y = TR.y - TL_in_warpim0.y;
	BL.x = BL.x - TL_in_warpim0.x;
	BL.y = BL.y - TL_in_warpim0.y;
	BR.x = BR.x - TL_in_warpim0.x;
	BR.y = BR.y - TL_in_warpim0.y;
	//变到im1坐标系
	Point TL_in_im0 = corners[1] - Point(0, 0);
	TL.x = TL.x - TL_in_im0.x;
	TL.y = TL.y - TL_in_im0.y;
	TR.x = TR.x - TL_in_im0.x;
	TR.y = TR.y - TL_in_im0.y;
	BL.x = BL.x - TL_in_im0.x;
	BL.y = BL.y - TL_in_im0.y;
	BR.x = BR.x - TL_in_im0.x;
	BR.y = BR.y - TL_in_im0.y;

	output.x = TL.x;
	output.y = TL.y;
	output.width = TR.x - TL.x + 1;
	output.height = BL.y - TL.y + 1;
	//保证输出矩形有效
	output.x = min(max(0, output.x), __img0.cols - 1);
	output.y = min(max(0, output.y), __img0.cols - 1);
	output.width = min(__img1.cols - 1 - output.x, output.width);
	output.height = min(__img1.rows - 1 - output.y, output.height);

	return output;
}


bool Stitching::initLocalFeasMats(string outDir, vector<Point> inSeamSection, Matrix<double, 3, 3, RowMajor> H, vector<ImageFeatures> &features, vector<MatchesInfo> &matches)
{
	_mkdir(outDir.c_str());
	features.clear();
	matches.clear();

	//找出缝隙段的矩形包络region
	int maxrow = 0, minrow = __seam_quality_map.rows - 1;
	int maxcol = 0, mincol = __seam_quality_map.cols - 1;
	for (auto &point : inSeamSection){
		int row = point.y, col = point.x;
		if (row < minrow)	minrow = row;
		if (row > maxrow)	maxrow = row;
		if (col < mincol)	mincol = col;
		if (col > maxcol)	maxcol = col;
	}
	Point rect_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	int rect_wd = maxcol - mincol + 1;
	int rect_ht = maxrow - minrow + 1;
	Point rect_tl(rect_center - Point(rect_wd / 2, rect_ht / 2));
	Rect orig_rect(rect_tl.x, rect_tl.y, rect_wd, rect_ht);
	//膨胀包络
	int rect_slide = max(rect_wd, rect_ht);
	rect_slide *= dilate_times;
	rect_tl = Point(rect_center - Point(rect_slide / 2, rect_slide / 2));
	Rect region(rect_tl.x, rect_tl.y, rect_slide, rect_slide);


	Rect orig_region_im0 = transCanvasRect0ToOriginalRect0(region,H);
	Rect orig_region_im1 = transCanvasRect1ToOriginalRect1(region,H);
	//设置局部区域掩膜mask0,mask1
	Mat mask0(__img0.size(), CV_8UC1, Scalar(0));
	Mat mask1(__img1.size(), CV_8UC1, Scalar(0));
	Mat region_mask0(mask0, orig_region_im0);
	Mat region_mask1(mask1, orig_region_im1);
	region_mask0.setTo(Scalar(255));
	region_mask1.setTo(Scalar(255));

	Mat im0_region;
	__img0(orig_region_im0).copyTo(im0_region);
	Mat im1_region;
	__img1(orig_region_im1).copyTo(im1_region);
	imwrite(outDir + "/Rect0.jpg", mask0);
	imwrite(outDir + "/Rect1.jpg", mask1);
	imwrite(outDir + "/im0_region.jpg", im0_region);
	imwrite(outDir + "/im1_region.jpg", im1_region);

	if (orig_region_im0.width < 30 || orig_region_im0.height < 30 || orig_region_im1.width < 30 || orig_region_im1.height < 30)
		return false;

	/*
	特征点选择有两种方法：
	1.直接从原来的__features中筛选出在mask0、mask1中的特征点
	2.重新对局部区域im0_region、im1_region进行特征点检测(不能直接检测)
	*/

	//筛选特征点方法2:先对img_region进行特征点检测，再改变每个特征点坐标到img坐标系上
	//vector<detail::ImageFeatures> features;
	features.resize(2);

	//SIFT特征点检测
	SIFT sift(1000);
	Mat gray;
	cvtColor(im0_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[0].keypoints, features[0].descriptors);
	features[0].img_idx = 0;
	features[0].img_size = __img0.size();
	cvtColor(im1_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[1].keypoints, features[1].descriptors);
	features[1].img_idx = 1;
	features[1].img_size = __img1.size();

	if (features[0].keypoints.size() == 0 || features[1].keypoints.size() == 0)
		return false;

	//features[0]转换坐标系
	Point2f rectTL_in_img0(orig_region_im0.x, orig_region_im0.y);
	for (int i = 0; i < features[0].keypoints.size(); ++i){
		KeyPoint& keypoint = features[0].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img0 + keypoint.pt;
	}
	//features[1]转换坐标系
	Point2f rectTL_in_img1(orig_region_im1.x, orig_region_im1.y);
	for (int i = 0; i < features[1].keypoints.size(); ++i){
		KeyPoint& keypoint = features[1].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img1 + keypoint.pt;
	}


	//特征点匹配
	float match_conf = 0.3f;
	//vector<detail::MatchesInfo> matches;
	matches.resize(4);
	detail::BestOf2NearestMatcher matcher(0, match_conf);
	matcher(features, matches);
	matcher.collectGarbage();

	bool is_valid = calcHomoFromMatches(matches[1], features[0], features[1], RANSAC_THRE1);
	if (is_valid){
		calcDualMatches(matches[2], matches[1]);
		return true;
	}
	else
		return false;
	
	
	//赋值全局变量_H、
	/*赋值_H前需要先判断H是否可靠！！！*/
	//double *homography = (double*)matches[1].H.data;//指向Mat型变量H的data的指针
	//for (int j = 0; j < 3; ++j)
	//for (int k = 0; k < 3; ++k)
	//{
	//	double tmp = homography[j * 3 + k];
	//	_H(j, k) = tmp;//用Mat型H的data对Matrix型_H赋值
	//}

	////调试，显示特征点和inliers匹配点
	//const vector<KeyPoint>& fea_points0 = features[0].keypoints;
	//Mat img0;
	//__img0.copyTo(img0);
	//for (int k = 0; k < features[0].keypoints.size(); k++)
	//{
	//	Point2f pt = fea_points0[k].pt;
	//	circle(img0, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/features0.jpg", img0);
	//const vector<KeyPoint>& fea_points1 = features[1].keypoints;
	//Mat img1;
	//__img1.copyTo(img1);
	//for (int k = 0; k < features[1].keypoints.size(); k++)
	//{
	//	Point2f pt = fea_points1[k].pt;
	//	circle(img1, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/features1.jpg", img1);
	//_mkdir(string(outDir + "/Matches").c_str());
	//RNG rng = theRNG();
	//const vector<DMatch>& valid_matches = matches[1].matches;
	//const vector<uchar>& inliers_mask = matches[1].inliers_mask;
	//int width = _width, height = 2 * _height;
	//Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	//Mat imgA(imgAB, Rect(0, 0, _width, _height));
	//__img0.copyTo(imgA);
	//Mat imgB(imgAB, Rect(0, _height, _width, _height));
	//__img1.copyTo(imgB);
	//const vector<KeyPoint>& points1 = features[0].keypoints;
	//const vector<KeyPoint>& points2 = features[1].keypoints;
	//for (int k = 0; k < matches[1].matches.size(); k++)
	//{
	//	if (!inliers_mask[k])
	//		continue;
	//	const DMatch& t = valid_matches[k];
	//	Point2f p1 = points1[t.queryIdx].pt;
	//	Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
	//	Scalar newvalue(rng(256), rng(256), rng(256));
	//	line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
	//	circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
	//	circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/Matches/inliers.jpg", imgAB);

	//调试：
	//__features.clear();
	//__features = features;
	//__matches.clear();
	//__matches = matches;
	//_num_matches.push_back(__matches[1].matches.size());
}

bool Stitching::addLocalFeasMats(string outDir, vector<Point> inSeamSection, Matrix<double, 3, 3, RowMajor> H, vector<ImageFeatures> &features, vector<MatchesInfo> &matches)
{
	_mkdir(outDir.c_str());
	features.clear();
	matches.clear();
	//构建corners
	vector<Point> corners(2);
	WarpImg(__img0, corners[0], H);
	corners[1] = Point(0, 0);

	//构建局部区域掩膜local_mask0
	vector<Point> seam_section0;
	int sum_x = 0, sum_y = 0;
	for (auto pt : inSeamSection){
		Point pt0 = canPt2orgPt0(pt, H, corners);
		seam_section0.push_back(pt0);
		sum_x += pt0.x;
		sum_y += pt0.y;
	}
	auto end_unique0 = unique(seam_section0.begin(), seam_section0.end(),
		[](const Point& a, const Point& b){return (a.x == b.x) && (a.y == b.y); });
	seam_section0.erase(end_unique0, seam_section0.end());
	Point center_pt0(sum_x / seam_section0.size(), sum_y / seam_section0.size());
	Mat local_mask0(__img0.size(), CV_8UC1, Scalar(0));
	for (int row = 0; row < __simg0.rows; ++row){
		for (int col = 0; col < __simg0.cols; ++col){
			Point p(col,row);
			Vec3b p_color = __simg0.at<Vec3b>(row, col);
			float aff = 0;
			//离缝隙中心太远的点不需要
			float dis2center = ((p.x - center_pt0.x)*(p.x - center_pt0.x) + (p.y - center_pt0.y)*(p.y - center_pt0.y)) / (2 * _sigma_spatial*_sigma_spatial);
			dis2center = exp(-dis2center);
			if (dis2center< 0.3)
				continue;

			//计算点p和q的位置距离与颜色距离，得到相似度
			for (auto q : seam_section0){
				Vec3b q_color = __simg0.at<Vec3b>(q.y, q.x);
				float color_dist = 0, spatial_dist = 0;
				spatial_dist = ((p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y)) / (2 * _sigma_spatial*_sigma_spatial);
				spatial_dist = exp(-spatial_dist);
				if (spatial_dist < 0.1)
					continue;
				for (int c = 0; c<__simg0.channels(); ++c){
					color_dist += (p_color[c] - q_color[c])*(p_color[c] - q_color[c]);
				}
				color_dist = (color_dist / (2 * _sigma_color*_sigma_color));
				color_dist = exp(-color_dist);
				aff += spatial_dist*color_dist;//用乘法？？？
			}
			if (aff > 1){
				local_mask0.at<uchar>(row, col) = 255;
			}
		}
	}
	dilate(local_mask0, local_mask0, Mat(), Point(-1, -1), 4);
	imwrite(outDir + "/local_mask0.jpg", local_mask0);

	//构建局部区域掩膜local_mask1
	vector<Point> seam_section1;
	sum_x = 0, sum_y = 0;
	for (auto pt : inSeamSection){
		Point pt1 = canPt2orgPt1(pt, corners);
		seam_section1.push_back(pt1);
		sum_x += pt1.x;
		sum_y += pt1.y;
	}
	auto end_unique1 = unique(seam_section1.begin(), seam_section1.end(),
		[](const Point& a, const Point& b){return (a.x == b.x) && (a.y == b.y); });
	seam_section1.erase(end_unique1, seam_section1.end());
	Point center_pt1(sum_x / seam_section1.size(), sum_y / seam_section1.size());
	Mat local_mask1(__img1.size(), CV_8UC1, Scalar(0));
	for (int row = 0; row < __simg1.rows; ++row){
		for (int col = 0; col < __simg1.cols; ++col){
			Point p(col, row);
			Vec3b p_color = __simg1.at<Vec3b>(row, col);
			float aff = 0;
			//离缝隙中心太远的点不需要
			float dis2center = ((p.x - center_pt1.x)*(p.x - center_pt1.x) + (p.y - center_pt1.y)*(p.y - center_pt1.y)) / (2 * _sigma_spatial*_sigma_spatial);
			dis2center = exp(-dis2center);
			if (dis2center< 0.3)
				continue;

			for (auto q : seam_section1){
				Vec3b q_color = __simg1.at<Vec3b>(q.y, q.x);
				float color_dist = 0, spatial_dist = 0;
				spatial_dist = ((p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y)) / (2 * _sigma_spatial*_sigma_spatial);
				spatial_dist = exp(-spatial_dist);
				if (spatial_dist < 0.1)
					continue;
				for (int c = 0; c<__simg1.channels(); ++c){
					color_dist += (p_color[c] - q_color[c])*(p_color[c] - q_color[c]);
				}
				color_dist = (color_dist / (2 * _sigma_color*_sigma_color));
				color_dist = exp(-color_dist);
				aff += spatial_dist*color_dist;//用乘法？？？
			}
			if (aff > 1){
				local_mask1.at<uchar>(row, col) = 255;
			}
		}
	}
	dilate(local_mask1, local_mask1, Mat(), Point(-1, -1), 4);
	imwrite(outDir + "/local_mask1.jpg", local_mask1);


	vector<Point> pt_region0 = getPointsfromMask(local_mask0), pt_region1 = getPointsfromMask(local_mask1);
	//找出local_mask0,local_mask1的矩形包络rect0,rect1
	int maxrow = 0, minrow = __img0.rows - 1;
	int maxcol = 0, mincol = __img0.cols - 1;
	for (auto &point : pt_region0){
		int row = point.y, col = point.x;
		if (row < minrow)	minrow = row;
		if (row > maxrow)	maxrow = row;
		if (col < mincol)	mincol = col;
		if (col > maxcol)	maxcol = col;
	}
	Point rect0_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	int rect0_wd = maxcol - mincol + 1;
	int rect0_ht = maxrow - minrow + 1;
	Point rect0_tl(rect0_center - Point(rect0_wd / 2, rect0_ht / 2)), rect0_br(rect0_tl.x + rect0_wd - 1, rect0_tl.y + rect0_ht - 1);
	rect0_tl.x = min(max(0, rect0_tl.x), __img0.cols - 1);
	rect0_tl.y = min(max(0, rect0_tl.y), __img0.rows - 1);
	rect0_br.x = min(max(0, rect0_br.x), __img0.cols - 1);
	rect0_br.y = min(max(0, rect0_br.y), __img0.rows - 1);
	Rect rect0(rect0_tl, rect0_br);

	maxrow = 0, minrow = __img1.rows - 1;
	maxcol = 0, mincol = __img1.cols - 1;
	for (auto &point : pt_region1){
		int row = point.y, col = point.x;
		if (row < minrow)	minrow = row;
		if (row > maxrow)	maxrow = row;
		if (col < mincol)	mincol = col;
		if (col > maxcol)	maxcol = col;
	}
	Point rect1_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	int rect1_wd = maxcol - mincol + 1;
	int rect1_ht = maxrow - minrow + 1;
	Point rect1_tl(rect1_center - Point(rect1_wd / 2, rect1_ht / 2)), rect1_br(rect1_tl.x + rect1_wd - 1, rect1_tl.y + rect1_ht - 1);
	rect1_tl.x = min(max(0, rect1_tl.x), __img1.cols - 1);
	rect1_tl.y = min(max(0, rect1_tl.y), __img1.rows - 1);
	rect1_br.x = min(max(0, rect1_br.x), __img1.cols - 1);
	rect1_br.y = min(max(0, rect1_br.y), __img1.rows - 1);
	Rect rect1(rect1_tl, rect1_br);


	//设置局部rect0_mask,rect1_mask
	Mat rect0_mask(__img0.size(), CV_8UC1, Scalar(0));
	Mat rect1_mask(__img1.size(), CV_8UC1, Scalar(0));
	Mat region_mask0(rect0_mask, rect0);
	Mat region_mask1(rect1_mask, rect1);
	region_mask0.setTo(Scalar(255));
	region_mask1.setTo(Scalar(255));

	Mat im0_region;
	__img0(rect0).copyTo(im0_region);
	Mat im1_region;
	__img1(rect1).copyTo(im1_region);
	//imwrite(outDir + "/Rect0.jpg", rect0_mask);
	//imwrite(outDir + "/Rect1.jpg", rect1_mask);
	imwrite(outDir + "/im0_region.jpg", im0_region);
	imwrite(outDir + "/im1_region.jpg", im1_region);

	if (rect0.width < 30 || rect0.height < 30 || rect1.width < 30 || rect1.height < 30)
		return false;

	//筛选特征点方法:先对局部区域img_region进行特征点检测，再把特征点局部坐标变换到img坐标系上
	features.resize(2);
	SIFT sift(500);
	Mat gray;
	cvtColor(im0_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[0].keypoints, features[0].descriptors);
	features[0].img_idx = 0;
	features[0].img_size = __img0.size();
	cvtColor(im1_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[1].keypoints, features[1].descriptors);
	features[1].img_idx = 1;
	features[1].img_size = __img1.size();

	if (features[0].keypoints.size() <= 5 || features[1].keypoints.size() <= 5)
		return false;

	//features[0]转换坐标系
	Point2f rectTL_in_img0(rect0.x, rect0.y);
	for (int i = 0; i < features[0].keypoints.size(); ++i){
		KeyPoint& keypoint = features[0].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img0 + keypoint.pt;
	}
	
	//features[1]转换坐标系
	Point2f rectTL_in_img1(rect1.x, rect1.y);
	for (int i = 0; i < features[1].keypoints.size(); ++i){
		KeyPoint& keypoint = features[1].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img1 + keypoint.pt;
	}


	if (__features.size() != 0){
		//----------------20170508 筛选特征点，去掉与__features[0]重复的特征点
		vector<Point2f> fea0_pts;//__features[0]特征点的位置信息
		for (auto &keypt : __features[0].keypoints){
			fea0_pts.push_back(keypt.pt);
		}
		detail::ImageFeatures unique_features0;//保存与__features[0]不重复的特征点
		unique_features0.img_idx = features[0].img_idx;
		unique_features0.img_size = features[0].img_size;
		for (int idx_fea = 0; idx_fea < features[0].keypoints.size(); ++idx_fea){
			KeyPoint& keypoint = features[0].keypoints[idx_fea];
			Mat Drow = features[0].descriptors.row(idx_fea);
			Point2f pt = keypoint.pt;
			auto it = find_if(fea0_pts.begin(), fea0_pts.end(),
				[pt](const Point2f &a)
			{
				float distance = (pt.x - a.x)*(pt.x - a.x) + (pt.y - a.y)*(pt.y - a.y);
				return distance < 2 * EPSILON * EPSILON;
			});//查看当前特征点pt是否已经存在于fea0_pts中，即是否为重复特征点
			if (it == fea0_pts.end()){//如果不重复，则当前特征点压入
				unique_features0.keypoints.push_back(keypoint);
				unique_features0.descriptors.push_back(Drow);
			}
		}
		features[0] = unique_features0;
		vector<Point2f> fea1_pts;//__features[1]特征点的位置信息
		for (auto &keypt : __features[1].keypoints){
			fea1_pts.push_back(keypt.pt);
		}
		detail::ImageFeatures unique_features1;//保存与__features[1]不重复的特征点
		unique_features1.img_idx = features[1].img_idx;
		unique_features1.img_size = features[1].img_size;
		for (int idx_fea = 0; idx_fea < features[1].keypoints.size(); ++idx_fea){
			KeyPoint& keypoint = features[1].keypoints[idx_fea];
			Mat Drow = features[1].descriptors.row(idx_fea);
			Point2f pt = keypoint.pt;
			auto it = find_if(fea1_pts.begin(), fea1_pts.end(),
				[pt](const Point2f &a)
			{
				float distance = (pt.x - a.x)*(pt.x - a.x) + (pt.y - a.y)*(pt.y - a.y);
				return distance < 2 * EPSILON * EPSILON;
			});//查看当前特征点pt是否已经存在于fea1_pts中，即是否为重复特征点
			if (it == fea1_pts.end()){//如果不重复，则当前特征点压入
				unique_features1.keypoints.push_back(keypoint);
				unique_features1.descriptors.push_back(Drow);
			}
		}
		features[1] = unique_features1;
	}

	//只选取在local_mask0和local_mask1上的特征点
	ImageFeatures feas0(features[0]), feas1(features[1]);
	features[0].keypoints.clear();
	features[0].descriptors = Mat();
	features[1].keypoints.clear();
	features[1].descriptors = Mat();
	for (auto it = feas0.keypoints.begin(); it != feas0.keypoints.end(); ++it){
		KeyPoint keypt = *it;
		Point pt0(keypt.pt);
		bool is_in_local0 = local_mask0.at<uchar>(pt0.y, pt0.x);
		Mat Drow = feas0.descriptors.row(it - feas0.keypoints.begin());
		if (is_in_local0){
			//选取该特征点
			features[0].keypoints.push_back(keypt);
			features[0].descriptors.push_back(Drow);
		}
	}
	for (auto it = feas1.keypoints.begin(); it != feas1.keypoints.end(); ++it){
		KeyPoint keypt = *it;
		Point pt1(keypt.pt);
		bool is_in_local1 = local_mask1.at<uchar>(pt1.y, pt1.x);
		Mat Drow = feas1.descriptors.row(it - feas1.keypoints.begin());
		if (is_in_local1){
			//选取该特征点
			features[1].keypoints.push_back(keypt);
			features[1].descriptors.push_back(Drow);
		}
	}

	cout << "features0 number: " << features[0].keypoints.size()<<endl;
	cout << "features1 number: " << features[1].keypoints.size() << endl;
	if (features[0].keypoints.size() < 5 || features[1].keypoints.size() < 5){
		return false;
	}
	float match_conf = 0.3f;
	detail::BestOf2NearestMatcher matcher(0, match_conf);
	matcher(features, matches);//仅对当前features[0]与features[1]特征点进行匹配
	matcher.collectGarbage();
	cout << "matches number: " << matches[1].matches.size() << endl;

	bool is_valid = calcHomoFromMatches(matches[1], features[0], features[1], RANSAC_THRE1);
	if (is_valid){
		calcDualMatches(matches[2], matches[1]);
		return true;
	}
	else
		return false;


	/*
	有两种方法来计算新的H：
	1.初始区域特征点加上新区域特征点得到全部特征点，对全部特征点用ransac法计算新H，这会可能导致（区域数目>2时）新区域里的特征点并没有匹配到
	2.每个区域各自检测特征点并各自用ransac法计算各自的inliers，然后对所有区域的inliers用8点法计算新的H
	*/
	//方法二
	//特征点匹配
	//float match_conf = 0.3f;
	//detail::BestOf2NearestMatcher matcher(0, match_conf);
	//matcher(features, matches);//仅对当前features[0]与features[1]特征点进行匹配
	//matcher.collectGarbage();
	////features[0]压入__features[0]中
	//int num_rectfeas0 = __features[0].keypoints.size();
	//for (int idx_fea = 0; idx_fea < features[0].keypoints.size(); ++idx_fea){
	//	KeyPoint& keypoint = features[0].keypoints[idx_fea];
	//	Mat Drow = features[0].descriptors.row(idx_fea);
	//	__features[0].keypoints.push_back(keypoint);
	//	__features[0].descriptors.push_back(Drow);
	//}
	////features[1]压入__features[1]中
	//int num_rectfeas1 = __features[1].keypoints.size();
	//for (int idx_fea = 0; idx_fea < features[1].keypoints.size(); ++idx_fea){
	//	KeyPoint& keypoint = features[1].keypoints[idx_fea];
	//	Mat Drow = features[1].descriptors.row(idx_fea);
	//	__features[1].keypoints.push_back(keypoint);
	//	__features[1].descriptors.push_back(Drow);
	//}
	////matches[1]压入__matches[1]中
	//for (int idx_mat = 0; idx_mat < matches[1].matches.size(); ++idx_mat){
	//	if (!matches[1].inliers_mask[idx_mat])
	//		continue;
	//	//更新matches[1]的matches、inliers_mask、num_inliers成员
	//	DMatch mat = matches[1].matches[idx_mat];
	//	mat.queryIdx += num_rectfeas0;//新增的inliers匹配点索引的新增特征点是从features[0].keypoints[num_rectfeas0]开始的
	//	mat.trainIdx += num_rectfeas1;
	//	__matches[1].matches.push_back(mat);
	//	__matches[1].inliers_mask.push_back(uchar(1));
	//	++__matches[1].num_inliers;
	//}
	//calcHomoFromMatches_8PointMethod(matches[1], features[0], features[1]);
	//calcDualMatches(matches[2], matches[1]);
	//_num_matches.push_back(__matches[1].matches.size());

	
	
	
	
	


	////赋值全局变量_H
	//double *homography = (double*)__matches[1].H.data;//指向Mat型变量H的data的指针
	//for (int j = 0; j < 3; ++j)
	//for (int k = 0; k < 3; ++k)
	//{
	//	double tmp = homography[j * 3 + k];
	//	_H(j, k) = tmp;//用Mat型H的data对Matrix型_H赋值
	//}


	//调试
	//const vector<KeyPoint>& fea_points0 = __features[0].keypoints;
	//Mat img0;
	//__img0.copyTo(img0);
	//for (int k = 0; k < __features[0].keypoints.size(); k++)
	//{
	//	Point2f pt = fea_points0[k].pt;
	//	circle(img0, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/features0.jpg", img0);
	//const vector<KeyPoint>& fea_points1 = __features[1].keypoints;
	//Mat img1;
	//__img1.copyTo(img1);
	//for (int k = 0; k < __features[1].keypoints.size(); k++)
	//{
	//	Point2f pt = fea_points1[k].pt;
	//	circle(img1, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar(255, 0, 0), CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/features1.jpg", img1);
	//_mkdir(string(outDir + "/Matches").c_str());
	//RNG rng = theRNG();
	//const vector<DMatch>& valid_matches = __matches[1].matches;
	//const vector<uchar>& inliers_mask = __matches[1].inliers_mask;
	//int width = _width, height = 2 * _height;
	//Mat imgAB = Mat::zeros(height, width, CV_8UC3);
	//Mat imgA(imgAB, Rect(0, 0, _width, _height));
	//__img0.copyTo(imgA);
	//Mat imgB(imgAB, Rect(0, _height, _width, _height));
	//__img1.copyTo(imgB);
	//const vector<KeyPoint>& points1 = __features[0].keypoints;
	//const vector<KeyPoint>& points2 = __features[1].keypoints;
	//for (int k = 0; k < __matches[1].matches.size(); k++)
	//{
	//	if (!inliers_mask[k])
	//		continue;
	//	const DMatch& t = valid_matches[k];
	//	Point2f p1 = points1[t.queryIdx].pt;
	//	Point2f p2 = points2[t.trainIdx].pt + Point2f(0, _height);
	//	Scalar newvalue(rng(256), rng(256), rng(256));
	//	line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
	//	circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
	//	circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
	//}
	//imwrite(outDir + "/Matches/inliers.jpg", imgAB);

}

bool Stitching::addLocalFeasMats(string outDir, vector<Point> inSeamSection, const Warping& meshWarper, vector<ImageFeatures> &features, vector<MatchesInfo> &matches)
/*
由缝隙段决定局部区域掩膜local_mask0和local_mask1（用到了综合考虑了缝隙的颜色和位置，即_sigma_color，_sigma_spatial）
局部区域掩膜即为选取的特征点区域：对local_mask0,local_mask1的矩形包络rect0,rect1检测特征点，保留掩膜上的特征点
然后，选择与上次迭代的特征点__features没有重复的那些特征点作为features
最后，进行匹配，得到matches
*/
{
	_mkdir(outDir.c_str());
	features.clear();
	matches.clear();
	
	//构建局部区域掩膜local_mask0
	vector<Point> seam_section0;
	int sum_x = 0, sum_y = 0;
	Matrix<double, 3, 3, RowMajor> H_inv = _H.inverse();
	for (auto pt : inSeamSection){
		Point pt0 = canPt2orgPt0(pt, meshWarper, _cpw_corners);//canvas坐标系pt点反变换，转换到img0_pre-warped坐标系
		pt0 = warpPt2orgPt0(pt0, H_inv.data(), _prewarp_corners);//img0_pre-warped坐标系pt点反变换，转换到img0坐标系
		if (pt0.x < 0 || pt0.x > __img0.cols - 1 || pt0.y < 0 || pt0.y > __img0.rows - 1)//无效的_homos[i]导致的无效的点
			continue;
		seam_section0.push_back(pt0);
		sum_x += pt0.x;
		sum_y += pt0.y;
	}
	auto end_unique0 = unique(seam_section0.begin(), seam_section0.end(),
		[](const Point& a, const Point& b){return (a.x == b.x) && (a.y == b.y); });
	seam_section0.erase(end_unique0, seam_section0.end());
	Point center_pt0(sum_x / seam_section0.size(), sum_y / seam_section0.size());
	Mat local_mask0(__img0.size(), CV_8UC1, Scalar(0));
	for (int row = 0; row < __simg0.rows; ++row){
		for (int col = 0; col < __simg0.cols; ++col){
			Point p(col, row);
			Vec3b p_color = __simg0.at<Vec3b>(row, col);
			float aff = 0;
			//离缝隙中心太远的点不需要
			float dis2center = ((p.x - center_pt0.x)*(p.x - center_pt0.x) + (p.y - center_pt0.y)*(p.y - center_pt0.y)) / (2 * _sigma_spatial*_sigma_spatial);
			dis2center = exp(-dis2center);
			if (dis2center< 0.3)
				continue;

			for (auto q : seam_section0){
				Vec3b q_color = __simg0.at<Vec3b>(q.y, q.x);
				float color_dist = 0, spatial_dist = 0;
				spatial_dist = ((p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y)) / (2 * _sigma_spatial*_sigma_spatial);
				spatial_dist = exp(-spatial_dist);
				if (spatial_dist < 0.1)
					continue;
				for (int c = 0; c<__simg0.channels(); ++c){
					color_dist += (p_color[c] - q_color[c])*(p_color[c] - q_color[c]);
				}
				color_dist = (color_dist / (2 * _sigma_color*_sigma_color));
				color_dist = exp(-color_dist);
				aff += spatial_dist*color_dist;//用乘法？？？
			}
			if (aff > 1){
				local_mask0.at<uchar>(row, col) = 255;
			}
		}
	}
	dilate(local_mask0, local_mask0, Mat(), Point(-1, -1), 4);
	imwrite(outDir + "/local_mask0.jpg", local_mask0);

	//构建局部区域掩膜local_mask1
	vector<Point> seam_section1;
	sum_x = 0, sum_y = 0;
	for (auto pt : inSeamSection){
		Point pt1 = canPt2orgPt1(pt, _cpw_corners);
		if (pt1.x < 0 || pt1.x > __img1.cols - 1 || pt1.y < 0 || pt1.y > __img1.rows - 1)//无效的的点
			continue;
		seam_section1.push_back(pt1);
		sum_x += pt1.x;
		sum_y += pt1.y;
	}
	auto end_unique1 = unique(seam_section1.begin(), seam_section1.end(),
		[](const Point& a, const Point& b){return (a.x == b.x) && (a.y == b.y); });
	seam_section1.erase(end_unique1, seam_section1.end());
	Point center_pt1(sum_x / seam_section1.size(), sum_y / seam_section1.size());
	Mat local_mask1(__img1.size(), CV_8UC1, Scalar(0));
	for (int row = 0; row < __simg1.rows; ++row){
		for (int col = 0; col < __simg1.cols; ++col){
			Point p(col, row);
			Vec3b p_color = __simg1.at<Vec3b>(row, col);
			float aff = 0;
			//离缝隙中心太远的点不需要
			float dis2center = ((p.x - center_pt1.x)*(p.x - center_pt1.x) + (p.y - center_pt1.y)*(p.y - center_pt1.y)) / (2 * _sigma_spatial*_sigma_spatial);
			dis2center = exp(-dis2center);
			if (dis2center< 0.3)
				continue;

			for (auto q : seam_section1){
				Vec3b q_color = __simg1.at<Vec3b>(q.y, q.x);
				float color_dist = 0, spatial_dist = 0;
				spatial_dist = ((p.x - q.x)*(p.x - q.x) + (p.y - q.y)*(p.y - q.y)) / (2 * _sigma_spatial*_sigma_spatial);
				spatial_dist = exp(-spatial_dist);
				if (spatial_dist < 0.1)
					continue;
				for (int c = 0; c<__simg1.channels(); ++c){
					color_dist += (p_color[c] - q_color[c])*(p_color[c] - q_color[c]);
				}
				color_dist = (color_dist / (2 * _sigma_color*_sigma_color));
				color_dist = exp(-color_dist);
				aff += spatial_dist*color_dist;//用乘法？？？
			}
			if (aff > 1){
				local_mask1.at<uchar>(row, col) = 255;
			}
		}
	}
	dilate(local_mask1, local_mask1, Mat(), Point(-1, -1), 4);
	imwrite(outDir + "/local_mask1.jpg", local_mask1);


	vector<Point> pt_region0 = getPointsfromMask(local_mask0), pt_region1 = getPointsfromMask(local_mask1);
	//找出local_mask0,local_mask1的矩形包络rect0,rect1
	int maxrow = 0, minrow = __img0.rows - 1;
	int maxcol = 0, mincol = __img0.cols - 1;
	for (auto &point : pt_region0){
		int row = point.y, col = point.x;
		if (row < minrow)	minrow = row;
		if (row > maxrow)	maxrow = row;
		if (col < mincol)	mincol = col;
		if (col > maxcol)	maxcol = col;
	}
	Point rect0_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	int rect0_wd = maxcol - mincol + 1;
	int rect0_ht = maxrow - minrow + 1;
	Point rect0_tl(rect0_center - Point(rect0_wd / 2, rect0_ht / 2)), rect0_br(rect0_tl.x + rect0_wd - 1, rect0_tl.y + rect0_ht - 1);
	rect0_tl.x = min(max(0, rect0_tl.x), __img0.cols - 1);
	rect0_tl.y = min(max(0, rect0_tl.y), __img0.rows - 1);
	rect0_br.x = min(max(0, rect0_br.x), __img0.cols - 1);
	rect0_br.y = min(max(0, rect0_br.y), __img0.rows - 1);
	Rect rect0(rect0_tl, rect0_br);

	maxrow = 0, minrow = __img1.rows - 1;
	maxcol = 0, mincol = __img1.cols - 1;
	for (auto &point : pt_region1){
		int row = point.y, col = point.x;
		if (row < minrow)	minrow = row;
		if (row > maxrow)	maxrow = row;
		if (col < mincol)	mincol = col;
		if (col > maxcol)	maxcol = col;
	}
	Point rect1_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	int rect1_wd = maxcol - mincol + 1;
	int rect1_ht = maxrow - minrow + 1;
	Point rect1_tl(rect1_center - Point(rect1_wd / 2, rect1_ht / 2)), rect1_br(rect1_tl.x + rect1_wd - 1, rect1_tl.y + rect1_ht - 1);
	rect1_tl.x = min(max(0, rect1_tl.x), __img1.cols - 1);
	rect1_tl.y = min(max(0, rect1_tl.y), __img1.rows - 1);
	rect1_br.x = min(max(0, rect1_br.x), __img1.cols - 1);
	rect1_br.y = min(max(0, rect1_br.y), __img1.rows - 1);
	Rect rect1(rect1_tl, rect1_br);


	//设置局部rect0_mask,rect1_mask
	Mat rect0_mask(__img0.size(), CV_8UC1, Scalar(0));
	Mat rect1_mask(__img1.size(), CV_8UC1, Scalar(0));
	Mat region_mask0(rect0_mask, rect0);
	Mat region_mask1(rect1_mask, rect1);
	region_mask0.setTo(Scalar(255));
	region_mask1.setTo(Scalar(255));

	Mat im0_region;
	__img0(rect0).copyTo(im0_region);
	Mat im1_region;
	__img1(rect1).copyTo(im1_region);
	//imwrite(outDir + "/Rect0.jpg", rect0_mask);
	//imwrite(outDir + "/Rect1.jpg", rect1_mask);
	imwrite(outDir + "/im0_region.jpg", im0_region);
	imwrite(outDir + "/im1_region.jpg", im1_region);

	if (rect0.width < 30 || rect0.height < 30 || rect1.width < 30 || rect1.height < 30)
		return false;

	//筛选特征点方法:先对局部区域img_region进行特征点检测，再把特征点局部坐标变换到img坐标系上
	features.resize(2);
	SIFT sift(500);
	Mat gray;
	cvtColor(im0_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[0].keypoints, features[0].descriptors);
	features[0].img_idx = 0;
	features[0].img_size = __img0.size();
	cvtColor(im1_region, gray, CV_BGR2GRAY);
	sift.operator()(gray, Mat(), features[1].keypoints, features[1].descriptors);
	features[1].img_idx = 1;
	features[1].img_size = __img1.size();

	if (features[0].keypoints.size() <= 5 || features[1].keypoints.size() <= 5)
		return false;

	//features[0]转换坐标系
	Point2f rectTL_in_img0(rect0.x, rect0.y);
	for (int i = 0; i < features[0].keypoints.size(); ++i){
		KeyPoint& keypoint = features[0].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img0 + keypoint.pt;
	}

	//features[1]转换坐标系
	Point2f rectTL_in_img1(rect1.x, rect1.y);
	for (int i = 0; i < features[1].keypoints.size(); ++i){
		KeyPoint& keypoint = features[1].keypoints[i];
		Point2f pt(keypoint.pt);
		keypoint.pt = rectTL_in_img1 + keypoint.pt;
	}


	if (__features.size() != 0){
		//----------------20170508 筛选特征点，去掉与__features[0]重复的特征点
		vector<Point2f> fea0_pts;//__features[0]特征点的位置信息
		for (auto &keypt : __features[0].keypoints){
			fea0_pts.push_back(keypt.pt);
		}
		detail::ImageFeatures unique_features0;//保存与__features[0]不重复的特征点
		unique_features0.img_idx = features[0].img_idx;
		unique_features0.img_size = features[0].img_size;
		for (int idx_fea = 0; idx_fea < features[0].keypoints.size(); ++idx_fea){
			KeyPoint& keypoint = features[0].keypoints[idx_fea];
			Mat Drow = features[0].descriptors.row(idx_fea);
			Point2f pt = keypoint.pt;
			auto it = find_if(fea0_pts.begin(), fea0_pts.end(),
				[pt](const Point2f &a)
			{
				float distance = (pt.x - a.x)*(pt.x - a.x) + (pt.y - a.y)*(pt.y - a.y);
				return distance < 2 * EPSILON * EPSILON;
			});//查看当前特征点pt是否已经存在于fea0_pts中，即是否为重复特征点
			if (it == fea0_pts.end()){//如果不重复，则当前特征点压入
				unique_features0.keypoints.push_back(keypoint);
				unique_features0.descriptors.push_back(Drow);
			}
		}
		features[0] = unique_features0;
		vector<Point2f> fea1_pts;//__features[1]特征点的位置信息
		for (auto &keypt : __features[1].keypoints){
			fea1_pts.push_back(keypt.pt);
		}
		detail::ImageFeatures unique_features1;//保存与__features[1]不重复的特征点
		unique_features1.img_idx = features[1].img_idx;
		unique_features1.img_size = features[1].img_size;
		for (int idx_fea = 0; idx_fea < features[1].keypoints.size(); ++idx_fea){
			KeyPoint& keypoint = features[1].keypoints[idx_fea];
			Mat Drow = features[1].descriptors.row(idx_fea);
			Point2f pt = keypoint.pt;
			auto it = find_if(fea1_pts.begin(), fea1_pts.end(),
				[pt](const Point2f &a)
			{
				float distance = (pt.x - a.x)*(pt.x - a.x) + (pt.y - a.y)*(pt.y - a.y);
				return distance < 2 * EPSILON * EPSILON;
			});//查看当前特征点pt是否已经存在于fea1_pts中，即是否为重复特征点
			if (it == fea1_pts.end()){//如果不重复，则当前特征点压入
				unique_features1.keypoints.push_back(keypoint);
				unique_features1.descriptors.push_back(Drow);
			}
		}
		features[1] = unique_features1;
	}

	//只选取在local_mask0和local_mask1上的特征点
	ImageFeatures feas0(features[0]), feas1(features[1]);
	features[0].keypoints.clear();
	features[0].descriptors = Mat();
	features[1].keypoints.clear();
	features[1].descriptors = Mat();
	for (auto it = feas0.keypoints.begin(); it != feas0.keypoints.end(); ++it){
		KeyPoint keypt = *it;
		Point pt0(keypt.pt);
		bool is_in_local0 = local_mask0.at<uchar>(pt0.y, pt0.x);
		Mat Drow = feas0.descriptors.row(it - feas0.keypoints.begin());
		if (is_in_local0){
			//选取该特征点
			features[0].keypoints.push_back(keypt);
			features[0].descriptors.push_back(Drow);
		}
	}
	for (auto it = feas1.keypoints.begin(); it != feas1.keypoints.end(); ++it){
		KeyPoint keypt = *it;
		Point pt1(keypt.pt);
		bool is_in_local1 = local_mask1.at<uchar>(pt1.y, pt1.x);
		Mat Drow = feas1.descriptors.row(it - feas1.keypoints.begin());
		if (is_in_local1){
			//选取该特征点
			features[1].keypoints.push_back(keypt);
			features[1].descriptors.push_back(Drow);
		}
	}

	cout << "features0 number: " << features[0].keypoints.size() << endl;
	cout << "features1 number: " << features[1].keypoints.size() << endl;
	if (features[0].keypoints.size() < 5 || features[1].keypoints.size() < 5){
		return false;
	}

	float match_conf = 0.3f;
	detail::BestOf2NearestMatcher matcher(0, match_conf);
	matcher(features, matches);//仅对当前features[0]与features[1]特征点进行匹配
	matcher.collectGarbage();
	cout << "matches number: " << matches[1].matches.size() << endl;

	bool is_valid = calcHomoFromMatches(matches[1], features[0], features[1], RANSAC_THRE1);
	if (is_valid){
		calcDualMatches(matches[2], matches[1]);
		return true;
	}
	else
		return false;
}

bool Stitching::calcHomoFromMatches_8PointMethod(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < 5)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion 
	//只用inliers来计算H
	vector<Point2f> src_points, dst_points;
	for (int j = 0; j < num_matched; j++)
	{
		if (!m.inliers_mask[j])
			continue;
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	}
	m.H = findHomography(src_points, dst_points, CV_FM_8POINT);
	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	// num of inliers
	m.num_inliers = 0;
	for (int j = 0; j < num_matched; j++)
	if (m.inliers_mask[j])
		m.num_inliers++;

	// confidence, copied from matchers.cpp
	m.confidence = m.num_inliers / (8 + 0.3 * num_matched);
	m.confidence = m.confidence > 3. ? 0. : m.confidence;

	return true;
}

vector<Point> Stitching::getSeamSection(string outDir, Mat &seamQualityMap, Mat &seamMask, int selConIdx)
{
	_mkdir(outDir.c_str());
	Mat bad_seam(seamQualityMap.size(), CV_8UC1,Scalar(0));//质量较差的缝隙为bad_seam
	
	//为bad_seam赋值
	for (int i = 0; i < seamQualityMap.rows; ++i){
		for (int j = 0; j < seamQualityMap.cols; ++j){
			if (seamMask.at<uchar>(i, j)){
				float aff = seamQualityMap.at<float>(i, j);
				if (aff < bad_thre){//选取质量较差的缝隙
					bad_seam.at<uchar>(i, j) = 255;
				}
			}		
		}
	}
	imwrite(outDir + "/bad_seam.jpg", bad_seam);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(bad_seam, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);//检测缝隙片段，contours[i]为输出的第i条线段

	

	//合并一些端点接近的缝隙片段
	vector<vector<Point>> contours_refined = contours;
	//sort(contours_refined.begin(), contours_refined.end(),
	//	[](const vector<Point> &a, const vector<Point> &b)
	//{return a.size() > b.size(); });//按照线段长度排序
	auto it0 = contours_refined.begin();
	while (it0 != contours_refined.end()){
		vector<Point> pts0 = *it0;
		vector<Point> end_pts0 = getEndPoints(pts0);
		//遍历pts1
		//auto it1 = it0 + 1;
		auto it1 = contours_refined.begin();
		while (it1 != contours_refined.end()){
			if (it1 == it0){
				++it1;
				continue;
			}
			vector<Point> &pts1 = *it1;
			vector<Point> end_pts1 = getEndPoints(pts1);
			bool perform_merge = 0;//合并标志，初始化置0
			//计算两个点集的端点之间的距离
			for (int i = 0; i < end_pts0.size(); ++i){
				for (int j = 0; j < end_pts1.size(); ++j){
					int distance = (end_pts0[i].x - end_pts1[j].x)*(end_pts0[i].x - end_pts1[j].x)
						+ (end_pts0[i].y - end_pts1[j].y)*(end_pts0[i].y - end_pts1[j].y);
					if (distance <= dis_thre*dis_thre){//距离小于dis_thre，则合并标志置1
						perform_merge = 1;
						break;
					}
				}
			}
			//根据计算结果判断是否进行合并操作
			if (perform_merge){//如果当前缝隙段pts0与缝隙段pts1接近，则把pts1合并到pts0
				//pts1压入pts0
				for (auto &pt : pts1){
					it0->push_back(pt);
				}
				//删除pts1
				if (it0 < it1){
					int pos = it0 - contours_refined.begin();
					it1 = contours_refined.erase(it1);
					it0 = contours_refined.begin() + pos;//容器大小发生了改变，需重新赋值it0：由于it1大于it0，删除it1元素不影响it0位置
				}
				else
				{
					int pos = it0 - contours_refined.begin();
					it1 = contours_refined.erase(it1);
					it0 = contours_refined.begin() + pos - 1;//容器大小发生了改变，需重新赋值it0：由于it1小于it0，删除it1元素it0位置前移1
				}
				//更新pts0以及端点end_pts1
				pts0 = *it0;
				end_pts0 = getEndPoints(pts0);
			}
			else{//否则，遍历下一个pts1
				++it1;
			}		
		}
		++it0;
	}

	if (selConIdx >= contours_refined.size()){
		return vector<Point>();
	}

	//找到最长的缝隙片段
	sort(contours_refined.begin(), contours_refined.end(),
		[](const vector<Point> &a, const vector<Point> &b)
	{return a.size() > b.size(); });//按照线段长度排序

	//调试
	Mat sel_con_img;
	__result.copyTo(sel_con_img);
	for (int i = 0; i < contours_refined[selConIdx].size(); ++i){
		circle(sel_con_img, contours_refined[selConIdx][i], 5, Scalar(0, 0, 255), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/selected_contour.jpg", sel_con_img);

	return contours_refined[selConIdx];
	



	////找出缝隙段的矩形包络
	//int maxrow = 0, minrow = bad_seam.rows - 1;
	//int maxcol = 0, mincol = bad_seam.cols - 1;
	//for (auto &point : contours_refined[selConIdx]){
	//	int row = point.y,col = point.x;
	//	if (row < minrow)	minrow = row;
	//	if (row > maxrow)	maxrow = row;
	//	if (col < mincol)	mincol = col;
	//	if (col > maxcol)	maxcol = col;
	//}
	//Point rect_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
	//int rect_wd = maxcol - mincol + 1;
	//int rect_ht = maxrow - minrow + 1;
	//Point rect_tl(rect_center - Point(rect_wd / 2, rect_ht / 2));
	//Rect orig_rect(rect_tl.x, rect_tl.y, rect_wd, rect_ht);

	////膨胀包络
	//int rect_slide = max(rect_wd, rect_ht);
	//rect_slide *= dilate_times;
	//rect_tl = Point(rect_center - Point(rect_slide / 2, rect_slide / 2));
	//return Rect(rect_tl.x, rect_tl.y, rect_slide, rect_slide);
}

vector<Point> Stitching::getEndPoints(vector<Point> &inPoints)
{
	Point left(inPoints[0]), right(inPoints[0]), up(inPoints[0]), down(inPoints[0]);
	for (auto &pt : inPoints){
		if (pt.x < left.x)	left = pt;
		if (pt.x > right.x)	right = pt;
		if (pt.y < down.y)	down = pt;
		if (pt.y > up.y)	up = pt;
	}
	vector<Point> outPoints{ left, right, up, down };
	return outPoints;
}

float Stitching::calEdgeCostWeight(Point &pt, vector<Point2f> matFeas)
{
	float sum_dis = 0;
	float sigma = 15, delta=0.1;
	for (auto &matfea : matFeas){
		float dis = (pt.x - matfea.x)*(pt.x - matfea.x) + (pt.y - matfea.y)*(pt.y - matfea.y);	
		float dis_norm = dis / (2 * sigma * sigma);
		dis_norm = exp(-dis_norm);
		sum_dis += dis_norm;
	}
	float ret = 1 / (sum_dis + delta);
	return ret;
}

void Stitching::FindSeambyGraphcut_WeightedPatchDifference(string outDir, const vector<Mat>& imgs, const vector<Mat>& masks, const vector<Point2i>& corners,
	vector<ImageFeatures> &feas, vector<MatchesInfo> &mats, const vector<Mat>& masks_avoid)
{
	_mkdir(outDir.c_str());
	int width0 = imgs[0].cols;
	int height0 = imgs[0].rows;
	int width1 = imgs[1].cols;
	int height1 = imgs[1].rows;
	int minrow = min(corners[0].y, corners[1].y);
	int maxrow = max(corners[0].y + height0, corners[1].y + height1);
	int mincol = min(corners[0].x, corners[1].x);
	int maxcol = max(corners[0].x + width0, corners[1].x + width1);
	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//构建canvas大小的im0_canvas，im1_canvas，mask0_canvas和mask1_canvas
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corners[0].x - mincol, corners[0].y - minrow, width0, height0));
	imgs[0].copyTo(img0_valid);
	Mat mask0_valid(mask0_canvas, Rect(corners[0].x - mincol, corners[0].y - minrow, width0, height0));
	masks[0].copyTo(mask0_valid);
	Mat img1_valid(im1_canvas, Rect(corners[1].x - mincol, corners[1].y - minrow, width1, height1));
	imgs[1].copyTo(img1_valid);
	Mat mask1_valid(mask1_canvas, Rect(corners[1].x - mincol, corners[1].y - minrow, width1, height1));
	masks[1].copyTo(mask1_valid);

	//char file_im0_canvas[256], file_mask0_canvas[256], file_im1_canvas[256], file_mask1_canvas[256];
	//sprintf(file_im0_canvas, "%s/canvas_im0.jpg", outDir.c_str());
	//imwrite(file_im0_canvas, im0_canvas);
	//sprintf(file_mask0_canvas, "%s/canvas_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0_canvas, mask0_canvas);
	//sprintf(file_im1_canvas, "%s/canvas_im1.jpg", outDir.c_str());
	//imwrite(file_im1_canvas, im1_canvas);
	//sprintf(file_mask1_canvas, "%s/canvas_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1_canvas, mask1_canvas);

	//------------------------------------20170605
	//构建canvas大小的mask_avoid
	Mat mask0_avoid_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat mask1_avoid_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat mask0_avoid_valid(mask0_avoid_canvas, Rect(corners[0].x - mincol, corners[0].y - minrow, width0, height0));
	masks_avoid[0].copyTo(mask0_avoid_valid);
	Mat mask1_avoid_valid(mask1_avoid_canvas, Rect(corners[1].x - mincol, corners[1].y - minrow, width1, height1));
	masks_avoid[1].copyTo(mask1_avoid_valid);
	Mat mask_avoid = mask0_avoid_canvas | mask1_avoid_canvas;
	imwrite(outDir+"/mask_avoid_canvas.jpg", mask_avoid);

	//------------------------------------20170512
	//构建canvas上匹配点集合aligned_points
	vector<Point2f> aligned_points;
	const vector<KeyPoint>& points0 = feas[0].keypoints;
	const vector<KeyPoint>& points1 = feas[1].keypoints;
	for (int idx_match = 0; idx_match < mats[1].matches.size(); ++idx_match){
		if (mats[1].inliers_mask[idx_match] == 0)
			continue;
		int idx_fea0 = mats[1].matches[idx_match].queryIdx;
		int idx_fea1 = mats[1].matches[idx_match].trainIdx;
		Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;
		Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
		calcPoint_after_H(pt0, pt0_warped, _H.data());
		pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
		pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
		pt0_canvas.x = pt0_in_warpImg0.x + __corners[0].x - mincol;
		pt0_canvas.y = pt0_in_warpImg0.y + __corners[0].y - minrow;
		pt1_canvas.x = pt1.x + __corners[1].x - mincol;
		pt1_canvas.y = pt1.y + __corners[1].y - minrow;
		//压入aligned_points	
		aligned_points.push_back(pt0_canvas);
		aligned_points.push_back(pt1_canvas);
	}



	//construct graph
	typedef Graph<int, int, int> GraphType;
	int num_nodes = width*height, num_edges = num_nodes * 2;
	GraphType *g = new GraphType(/*estimated # of nodes*/num_nodes, /*estimated # of edges*/ num_edges);
	int terminal_cost = INT_MAX;

	g->add_node(num_nodes);

	//define egdes(excluding t-links)
	int psize = 2;
	uchar *mask0_canvas_data = mask0_canvas.data;
	uchar *mask1_canvas_data = mask1_canvas.data;	
	uchar *im0_data = im0_canvas.data;
	uchar *im1_data = im1_canvas.data;
	//-------------------------------------------20170515 用colored_edge设置代价
	//Mat im0_canvas_cedge = convertoColoredEdgeImage(im0_canvas);
	//Mat im1_canvas_cedge = convertoColoredEdgeImage(im1_canvas);
	//uchar *im0_data = im0_canvas_cedge.data;
	//uchar *im1_data = im1_canvas_cedge.data;
	Mat im0_edge = convertoColoredEdgeImage(im0_canvas);
	uchar *im0_edge_data = im0_edge.data;
	Mat im1_edge = convertoColoredEdgeImage(im1_canvas);
	uchar *im1_edge_data = im1_edge.data;
	for (int col = 0; col < width; ++col)
	{
		bool right_exist = (col != (width - 1));//当col=width-1时，该node没有right_edge
		for (int row = 0; row < height; ++row)
		{
			bool below_exist = (row != (height - 1));
			int x = col, y = row;
			int chs = 3;
			int pos = row*width + col;
			bool node_in0 = mask0_canvas_data[pos], node_in1 = mask1_canvas_data[pos];
			bool node_in_ol = node_in0 && node_in1;
			if (right_exist)
			{
				bool node_right_in0 = mask0_canvas_data[pos + 1], node_right_in1 = mask1_canvas_data[pos + 1];
				bool node_right_in_ol = node_right_in0 && node_right_in1;
				if (node_in_ol && node_right_in_ol)//当前点(x,y)与(x+1,y)在重叠区
				{
					if (mask_avoid.at<uchar>(row, col) > 0){//如果当前点被标记为不佳区域，则代价为无穷大
						g->add_edge(pos, pos + 1, 100, 100);
					}
					else{
						float d1 = 0, d2 = 0;
						for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
						{
							for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
							{
								for (int c = 0; c < chs; c++)
								{
									d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
									d2 += pow(im0_data[(p*width + q + 1)*chs + c] - im1_data[(p*width + q + 1)*chs + c], 2);
								}
								//增加梯度通道
								d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
								d1 += pow(im0_edge_data[p*width + q + 1] - im1_edge_data[p*width + q + 1], 2);
							}
						}
						float weight1 = calEdgeCostWeight(Point(x, y), aligned_points),
							weight2 = calEdgeCostWeight(Point(x + 1, y), aligned_points);
						float dd = (sqrt(d1)*weight1 + sqrt(d2)*weight2);//edge(pi,pj)=cost(pi)+cost(pj)
						dd = max(dd, 1.0f);
						//int dd = d1 + d2;
						g->add_edge(pos, pos + 1, int(dd), int(dd));
					}		
				}
				if (node_in_ol^node_right_in_ol){//当前点(x,y)与(x+1,y)有一点在重叠区
					g->add_edge(pos, pos + 1, terminal_cost, terminal_cost);
				}

			}

			if (below_exist)
			{
				bool node_below_in0 = mask0_canvas_data[pos + width], node_below_in1 = mask1_canvas_data[pos + width];
				bool node_below_in_ol = node_below_in0 && node_below_in1;
				if (node_in_ol && node_below_in_ol)//当前点(x,y)与(x,y+1)在重叠区
				{
					if (mask_avoid.at<uchar>(row, col) > 0){//如果当前点被标记为不佳区域，则代价为无穷大
						g->add_edge(pos, pos + width, 100, 100);
					}
					else{
						float d1 = 0, d2 = 0;
						for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
						{
							for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
							{
								for (int c = 0; c < chs; c++)
								{
									d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
									d2 += pow(im0_data[((p + 1)*width + q)*chs + c] - im1_data[((p + 1)*width + q)*chs + c], 2);
								}
								//增加梯度通道
								d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
								d1 += pow(im0_edge_data[(p + 1)*width + q] - im1_edge_data[(p + 1)*width + q], 2);
							}
						}
						//float dd = (sqrt(d1) + sqrt(d2));
						float weight1 = calEdgeCostWeight(Point(x, y), aligned_points),
							weight2 = calEdgeCostWeight(Point(x, y + 1), aligned_points);
						float dd = (sqrt(d1)*weight1 + sqrt(d2)*weight2);
						dd = max(dd, 1.0f);
						g->add_edge(pos, pos + width, int(dd), int(dd));
					}		
				}
				if (node_in_ol^node_below_in_ol){//当前点(x,y)与(x,y+1)有一点在重叠区
					g->add_edge(pos, pos + width, terminal_cost, terminal_cost);
				}
			}
		}
	}

	//define t-links
	Mat source_area = Mat::zeros(height, width, CV_8UC1);
	Mat sink_area = Mat::zeros(height, width, CV_8UC1);
	uchar *source_data = source_area.data;
	uchar *sink_data = sink_area.data;

	for (int col = 0; col < width; ++col)
	{
		for (int row = 0; row < height; ++row)
		{
			int pose = row*width + col;
			bool in0 = mask0_canvas_data[pose];
			bool in1 = mask1_canvas_data[pose];
			if (in0 && !in1)
			{
				g->add_tweights(pose,   /* capacities */  terminal_cost, 0);//link to source
				source_data[pose] = 255;
			}
			else if (!in0 && in1)
			{
				g->add_tweights(pose,   /* capacities */  0, terminal_cost);//link to sink
				sink_data[pose] = 255;
			}
			else if (in0 && in1)
			{
			}
			else
			{
				g->add_tweights(pose,   /* capacities */  1, 1);
				source_data[pose] = 120;
				sink_data[pose] = 120;
			}
		}
	}
	//char file_source_area[256], file_sink_area[256];
	//sprintf(file_source_area, "%s/source_area.jpg", outDir.c_str());
	//imwrite(file_source_area, source_area);
	//sprintf(file_sink_area, "%s/sink_area.jpg", outDir.c_str());
	//imwrite(file_sink_area, sink_area);


	int flow = g->maxflow();
	Mat result_mask0_canvas = mask0_canvas.clone(), result_mask1_canvas = mask1_canvas.clone();
	uchar *result_mask0_data = result_mask0_canvas.data, *result_mask1_data = result_mask1_canvas.data;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			int pos = row*width + col;
			//source set refers to im0		
			if (g->what_segment(pos) == GraphType::SOURCE)//pixel at pos belongs to source set
			{
				result_mask0_data[pos] = 255;
				result_mask1_data[pos] = 0;
			}
			else
			{
				result_mask0_data[pos] = 0;
				result_mask1_data[pos] = 255;
			}
		}
	}
	delete g;

	result_mask0_canvas = result_mask0_canvas & mask0_canvas;
	result_mask1_canvas = result_mask1_canvas & mask1_canvas;
	Mat result_mask0(result_mask0_canvas, Rect(corners[0].x - mincol, corners[0].y - minrow, width0, height0));
	Mat result_mask1(result_mask1_canvas, Rect(corners[1].x - mincol, corners[1].y - minrow, width1, height1));
	result_mask0.copyTo(masks[0]);
	result_mask1.copyTo(masks[1]);

	//char file_mask0[256], file_mask1[256];
	//sprintf(file_mask0, "%s/result_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0, result_mask0);
	//sprintf(file_mask1, "%s/result_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1, result_mask1);

}

void Stitching::FindSeambyGraphcut_WeightedPatchDifference(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1)
{
	_mkdir(outDir.c_str());
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//construct im0_canvas,im1_canvas and mask0_canvas,mask1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat mask0_valid(mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	mask0.copyTo(mask0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);
	Mat mask1_valid(mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	mask1.copyTo(mask1_valid);

	//char file_im0_canvas[256], file_mask0_canvas[256], file_im1_canvas[256], file_mask1_canvas[256];
	//sprintf(file_im0_canvas, "%s/canvas_im0.jpg", outDir.c_str());
	//imwrite(file_im0_canvas, im0_canvas);
	//sprintf(file_mask0_canvas, "%s/canvas_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0_canvas, mask0_canvas);
	//sprintf(file_im1_canvas, "%s/canvas_im1.jpg", outDir.c_str());
	//imwrite(file_im1_canvas, im1_canvas);
	//sprintf(file_mask1_canvas, "%s/canvas_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1_canvas, mask1_canvas);

	//------------------------------------20170512
	//匹配点在canvas上的坐标aligned_points
	vector<Point2f> aligned_points;
	const vector<KeyPoint>& points0 = __features[0].keypoints;
	const vector<KeyPoint>& points1 = __features[1].keypoints;
	for (int idx_match = 0; idx_match < __matches[1].matches.size(); ++idx_match){
		int idx_fea0 = __matches[1].matches[idx_match].queryIdx;
		int idx_fea1 = __matches[1].matches[idx_match].trainIdx;
		Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;
		Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
		calcPoint_after_H(pt0, pt0_warped, _H.data());
		pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
		pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
		convert_to_CanvasCoordinate(__corners, pt0_in_warpImg0, pt0_canvas, 0);
		convert_to_CanvasCoordinate(__corners, pt1, pt1_canvas, 1);
		aligned_points.push_back(pt0_canvas);
		aligned_points.push_back(pt1_canvas);
	}


	//construct graph
	typedef Graph<int, int, int> GraphType;
	int num_nodes = width*height, num_edges = num_nodes * 2;
	GraphType *g = new GraphType(/*estimated # of nodes*/num_nodes, /*estimated # of edges*/ num_edges);
	int terminal_cost = INT_MAX;

	g->add_node(num_nodes);

	//define egdes(excluding t-links)
	int psize = 2;
	uchar *mask0_canvas_data = mask0_canvas.data;
	uchar *mask1_canvas_data = mask1_canvas.data;
	uchar *im0_data = im0_canvas.data;
	uchar *im1_data = im1_canvas.data;
	//-------------------------------------------20170515 用colored_edge设置代价
	//Mat im0_canvas_cedge = convertoColoredEdgeImage(im0_canvas);
	//Mat im1_canvas_cedge = convertoColoredEdgeImage(im1_canvas);
	//uchar *im0_data = im0_canvas_cedge.data;
	//uchar *im1_data = im1_canvas_cedge.data;
	Mat im0_edge = convertoColoredEdgeImage(im0_canvas);
	uchar *im0_edge_data = im0_edge.data;
	Mat im1_edge = convertoColoredEdgeImage(im1_canvas);
	uchar *im1_edge_data = im1_edge.data;
	for (int col = 0; col < width; ++col)
	{
		bool right_exist = (col != (width - 1));//当col=width-1时，该node没有right_edge
		for (int row = 0; row < height; ++row)
		{
			bool below_exist = (row != (height - 1));
			int x = col, y = row;
			int chs = 3;
			int pos = row*width + col;
			bool node_in0 = mask0_canvas_data[pos], node_in1 = mask1_canvas_data[pos];
			bool node_in_ol = node_in0 && node_in1;
			if (right_exist)
			{
				bool node_right_in0 = mask0_canvas_data[pos + 1], node_right_in1 = mask1_canvas_data[pos + 1];
				bool node_right_in_ol = node_right_in0 && node_right_in1;
				if (node_in_ol && node_right_in_ol)//当前点(x,y)与(x+1,y)在重叠区
				{				
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[(p*width + q + 1)*chs + c] - im1_data[(p*width + q + 1)*chs + c], 2);
							}
							//增加梯度通道
							d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
							d1 += pow(im0_edge_data[p*width + q + 1] - im1_edge_data[p*width + q + 1], 2);
						}
					}
					float weight1 = calEdgeCostWeight(Point(x, y), aligned_points),
						weight2 = calEdgeCostWeight(Point(x + 1, y), aligned_points);
					float dd = (sqrt(d1)*weight1 + sqrt(d2)*weight2);//edge(pi,pj)=cost(pi)+cost(pj)
					dd = max(dd, 1.0f);
					//int dd = d1 + d2;
					g->add_edge(pos, pos + 1, int(dd), int(dd));
					
				}
				if (node_in_ol^node_right_in_ol){//当前点(x,y)与(x+1,y)有一点在重叠区
					g->add_edge(pos, pos + 1, terminal_cost, terminal_cost);
				}

			}

			if (below_exist)
			{
				bool node_below_in0 = mask0_canvas_data[pos + width], node_below_in1 = mask1_canvas_data[pos + width];
				bool node_below_in_ol = node_below_in0 && node_below_in1;
				if (node_in_ol && node_below_in_ol)//当前点(x,y)与(x,y+1)在重叠区
				{		
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[((p + 1)*width + q)*chs + c] - im1_data[((p + 1)*width + q)*chs + c], 2);
							}
							//增加梯度通道
							d1 += pow(im0_edge_data[p*width + q] - im1_edge_data[p*width + q], 2);
							d1 += pow(im0_edge_data[(p + 1)*width + q] - im1_edge_data[(p + 1)*width + q], 2);
						}
					}
					//float dd = (sqrt(d1) + sqrt(d2));
					float weight1 = calEdgeCostWeight(Point(x, y), aligned_points),
						weight2 = calEdgeCostWeight(Point(x, y + 1), aligned_points);
					float dd = (sqrt(d1)*weight1 + sqrt(d2)*weight2);
					dd = max(dd, 1.0f);
					g->add_edge(pos, pos + width, int(dd), int(dd));
					
				}
				if (node_in_ol^node_below_in_ol){//当前点(x,y)与(x,y+1)有一点在重叠区
					g->add_edge(pos, pos + width, terminal_cost, terminal_cost);
				}
			}
		}
	}

	//define t-links
	Mat source_area = Mat::zeros(height, width, CV_8UC1);
	Mat sink_area = Mat::zeros(height, width, CV_8UC1);
	uchar *source_data = source_area.data;
	uchar *sink_data = sink_area.data;

	for (int col = 0; col < width; ++col)
	{
		for (int row = 0; row < height; ++row)
		{
			int pose = row*width + col;
			bool in0 = mask0_canvas_data[pose];
			bool in1 = mask1_canvas_data[pose];
			if (in0 && !in1)
			{
				g->add_tweights(pose,   /* capacities */  terminal_cost, 0);//link to source
				source_data[pose] = 255;
			}
			else if (!in0 && in1)
			{
				g->add_tweights(pose,   /* capacities */  0, terminal_cost);//link to sink
				sink_data[pose] = 255;
			}
			else if (in0 && in1)
			{
			}
			else
			{
				g->add_tweights(pose,   /* capacities */  1, 1);
				source_data[pose] = 120;
				sink_data[pose] = 120;
			}
		}
	}
	//char file_source_area[256], file_sink_area[256];
	//sprintf(file_source_area, "%s/source_area.jpg", outDir.c_str());
	//imwrite(file_source_area, source_area);
	//sprintf(file_sink_area, "%s/sink_area.jpg", outDir.c_str());
	//imwrite(file_sink_area, sink_area);


	int flow = g->maxflow();
	Mat result_mask0_canvas = mask0_canvas.clone(), result_mask1_canvas = mask1_canvas.clone();
	uchar *result_mask0_data = result_mask0_canvas.data, *result_mask1_data = result_mask1_canvas.data;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			int pos = row*width + col;
			//source set refers to im0		
			if (g->what_segment(pos) == GraphType::SOURCE)//pixel at pos belongs to source set
			{
				result_mask0_data[pos] = 255;
				result_mask1_data[pos] = 0;
			}
			else
			{
				result_mask0_data[pos] = 0;
				result_mask1_data[pos] = 255;
			}
		}
	}
	delete g;

	result_mask0_canvas = result_mask0_canvas & mask0_canvas;
	result_mask1_canvas = result_mask1_canvas & mask1_canvas;
	Mat result_mask0(result_mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	Mat result_mask1(result_mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	result_mask0.copyTo(mask0);
	result_mask1.copyTo(mask1);

	//char file_mask0[256], file_mask1[256];
	//sprintf(file_mask0, "%s/result_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0, result_mask0);
	//sprintf(file_mask1, "%s/result_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1, result_mask1);

}


void Stitching::FindSeambyGraphcut_WeightedRect(string outDir, const Mat& im0, const Mat& im1, const Mat& mask0, const Mat& mask1, const Point2i& corner0, const Point2i& corner1)
{
	_mkdir(outDir.c_str());
	int width0 = im0.cols;
	int height0 = im0.rows;
	int width1 = im1.cols;
	int height1 = im1.rows;
	int minrow = min(corner0.y, corner1.y);
	int maxrow = max(corner0.y + height0, corner1.y + height1);
	int mincol = min(corner0.x, corner1.x);
	int maxcol = max(corner0.x + width0, corner1.x + width1);

	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;

	//construct im0_canvas,im1_canvas and mask0_canvas,mask1_canvas in canvas size
	Mat im0_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask0_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat im1_canvas = Mat::zeros(height, width, CV_8UC3);
	Mat mask1_canvas = Mat::zeros(height, width, CV_8UC1);
	Mat img0_valid(im0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	im0.copyTo(img0_valid);
	Mat mask0_valid(mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	mask0.copyTo(mask0_valid);
	Mat img1_valid(im1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	im1.copyTo(img1_valid);
	Mat mask1_valid(mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	mask1.copyTo(mask1_valid);

	//char file_im0_canvas[256], file_mask0_canvas[256], file_im1_canvas[256], file_mask1_canvas[256];
	//sprintf(file_im0_canvas, "%s/canvas_im0.jpg", outDir.c_str());
	//imwrite(file_im0_canvas, im0_canvas);
	//sprintf(file_mask0_canvas, "%s/canvas_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0_canvas, mask0_canvas);
	//sprintf(file_im1_canvas, "%s/canvas_im1.jpg", outDir.c_str());
	//imwrite(file_im1_canvas, im1_canvas);
	//sprintf(file_mask1_canvas, "%s/canvas_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1_canvas, mask1_canvas);

	//------------------------------------20170510
	/*
	计算局部配准区域，对局部配准区域内的代价加权
	*/
	vector<Rect> aligned_region;
	const vector<KeyPoint>& points0 = __features[0].keypoints;
	const vector<KeyPoint>& points1 = __features[1].keypoints;
	int num_region = _num_matches.size() - 1;
	for (int idx_region = 0; idx_region < num_region; ++idx_region){
		int match_beg = _num_matches[idx_region];
		int match_end = _num_matches[idx_region + 1];
		//找出当前匹配点集合的矩形包络
		int maxrow = 0, minrow = width - 1;
		int maxcol = 0, mincol = height - 1;
		//找出当前匹配点集合的矩形包络范围
		for (int idx_match = match_beg; idx_match < match_end; ++idx_match){
			int idx_fea0 = __matches[1].matches[idx_match].queryIdx;
			int idx_fea1 = __matches[1].matches[idx_match].trainIdx;
			Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;
			Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
			calcPoint_after_H(pt0, pt0_warped, _H.data());
			pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
			pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
			convert_to_CanvasCoordinate( __corners, pt0_in_warpImg0, pt0_canvas, 0);
			convert_to_CanvasCoordinate( __corners, pt1, pt1_canvas, 1);
			vector<Point2f> aligned_points{ pt0_canvas, pt1_canvas };
			for (auto &point : aligned_points){
				int row = point.y, col = point.x;
				if (row < minrow)	minrow = row;
				if (row > maxrow)	maxrow = row;
				if (col < mincol)	mincol = col;
				if (col > maxcol)	maxcol = col;
			}
		}
		Point rect_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
		int rect_wd = maxcol - mincol + 1;
		int rect_ht = maxrow - minrow + 1;
		Point rect_tl(rect_center - Point(rect_wd / 2, rect_ht / 2));
		Rect orig_rect(rect_tl.x, rect_tl.y, rect_wd, rect_ht);
		aligned_region.push_back(orig_rect);
	}
	//构建局部配准区域掩膜
	Mat aligned_region_mask = Mat::zeros(height, width, CV_8UC1);
	for (auto &roi : aligned_region){
		for (int i = roi.y; i < roi.y + roi.height; ++i){
			for (int j = roi.x; j < roi.x + roi.width; ++j){
				aligned_region_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	imwrite(outDir + "/aligned_region.jpg", aligned_region_mask);
	int cost_weight = 10;


	//construct graph
	typedef Graph<int, int, int> GraphType;
	int num_nodes = width*height, num_edges = num_nodes * 2;
	GraphType *g = new GraphType(/*estimated # of nodes*/num_nodes, /*estimated # of edges*/ num_edges);
	int terminal_cost = INT_MAX;

	g->add_node(num_nodes);

	//define egdes(excluding t-links)
	int psize = 2;
	uchar *mask0_canvas_data = mask0_canvas.data;
	uchar *mask1_canvas_data = mask1_canvas.data;
	uchar *im0_data = im0_canvas.data;
	uchar *im1_data = im1_canvas.data;
	for (int col = 0; col < width; ++col)
	{
		bool right_exist = (col != (width - 1));//当col=width-1时，该node没有right_edge
		for (int row = 0; row < height; ++row)
		{
			bool below_exist = (row != (height - 1));
			int x = col, y = row;
			int chs = 3;
			int pos = row*width + col;
			bool node_in0 = mask0_canvas_data[pos], node_in1 = mask1_canvas_data[pos];
			bool node_in_ol = node_in0 && node_in1;
			if (right_exist)
			{
				bool node_right_in0 = mask0_canvas_data[pos + 1], node_right_in1 = mask1_canvas_data[pos + 1];
				bool node_right_in_ol = node_right_in0 && node_right_in1;
				if (node_in_ol && node_right_in_ol)//当前点(x,y)与(x+1,y)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[(p*width + q + 1)*chs + c] - im1_data[(p*width + q + 1)*chs + c], 2);
							}
						}
					}

					float dd = (sqrt(d1) + sqrt(d2));
					bool is_in_aligned_region = aligned_region_mask.at<uchar>(row, col);
					if (!is_in_aligned_region){//不在局部配准区域，则代价越大
						dd = dd * cost_weight;
					}
					//int dd = d1 + d2;
					g->add_edge(pos, pos + 1, int(dd), int(dd));
				}
				if (node_in_ol^node_right_in_ol){//当前点(x,y)与(x+1,y)有一点在重叠区
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 1); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 2); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[(p*width + q + 1)*chs + c] - im1_data[(p*width + q + 1)*chs + c], 2);
							}
						}
					}
					float dd = node_in_ol ? (2 * sqrt(d1)) : (2 * sqrt(d2));//edge(pi,pj)=2*cost(pk),pk为pi和pj中位于重叠区的点
					g->add_edge(pos, pos + 1, int(dd), int(dd));
					//g->add_edge(pos, pos + 1, terminal_cost, terminal_cost);
				}

			}

			if (below_exist)
			{
				bool node_below_in0 = mask0_canvas_data[pos + width], node_below_in1 = mask1_canvas_data[pos + width];
				bool node_below_in_ol = node_below_in0 && node_below_in1;
				if (node_in_ol && node_below_in_ol)//当前点(x,y)与(x,y+1)在重叠区
				{
					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[((p + 1)*width + q)*chs + c] - im1_data[((p + 1)*width + q)*chs + c], 2);
							}
						}
					}
					float dd = (sqrt(d1) + sqrt(d2));
					bool is_in_aligned_region = aligned_region_mask.at<uchar>(row, col);
					if (!is_in_aligned_region){
						dd = dd * cost_weight;
					}
					//int dd = d1 + d2;
					g->add_edge(pos, pos + width, int(dd), int(dd));
				}
				if (node_in_ol^node_below_in_ol){//当前点(x,y)与(x,y+1)有一点在重叠区

					float d1 = 0, d2 = 0;
					for (int p = max(y - psize, 0); p <= min(y + psize, height - 2); p++)
					{
						for (int q = max(x - psize, 0); q <= min(x + psize, width - 1); q++)
						{
							for (int c = 0; c < chs; c++)
							{
								d1 += pow(im0_data[(p*width + q)*chs + c] - im1_data[(p*width + q)*chs + c], 2);
								d2 += pow(im0_data[((p + 1)*width + q)*chs + c] - im1_data[((p + 1)*width + q)*chs + c], 2);
							}
						}
					}
					float dd = node_in_ol ? (2 * sqrt(d1)) : (2 * sqrt(d2));//edge(pi,pj)=2*cost(pk),pk为pi和pj中位于重叠区的点
					g->add_edge(pos, pos + width, int(dd), int(dd));
					//g->add_edge(pos, pos + width, terminal_cost, terminal_cost);
				}

			}

		}
	}

	//define t-links
	Mat source_area = Mat::zeros(height, width, CV_8UC1);
	Mat sink_area = Mat::zeros(height, width, CV_8UC1);
	uchar *source_data = source_area.data;
	uchar *sink_data = sink_area.data;

	for (int col = 0; col < width; ++col)
	{
		for (int row = 0; row < height; ++row)
		{
			int pose = row*width + col;
			bool in0 = mask0_canvas_data[pose];
			bool in1 = mask1_canvas_data[pose];
			if (in0 && !in1)
			{
				g->add_tweights(pose,   /* capacities */  terminal_cost, 0);//link to source
				source_data[pose] = 255;
			}
			else if (!in0 && in1)
			{
				g->add_tweights(pose,   /* capacities */  0, terminal_cost);//link to sink
				sink_data[pose] = 255;
			}
			else if (in0 && in1)
			{
			}
			else
			{
				g->add_tweights(pose,   /* capacities */  1, 1);
				source_data[pose] = 120;
				sink_data[pose] = 120;
			}
		}
	}
	//char file_source_area[256], file_sink_area[256];
	//sprintf(file_source_area, "%s/source_area.jpg", outDir.c_str());
	//imwrite(file_source_area, source_area);
	//sprintf(file_sink_area, "%s/sink_area.jpg", outDir.c_str());
	//imwrite(file_sink_area, sink_area);


	int flow = g->maxflow();
	Mat result_mask0_canvas = mask0_canvas.clone(), result_mask1_canvas = mask1_canvas.clone();
	uchar *result_mask0_data = result_mask0_canvas.data, *result_mask1_data = result_mask1_canvas.data;
	for (int row = 0; row < height; ++row)
	{
		for (int col = 0; col < width; ++col)
		{
			int pos = row*width + col;
			//source set refers to im0		
			if (g->what_segment(pos) == GraphType::SOURCE)//pixel at pos belongs to source set
			{
				result_mask0_data[pos] = 255;
				result_mask1_data[pos] = 0;
			}
			else
			{
				result_mask0_data[pos] = 0;
				result_mask1_data[pos] = 255;
			}
		}
	}
	delete g;

	result_mask0_canvas = result_mask0_canvas & mask0_canvas;
	result_mask1_canvas = result_mask1_canvas & mask1_canvas;
	Mat result_mask0(result_mask0_canvas, Rect(corner0.x - mincol, corner0.y - minrow, width0, height0));
	Mat result_mask1(result_mask1_canvas, Rect(corner1.x - mincol, corner1.y - minrow, width1, height1));
	result_mask0.copyTo(mask0);
	result_mask1.copyTo(mask1);

	//char file_mask0[256], file_mask1[256];
	//sprintf(file_mask0, "%s/result_mask0.jpg", outDir.c_str());
	//imwrite(file_mask0, result_mask0);
	//sprintf(file_mask1, "%s/result_mask1.jpg", outDir.c_str());
	//imwrite(file_mask1, result_mask1);

}

Mat Stitching::convertoColoredEdgeImage(const Mat& image)
{
	Mat gray, edge, cedge;
	int edgeThresh = 40;

	cvtColor(image, gray, COLOR_BGR2GRAY);

	// Reduce noise with a kernel 3x3
	//blur(gray, edge, Size(3, 3));

	// Run the edge detector on grayscale
	Canny(gray, edge, edgeThresh, edgeThresh * 3, 3);
	//dilate(edge, edge, Mat());

	cedge.create(image.size(), image.type());
	cedge = Scalar::all(0);
	image.copyTo(cedge, edge);

	//return cedge;
	return edge;
}

void Stitching::ShowAffinityImg_4ch(string outDir, const vector<uchar> &clusterMask, const Mat& seamMask)
{
	_mkdir(outDir.c_str());
	int num_mat = __matches[1].matches.size();

	const vector<KeyPoint>& points0 = __features[0].keypoints;
	const vector<KeyPoint>& points1 = __features[1].keypoints;
	Mat canvas_img;
	__result.copyTo(canvas_img);

	Mat img0_canvas, img1_canvas;
	generateCanvasImgs(__images_warped[0], __images_warped[1], __corners[0], __corners[1], img0_canvas, img1_canvas);

	float sigma_normalized = 0.10f;
	float sigma = sqrt(255 * 255 * 4)*sigma_normalized;
	Mat img0_edge = convertoColoredEdgeImage(img0_canvas);
	Mat img1_edge = convertoColoredEdgeImage(img1_canvas);
	for (int idx_mat = 0; idx_mat < num_mat; idx_mat++)
	{
		//if (_select(idx_mat) == 0)
		//	continue;
		if (clusterMask[idx_mat] == 0) continue;

		int idx_fea0 = __matches[1].matches[idx_mat].queryIdx;
		int idx_fea1 = __matches[1].matches[idx_mat].trainIdx;
		Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;
		Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
		calcPoint_after_H(pt0, pt0_warped, _H.data());
		pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
		pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
		convert_to_CanvasCoordinate( __corners, pt0_in_warpImg0, pt0_canvas, 0);
		convert_to_CanvasCoordinate( __corners, pt1, pt1_canvas, 1);

		VectorXf Patch0(_pnum*3), Patch0_edge(_pnum);
		float* patchData0 = Patch0.data();//颜色通道patch
		float* patchData0_edge = Patch0_edge.data();//edge通道patch
		PatchInitialization_on_Pixel(Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), img0_canvas, patchData0);
		PatchInitialization_on_Pixel(Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), img0_edge, patchData0_edge);
		VectorXf Patch1(_pnum * 3), Patch1_edge(_pnum);
		float* patchData1 = Patch1.data();
		float* patchData1_edge = Patch1_edge.data();
		PatchInitialization_on_Pixel(Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), img1_canvas, patchData1);
		PatchInitialization_on_Pixel(Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), img1_edge, patchData1_edge);
		//float aff = calAffinitybtPatches(patchData_dst, patchData0);
		
		//float sigma_color = sqrt(255 * 255 * _channels)*sigma_normalized;
		//float sigma_edge = sqrt(255 * 255)*sigma_normalized;
		//float thre_dist2 = -log(0.1f);// exp(-dist2)>=0.1f;
		float dist_color = 0,dist_edge = 0;
		for (int c = 0; c < _pdim; c++) // 指针操作更快--------------------by Kai Li
		{
			dist_color += (patchData0[c] - patchData1[c])*(patchData0[c] - patchData1[c]);
		}
		for (int c = 0; c < _pnum; ++c){
			dist_edge += (patchData0_edge[c] - patchData1_edge[c])*(patchData0_edge[c] - patchData1_edge[c]);
		}
		int dist = dist_color + dist_edge;
		
		float aff = exp(-(dist_color / (_pnum * 2 * sigma*sigma)));
		float c = min(max(aff, 0), 1);
		circle(canvas_img, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 5, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img0_canvas, Point(cvRound(pt0_canvas.x), cvRound(pt0_canvas.y)), 5, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(canvas_img, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 5, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
		circle(img1_canvas, Point(cvRound(pt1_canvas.x), cvRound(pt1_canvas.y)), 5, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
	}
	imwrite(outDir + "/img0_canvas.jpg", img0_canvas);
	imwrite(outDir + "/img1_canvas.jpg", img1_canvas);


	//求重叠区的img0_canvas，img1_canvas，解决patch处在重叠区边界的问题
	Mat AllWhite(__img0.size(), CV_8UC1, Scalar(255));
	Mat mask0, mask1;
	mask0 = WarpImg(AllWhite, Point2i(), _H);
	mask1 = AllWhite;
	Mat mask0_canvas, mask1_canvas;
	generateCanvasMasks(mask0, mask1, __corners[0], __corners[1], mask0_canvas, mask1_canvas);
	Mat overlap_region = mask0_canvas & mask1_canvas;
	for (int i = 0; i < canvas_img.rows; ++i){
		for (int j = 0; j < canvas_img.cols; ++j){
			bool is_valid = overlap_region.at<uchar>(i, j);
			if (!is_valid){
				img0_canvas.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				img1_canvas.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}

	__seam_quality_map = Mat(seamMask.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < canvas_img.rows; ++i){
		for (int j = 0; j < canvas_img.cols; ++j){
			Point2i pt(j, i);
			bool is_seam = seamMask.at<uchar>(i, j);
			if (is_seam){
				VectorXf Patch0(_pdim), Patch0_edge(_pnum);
				float* patchData0 = Patch0.data();
				float* patchData0_edge = Patch0_edge.data();//edge通道patch
				PatchInitialization_on_Pixel(pt, img0_canvas, patchData0);
				PatchInitialization_on_Pixel(pt, img0_edge, patchData0_edge);
				VectorXf Patch1(_pdim), Patch1_edge(_pnum);
				float* patchData1 = Patch1.data();
				float* patchData1_edge = Patch1_edge.data();
				PatchInitialization_on_Pixel(pt, img1_canvas, patchData1);		
				PatchInitialization_on_Pixel(pt, img1_edge, patchData1_edge);
				float dist_color = 0, dist_edge = 0;
				for (int c = 0; c < _pdim; c++) // 指针操作更快--------------------by Kai Li
				{
					dist_color += (patchData0[c] - patchData1[c])*(patchData0[c] - patchData1[c]);
				}
				for (int c = 0; c < _pnum; ++c){
					dist_edge += (patchData0_edge[c] - patchData1_edge[c])*(patchData0_edge[c] - patchData1_edge[c]);
				}
				int dist = dist_color + dist_edge;
				float aff = exp(-(dist_color / (_pnum * 2 * sigma*sigma)));
				float c = min(max(aff, 0), 1);
				__seam_quality_map.at<float>(i, j) = c;
				circle(canvas_img, pt, 3, Scalar(c * 255, 0, (1 - c) * 255), CV_FILLED, CV_AA);
			}
		}
	}
	imwrite(outDir + "/AffinityImg.jpg", canvas_img);
}

Mat Stitching::wellAlignedRegion(string outDir, vector<detail::ImageFeatures> &fea, vector<detail::MatchesInfo> &mat)
{
	double *h_data = (double*)mat[1].H.data;
	Matrix<double, 3, 3, RowMajor> H;
	for (int j = 0; j < 3; ++j)
	for (int k = 0; k < 3; ++k)
	{
		double tmp = h_data[j * 3 + k];
		H(j, k) = tmp;//用Mat型H的data对Matrix型H赋值
	}

	int width0 = __images_warped[0].cols;
	int height0 = __images_warped[0].rows;
	int width1 = __images_warped[1].cols;
	int height1 = __images_warped[1].rows;
	int minrow = min(__corners[0].y, __corners[1].y);
	int maxrow = max(__corners[0].y + height0, __corners[1].y + height1);
	int mincol = min(__corners[0].x, __corners[1].x);
	int maxcol = max(__corners[0].x + width0, __corners[1].x + width1);
	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;
	//构建局部配准区域掩膜
	Mat aligned_region_mask = Mat::zeros(height, width, CV_8UC1);
	/*
	计算局部配准区域
	*/
	vector<Rect> aligned_region;
	const vector<KeyPoint>& points0 = fea[0].keypoints;
	const vector<KeyPoint>& points1 = fea[1].keypoints;
	int num_region = _num_matches.size();
	if (num_region == 0)//如果无匹配点，则返回全255图像
		return Mat::ones(height, width, CV_8UC1);
	for (int idx_region = 0; idx_region < num_region; ++idx_region){
		int match_beg;
		int match_end;
		if (idx_region == 0){
			match_beg = 0;
			match_end = _num_matches[idx_region];
		}
		else{
			match_beg = _num_matches[idx_region - 1];
			match_end = _num_matches[idx_region];
		}
		//找出当前匹配点集合的矩形包络
		int maxrow = 0, minrow = width - 1;
		int maxcol = 0, mincol = height - 1;
		//找出当前匹配点集合的矩形包络范围
		for (int idx_match = match_beg; idx_match < match_end; ++idx_match){
			if (mat[1].inliers_mask[idx_match] == 0)
				continue;
			int idx_fea0 = mat[1].matches[idx_match].queryIdx;
			int idx_fea1 = mat[1].matches[idx_match].trainIdx;
			Point2f pt0 = points0[idx_fea0].pt, pt0_canvas, pt0_warped, pt0_in_warpImg0;	
			calcPoint_after_H(pt0, pt0_warped, H.data());
			pt0_in_warpImg0.x = pt0_warped.x - __corners[0].x;
			pt0_in_warpImg0.y = pt0_warped.y - __corners[0].y;
			convert_to_CanvasCoordinate( __corners, pt0_in_warpImg0, pt0_canvas, 0);
			Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
			convert_to_CanvasCoordinate( __corners, pt1, pt1_canvas, 1);
			vector<Point2f> aligned_points{ pt0_canvas, pt1_canvas };
			for (auto &point : aligned_points){
				int row = point.y, col = point.x;
				if (row < minrow)	minrow = row;
				if (row > maxrow)	maxrow = row;
				if (col < mincol)	mincol = col;
				if (col > maxcol)	maxcol = col;
			}

		}
		Point rect_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
		int rect_wd = maxcol - mincol + 1;
		int rect_ht = maxrow - minrow + 1;
		Point rect_tl(rect_center - Point(rect_wd / 2, rect_ht / 2));
		Rect orig_rect(rect_tl.x, rect_tl.y, rect_wd, rect_ht);
		aligned_region.push_back(orig_rect);
	}
	
	for (auto &roi : aligned_region){
		for (int i = roi.y; i < roi.y + roi.height; ++i){
			for (int j = roi.x; j < roi.x + roi.width; ++j){
				aligned_region_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	//imwrite(outDir + "/aligned_region_mask.jpg", aligned_region_mask);
	return aligned_region_mask;
}

Mat Stitching::wellAlignedRegion(string outDir, vector<detail::ImageFeatures> &fea, vector<detail::MatchesInfo> &mat,
	const Warping& meshWarper)
{
	double *h_data = (double*)mat[1].H.data;
	Matrix<double, 3, 3, RowMajor> H;
	for (int j = 0; j < 3; ++j)
	for (int k = 0; k < 3; ++k)
	{
		double tmp = h_data[j * 3 + k];
		H(j, k) = tmp;//用Mat型H的data对Matrix型H赋值
	}

	int width0 = __images_warped[0].cols;
	int height0 = __images_warped[0].rows;
	int width1 = __images_warped[1].cols;
	int height1 = __images_warped[1].rows;
	int minrow = min(__corners[0].y, __corners[1].y);
	int maxrow = max(__corners[0].y + height0, __corners[1].y + height1);
	int mincol = min(__corners[0].x, __corners[1].x);
	int maxcol = max(__corners[0].x + width0, __corners[1].x + width1);
	//size of canvas 
	int width = maxcol - mincol + 1, height = maxrow - minrow + 1;
	//构建局部配准区域掩膜
	Mat aligned_region_mask = Mat::zeros(height, width, CV_8UC1);
	/*
	计算局部配准区域
	*/
	vector<Rect> aligned_region;
	const vector<KeyPoint>& points0 = fea[0].keypoints;
	const vector<KeyPoint>& points1 = fea[1].keypoints;
	int num_region = _num_matches.size();
	if (num_region == 0)//如果无匹配点，则返回全255图像
		return Mat::ones(height, width, CV_8UC1);
	for (int idx_region = 0; idx_region < num_region; ++idx_region){
		int match_beg;
		int match_end;
		if (idx_region == 0){
			match_beg = 0;
			match_end = _num_matches[idx_region];
		}
		else{
			match_beg = _num_matches[idx_region - 1];
			match_end = _num_matches[idx_region];
		}
		//找出当前匹配点集合的矩形包络
		int maxrow = 0, minrow = width - 1;
		int maxcol = 0, mincol = height - 1;
		//找出当前匹配点集合的矩形包络范围
		for (int idx_match = match_beg; idx_match < match_end; ++idx_match){
			if (mat[1].inliers_mask[idx_match] == 0)
				continue;
			int idx_fea0 = mat[1].matches[idx_match].queryIdx;
			int idx_fea1 = mat[1].matches[idx_match].trainIdx;
			Point2f pt0 = points0[idx_fea0].pt, pt0_canvas;	
			pt0_canvas = orgPt02canPt(pt0, meshWarper, __corners);
			pt0_canvas.x = min(max(pt0_canvas.x, 0), width - 1);
			pt0_canvas.y = min(max(pt0_canvas.y, 0), height - 1);
			Point2f pt1 = points1[idx_fea1].pt, pt1_canvas;
			convert_to_CanvasCoordinate(__corners, pt1, pt1_canvas, 1);
			vector<Point2f> aligned_points{ pt0_canvas, pt1_canvas };
			for (auto &point : aligned_points){
				int row = point.y, col = point.x;
				if (row < minrow)	minrow = row;
				if (row > maxrow)	maxrow = row;
				if (col < mincol)	mincol = col;
				if (col > maxcol)	maxcol = col;
			}
		}
		Point rect_center((mincol + maxcol) / 2, (minrow + maxrow) / 2);
		int rect_wd = maxcol - mincol + 1;
		int rect_ht = maxrow - minrow + 1;
		Point rect_tl(rect_center - Point(rect_wd / 2, rect_ht / 2));
		Rect orig_rect(rect_tl.x, rect_tl.y, rect_wd, rect_ht);
		aligned_region.push_back(orig_rect);
	}

	for (auto &roi : aligned_region){
		for (int i = roi.y; i < roi.y + roi.height; ++i){
			for (int j = roi.x; j < roi.x + roi.width; ++j){
				aligned_region_mask.at<uchar>(i, j) = 255;
			}
		}
	}
	//imwrite(outDir + "/aligned_region_mask.jpg", aligned_region_mask);
	return aligned_region_mask;
}

bool Stitching::isSeamPassAlignedRegion(Mat &seam, Mat &region)
{
	Mat seam_in_region = seam & region;
	int num_white=0;
	for (int row = 0; row < seam_in_region.rows; ++row){
		for (int col = 0; col < seam_in_region.cols; ++col){
			uchar value = seam_in_region.at<uchar>(row, col);
			if (value>0){
				++num_white;
			}
		}
	}

	if (num_white > 0)
		return true;
	else
		return false;
}

bool Stitching::isHomoCorrect(double* h_data)
{
	if (!h_data)
		return false;
	Point2f tl(0, 0), tr(_width - 1, 0), bl(0, _height - 1), br(_width - 1, _height - 1);
	calcPoint_after_H(tl, tl, h_data);
	calcPoint_after_H(tr, tr, h_data);
	calcPoint_after_H(bl, bl, h_data);
	calcPoint_after_H(br, br, h_data);
	float x_min, x_max, y_min, y_max;
	x_min = min(tl.x, min(tr.x, min(bl.x, br.x)));
	y_min = min(tl.y, min(tr.y, min(bl.y, br.y)));
	x_max = max(tl.x, max(tr.x, max(bl.x, br.x)));
	y_max = max(tl.y, max(tr.y, max(bl.y, br.y)));
	int canvas_wd = x_max - x_min + 1;
	int canvas_ht = y_max - y_min + 1;
	float wd_times = float(canvas_wd) / float(_width), ht_times = float(canvas_ht) / float(_height);
	if (tl.x > tr.x || tl.y > bl.y || tr.y > br.y || bl.x > br.x)//如果发生翻转，则H无效
		return false;
	else{
		if (abs(h_data[0]) > 2 || abs(h_data[1]) > 2 || abs(h_data[3]) > 2 || abs(h_data[4]) > 2)
			return false;
		else if (wd_times > 2 || ht_times > 2 || wd_times < 0.5 || ht_times < 0.5)
			return false;
		else
			return true;
	}	
}

Mat Stitching::getMaskfromPoints(Size inSize, vector<Point> &pts)
{

	Mat mask(inSize, CV_8UC1, Scalar(0));
	for (auto &pt : pts){
		pt.x = min(max(pt.x, 0), inSize.width - 1);
		pt.y = min(max(pt.y, 0), inSize.height - 1);
		mask.at<uchar>(pt.y, pt.x) = 255;
	}
	/*Mat mask_dilated;
	dilate(mask, mask_dilated, Mat(),Point(-1,-1),5);*/
	//dilate(mask1, mask1_dilated, Mat());
	return mask;
}

vector<Point> Stitching::getPointsfromMask(Mat &img)
{
	vector<Point> pts;
	for (int row = 0; row < img.rows; ++row){
		for (int col = 0; col < img.cols; ++col){
			Point pt(col, row);
			if (img.at<uchar>(row, col)>0){
				pts.push_back(pt);
			}
		}
	}
	return pts;
}

void Stitching::MixFeasMats(vector<ImageFeatures> &inFeas, vector<MatchesInfo> &inMats,
	vector<ImageFeatures> &addFeas, vector<MatchesInfo> &addMats)
{
	if (inFeas.size() == 0 || inMats.size() == 0){
		inFeas = addFeas;
		inMats = addMats;
		return;
	}

	//addFeas[0]压入inFeas[0]中
	int num_rectfeas0 = inFeas[0].keypoints.size();
	for (int idx_fea = 0; idx_fea < addFeas[0].keypoints.size(); ++idx_fea){
		KeyPoint& keypoint = addFeas[0].keypoints[idx_fea];
		Mat Drow = addFeas[0].descriptors.row(idx_fea);
		inFeas[0].keypoints.push_back(keypoint);
		inFeas[0].descriptors.push_back(Drow);
	}
	//addFeas[1]压入inFeas[1]中
	int num_rectfeas1 = inFeas[1].keypoints.size();
	for (int idx_fea = 0; idx_fea < addFeas[1].keypoints.size(); ++idx_fea){
		KeyPoint& keypoint = addFeas[1].keypoints[idx_fea];
		Mat Drow = addFeas[1].descriptors.row(idx_fea);
		inFeas[1].keypoints.push_back(keypoint);
		inFeas[1].descriptors.push_back(Drow);
	}
	//addMats[1]压入inMats[1]中
	for (int idx_mat = 0; idx_mat < addMats[1].matches.size(); ++idx_mat){
		if (!addMats[1].inliers_mask[idx_mat])
			continue;
		//更新addMats[1]的matches、inliers_mask、num_inliers成员
		DMatch mat = addMats[1].matches[idx_mat];
		mat.queryIdx += num_rectfeas0;//新增的inliers匹配点索引的新增特征点是从addFeas[0].keypoints[num_rectfeas0]开始的
		mat.trainIdx += num_rectfeas1;
		inMats[1].matches.push_back(mat);
		inMats[1].inliers_mask.push_back(uchar(1));
		++inMats[1].num_inliers;
	}
	calcHomoFromMatches_8PointMethod(inMats[1], inFeas[0], inFeas[1]);
	calcDualMatches(inMats[2], inMats[1]);
}

bool Stitching::iteration(string outDir, int idx)
{
	//dilate_times = 1.5;
	_sigma_spatial = sigma_spatial_init;
	selcon_idx = 0;
	H_flag = false;
	Pass_flag = false;
	Dilate_flag = false;
	feas.clear();
	mats.clear();
	seam_section.clear();
	mask0_avoid = Mat(__img0.size(), CV_8UC1, Scalar(0));//初始时，不用避开某片区域
	mask1_avoid = Mat(__img1.size(), CV_8UC1, Scalar(0));


	cout << endl << endl;
	cout << "!!!iteration " << idx << ":" << endl;
	cout << "Select Seam Section ====>No." << selcon_idx << endl;
	cout << "sigma_spatial--------->" << _sigma_spatial << endl;
	char dir_name[100];//本次迭代文件夹名
	sprintf(dir_name, "%s/LocalArea%d", outDir.c_str(), idx);
	_mkdir(dir_name);
	_mkdir((string(dir_name)+"/Result").c_str());
	char preit_dir_name[100];//上次迭代文件夹名
	if (idx == 0){
		sprintf(preit_dir_name, "%s/GlobalResult_OpenCV", outDir.c_str(), idx - 1);
	}
	else
	{
		sprintf(preit_dir_name, "%s/LocalArea%d", outDir.c_str(), idx - 1);
	}
	

	
	while (!(H_flag&&Pass_flag)){//当H不符合要求，或者缝隙没有穿过配准区域时，循环		
		seam_section = getSeamSection(string(preit_dir_name) + "/Result", __seam_quality_map, __seam_mask, selcon_idx);//找缝隙段
		if (seam_section.size() == 0){
			return 1;
		}
		bool is_Homo_valid;
		if (idx == 0){//以缝隙段初始化特征点和匹配点
			is_Homo_valid = addLocalFeasMats(dir_name, seam_section, _H, feas, mats);
		}
		else{
			is_Homo_valid = addLocalFeasMats(dir_name, seam_section, _meshwarper, feas, mats);
		}
		
		cout << "AddLocalFeasMats" << endl;
		if (is_Homo_valid){
			ShowMatches(string(dir_name) + "/addFeasMats", feas, mats);
			cout << "ShowNewFeasMats" << endl;
		}
			
		//判断一
		//检查当前mats的H是否有效
		if (mats.size() == 0)
			H_flag = false;
		else
			H_flag = is_Homo_valid && isHomoCorrect((double*)mats[1].H.data);
		if (!H_flag){//如果H无效，则选取次差的缝隙段
			cout << "Invalid Homography" << endl;
			if (_sigma_spatial < 90){//未到sigma上界
				_sigma_spatial += 10;
				cout << "sigma_spatial--------->" << _sigma_spatial << endl;
			}
			else{//达到sigma上界
				_sigma_spatial = sigma_spatial_init;
				if (selcon_idx < seam_section.size()){//换一条缝隙，直到缝隙遍历完
					/*
					标记导致H无效的缝隙片段区域，得到mask0_avoid、mask1_avoid
					*/
					vector<Point> pts0, pts1;
					//把seam_section进行坐标变换，得到点集pts0和pts1
					if (idx == 0){//上次迭代为全局配准
						for (auto &pt : seam_section){
							Point pt0 = canPt2orgPt0(pt, _H, __corners);
							Point pt1 = canPt2orgPt1(pt, __corners);
							pts0.push_back(pt0);
							pts1.push_back(pt1);
						}
					}
					else{//上次迭代为局部配准
						for (auto &pt : seam_section){
							Point pt0 = canPt2orgPt0(pt, _meshwarper, __corners);
							Point pt1 = canPt2orgPt1(pt, __corners);
							pts0.push_back(pt0);
							pts1.push_back(pt1);
						}
					}
					//从pts0、pts1点集得到mask0_avoid，mask1_avoid
					Mat pts0_avoid = getMaskfromPoints(__img0.size(), pts0);
					dilate(pts0_avoid, pts0_avoid, Mat(), Point(-1, -1), 5);//膨胀pts0_avoid	
					Mat pts1_avoid = getMaskfromPoints(__img1.size(), pts1);
					dilate(pts1_avoid, pts1_avoid, Mat(), Point(-1, -1), 5);//膨胀mask1_avoid
					mask0_avoid = mask0_avoid | pts0_avoid;
					mask1_avoid = mask1_avoid | pts1_avoid;
					imwrite(string(dir_name) + "/Result/mask0_avoid.jpg", mask0_avoid);
					imwrite(string(dir_name) + "/Result/mask1_avoid.jpg", mask1_avoid);

					selcon_idx++;
					cout << endl;
					cout << "Select Seam Section ====>No." << selcon_idx << endl;
				}		
				else{//缝隙遍历完,迭代终止
					return 1;
				}		
			}
		}
		else{//当前H有效，则进行配准、找缝隙
			cout << "Valid Homography" << endl;
			vector<ImageFeatures> mixed_feas(__features);
			vector<MatchesInfo> mixed_mats(__matches);
			MixFeasMats(mixed_feas, mixed_mats, feas, mats);
			ShowMatches(string(dir_name) + "/mixedFeasMats", mixed_feas, mixed_mats);
			double* h_data = (double*)mixed_mats[1].H.data;
			Matrix<double, 3, 3, RowMajor> H;
			for (int j = 0; j < 3; ++j)
			for (int k = 0; k < 3; ++k)
			{
				double tmp = h_data[j * 3 + k];
				H(j, k) = tmp;//用Mat型H的data对Matrix型H赋值
			}

			//先进行投影变换，预配准
			cout << "Pre-warping by Global H:" << endl;
			if (isHomoCorrect(h_data)){
				cout << "H is correct:" << endl;
				//cout << mixed_mats[1].H << endl;
			}
			else{
				cout << "H is wrong:" << endl;
				//cout << mixed_mats[1].H << endl;
				_sigma_spatial += 5;
				selcon_idx = 0;
				cout << endl << endl;
				cout << "sigma_spatial :" << _sigma_spatial << endl;
				continue;
			}
			AlignmentbyH(H);
			vector<Point> prewarp_corners = __corners;
			imwrite(string(dir_name) + "/Result/img0_pre-warped.jpg", __images_warped[0]);

			//再进行CPW，精细配准
			cout << "Content Preserving Warping by Local H:" << endl;
			Mat img0_pre_warped = __images_warped[0], mask0_pre_warped = __masks_warped[0];
			erode(mask0_pre_warped, mask0_pre_warped, Mat());
			vector<ImageFeatures> feas_pre_warped(mixed_feas);
			for (auto &keypt : feas_pre_warped[0].keypoints){
				Point2f &pt = keypt.pt;
				pt = orgPt02warpPt(pt, H.data(), __corners);
			}
			Warping meshwarper(img0_pre_warped.size(), img0_pre_warped.size(), xquads, yquads, more_rigid);
			meshwarper.solve(mixed_mats[1], feas_pre_warped[0], feas_pre_warped[1]);
			Mat grid_warped = gridWarpImg(meshwarper);
			imwrite(string(dir_name) + "/Result/grid_warped.jpg", grid_warped);
			AlignmentbyMesh(meshwarper, img0_pre_warped, mask0_pre_warped);
			vector<Point> cpw_corners = __corners;
			Mat overlap_mask = getOverlapRegionMask(__masks_warped, __corners);

			Mat result_lin_bld = CalLinearBlend(__images_warped[0], __images_warped[1],
				__masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);//直接线性融合
			imwrite(string(dir_name) + "/Result/result_lin_bld.jpg", result_lin_bld);
			imwrite(string(dir_name) + "/Result/img0_CPW.jpg", __images_warped[0]);

			/*
			我的graphcut方法
			*/
			////融合前找缝隙
			//Mat mask0_avoid_warped, mask1_avoid_warped(mask1_avoid);
			//WarpImg(mask0_avoid, mask0_avoid_warped, Point(), H);
			//meshwarper.warp(mask0_avoid_warped, mask0_avoid);
			//FindSeambyGraphcut_WeightedPatchDifference(string(dir_name) + "/Result", __images_warped, __masks_warped,
			//	__corners, mixed_feas, mixed_mats, vector<Mat>{mask0_avoid_warped, mask1_avoid_warped});//找缝隙
			////融合
			//Mat result;
			//Blend(result, Mat(), __images_warped, __masks_warped, __corners, detail::Blender::NO);//blender融合		
			//imwrite(string(dir_name) + "/Result/result.jpg", result);
			/*
			OpenCV方法
			*/
			cout << "Seam Finding by Opencv" << endl;
			int blend_type = detail::Blender::NO;
			Mat result, result_mask;
			Findseam_and_Blend(result, result_mask, __images_warped, __masks_warped, __corners, blend_type);
			cout << "Blending by Opencv" << endl;
			imwrite(string(dir_name) + "/Result/result.jpg", result);

			Mat seam_mask = generateSeamMask_on_Canvas(string(dir_name) + "/Result");
			Mat seam_quality_map = getSeamQualityMat(seam_mask, overlap_mask);
			ShowAlignQuality(string(dir_name) + "/Result", feas_pre_warped, mixed_mats, seam_mask, seam_quality_map, result, meshwarper);
			_num_matches.push_back(mixed_mats[1].matches.size());
			aligned_region_mask = wellAlignedRegion(string(dir_name) + "/Result", feas_pre_warped, mixed_mats, meshwarper);//找匹配点区域	
			dilate(aligned_region_mask, aligned_region_mask, Mat(), Point(-1, -1), 30);
			imwrite(string(dir_name) + "/Result/aligned_region_mask.jpg", aligned_region_mask);
			Pass_flag = isSeamPassAlignedRegion(seam_mask, aligned_region_mask);
			//判断二
			//是否穿过配准区域
			if (!Pass_flag){//若没有穿过，则下次循环将区域扩大
				cout << "Seam Doesn't Pass through Aligned Area" << endl;
				_sigma_spatial += 10;
				cout << endl << endl;
				cout << "sigma_spatial--------->" << _sigma_spatial << endl;
				_num_matches.pop_back();
			}
			else{//进入下一次迭代
				cout << "Seam Pass through Aligned Area" << endl;
				__seam_quality_map = seam_quality_map;//用下一次迭代找匹配点的缝隙依据
				__seam_mask = seam_mask;
				__features.clear();
				__features = mixed_feas;
				__matches.clear();
				__matches = mixed_mats;
				_H = H;
				__result = result;
				_meshwarper = meshwarper;
				_cpw_corners = cpw_corners;
				_prewarp_corners = prewarp_corners;
				return 0;
			}
		}
	}
}

Mat Stitching::getSeamQualityMat(const Mat& seamMask, const Mat& overlapMask)
{
	//只需要img0_canvas和img1_canvas的重叠区部分来计算patch
	Mat img0_canvas, img1_canvas;
	generateCanvasImgs(__images_warped[0], __images_warped[1], __corners[0], __corners[1], img0_canvas, img1_canvas);
	for (int i = 0; i < img0_canvas.rows; ++i){
		for (int j = 0; j < img0_canvas.cols; ++j){
			bool is_valid = overlapMask.at<uchar>(i, j);
			if (!is_valid){
				img0_canvas.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				img1_canvas.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
		}
	}
	Mat result = Mat(seamMask.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < img0_canvas.rows; ++i){
		for (int j = 0; j < img0_canvas.cols; ++j){
			Point2i pt(j, i);
			bool is_seam = seamMask.at<uchar>(i, j);
			if (is_seam){
				VectorXf Patch0(_pdim);
				float* patchData0 = Patch0.data();
				PatchInitialization_on_Pixel(pt, img0_canvas, patchData0);
				VectorXf Patch_dst(_pdim);
				float* patchData_dst = Patch_dst.data();
				PatchInitialization_on_Pixel(pt, img1_canvas, patchData_dst);
				float aff = calAffinitybtPatches(patchData_dst, patchData0);
				float c = min(max(aff, 0), 1);
				result.at<float>(i, j) = c;
			}
		}
	}

	return result;
}

Mat Stitching::getOverlapRegionMask(const vector<Mat>& masksWarped, const vector<Point>& corners)
{
	Mat mask0_canvas, mask1_canvas;
	generateCanvasMasks(masksWarped[0], masksWarped[1], corners[0], corners[1], mask0_canvas, mask1_canvas);
	Mat overlap_region = mask0_canvas & mask1_canvas;
	return overlap_region;
}

Point Stitching::canPt2orgPt0(Point canPt, Matrix<double, 3, 3, RowMajor> H, vector<Point> corners)
{
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	//变到im0坐标系
	Point TL_in_im0 = corners[1] - Point(mincol, minrow);//corners[1]的canvas坐标
	canPt.x = canPt.x - TL_in_im0.x;
	canPt.y = canPt.y - TL_in_im0.y;
	//反变换
	Point2f Pt_dst;
	Matrix<double, 3, 3, RowMajor> H_inv = H.inverse();
	double *homo = H_inv.data();
	calcPoint_after_H(Point2f(canPt), Pt_dst, homo);

	Point orgPt(Pt_dst);
	//保证输出矩形有效
	orgPt.x = min(max(0, orgPt.x), __img0.cols - 1);
	orgPt.y = min(max(0, orgPt.y), __img0.rows - 1);
	return orgPt;
}

Point Stitching::canPt2orgPt0(Point canPt, const Warping& meshWarper, vector<Point> &corners)
{
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	//变到im0坐标系
	Point TL_in_im0 = corners[1] - Point(mincol, minrow);//corners[1]的canvas坐标
	Point warpedPt;
	warpedPt.x = canPt.x - TL_in_im0.x;
	warpedPt.y = canPt.y - TL_in_im0.y;

	//判断Pt_warped在哪个quad内
	Point2f* V2 = (Point2f *)meshWarper.__vertices2.data();
	int xquads = meshWarper._xquads;
	int yquads = meshWarper._yquads;
	int xctrls = meshWarper._xquads + 1;
	int yctrls = meshWarper._yquads + 1;
	Size orgSize = meshWarper._src_size;
	for (int i = 0; i < yquads; i++)
	{
		for (int j = 0; j < xquads; j++)
		{
			vector<Point2f> vpts;//该quad四个顶点（float型坐标）
			vpts.push_back(V2[i*xctrls + j]);
			vpts.push_back(V2[i*xctrls + j + 1]);
			vpts.push_back(V2[(i + 1)*xctrls + j + 1]);
			vpts.push_back(V2[(i + 1)*xctrls + j]);

			if (pointPolygonTest(vpts, warpedPt, false)<0)//检查warpedPt是否在该quad外部
				continue;
			if (meshWarper.__homos[i*xquads + j].empty()){//检查该quad的H是否为空
				return Point(-1,-1);//返回(-1,-1)表示无效的点
			}			
			double *h = (double *)meshWarper.__homos[i*xquads + j].data;
			Point2f orgPt;
			calcPoint_after_H(Point2f(warpedPt), orgPt, h);

			orgPt.x = min(max(orgPt.x, 0), orgSize.width - 1);
			orgPt.y = min(max(orgPt.y, 0), orgSize.height - 1);
			return Point(orgPt);
		}
	}
}

Point Stitching::canPt2orgPt1(Point canPt, vector<Point> corners)
{
	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	//变到im1坐标系
	Point TL_canvas = corners[1] - Point(mincol, minrow);//corners[1]的canvas坐标
	Point orgPt;
	orgPt.x = canPt.x - TL_canvas.x;
	orgPt.y = canPt.y - TL_canvas.y;

	//保证输出矩形有效
	orgPt.x = min(max(0, orgPt.x), __img1.cols - 1);
	orgPt.y = min(max(0, orgPt.y), __img1.rows - 1);
	return orgPt;
}

Point Stitching::orgPt02canPt(Point orgPt, const Warping& meshWarper, vector<Point> &corners)
{

	int gx = int(orgPt.x / meshWarper._quadWidth);//特征点p1的x轴索引（以_quadWidth为步长）
	int gy = int(orgPt.y / meshWarper._quadHeight);//特征点p1的y轴索引（以_quadHeight为步长）
	int xquads = meshWarper._xquads;
	double *h = (double *)meshWarper.__homos[gy*xquads + gx].data;

	Matrix<double, 3, 3, RowMajor> homo, homo_inv;
	for (int i = 0; i < 3; ++i){
		for (int j = 0; j < 3; ++j){
			homo(i,j) = h[i * 3 + j];
		}
	}
	homo_inv = homo.inverse();
	h = homo_inv.data();
	Point2f warpPt;
	calcPoint_after_H(Point2f(orgPt), warpPt, h);

	int minrow = min(corners[0].y, corners[1].y);
	int mincol = min(corners[0].x, corners[1].x);
	Point canPt = Point(warpPt) + corners[1] - Point(mincol,minrow);
	return canPt;
}

Point2f Stitching::orgPt02warpPt(Point2f orgPt, const double* homo, vector<Point> &corners)
{
	Point2f warpPt;
	calcPoint_after_H(orgPt, warpPt, homo);
	warpPt = warpPt - Point2f(corners[0]);
	return warpPt;
}

Point2f Stitching::warpPt2orgPt0(Point2f warpPt, const double* invHomo, vector<Point> &corners)
{
	Point2f orgPt;
	warpPt = warpPt + Point2f(corners[0]);
	calcPoint_after_H(warpPt, orgPt, invHomo);
	return orgPt;
}

float Stitching::getSeamQuality(const Mat& seamMask, const Mat& seamQualityMat)
{
	float value, sum = 0;
	int count = 0;
	for (int i = 0; i < seamMask.rows; ++i){
		for (int j = 0; j < seamMask.cols; ++j){
			Point2i pt(j, i);
			bool is_seam = seamMask.at<uchar>(i, j);
			if (is_seam){
				sum += seamQualityMat.at<float>(i, j);
				count++;
			}
		}
	}
	value = sum / count;
	return value;
}

bool Stitching::Stitch(string outDir)
{
	//ShowMatches(outDir + "/Matches", true);
	//ShowMatches(outDir + "/Matches", false);
	
	//分块局部配准---------------------------------------------------20170622
	//Warping meshwarper(__img0.size(), __images_warped[0].size(), xquads, yquads, more_rigid);
	//AlignmentbyMesh(meshwarper);
	//imwrite(outDir + "/GlobalResult_OpenCV/Result/img0_warped.jpg", __images_warped[0]);
	//Mat grid_warped = gridWarpImg(meshwarper);
	//imwrite(outDir + "/GlobalResult_OpenCV/Result/grid_warped.jpg", grid_warped);
	//Mat result_lin_bld = CalLinearBlend(__images_warped[0], __images_warped[1],
	//	__masks_warped[0], __masks_warped[1], __corners[0], __corners[1]);//直接线性融合
	//imwrite(outDir + "/GlobalResult_OpenCV/Result/result_lin_bld.jpg", result_lin_bld);
	//Composite_Opencv(outDir + "/GlobalResult_OpenCV/Result");

	/***************
	全局配准
	****************/
	AlignmentbyH(_H);
	/*Warping meshwarper(__img0.size(), __images_warped[0].size(), xquads, yquads, more_rigid);
	meshWarper.solve(__matches[1], __features[0], __features[1]);
	AlignmentbyMesh(meshwarper);*/
	Mat overlap_mask = getOverlapRegionMask(__masks_warped,__corners);
	Composite_Opencv(outDir + "/GlobalResult_OpenCV/Result");
	

	//显示缝隙和匹配点质量
	__seam_mask=generateSeamMask_on_Canvas(outDir + "/GlobalResult_OpenCV/Result");
	__seam_quality_map = getSeamQualityMat(__seam_mask, overlap_mask);
	ShowAlignQuality(outDir + "/GlobalResult_OpenCV/Result", __features, __matches, __seam_mask, __seam_quality_map, __result);//计算缝隙质量

	float global_seam_quality=getSeamQuality(__seam_mask, __seam_quality_map);
	cout << "拼接结果质量分数：" << global_seam_quality << endl;

	__features.clear();
	__matches.clear();

	//调试
	ofstream f_out(outDir + "/resultQuality.txt");
	f_out << "全局拼接结果缝隙质量： "<<global_seam_quality << endl;
	f_out << endl;

	/**************
	迭代局部配准
	***************/
	for (int i = 0; i < 5; ++i){
		if (iteration(outDir, i)){//本次迭代终止
			cout << "Project Failed!!!" << endl;
			return 1;	
		}
		else{//本次迭代顺利
			float seam_quality = getSeamQuality(__seam_mask, __seam_quality_map);
			cout << "拼接结果质量分数：" << seam_quality << endl;
			/*if (seam_quality - global_seam_quality > 0.1)
			break;*/
			f_out << "第" << i << "次迭代缝隙质量： " << seam_quality << endl;
		}	
	}
	f_out.close();
	cout << "Project Succeeded!!!" << endl;
	return 0;	
	//iteration(outDir, 0);
	//iteration(outDir, 1);
	//iteration(outDir, 2);
	//iteration(outDir, 3);
	//iteration(outDir, 4);
}