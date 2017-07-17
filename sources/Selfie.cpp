
#include "OpticalFlow.h"
#include "qpOASES.hpp"
//#include "PImage.h"
//#include "permutohedral.h"
#include "matting.h"
#include "TImage.h"
#include "gkdtree.h"
#include "Selfie.h"
#include "Warping.h"
#include "Composite.h"
#include "fibheap.h"


const int Selfie::_max_num_faces = 1;
const int Selfie::_req_num_inliers = 9;
const float Selfie::_req_homo_conf = 0.80f;
const float Selfie::_req_classifier_conf = 0.80f;
const float Selfie::_thre_valid_bk = 0.20f;
const int Selfie::_req_same_labels = 1;
const float Selfie::_req_ncc = 0.80f;
const float Selfie::_sigma_normalized = 0.05f;


Selfie::Selfie(string imDir, string imExt, string feaExt)
{
	//vector<Mat> imgs, masks;
	//imgs.push_back(imread("s.png"));
	//imgs.push_back(imread("t.jpg"));
	//vector<detail::ImageFeatures> features(2);
	//Ptr<detail::FeaturesFinder> finder = new detail::SurfFeaturesFinder();
	//(*finder)(imgs[0], features[0]);
	//(*finder)(imgs[1], features[1]);

	//detail::BestOf2NearestMatcher matcher;
	//vector<detail::MatchesInfo> matches;
	//matcher(features, matches);
	//Warping meshwarper(imgs[0].size(), imgs[1].size(), 8, 8, 0.05);
	//meshwarper.solve(matches[0*2+1], features[0], features[1]);

	//Mat warped;
	//meshwarper.warp(warped, imgs[0]);
	//imwrite("warped.png", warped);

	// scan the image directory, load all images
	__indir = imDir;
	ScanDirectory(imDir, imExt, __imFiles, __imNames);
	_num_frames = __imFiles.size();
	if (_num_frames < 2)	
	{
		printf("ERROR: please load more than two images");
		exit(0);
	}
	printf("load %d images in %s/*%s\n", _num_frames, imDir.c_str(), imExt.c_str());
	__imgs.resize(_num_frames);
	for (int i = 0; i < _num_frames; i++)
		__imgs[i] = imread(__imFiles[i]);
	_width = __imgs[0].cols;
	_height = __imgs[0].rows;
	_channels = __imgs[0].channels();

	// parameers
	_chs = _channels+1;
	_sigma_color = sqrt(255*255*_chs*_sigma_normalized*_sigma_normalized);
	_sigma_spatial = sqrt(((_width-1)*(_width-1)+(_height-1)*(_height-1))*_sigma_normalized*_sigma_normalized);
	_ransac_reject = min(_width, _height)*0.02;
	_psize = cvRound(min(_width,_height)*0.005);

	// get the mask file
	vector<string> mask_names;
	ScanDirectory(imDir+"/Mask", "png", __maskFiles, mask_names);
	__masks.resize(min(_num_frames,__maskFiles.size()));
	for (int i = 0; i < __masks.size(); i++)
		__masks[i] = imread(__maskFiles[i], 0);

	// find features and matches
	vector<string> fea_path, fea_name;
	if (strcmp(feaExt.c_str(),"pnt")==0)	
	{
		if (ScanDirectory(imDir, feaExt, fea_path, fea_name))
			readFeaturesMatchesFromPNTs(fea_path);
		else
			printf("ERROR: please specify the *.pnt files\n");
	} 
	else if (strcmp(feaExt.c_str(),"if")==0) 
	{
		if (ScanDirectory(imDir, feaExt, fea_path, fea_name))
			readFeaturesMatchesFromIF(fea_path[0]);
		else
			printf("ERROR: please specify the *.if file\n");
	} 
	else
	{
		//findKLTFeaturesMatches();
		//findSIFTFeaturesMatches();

		// calculate optical flow, use the original image first
		vector<string> im_files, im_names;
		ScanDirectory(imDir, "origin.png", im_files, im_names);
		for (int i = 0; i < _num_frames; i++)
			__imgs[i] = imread(im_files[i]);
		vector<DImage> __flows;
		getOpticalFlow(__flows, imDir+"/Flow");

		for (int i = 0; i < _num_frames; i++)
			__imgs[i] = imread(__imFiles[i]);
		// get sampled features from optical flow
		findFlowFeaturesMatches(__flows);
	}
	ScanDirectory(imDir+"/Flow", "txt", __flowFiles, fea_name);
	ScanDirectory(imDir+"/Flow", "visual.jpg", __visFiles, fea_name);

	// index the the features
	//removeShortTrajectories();
	indexFeaturesMatches();
}

Selfie::~Selfie()
{

}

MatrixXf Selfie::readMatrixFromTXT(string filename)
{
	MatrixXf M;
	ifstream fp(filename.c_str());
	if (fp.fail())
	{
		printf("can not open %s\n!", filename.c_str());
		return M;
	}
	int rows, cols, elems;
	fp >> cols >> rows;
	elems = rows*cols;
	M.resize(rows, cols);
	for (int i = 0; i < elems && !fp.eof(); i++)
		fp >> M(i/cols, i%cols);
	fp.close();
	return M;
}

void Selfie::writeMatrixToTXT(string filename, const MatrixXf& M)
{
	ofstream fp(filename.c_str());
	int rows = M.rows(), cols = M.cols();
	fp << cols << " " << rows << endl;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
			fp << M(i, j) << " ";
		fp << endl;
	}
	fp.close();
}

void Selfie::readFeaturesMatchesFromPNTs(const vector<string>& filenames)
{
	printf("load features and matches\n");
	__features.clear();
	__matches.clear();
	__features.resize(_num_frames);
	__matches.resize(_num_frames*_num_frames);

	for (int i = 0; i < _num_frames; i++)
	{
		detail::ImageFeatures& F = __features[i];
		detail::ImageFeatures& pF = (i>0) ? __features[i-1] : __features[0];
		detail::MatchesInfo& M = (i>0) ? __matches[(i-1)*_num_frames+i] : __matches[0];
		detail::MatchesInfo& dM = (i>0) ? __matches[i*_num_frames+i-1] : __matches[0];
		F.img_idx = i;
		F.img_size = Size(_width,_height);

		// read pnt format file	
		ifstream fp(filenames[i].c_str());
		if (fp.fail()) 
		{
			printf("ERROR: cannot read file: %s\n", filenames[i].c_str());
			continue;
		}
		string tmp_str;		
		getline(fp, tmp_str);
		vector<tracking_pnt> cc;
		while (!fp.eof())	
		{
			tracking_pnt t;
			fp >> t.x >> t.y >> t.manual >> t.type3d >> t.px >> t.py >> t.pz >> t.ident >> t.hasprev >> t.pcx >> t.pcy >> t.support;
			if (fp.eof())
				break;
			cc.push_back(t);
		}
		fp.close();

		// convert tracking_pnt to ImageFeatures & MatchesInfo	
		int num_current = cc.size(), valid_idx = 0;
		for (int j = 0; j < num_current; j++)
		{
			// ImageFeatures
			tracking_pnt t = cc[j];
			if (t.x < 0 || t.y < 0)
				continue;
			F.keypoints.push_back(KeyPoint(t.x, t.y, 0));
			valid_idx++;

			// MatchesInfo
			if (!t.hasprev)
				continue;
			int num_previous = pF.keypoints.size(); 
			int m = min(j, num_previous-1);
			Point2f p = pF.keypoints[m].pt;
			if ((t.pcx-p.x)*(t.pcx-p.x)+(t.pcy-p.y)*(t.pcy-p.y) > 1.0)
			{
				for (m=0; m<num_previous; m++)
				{
					p = pF.keypoints[m].pt;
					if ((t.pcx-p.x)*(t.pcx-p.x)+(t.pcy-p.y)*(t.pcy-p.y) < 1.0)
						break;
				}
			}
			M.matches.push_back(DMatch(m, valid_idx-1, -1, 0));
		}
		if (i <= 0)
			continue;
		calcHomoFromMatches(M, pF, F);
		calcDualMatches(dM, M);
	}

	// matches between non-adjacent frames
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+2; j <_num_frames; j++)
		{
			detail::MatchesInfo& M = __matches[i*_num_frames+j];
			detail::MatchesInfo& dM = __matches[j*_num_frames+i];
			const detail::MatchesInfo& pM = __matches[(j-1)*_num_frames+j];
			const detail::MatchesInfo& ppM = __matches[i*_num_frames+j-1];

			vector<int> pInd(__features[j-1].keypoints.size(), -1);
			for (int k = 0; k < pM.matches.size(); k++)
			{
				const DMatch& t = pM.matches[k];
				pInd[t.queryIdx] = t.trainIdx;
			}
			for (int k = 0; k < ppM.matches.size(); k++)
			{
				const DMatch& t = ppM.matches[k];
				if (pInd[t.trainIdx]>=0)
					M.matches.push_back(DMatch(t.queryIdx, pInd[t.trainIdx], -1, 0));
			}
			calcHomoFromMatches(M, __features[i], __features[j]);
			calcDualMatches(dM, M);
		}
	}
}

void Selfie::readFeaturesMatchesFromIF(string filename)
{
	printf("load features and matches\n");
	ifstream fp(filename.c_str());
	if (fp.fail())
	{
		printf("ERROR: can not open file %s\n", filename.c_str());
		return;
	}
	__features.clear();
	__matches.clear();
	__features.resize(_num_frames);
	__matches.resize(_num_frames*_num_frames);

	// read features
	int size_features, num_features;
	fp >> size_features;
	for (int i = 0; i < size_features; i++)
	{
		detail::ImageFeatures F;
		fp >> F.img_idx >> F.img_size.width >> F.img_size.height >> num_features;
		for (int j = 0; j < num_features; j++)
		{
			KeyPoint p;
			fp >> p.pt.x >> p.pt.y >> p.size >> p.angle >> p.response >> p.octave >> p.class_id;
			F.keypoints.push_back(p);
		}
		__features[F.img_idx] = F;
	}
	// read matches
	int size_matches, num_matches;
	fp >> size_matches;
	for (int i = 0; i < size_matches && !fp.eof(); i++)
	{
		detail::MatchesInfo M;
		fp >> M.src_img_idx >> M.dst_img_idx >> num_matches >> M.num_inliers >> M.confidence;
		M.H.create(3, 3, CV_64F);
		for (int j = 0; j < 3; j++)
			for (int k = 0; k<3; k++)
				fp >> M.H.at<double>(j, k);
		
		for (int j = 0; j < num_matches ;j++)
		{
			DMatch t;
			uchar inlier;
			fp >> t.queryIdx >> t.trainIdx >> t.imgIdx >> t.distance >> inlier;
			M.matches.push_back(t);
			M.inliers_mask.push_back(inlier);
		}
		__matches[M.src_img_idx*_num_frames+M.dst_img_idx] = M;
	}
	fp.close();
}

void Selfie::writeFeaturesMatchesToIF(string filename)
{
	ofstream fp(filename.c_str());
	fp << __features.size() << endl;
	for (int i = 0; i < __features.size(); i++)
	{
		const detail::ImageFeatures& F = __features[i];
		fp << F.img_idx << " " << F.img_size.width << " " << F.img_size.height << " " << F.keypoints.size() << endl;
		for (int j = 0; j < F.keypoints.size(); j++)
		{
			const KeyPoint& p = F.keypoints[j];
			fp << p.pt.x << " " << p.pt.y << " " << p.size << " " << p.angle << " " << p.response << " " << p.octave << " " << p.class_id <<endl; 
		}
	}

	int size_matches = 0;
	for (int i = 0; i < __matches.size(); i++)
		if (__matches[i].matches.size()>0)
			size_matches++;

	fp << size_matches << endl;
	for (int i = 0; i < __matches.size(); i++)
	{
		if (__matches[i].matches.size()<=0)
			continue;
		const detail::MatchesInfo& M = __matches[i];
		fp << M.src_img_idx << " " << M.dst_img_idx << " " << M.matches.size() << " " << M.num_inliers << " " << M.confidence << endl;
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k<3; k++) {
				fp << M.H.at<double>(j, k) << " ";
			}
			fp << endl;
		}
		for (int j = 0; j < M.matches.size(); j++)
		{
			const DMatch& t = M.matches[j];
			fp << t.queryIdx << " " << t.trainIdx << " " << t.imgIdx <<  " " << t.distance << " " << M.inliers_mask[j] << endl;
		}
	}
	fp.close();
}

void Selfie::readOpticalFlow(DImage& flow, string filename)
{
	ifstream fp(filename.c_str());
	int W, H, C;
	fp >> W >> H >> C;
	if (!flow.matchDimension(W, H, C))
		flow.allocate(W, H, C);
	double *flow_data = flow.data();
	for (int i = 0; i < H; i++) 
	{
		for (int j = 0; j < W; j++) 
		{
			for (int k = 0; k < C; k++)
				fp >> flow_data[(i*W+j)*C+k];
		}
	}
	fp.close();
}

void Selfie::writeOpticalFlow(string filename, const DImage& flow)
{
	ofstream fp(filename.c_str());
	int W = flow.width(), H = flow.height(), C = flow.nchannels();
	fp << W << " " << H << " " << C << endl;
	const double *flow_data = flow.data();
	for (int i = 0; i < H; i++) 
	{
		for (int j = 0; j < W; j++) 
		{
			for (int k = 0; k < C; k++)
				fp << flow_data[(i*W+j)*C+k] << " ";		
		}
	}
	fp.close();
}

template<class T>
void Selfie::saveFloatImage(string filename, const T* data, int width, int height, int channels)
{
	Mat img(height, width, CV_8UC(channels));
	saveFloatImage(img, data, width, height, channels);
	imwrite(filename, img);
}

template<class T>
void Selfie::saveFloatImage(Mat& img, const T* data, int width, int height, int channels)
{
	if (img.cols != width || img.rows != height || img.type() != CV_8UC(channels))
		img.create(height, width, CV_8UC(channels));
	uchar *im_data = img.data;
	int num_elems = width*height*channels;
	for (int k = 0; k < num_elems; k++)
		im_data[k] = min(max(cvRound(data[k]*255), 0), 255);
}

template<class T>
double Selfie::calcNCC(const T* a, const T* b, int size)
{
	double mu_a, mu_b;
	ComputeMean(1, size, a, &mu_a, NULL);
	ComputeMean(1, size, b, &mu_b, NULL);
	double sum_ab = 0, norm_a = 0, norm_b = 0, tmpa, tmpb;
	for (int j = 0; j < size; j++)
	{
		tmpa = a[j] - mu_a;
		tmpb = b[j] - mu_b;
		sum_ab += tmpa*tmpb;
		norm_a += tmpa*tmpa;
		norm_b += tmpb*tmpb;
	}
	double ncc = sum_ab / sqrt(norm_a) / sqrt(norm_b);
	return ncc;
}

bool Selfie::calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < _req_num_inliers)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;

	// calculate the geometric motion
	vector<Point2f> src_points, dst_points;
	Point2f center(0.5*_width, 0.5*_height);
	for (int j = 0; j < num_matched; j++)
	{
		const DMatch& t = m.matches[j];
		src_points.push_back(f1.keypoints[t.queryIdx].pt - center);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt - center);
	}
	m.H = findHomography(src_points, dst_points, m.inliers_mask, CV_RANSAC, _ransac_reject);	
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
	//m.confidence = m.confidence > 3. ? 0. : m.confidence;

	// refine the homography matrix
	if (m.num_inliers < _req_num_inliers || m.confidence < _req_homo_conf)
	{
		m = detail::MatchesInfo();
		return false;
	}
	//src_points.create(1, m.num_inliers, CV_32FC2);
	//dst_points.create(1, m.num_inliers, CV_32FC2);
	//int inlier_idx = 0;
	//for (int j = 0; j < num_matched; j++)
	//{
	//	if (!m.inliers_mask[j])
	//		continue;
	//	const DMatch& t = m.matches[j];
	//	src_points.at<Point2f>(inlier_idx) = f1.keypoints[t.queryIdx].pt - center;
	//	dst_points.at<Point2f>(inlier_idx) = f2.keypoints[t.trainIdx].pt - center;
	//	inlier_idx++;
	//}
	//m.H = findHomography(src_points, dst_points, CV_RANSAC, _ransac_reject); // do NOT output inlier mask
	return true;
}

bool Selfie::calcHomoFromMatches(detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, const float* c1, const float* c2)
{
	// compute other elements of MatchesInfo
	int num_matched = m.matches.size();
	if (num_matched < _req_num_inliers)
	{
		m = detail::MatchesInfo();
		return false;
	}
	m.src_img_idx = f1.img_idx;
	m.dst_img_idx = f2.img_idx;
	vector<Point2f> src_points, dst_points;
	vector<int> valid_ind(num_matched, -1);
	Point2f center(0.5*_width, 0.5*_height);
	int num_valid = 0;
	for (int j = 0; j < num_matched; j++)
	{
		const DMatch& t = m.matches[j];
		if (c1[t.queryIdx] > _thre_valid_bk || c2[t.trainIdx] > _thre_valid_bk)
			continue;
		src_points.push_back(f1.keypoints[t.queryIdx].pt - center);
		dst_points.push_back(f2.keypoints[t.trainIdx].pt - center);
		valid_ind[j] = num_valid;
		num_valid++;
	}
	if (num_valid < _req_num_inliers)
	{
		m = detail::MatchesInfo();
		return false;
	}

	// compute homography
	vector<uchar> inlier_ind;
	m.H = findHomography(Mat(src_points), Mat(dst_points), inlier_ind, CV_RANSAC, _ransac_reject);	
	if (std::abs(determinant(m.H)) < numeric_limits<double>::epsilon())
	{
		m = detail::MatchesInfo();
		return false;
	}
	// num of inliers
	m.num_inliers = 0;
	for (int j = 0; j < num_valid; j++)
		if (inlier_ind[j])
			m.num_inliers++;

	// confidence, copied from matchers.cpp
	m.confidence = m.num_inliers / (8 + 0.3 * num_valid);
	//m.confidence = m.confidence > 3. ? 0. : m.confidence;

	// refine the homography matrix
	if (m.num_inliers < _req_num_inliers || m.confidence < _req_homo_conf)
	{
		m = detail::MatchesInfo();
		return false;
	}
	// inlier mask for m
	m.inliers_mask.resize(num_matched);
	for (int j = 0; j < num_matched; j++)
	{
		if (valid_ind[j] >= 0 && inlier_ind[valid_ind[j]])
			m.inliers_mask[j] = 1;
		else
			m.inliers_mask[j] = 0;
	}

	//src_points.resize(m.num_inliers);
	//dst_points.resize(m.num_inliers);
	//int inlier_idx = 0;
	//for (int j = 0; j < num_matched; j++)
	//{
	//	if (!m.inliers_mask[j])
	//		continue;
	//	const DMatch& t = m.matches[j];
	//	src_points[inlier_idx] = f1.keypoints[t.queryIdx].pt - center;
	//	dst_points[inlier_idx] = f2.keypoints[t.trainIdx].pt - center;
	//	inlier_idx++;
	//}
	//m.H = findHomography(Mat(src_points), Mat(dst_points), CV_RANSAC, _ransac_reject); // do NOT output inlier mask
	return true;
}

void Selfie::calcDualMatches(detail::MatchesInfo& dm, const detail::MatchesInfo& m)
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

void Selfie::getImgContour(Mat& cont, const Mat& img, float thre_edge, float thre_length, int ksize)
{

	Mat src;
	if (thre_edge<0) 
	{
		if (img.channels()==1)
			img.copyTo(src);
		else
			cvtColor(img, src, CV_BGR2GRAY);
	} else /// Detect edges using canny
		getImgEdge(src, img, thre_edge, 0);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	/// Find contours
	findContours(src, contours, hierarchy, RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	// iterate through all the top-level contours,
	cont = Mat::zeros(img.size(), CV_8U);
	for( int i = 0; i>=0 && i<contours.size(); i = hierarchy[i][0]) 
	{
		if (arcLength(contours[i], true)>=thre_length)
		{
			drawContours(cont, contours, i, Scalar(255,255,255), ksize, 8, hierarchy, 0);
		}
	}
	//imwrite("Contours.png", cont);
}
void Selfie::getImgEdge(Mat& conf, const Mat& img, float thre, int ksize)
{	
	Mat src, gray, edge;
	blur(img, src, Size(3,3));
	if (img.channels()!=1)
		cvtColor(src, gray, CV_BGR2GRAY);
	else
		src.copyTo(gray);
	Canny(gray, edge, thre, thre*2);
	if (ksize>0) 
	{
		Mat elem = getStructuringElement(MORPH_RECT, Size(ksize,ksize));
		morphologyEx(edge, conf, MORPH_DILATE, elem);
	} else
		edge.copyTo(conf);
	//vector<Mat> edges(img.channels());
	//vector<Mat> planes;
	//split(img, planes);
	//for (int i = 0; i < planes.size(); i++)
	//{
	//	blur(planes[i], planes[i], Size(3,3));
	//	Canny(planes[i], edges[i], thre, thre*2);
	//	if (ksize>0) {
	//		Mat elem = getStructuringElement(MORPH_RECT, Size(ksize,ksize));
	//		morphologyEx(edges[i], edges[i], MORPH_DILATE, elem);
	//	}
	//}
	//edges[0].copyTo(conf);
	//for (int i = 1; i < planes.size(); i++)
	//	conf &= edges[i];
}

void Selfie::getGeneralImg(Mat& gimg, const Mat& img)
{	
	int channels = img.channels();
	Mat src, gray;
	GaussianBlur(img, src, Size(3,3), 0, 0, BORDER_DEFAULT);
	if (img.channels()!=1)
		cvtColor(src, gray, CV_BGR2GRAY);
	else
		src.copyTo(gray);

	vector<Mat> bgr;
	split(img, bgr);

	Mat gradx, grady, mag2, mag_f, mag_c;
	Sobel(gray, gradx, CV_32F, 1, 0, 3);
	Sobel(gray, grady, CV_32F, 0, 1, 3);
	mag2 = gradx.mul(gradx)+grady.mul(grady);
	sqrt(mag2, mag_f);
	mag_f.convertTo(mag_c, CV_8U);
	bgr.push_back(mag_c);
	merge(bgr, gimg);
}

void Selfie::visualizeFlow(Mat& img, DImage& flow)
{
	Mat uv = flow.Image2Mat();
	vector<Mat> motion(uv.channels());
	split(uv, motion);
	Mat mag = motion[0].mul(motion[0])+motion[1].mul(motion[1]);
	sqrt(mag, mag);
	Mat ang = motion[1].mul(1/motion[0]);
	double maxm;
	minMaxLoc(mag, 0, &maxm);

	Mat hsv(_height, _width, CV_8UC3, Scalar(255,255,255));
	float *ux_data = (float *)motion[0].data;
	float *mag_data = (float *)mag.data;
	float *ang_data = (float *)ang.data;
	uchar *hsv_data = hsv.data;
	int num_pixels = _height*_width;
	for (int k=0; k < num_pixels; k++)
	{
		float tmp = atan(ang_data[k])*180/PI+90;
		if (ux_data[k]<0)
			tmp += 180;
		hsv_data[k*3+0] = min(max(cvRound(tmp*0.5),0),180);
		hsv_data[k*3+1] = min(max(cvRound(mag_data[k]/maxm*255),0),255);
	}
	cvtColor(hsv, img, CV_HSV2BGR);
}

void Selfie::findSIFTFeaturesMatches()
{
	printf("detect features\n");
	__features.clear();
	__features.resize(_num_frames);
	
	int gridx = 4, gridy = 4; // divide the image into a 4*4 grid
	int maxNumInGrid = MaxFeaNum/(gridx*gridy);
	int reserveFeaNum = MaxFeaNum*10;

	//SIFT finder(reserveFeaNum, 3, 0.01);
	detail::SurfFeaturesFinder finder(50);

	for (int i = 0; i < _num_frames; i++)
	{
		detail::ImageFeatures& F = __features[i];
		Mat gray;
		cvtColor(__imgs[i], gray, CV_BGR2GRAY);

		//vector<KeyPoint> keypoints;
		//Mat descriptors;
		//finder(gray, Mat(), keypoints, descriptors);
		detail::ImageFeatures tmp;
		finder(gray, tmp);
		vector<KeyPoint>& keypoints = tmp.keypoints;
		Mat& descriptors = tmp.descriptors;

		F.keypoints.clear();
		F.descriptors.create(descriptors.rows, descriptors.cols, descriptors.type());
		MatrixXi cnt_grid(gridy, gridx);
		cnt_grid.setZero();
		int num_detected = keypoints.size();
		for (int j = 0, c = 0; j < num_detected; j++) 
		{
			const KeyPoint& p = keypoints[j];
			int m = p.pt.y/_height*gridy, n = p.pt.x/_width*gridx;
			if (cnt_grid(m, n) < maxNumInGrid) 
			{
				F.keypoints.push_back(p);
				descriptors.row(j).copyTo(F.descriptors.row(c));
				cnt_grid(m, n)++;
				c++;
			}
		}
		F.descriptors.resize(F.keypoints.size());
		F.img_idx = i;
		F.img_size = __imgs[i].size();
	}
	//detail::SurfFeaturesFinder finder(400, 4, 2, 4, 2);
	//for (int i = 0; i < _num_frames; i++)
	//{
	//	Mat gray;
	//	cvtColor(__imgs[i], gray, CV_BGR2GRAY);
	//	finder(gray, __features[i]);
	//	__features[i].img_idx = i;
	//}
	//SIFT sift(1000);
	//for (int i = 0; i < _num_frames; i++)
	//{
	//	Mat gray;
	//	cvtColor(__imgs[i], gray, CV_BGR2GRAY);
	//	sift(gray, Mat(), __features[i].keypoints, __features[i].descriptors);
	//	__features[i].img_idx = i;
	//	__features[i].img_size = gray.size();
	//}

	printf("match features\n");
	__matches.clear();
	__matches.resize(_num_frames*_num_frames);
	detail::BestOf2NearestMatcher matcher(FALSE, 0.4f);
	matcher(__features, __matches);
	for (int i = 0; i < __matches.size(); i++)
		if (__matches[i].num_inliers < _req_num_inliers || __matches[i].confidence < _req_homo_conf)
			__matches[i] = detail::MatchesInfo();

	//remove outlier matches
	removeOutlierMatches();

	string filename = __imFiles[0];
	filename.erase(filename.find_last_of('_'));
	writeFeaturesMatchesToIF(filename+".if");
}

void Selfie::clearKLTFeature(KLT_Feature& f)
{
	f->x   = -1.0;
	f->y   = -1.0;
	f->val = -1;
	f->aff_x = -1.0;
	f->aff_y = -1.0;
	f->aff_Axx = 1.0;
	f->aff_Ayx = 0.0;
	f->aff_Axy = 0.0;
	f->aff_Ayy = 1.0;
	// free image and gradient for lost feature
	if (f->aff_img)						
		_KLTFreeFloatImage(f->aff_img);
	if (f->aff_img_gradx)
		_KLTFreeFloatImage(f->aff_img_gradx);
	if (f->aff_img_grady)	
		_KLTFreeFloatImage(f->aff_img_grady);
	f->aff_img = NULL;
	f->aff_img_gradx = NULL;
	f->aff_img_grady = NULL;
}

void Selfie::findKLTFeaturesMatches()
{
	printf("detect features\n");
	__features.clear();
	__features.resize(_num_frames);

	// setup the parameters
	int gridx = 4, gridy = 4;
	int maxNumInGrid = MaxFeaNum/(gridx*gridy);
	int reserveFeaNum = 10000;
	KLT_TrackingContext context = KLTCreateTrackingContext();
	KLT_FeatureList framelist = KLTCreateFeatureList(reserveFeaNum);
	KLT_FeatureTable table = KLTCreateFeatureTable(_num_frames, reserveFeaNum);
	context->sequentialMode = TRUE;
	context->writeInternalImages = FALSE;
	context->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
	context->min_eigenvalue = 50;
	context->lighting_insensitive = FALSE;
	context->window_height = 9;
	context->window_width = 9;
	context->mindist = 20;
	context->min_determinant = 0.01f;
	context->min_displacement = 0.01f;
	KLTChangeTCPyramid(context, 15);
	KLTUpdateTCBorder(context);

	// find good features to track
	printf("  00 of %02d\r", _num_frames);
	vector<Mat> grays(_num_frames);
	for (int i = 0; i < _num_frames; i++)
		cvtColor(__imgs[i], grays[i], CV_BGR2GRAY);
	KLTSelectGoodFeatures(context, grays[0].data, _width, _height, framelist);
	MatrixXi cnt_grid(gridy, gridx);
	cnt_grid.setZero();
	for (int j = 0; j < reserveFeaNum; j++)  
	{
		const KLT_Feature& f = framelist->feature[j];
		if (f->x<0 || f->y<0 || f->val<0)
			continue;
		int m = f->y/_height*gridy, n = f->x/_width*gridx;
		if (cnt_grid(m, n) < maxNumInGrid)
			cnt_grid(m,n)++;
		else
			clearKLTFeature(framelist->feature[j]);
	}
	KLTStoreFeatureList(framelist, table, 0);

	// track the features
	for (int i = 1; i < _num_frames; i++)
	{
		printf("  %02d of %02d\r", i, _num_frames);
		KLTTrackFeatures(context, grays[i-1].data, grays[i].data, _width, _height, framelist);
		KLTReplaceLostFeatures(context, grays[i].data, _width, _height, framelist);
		cnt_grid.setZero();
		for (int j = 0, c = 0; j < reserveFeaNum; j++) 
		{
			const KLT_Feature& f = framelist->feature[j];
			if (f->x<0 || f->y<0 || f->val<0)
				continue;
			int m = f->y/_height*gridy, n = f->x/_width*gridx;
			if (cnt_grid(m, n) < maxNumInGrid)
				cnt_grid(m,n)++;
			else
				clearKLTFeature(framelist->feature[j]);
		}	
		KLTStoreFeatureList(framelist, table, i);
	}

	printf("match features\n");
	__matches.clear();
	__matches.resize(_num_frames*_num_frames);

	// convert KLTFeatureList to ImageFeatures
	MatrixXf ind(table->nFrames, table->nFeatures);
	ind.setConstant(-1);
	for (int i = 0; i < _num_frames; i ++)
	{
		detail::ImageFeatures& F = __features[i];
		detail::MatchesInfo& M = (i>0) ? __matches[(i-1)*_num_frames+i] : __matches[0];
		F.img_idx = i;
		F.img_size = Size(_width,_height);
		KLT_FeatureList tmplist = KLTCreateFeatureList(table->nFeatures);
		KLTExtractFeatureList(tmplist, table, i);
		for (int j = 0, cnt = 0; j < tmplist->nFeatures; j++)	
		{
			const KLT_Feature& f = tmplist->feature[j];
			if (f->x < 0 || f->y<0 || f->val<0)
				continue;
			F.keypoints.push_back(KeyPoint(f->x, f->y, 0));
			ind(i, j) = cnt;
			cnt++;
			if (i && f->val==0)
				M.matches.push_back(DMatch(ind(i-1,j), ind(i,j), -1, 0));
		}
		KLTFreeFeatureList(tmplist);
		if (i==0)
			continue;
		calcHomoFromMatches(M, __features[i-1], F);
		calcDualMatches(__matches[i*_num_frames+i-1], M);
	}

	// matches between non-adjacent frames
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+2; j <_num_frames; j++)
		{
			detail::MatchesInfo& M = __matches[i*_num_frames+j];
			detail::MatchesInfo& dM = __matches[j*_num_frames+i];
			const detail::MatchesInfo& pM = __matches[(j-1)*_num_frames+j];
			const detail::MatchesInfo& ppM = __matches[i*_num_frames+j-1];

			vector<int> pInd(__features[j-1].keypoints.size(), -1);
			for (int k = 0; k < pM.matches.size(); k++)
			{
				const DMatch& t = pM.matches[k];
				pInd[t.queryIdx] = t.trainIdx;
			}
			for (int k = 0; k < ppM.matches.size(); k++)
			{
				const DMatch& t = ppM.matches[k];
				if (pInd[t.trainIdx]>=0)
					M.matches.push_back(DMatch(t.queryIdx, pInd[t.trainIdx], -1, 0));
			}
			calcHomoFromMatches(M, __features[i], __features[j]);
			calcDualMatches(dM, M);
		}
	}

	// remove outlier matches
	//removeOutlierMatches();

	string filename = __imFiles[0];
	filename.erase(filename.find_last_of('_'));
	writeFeaturesMatchesToIF(filename+".if");
	
	KLTFreeFeatureList(framelist);
	KLTFreeFeatureTable(table);
	KLTFreeTrackingContext(context);
}

void Selfie::removeOutlierMatches()
{
	Point2f center(0.5*_width, 0.5*_height);
	for (int i = 0; i < _num_frames-1; i++)
	{
		const vector<KeyPoint>& points1 = __features[i].keypoints;
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched<=0)
				continue;
			detail::MatchesInfo&  M = __matches[i*_num_frames+j];
			detail::MatchesInfo& dM = __matches[j*_num_frames+i];
			const vector<KeyPoint>& points2 = __features[j].keypoints;
			const double* h = (double *)M.H.data;
			Mat w_error(1, num_matched, CV_32F);
			float *werror = (float *)w_error.data;
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M.matches[k];
				Point2f p1 = points1[t.queryIdx].pt - center;
				Point2f p2 = points2[t.trainIdx].pt - center;
				Point2f wp1 = warpPoint(p1, h);
				werror[k] = sqrt((wp1.x-p2.x)*(wp1.x-p2.x)+(wp1.y-p2.y)*(wp1.y-p2.y));
			}
			Scalar mu, sigma;
			meanStdDev(w_error, mu, sigma);
			vector<DMatch> valid_matches;
			for (int k = 0; k < num_matched; k++)
				if (abs(werror[k]-mu[0])<=2*sigma[0])
					valid_matches.push_back(M.matches[k]);
			
			M.matches = valid_matches;
			num_matched = M.matches.size();
			Mat src_points(1, num_matched, CV_32FC2);
			Mat dst_points(1, num_matched, CV_32FC2);
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = valid_matches[k];
				src_points.at<Point2f>(k) = points1[t.queryIdx].pt - center;
				dst_points.at<Point2f>(k) = points2[t.trainIdx].pt - center;
			}
			M.H = findHomography(src_points, dst_points, M.inliers_mask, CV_RANSAC, _ransac_reject);
			if (std::abs(determinant(M.H)) < numeric_limits<double>::epsilon())
			{
				M = detail::MatchesInfo();			
				dM = detail::MatchesInfo();
				continue;
			}
			M.num_inliers = 0;
			for (int j = 0; j < num_matched; j++)
			if (M.inliers_mask[j])
				M.num_inliers++;
			
			// confidence, copied from matchers.cpp
			M.confidence = M.num_inliers / (8 + 0.3 * num_matched);
			M.confidence = M.confidence > 3. ? 0. : M.confidence;
			
			if (M.num_inliers < _req_num_inliers || M.confidence < _req_homo_conf)
			{
				M = detail::MatchesInfo();			
				dM = detail::MatchesInfo();
				continue;
			}
			calcDualMatches(dM, M);
		}
	}
}

void Selfie::removeShortTrajectories(int length)
{
	for (int i = 0; i < _num_frames; i++)
	{
		vector<KeyPoint> &points = __features[i].keypoints, vpoints;
		int num_points = points.size();
		VectorXi cnt(points.size());
		cnt.setZero();
		for (int j = 0; j < _num_frames; j++)
		{
			const vector<DMatch> &M = __matches[i*_num_frames+j].matches;
			int num_matched = M.size();
			if (num_matched<=0)
				continue;
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M[k];
				cnt(t.queryIdx)++;
			}
		}
		VectorXi idx(num_points);
		idx.setConstant(-1);
		for (int k = 0; k < num_points; k++) 	
		{
			if (cnt(k)>=length) {
				idx(k) = vpoints.size();
				vpoints.push_back(points[k]);
			}
		}
		points = vpoints;
		for (int j = 0; j < _num_frames; j++)
		{
			vector<DMatch> &matches = __matches[i*_num_frames+j].matches, vmatches;
			if (matches.size()<=0)
				continue;
			for (int k = 0; k < matches.size(); k++)
			{
				DMatch t = matches[k];
				if (idx(t.queryIdx)>=0) 
				{
					t.queryIdx = idx(t.queryIdx);
					vmatches.push_back(t);
				}
			}
			matches = vmatches;
		}
	}
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			calcHomoFromMatches(__matches[i*_num_frames+j], __features[i], __features[j]);
			calcDualMatches(__matches[j*_num_frames+i], __matches[i*_num_frames+j]);
		}
	}
}

void Selfie::indexFeaturesMatches()
{	
	// parameters
	int psize = _psize;
	int chs = _chs;

	// index for features 
	__starts.resize(_num_frames);
	_num_variables = 0;
	for (int i = 0; i < _num_frames; i++)
	{
		__starts[i] = _num_variables;
		_num_variables += __features[i].keypoints.size();
	}

	// color value for each feature point
	int wsize = 2*psize+1, cdim = chs*wsize*wsize;
	__color.resize(_num_variables, cdim);
	vector<Mat> gimgs(_num_frames);
	for (int i = 0; i < _num_frames; i++)
	{
		if (chs == _channels)
			gimgs[i] = __imgs[i].clone();
		else
			getGeneralImg(gimgs[i], __imgs[i]);
	}
	for (int i = 0; i < _num_frames; i++)
	{
		uchar *im = (uchar *)gimgs[i].data;
		const vector<KeyPoint>& points = __features[i].keypoints;
		int num_points = points.size();
		int offset = __starts[i];
		for (int j = 0; j < num_points; j++)
		{
			const Point2f& pt = points[j].pt;
			for (int u = -psize; u <= psize; u++)
				for (int v = -psize; v <= psize; v++)		
					bilinearInterpolate(&__color(offset+j, ((u+psize)*wsize+v+psize)*chs), Point2f(pt.x+v,pt.y+u), im, _width, _height, chs);
		}
	}

	// features -> trajectory
	__trajectories.clear();
	__trajIndex.resize(_num_frames);
	_num_trajs = 0;
	for (int i = 0; i < _num_frames; i++)
	{
		vector<int>& idx = __trajIndex[i];
		const vector<KeyPoint>& points = __features[i].keypoints;
		int num_points = points.size();
		idx.resize(num_points);

		// initialization for the first frame
		if (i==0)
		{
			for (int k = 0; k < num_points; k++)
			{		
				__trajectories.push_back(tracking_trajectory(i, points[k].pt, k));
				idx[k] = _num_trajs;
				_num_trajs++;
			}
			continue;
		}

		// matches from previous to current frame
		const vector<int>& pIdx = __trajIndex[i-1];
		const detail::MatchesInfo& M = __matches[(i-1)*_num_frames+i];
		int num_matched = M.matches.size();
		if (num_matched<=0)
			continue;
		// matches
		vector<int> flag(num_points, 0);
		for (int k = 0; k < num_matched; k++)
		{
			const DMatch& dm = M.matches[k];
			tracking_trajectory& tt = __trajectories[pIdx[dm.queryIdx]];			
			tt.xy.push_back(points[dm.trainIdx].pt);
			tt.feature_idx.push_back(dm.trainIdx);
			idx[dm.trainIdx] = pIdx[dm.queryIdx];
			flag[dm.trainIdx] = 1;
		}
		// new features
		for (int k = 0; k < num_points; k++) 
		{
			if (flag[k]<=0)
			{
				__trajectories.push_back(tracking_trajectory(i, points[k].pt, k));
				idx[k] = _num_trajs;
				_num_trajs++;
			}
		}
	}
	// update the other information in the trajectory
	for (int i = 0; i < _num_trajs; i++)
	{
		tracking_trajectory& tt = __trajectories[i];
		tt.last_frame = tt.first_frame + tt.xy.size();
		tt.length = tt.xy.size();
		tt.change.clear();
		for (int k = 0; k < tt.length-1; k++)
			tt.change.push_back(tt.xy[k+1]-tt.xy[k]);
	}
	
	// show the statistics
	int num_pairs = 0, num_matches = 0;
	for (int i = 0; i < _num_frames; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			num_pairs++;
			num_matches += __matches[i*_num_frames+j].matches.size();
		}
	}
	printf("    %d features, %d trajectories, %d pairs, %d matches\n", _num_variables, _num_trajs, num_pairs, num_matches);
}

void Selfie::getOpticalFlow(vector<DImage>& flows, string flowDir)
{
	printf("calc optical flow\n");
	double alpha = 0.012;
	double ratio = 0.75;
	int minWidth = 20;
	int nOuterFPIters = 7;
	int nInnerFPIters = 1;
	int nCGIters = 30;
	int num_frames = __imNames.size();

	char filename[512];
	_mkdir(flowDir.c_str());
	flows.resize((num_frames-1)*2);

	DImage im0, im1, warpI2, ux, vy;
	for (int i = 0; i < num_frames-1; i++)
	{
		printf("  %d of %d\r", i, num_frames);				
		im0.Mat2Image(__imgs[i]);
		im1.Mat2Image(__imgs[i+1]);	

		// forward flow
		sprintf(filename, "%s/%s-%s.txt", flowDir.c_str(), __imNames[i].c_str(), __imNames[i+1].c_str());
		if (_access(filename, 0) != -1)
			readOpticalFlow(flows[i*2], filename);
		else
		{
			OpticalFlow::Coarse2FineFlow(ux, vy, warpI2, im0, im1, alpha, ratio, minWidth, nOuterFPIters, nInnerFPIters, nCGIters);
			OpticalFlow::AssembleFlow(ux, vy, flows[i*2]);
			writeOpticalFlow(filename, flows[i*2]);
			sprintf(filename, "%s/%s-%s_warp.jpg", flowDir.c_str(), __imNames[i].c_str(), __imNames[i+1].c_str());
			warpI2.imwrite(filename);
			sprintf(filename, "%s/%s-%s_visual.jpg", flowDir.c_str(), __imNames[i].c_str(), __imNames[i+1].c_str());
			Mat visual;
			visualizeFlow(visual, flows[i*2]);
			imwrite(filename, visual);
		}
		// backward flow
		sprintf(filename, "%s/%s-%s.txt", flowDir.c_str(), __imNames[i+1].c_str(), __imNames[i].c_str());
		if (_access(filename, 0) != -1)
			readOpticalFlow(flows[i*2+1], filename);
		else
		{
			OpticalFlow::Coarse2FineFlow(ux, vy, warpI2, im1, im0, alpha, ratio, minWidth, nOuterFPIters, nInnerFPIters, nCGIters);
			OpticalFlow::AssembleFlow(ux, vy, flows[i*2+1]);
			writeOpticalFlow(filename, flows[i*2+1]);
			sprintf(filename, "%s/%s-%s_warp.jpg", flowDir.c_str(), __imNames[i+1].c_str(), __imNames[i].c_str());
			warpI2.imwrite(filename);		
			sprintf(filename, "%s/%s-%s_visual.jpg", flowDir.c_str(), __imNames[i+1].c_str(), __imNames[i].c_str());
			Mat visual;
			visualizeFlow(visual, flows[i*2+1]);
			imwrite(filename, visual);
		}
	}
}

void Selfie::findFlowFeaturesMatches(vector<DImage>& flows)
{
	printf("find features and matches\n");
	// parameters
	int grid = cvRound(min(_width,_height)*0.05); //18*18
	int psize = _psize; // cvRound(min(_width,_height)*0.005) 5*5;
	int chs = _chs;
	float sigma_times2 = -log(0.1f);
	float thre_img_edge = 48;
	float thre_flow_edge = 24;
	int ksize = 2*psize+1;
	int numgx = cvCeil(double(_width-grid/2)/grid), numgy = cvCeil(double(_height-grid/2)/grid);
	int wsize = 2*psize+1, pnum = wsize*wsize, cdim = chs*pnum;
	int num_reserved = 10*numgx*numgy*_num_frames, num_total = 0;
	char filename[512];
	
	__features.clear();
	__features.resize(_num_frames);
	__matches.clear();
	__matches.resize(_num_frames*_num_frames);
	__color.resize(num_reserved, cdim);
	__color.setZero();
	__starts.resize(_num_frames);
	RNG rng(getTickCount()); // = theRNG();

	// compute the general image and edges
	vector<Mat> gimgs(_num_frames), cimgs(_num_frames);
	for (int i = 0; i < _num_frames; i++)
	{
		if (chs == _channels)
			gimgs[i] = __imgs[i].clone();
		else
			getGeneralImg(gimgs[i], __imgs[i]);
		getImgEdge(cimgs[i], __imgs[i], thre_img_edge, ksize);
		//sprintf(filename, "img_conf_%02d.png", i);
		//imwrite(filename, cimgs[i]);
	}
	// compute the flow edge
	vector<Mat> cflows(flows.size());
	for (int i = 0; i < flows.size(); i++)
	{
		Mat img;
		visualizeFlow(img, flows[i]);
		getImgEdge(cflows[i], img, thre_flow_edge, ksize);
		//sprintf(filename, "flow_conf_%02d.png", i);
		//imwrite(filename, cflows[i]);
	}

	// equally sample the pixels in the first image
	__features[0].img_idx = 0;
	__features[0].img_size = Size(_width, _height);
	__starts[0] = 0;
	uchar *im = gimgs[0].data;
	uchar *cim = cimgs[0].data;
	uchar *cflow = cflows[0].data;
	for (int m = 0; m < numgy; m++) {
		for (int n = 0; n < numgx; n++) {
			int xx = grid/2+n*grid, yy = grid/2+m*grid, r = 0;
			if (cim[yy*_width+xx]>0) {
				for (r = 0; r < grid*grid; r++) {
					xx = rng(min(grid,_width-n*grid))+n*grid;
					yy = rng(min(grid,_height-m*grid))+m*grid;
					if (cim[yy*_width+xx]<=0 && cflow[yy*_width+xx]<=0)
						break;
				}
			}
			__features[0].keypoints.push_back(KeyPoint(xx, yy, 0));
			for (int p = -psize; p <= psize; p++) {
				for (int q = -psize; q <= psize; q++)
					bilinearInterpolate(&__color(num_total, ((p+psize)*wsize+q+psize)*chs), Point2f(xx+q, yy+p), im, _width, _height, chs);
			}
			num_total++;
		}
	}

	// get the features, direct matches, color values
	for (int i = 1; i < _num_frames; i++)
	{
		__starts[i] = num_total;
		const double *flow0 = flows[(i-1)*2].data();
		const double *flow1 = flows[(i-1)*2+1].data();
		uchar *im0 = gimgs[i-1].data;
		uchar *im1 = gimgs[i].data;
		cim = cimgs[i].data;
		uchar *cflow0 = cflows[(i-1)*2].data;
		uchar *cflow1 = cflows[(i-1)*2+1].data;
		detail::ImageFeatures& F1 = __features[i];
		const detail::ImageFeatures& F0 = __features[i-1];
		detail::MatchesInfo& M = __matches[(i-1)*_num_frames+i];		
		const vector<KeyPoint>& points0 = F0.keypoints;
		vector<KeyPoint>& points1 = F1.keypoints;
		int offset0 = __starts[i-1], offset1 = __starts[i];

		F1.img_idx = i;
		F1.img_size = Size(_width, _height);
		int num_points = points0.size();
		MatrixXi flag(numgy, numgx);
		flag.setZero();
		float *uv = new float[2], *c0, *c1 = new float[cdim], *c2 = new float[cdim];
		Point2f p0, p1;
		float cf, ncc;

		// track the feature points via optical flow
		for (int j = 0; j < num_points; j++)
		{
			p0 = points0[j].pt;
			c0 = &__color(offset0+j, 0);
			
			// check foward flow
			bilinearInterpolate(uv, p0, flow0, _width, _height, 2);
			p1.x = p0.x+uv[0];
			p1.y = p0.y+uv[1];
			if (p1.x<0 || p1.x>_width-1 || p1.y<0 || p1.y>_height-1)
				continue;
			bilinearInterpolate(&cf, p1, cflow1, _width, _height, 1);
			if (cf>EPSILON)
				continue;

			// check foward appearance
			for (int p = -psize; p <= psize; p++) {
				for (int q = -psize; q <= psize; q++)
					bilinearInterpolate(c1+((p+psize)*wsize+q+psize)*chs, Point2f(p1.x+q, p1.y+p), im1, _width, _height, chs);
			}
			float dd = 0;
			for (int c = 0; c < cdim; c++)
				dd += (c0[c]-c1[c])*(c0[c]-c1[c])/(2*pnum*_sigma_color*_sigma_color);
			if (dd > sigma_times2)
				continue;
			ncc = calcNCC(c0, c1, cdim);
			if (ncc < _req_ncc)
				continue;

			// check background flow
			bilinearInterpolate(uv, p1, flow1, _width, _height, 2);
			if ((p0.x-p1.x-uv[0])*(p0.x-p1.x-uv[0])+(p0.y-p1.y-uv[1])*(p0.y-p1.y-uv[1])>_ransac_reject*_ransac_reject)
				continue;
			p0.x = p1.x+uv[0];
			p0.y = p1.y+uv[1];
			if (p0.x<0 || p0.x>_width-1 || p0.y<0 || p0.y>_height-1)
				continue;
			bilinearInterpolate(&cf, p0, cflow0, _width, _height, 1);
			if (cf>EPSILON)
				continue;

			// check background appearance
			for (int p = -psize; p <= psize; p++) {
				for (int q = -psize; q <= psize; q++)
					bilinearInterpolate(c2+((p+psize)*wsize+q+psize)*chs, Point2f(p0.x+q, p0.y+p), im0, _width, _height, chs);
			}
			dd = 0;
			for (int c = 0; c < cdim; c++)
				dd += (c0[c]-c2[c])*(c0[c]-c2[c])/(2*pnum*_sigma_color*_sigma_color);
			if (dd > sigma_times2)
				continue;
			ncc = calcNCC(c0, c2, cdim);
			if (ncc < _req_ncc)
				continue;

			// push back the match
			points1.push_back(KeyPoint(p1.x, p1.y, 0));
			for (int c = 0; c < cdim; c++)
				__color(num_total, c) = c1[c];
			flag(min(max(int(p1.y/grid),0),numgy-1), min(max(int(p1.x/grid),0),numgx-1))++;
			M.matches.push_back(DMatch(j, num_total-offset1, -1, 0));	
			num_total++;
		}

		// replace lost features
		for (int m = 0; m < numgy; m++) {
			for (int n = 0; n < numgx; n++) {
				if (flag(m,n))	
					continue;
				int xx = grid/2+n*grid, yy = grid/2+m*grid, r = 0;
				if (cim[yy*_width+xx]>0 || cflow1[yy*_width+xx]>0) {
					for (r = 0; r < grid*grid; r++) {
						xx = rng(min(grid,_width-n*grid))+n*grid;
						yy = rng(min(grid,_height-m*grid))+m*grid;
						if (cim[yy*_width+xx]<=0 && cflow1[yy*_width+xx]<=0)
							break;
					}
				}
				points1.push_back(KeyPoint(xx,yy,0));
				for (int p = -psize; p <= psize; p++) {
					for (int q = -psize; q <= psize; q++)
						bilinearInterpolate(&__color(num_total,((p+psize)*wsize+q+psize)*chs), Point2f(xx+q, yy+p), im1, _width, _height, chs);
				}
				num_total++;
			}
		}
		delete[] uv;
		delete[] c1;
		delete[] c2;

		calcHomoFromMatches(M, F0, F1);
		calcDualMatches(__matches[i*_num_frames+i-1], M);
	}
	__color.resize(num_total, cdim);
	_num_variables = num_total;

	// matches between non-adjacent frames
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+2; j <_num_frames; j++)
		{
			detail::MatchesInfo& M = __matches[i*_num_frames+j];
			detail::MatchesInfo& dM = __matches[j*_num_frames+i];
			const detail::MatchesInfo& pM = __matches[(j-1)*_num_frames+j];
			const detail::MatchesInfo& ppM = __matches[i*_num_frames+j-1];

			vector<int> pInd(__features[j-1].keypoints.size(), -1);
			for (int k = 0; k < pM.matches.size(); k++)
			{
				const DMatch& t = pM.matches[k];
				pInd[t.queryIdx] = t.trainIdx;
			}
			for (int k = 0; k < ppM.matches.size(); k++)
			{
				const DMatch& t = ppM.matches[k];
				if (pInd[t.trainIdx]>=0)
					M.matches.push_back(DMatch(t.queryIdx, pInd[t.trainIdx], -1, 0));
			}
			calcHomoFromMatches(M, __features[i], __features[j]);
			calcDualMatches(dM, M);
		}
	}

	string feafile = __imFiles[0];
	feafile.erase(feafile.find_last_of('_'));
	writeFeaturesMatchesToIF(feafile+".if");
}

// face detection for initializing the labels is done before clustering
void Selfie::clustering(string outDir)
{	
	printf("clustering feature motion\n");
	int clusterCount = 2;
	_mkdir(outDir.c_str());
	vector<string> paths, names;
	if (ScanDirectory(outDir, "jpg", paths, names)){
		for (int i = 0; i < paths.size(); i++)
			remove(paths[i].c_str());
	}

	// create the data
	Scalar colorTab[] = {Scalar(0, 0, 255), Scalar(0,255,0), Scalar(255,100,100), Scalar(255,0,255), Scalar(0,255,255)};
	__motionMix.resize(_num_frames*_num_frames);
	__categories.resize(_num_frames);
	for (int i = 0; i < _num_frames; i++)
	{
		int num_points = __features[i].keypoints.size();
		__categories[i].create(num_points, num_points, CV_32S);
		__categories[i].setTo(Scalar(2*_num_frames));
	}

	Point2f center(0.5*_width, 0.5*_height);
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched<=0)
				continue;

			GMMInfo& G = __motionMix[i*_num_frames+j];
			const detail::MatchesInfo& M = __matches[i*_num_frames+j];
			const vector<KeyPoint>& points1 = __features[i].keypoints;
			const vector<KeyPoint>& points2 = __features[j].keypoints;

			int offset1 = __starts[i], offset2 = __starts[j];
			int cnt_positive = 0, cnt_negative = 0;
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M.matches[k];
				if (min(__labels[offset1+t.queryIdx], __labels[offset2+t.trainIdx])<= _thre_valid_bk)
					cnt_negative++;
				else if (max(__labels[offset1+t.queryIdx], __labels[offset2+t.trainIdx])>=1-_thre_valid_bk)
					cnt_positive++;
			}
			if (cnt_negative<_req_num_inliers || cnt_positive<_req_num_inliers)
				continue;

			//const double* h = (double *)M.H.data;
			Mat samples(num_matched, 2, CV_64F);
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M.matches[k];
				Point2f p1 = points1[t.queryIdx].pt - center;
				Point2f p2 = points2[t.trainIdx].pt - center;
				samples.at<double>(k, 0) = p2.x-p1.x;
				samples.at<double>(k, 1) = p2.y-p1.y;
				//Point2f wp1 = warpPoint(p1, h);
				//samples.at<float>(k, 2) = sqrt((wp1.x-p2.x)*(wp1.x-p2.x)+(wp1.y-p2.y)*(wp1.y-p2.y));
			}
	
			// clustering by k-means
			//Mat labels, clusters(clusterCount, 1, samples.type());
			//kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 0.01), 10, KMEANS_PP_CENTERS, clusters);

			// classification by GMMs
			Mat likely, labels;
			G.model.set("nclusters", clusterCount);
			G.model.set("maxIters", MaxIters);
			G.model.set("epsilon", EPSILON);
			G.model.train(samples, likely, G.labels, G.probs);
			labels = G.labels.clone();

			//Mat img = __imgs[i].clone();
			//for (int k = 0; k < num_matched; k++ )
			//{
			//	int clusterIdx = labels.at<int>(k);
			//	const DMatch& t = M.matches[k];
			//	Point ipt = points1[t.queryIdx].pt;
			//	circle( img, ipt, 3, colorTab[clusterIdx], CV_FILLED, CV_AA );
			//}
			//char filename[512];
			//sprintf(filename, "%s/%s-%s.jpg", outDir.c_str(), __imNames[i].c_str(), __imNames[j].c_str());
			//imwrite(filename, img);

			for (int m = 0; m < num_matched; m++)
			{
				int id1 = labels.at<int>(m);
				const DMatch& t1 = M.matches[m];
				for (int n = m+1; n < num_matched; n++)
				{
					int id2 = labels.at<int>(n);
					const DMatch& t2 = M.matches[n];
					if (__categories[i].at<int>(t1.queryIdx, t2.queryIdx) > _num_frames) {
						__categories[i].at<int>(t1.queryIdx, t2.queryIdx) = 0;
						__categories[i].at<int>(t2.queryIdx, t1.queryIdx) = 0;
					}
					if (__categories[j].at<int>(t1.trainIdx, t2.trainIdx) > _num_frames) {
						__categories[j].at<int>(t1.trainIdx, t2.trainIdx) = 0;
						__categories[j].at<int>(t2.trainIdx, t1.trainIdx) = 0;
					}
					if (id1 == id2) {
						__categories[i].at<int>(t1.queryIdx, t2.queryIdx)++;
						__categories[i].at<int>(t2.queryIdx, t1.queryIdx)++;
						__categories[j].at<int>(t1.trainIdx, t2.trainIdx)++;
						__categories[j].at<int>(t2.trainIdx, t1.trainIdx)++;
					} else {
						__categories[i].at<int>(t1.queryIdx, t2.queryIdx)--;
						__categories[i].at<int>(t2.queryIdx, t1.queryIdx)--;
						__categories[j].at<int>(t1.trainIdx, t2.trainIdx)--;
						__categories[j].at<int>(t2.trainIdx, t1.trainIdx)--;
					}
				}
			}
		}
	}
}

void Selfie::affinities()
{
	//float sigma_times1 = -log(0.3f);
	float sigma_times2 = -log(0.1f);
	float match_times = log(1.f);
	printf("calc affinities: %d nodes, ", _num_variables);
	const float* color_data = __color.data();
	int cdim = __color.cols(), pnum = (2*_psize+1)*(2*_psize+1);

	// affinities between pixels in the same image
	vector<Triplet<float>> triplet;
	//triplet.reserve(_num_variables*_num_variables*0.01);
	for (int i = 0; i < _num_frames; i++)
	{
		const vector<KeyPoint>& points = __features[i].keypoints;
		const int* category_data = (int *)__categories[i].data;
		int num_points = points.size();
		int offset = __starts[i];
		for (int j = 0; j < num_points-1; j++)
		{
			const float* c1 = color_data+(offset+j)*cdim;
			const Point2f& p1 = points[j].pt;
			Scalar sum_category = sum(abs(__categories[i].row(j)));

			for (int k=j+1; k < num_points; k++)
			{
				if (category_data[j*num_points+k] < _req_same_labels)
					continue;

				const float* c2 = color_data+(offset+k)*cdim;
				const Point2f& p2 = points[k].pt;
				
				float dd = 0;
				for (int c = 0; c < cdim; c++)
					dd += (c1[c]-c2[c])*(c1[c]-c2[c])/(2*pnum*_sigma_color*_sigma_color);
				if (dd > sigma_times2)
					continue;

				if (category_data[j*num_points+k] > _num_frames)
				{
					dd += ((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y))/(2*_sigma_spatial*_sigma_spatial);
					if (dd > sigma_times2)
						continue;
				}

				triplet.push_back(Triplet<float>(offset+j, offset+k, dd));
				triplet.push_back(Triplet<float>(offset+k, offset+j, dd));
			}
		}
	}
	printf("1\n");
	// affinities between matched points
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched <= 0)
				continue;
			const vector<DMatch>& M = __matches[i*_num_frames+j].matches;
			int offset1 = __starts[i];
			int offset2 = __starts[j];
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M[k];
				const float* c1 = color_data+(offset1+t.queryIdx)*cdim;
				const float* c2 = color_data+(offset2+t.trainIdx)*cdim;
				float dd = 0;
				for (int c = 0; c < cdim; c++)
					dd += (c1[c]-c2[c])*(c1[c]-c2[c])/(2*pnum*_sigma_color*_sigma_color);
				if (dd > sigma_times2)
					continue;
				triplet.push_back(Triplet<float>(offset1+t.queryIdx, offset2+t.trainIdx, dd-match_times));
				triplet.push_back(Triplet<float>(offset2+t.trainIdx, offset1+t.queryIdx, dd-match_times));
			}
		}
	}

	int num_triplets = triplet.size();
	printf("%d edges\n", num_triplets/2);
	__sparseD.resize(_num_variables, _num_variables);
	__sparseD.setFromTriplets(triplet.begin(), triplet.end());
	//__sparseD.makeCompressed();
	
	//__affinities.resize(_num_variables, _num_variables);
	//__affinities.setZero();
	//for (int i = 0; i < num_triplets; i++)
	//{
	//	const Triplet<float>& t = triplet[i];
	//	__affinities(t.row(), t.col()) = exp(-t.value());
	//}
}

void Selfie::refreshHomosAfterLabeling()
{
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			if (__matches[i*_num_frames+j].matches.size()<=0)
				continue;
			calcHomoFromMatches(__matches[i*_num_frames+j], __features[i], __features[j], __labels.data()+__starts[i], __labels.data()+__starts[j]);
			calcDualMatches(__matches[j*_num_frames+i], __matches[i*_num_frames+j]);
		}
	}
}

void Selfie::estimateGroundtruthHomos()
{
	__gtH.clear();
	__gtH.resize(__matches.size());	
	Point2f center(0.5*_width, 0.5*_height);
	for (int i = 0; i < _num_frames-1; i++)
	{ 
		const unsigned char* mask1 = __masks[i].data;
		const vector<KeyPoint>& points1 = __features[i].keypoints;
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched <= 0 )
				continue;

			const vector<DMatch>& valid_matches = __matches[i*_num_frames+j].matches;
			const unsigned char* mask2 = __masks[j].data;
			const vector<KeyPoint>& points2 = __features[j].keypoints;
			vector<Point2f> src_points, dst_points;
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = valid_matches[k];
				Point2f p1 = points1[t.queryIdx].pt;
				Point2f p2 = points2[t.trainIdx].pt;
				if (mask1[int(p1.y)*_width+int(p1.x)] < 128 || mask2[int(p2.y)*_width+int(p2.x)] < 128)
					continue;
				src_points.push_back(p1-center);
				dst_points.push_back(p2-center);
			}
			if (src_points.size()>=_req_num_inliers && dst_points.size()>=_req_num_inliers)
			{
				__gtH[i*_num_frames+j] = findHomography(Mat(src_points), Mat(dst_points), CV_RANSAC, _ransac_reject);
				invert(__gtH[i*_num_frames+j], __gtH[j*_num_frames+i]);
			}
		}
	}
}

void Selfie::showReprojHistogram(string outDir, bool gt/*=false*/)
{
	_mkdir(outDir.c_str());
	gt &= (__gtH.size()==__matches.size());
	Point2f center(0.5*_width, 0.5*_height);

	for (int i = 0; i < _num_frames-1; i++)
	{
		const unsigned char* mask1 = __masks[i].data;
		const vector<KeyPoint>& points1 = __features[i].keypoints;
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched <= 0 )
				continue;
			
			const detail::MatchesInfo& M = __matches[i*_num_frames+j];
			const unsigned char* mask2 = __masks[j].data;
			const vector<KeyPoint>& points2 = __features[j].keypoints;
		
			//calculate the reprojection errors
			const double *h = NULL;
			if (gt) {
				if (__gtH[i*_num_frames+j].empty())
					continue;
				h = (double *)__gtH[i*_num_frames+j].data;
			}else
				h = (double *)M.H.data;
			vector<double> bk_errors, fg_errors;
			Mat bk_hist(1, _width, CV_32F, Scalar(0,0,0)), fg_hist(1, _width, CV_32F, Scalar(0,0,0));
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M.matches[k];
				Point2f p1 = points1[t.queryIdx].pt;
				Point2f p2 = points2[t.trainIdx].pt;
				Point2f wp1 = warpPoint(p1-center, h);
				double dd = sqrt((wp1.x-p2.x+center.x)*(wp1.x-p2.x+center.x)+(wp1.y-p2.y+center.y)*(wp1.y-p2.y+center.y));
				if (mask1[int(p1.y)*_width+int(p1.x)] < 128 || mask2[int(p2.y)*_width+int(p2.x)] < 128)	{
					fg_errors.push_back(dd);
					fg_hist.at<float>(min(int(dd),_width-1)) += 1;
				} else {
					bk_errors.push_back(dd);
					bk_hist.at<float>(min(int(dd),_width-1)) += 1;
				}
			}
			Mat histImage(_height, _width, CV_8UC3, Scalar(0,0,0));
			normalize(bk_hist, bk_hist, 0, _height/2-1, NORM_MINMAX, -1, Mat());
			normalize(fg_hist, fg_hist, 0, _height/2-1, NORM_MINMAX, -1, Mat());
			for( int k = 1; k < _width; k++)
			{
				line( histImage, Point(k-1, _height/2-1 - cvRound(bk_hist.at<float>(k-1))),
                       Point(k, _height/2-1 - cvRound(bk_hist.at<float>(k))), Scalar(255, 0, 0), 1, CV_AA);
				line( histImage, Point(k-1, _height-1 - cvRound(fg_hist.at<float>(k-1))) ,
                       Point(k, _height-1 - cvRound(fg_hist.at<float>(k))), Scalar(0, 0, 255), 1, CV_AA);
			}
			string filename = outDir + "/" +__imNames[i] + " - " + __imNames[j] + "_reproj.jpg";
			imwrite(filename, histImage);
		}
	}
}

void Selfie::showFlowHistogram(string outDir)
{
	_mkdir(outDir.c_str());	
	Point2f center(0.5*_width, 0.5*_height);
	vector<Mat> masks(_num_frames);
	for (int i = 0; i < _num_frames; i++)
		masks[i] = imread(__maskFiles[i], 0);

	for (int i = 0; i < _num_frames-1; i++)
	{
		const unsigned char* mask1 = masks[i].data;
		const vector<KeyPoint>& points1 = __features[i].keypoints;
		for (int j = i+1; j < _num_frames; j++)//int j = i+1;
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched <= 0 )
				continue;

			const vector<DMatch>& valid_matches = __matches[i*_num_frames+j].matches;
			const unsigned char* mask2 = masks[j].data;
			const vector<KeyPoint>& points2 = __features[j].keypoints;
			Mat bk_hist(1, 2*_width+1, CV_32F, Scalar(0,0,0)), fg_hist(1, 2*_width+1, CV_32F, Scalar(0,0,0));
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = valid_matches[k];
				Point2f p1 = points1[t.queryIdx].pt;
				Point2f p2 = points2[t.trainIdx].pt;
				int u = min(max(int(p2.x-p1.x), -_width),_width);
				if (mask1[int(p1.y)*_width+int(p1.x)] < 128 || mask2[int(p2.y)*_width+int(p2.x)] < 128)
					fg_hist.at<float>(u+_width) += 1;
				else
					bk_hist.at<float>(u+_width) += 1;
			}
			Mat histImage(_height, 2*_width+1, CV_8UC3, Scalar(0,0,0));
			normalize(bk_hist, bk_hist, 0, _height/2-1, NORM_MINMAX, -1, Mat());
			normalize(fg_hist, fg_hist, 0, _height/2-1, NORM_MINMAX, -1, Mat());
			for( int k = 1; k < 2*_width+1; k++)
			{
				line( histImage, Point(k-1, _height/2-1 - cvRound(bk_hist.at<float>(k-1))),
                       Point(k, _height/2-1 - cvRound(bk_hist.at<float>(k))), Scalar(255, 0, 0), 1, CV_AA);
				line( histImage, Point(k-1, _height-1 - cvRound(fg_hist.at<float>(k-1))) ,
                       Point(k, _height-1 - cvRound(fg_hist.at<float>(k))), Scalar(0, 0, 255), 1, CV_AA);
			}
			string filename = outDir + "/" +__imNames[i] + " - " + __imNames[j] + "_u.jpg";
			imwrite(filename, histImage);
		}
	}
}

void Selfie::showAffinities(string outDir)
{
	printf("show affitinies\n");
	int radius = cvRound(min(_width, _height)*0.05);	
	_mkdir(outDir.c_str());

	for (int i = 0; i < _num_frames; i++)
	{
		printf("%d of %d\r", i, _num_frames);
		const vector<KeyPoint>& points = __features[i].keypoints;
		int offset = __starts[i];
		int num_points = points.size();
		char frameDir[512];
		sprintf(frameDir, "%s/%s", outDir.c_str(), __imNames[i].c_str());
		_mkdir(frameDir);

		vector<string> paths, names;
		if (	ScanDirectory(frameDir, "jpg", paths, names)) {
			for (int k = 0; k < paths.size(); k++)
				remove(paths[k].c_str());
		}

		for (int k = 0; k < num_points; k++)
		{
			const Point2f& p1 = points[k].pt;
			Mat img = __imgs[i].clone();
			rectangle(img, Rect(p1.x-radius/2,p1.y-radius/2,radius,radius), Scalar(255,0,0), 1, CV_AA);
			int minx = p1.x-2*radius, miny = p1.y-2*radius, maxx = p1.x+2*radius, maxy = p1.y+2*radius;

			for (SparseMatrix<float, RowMajor>::InnerIterator it(__sparseD, offset+k); it; ++it)
			{
				int m = it.col()-offset;
				if (m < 0 || m >= num_points)
					continue;
				float aa = exp(-it.value());
				int rr = cvRound(aa*radius);
				if (rr <= 0)
					continue;
				const Point2f& p2 = points[m].pt;
				circle(img, p2, rr, Scalar(0,255,0), 1, CV_AA);
				minx = min(minx, p2.x-rr);
				miny = min(miny, p2.y-rr);
				maxx = max(maxx, p2.x+rr);
				maxy = max(maxy, p2.y+rr);
			}
			
			minx = max(minx, 0);
			miny = max(miny, 0);
			maxx = min(maxx, _width);
			maxy = min(maxy, _height);
			Mat region(img, Rect(minx,miny,maxx-minx,maxy-miny));
			char filename[512];
			sprintf(filename, "%s/%05d.jpg", frameDir, k);
			imwrite(filename, region);
		}
	}
}

void Selfie::showFeatures(string outDir, const VectorXf& labels)
{
	_mkdir(outDir.c_str());
	vector<string> paths, names;
	if (ScanDirectory(outDir, "jpg", paths, names)) {
		for (int i = 0; i < paths.size(); i++)
			remove(paths[i].c_str());
	}

	for (int i = 0; i < _num_frames; i++)
	{
		const vector<KeyPoint>& points = __features[i].keypoints;
		int num_points = points.size();
		int offset = __starts[i];
		Mat img;
		__imgs[i].copyTo(img);
		for (int k = 0; k < num_points; k++)
		{
			Point2f pt = points[k].pt;
			float c = min(max(labels[offset+k], 0), 1);
			circle(img, Point(cvRound(pt.x), cvRound(pt.y)), 3, Scalar((1-c)*255,0,c*255), CV_FILLED, CV_AA);
		}
		char fea_name[512];
		sprintf(fea_name, "%s/%s.jpg", outDir.c_str(),  __imNames[i].c_str());
		imwrite(fea_name, img);
	}
}

void Selfie::showMatches(string outDir, bool show_outlier/*=true*/)
{
	_mkdir(outDir.c_str());
	vector<string> paths, names;
	if (ScanDirectory(outDir, "jpg", paths, names)) {
		for (int i = 0; i < paths.size(); i++)
			remove(paths[i].c_str());
	}

	RNG rng = theRNG();
	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			int num_matched = __matches[i*_num_frames+j].matches.size();
			if (num_matched <= 0)
				continue;

			const vector<DMatch>& valid_matches = __matches[i*_num_frames+j].matches;
			const vector<uchar>& inliers_mask = __matches[i*_num_frames+j].inliers_mask;
			int widthA = __imgs[i].cols, heightA = __imgs[i].rows;
			int widthB = __imgs[j].cols, heightB = __imgs[j].rows;
			int width = max(widthA, widthB), height = heightA+heightB;
			Mat imgAB = Mat::zeros(height, width, CV_8UC3);
			Mat imgA(imgAB, Rect(0, 0, widthA, heightA));
			__imgs[i].copyTo(imgA);
			Mat imgB(imgAB, Rect(0, heightA, widthB, heightB));
			__imgs[j].copyTo(imgB);

			const vector<KeyPoint>& points1 = __features[i].keypoints;
			const vector<KeyPoint>& points2 = __features[j].keypoints;
			int num_points1 = points1.size();
			int num_points2 = points2.size();
			int offset1 = __starts[i];
			int offset2 = __starts[j];

			VectorXi flag1(num_points1), flag2(num_points2);
			flag1.setZero();
			flag2.setZero();
			for (int k = 0; k < num_matched; k++)
			{
				if (!show_outlier && !inliers_mask[k])
					continue;
				const DMatch& t = valid_matches[k];
				Point2f p1 = points1[t.queryIdx].pt;
				Point2f p2 = points2[t.trainIdx].pt+Point2f(0,heightA);
				Scalar newvalue(rng(256),rng(256),rng(256));
				line(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), Point(cvRound(p2.x), cvRound(p2.y)), newvalue, 1, CV_AA);
				circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, newvalue, CV_FILLED, CV_AA);
				circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, newvalue, CV_FILLED, CV_AA);
				flag1[t.queryIdx] = 1;
				flag2[t.trainIdx] = 1;
			}
			for (int k = 0; k < num_points1; k++)
			{
				if (flag1[k])		continue;
				Point2f p1 = points1[k].pt;
				circle(imgAB, Point(cvRound(p1.x), cvRound(p1.y)), 3, Scalar(0,255,0), CV_FILLED, CV_AA);
			}
			for (int k = 0; k < num_points2; k++)
			{
				if (flag2[k])		continue;
				Point2f p2 = points2[k].pt + Point2f(0,heightA);
				circle(imgAB, Point(cvRound(p2.x), cvRound(p2.y)), 3, Scalar(0,255,0), CV_FILLED, CV_AA);
			}
			char fea_name[512];
			sprintf(fea_name, "%s/%s-%s.jpg", outDir.c_str(),  __imNames[i].c_str(), __imNames[j].c_str());
			imwrite(fea_name, imgAB);
		}
	}
}

// run face detection 
void Selfie::faceDetection(string faceDir)
{	
	float shscale = 2;
	float shangle = 30.f/180.f*PI;

	cout << "detect faces" << endl;
	string cascade_name = "C:/Program Files/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"; 
	// haarcascade_frontalface_alt.xml achieves the best detection result
	_mkdir(faceDir.c_str());
	CascadeClassifier cascade;
	if (!cascade.load(cascade_name))
	{
		printf("ERROR: Could not load classifier cascade\n" );;
		return;
	}

	__faces.clear();
	__faces.resize(_num_frames);
	for (int i=0; i<_num_frames; i++)
	{
		printf("  %02d of %02d\r", i, _num_frames);
		Mat img, gray;
		__imgs[i].copyTo(img);
		cvtColor(img, gray, CV_BGR2GRAY);
		equalizeHist(gray, gray);

		vector<Rect> faces;
		cascade.detectMultiScale(gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(100, 100));

		int num_faces = min(faces.size(), _max_num_faces);
		vector<Rect> valid(num_faces, Rect(0,0,0,0));
		for (int j = 0; j < faces.size(); j++)
		{
			Rect f = faces[j];
			for (int k = 0; k < num_faces; k++)
			{
				Rect v = valid[k];
				if (f.area() > v.area())
				{
					for (int m = num_faces-1; m > k; m--)
						valid[m] = valid[m-1];
					valid[k] = f;
					break;
				}
			}
		}
		Mat upper(num_faces, 8, CV_32SC2);
		for (int j = 0; j < num_faces; j++)
		{
			Rect f = valid[j];
			upper.at<Point>(j, 0) = Point(f.x, f.y);
			upper.at<Point>(j, 1) = Point(f.x+f.width-1, f.y);
			upper.at<Point>(j, 2) = Point(f.x+f.width-1, f.y+f.height-1);
			upper.at<Point>(j, 3) = Point(min(f.x+(shscale/2+0.5)*f.width-1,_width-1), f.y+f.height+f.width*(shscale/2-0.5)*tan(shangle)-1);
			upper.at<Point>(j, 4) = Point(min(f.x+(shscale/2+0.5)*f.width-1,_width-1), max(f.y+f.height+f.width*(shscale/2-0.5)*tan(shangle),_height)-1);
			upper.at<Point>(j, 5) = Point(max(f.x-(shscale/2-0.5)*f.width,0), max(f.y+f.height+f.width*(shscale/2-0.5)*tan(shangle),_height)-1);
			upper.at<Point>(j, 6) = Point(max(f.x-(shscale/2-0.5)*f.width,0), f.y+f.height+f.width*(shscale/2-0.5)*tan(shangle)-1);
			upper.at<Point>(j, 7) = Point(f.x, f.y+f.height-1);
			polylines(img, upper.row(j), true, Scalar(0, 0, 255), 1, CV_AA);
		}
		upper.copyTo(__faces[i]);
		imwrite(faceDir+"/"+__imNames[i]+".jpg", img);
	}
}

void Selfie::initializeLabelsFromFace(VectorXf& labels)
{	
	printf("initialize labels from face\n");
	float slack = 0.25;
	// initialize labels
	if (labels.size() != _num_variables)
		labels.resize(_num_variables);
	labels.setConstant(0.5);

	VectorXi Cbk(_num_variables), Cfg(_num_variables);
	Cbk.setZero();
	Cfg.setZero();
	
	for (int i = 0; i < _num_frames; i++)
	{
		const vector<KeyPoint>& points = __features[i].keypoints;
		const Mat& faces = __faces[i];
		int num_points = points.size();
		int offset = __starts[i];

		// inner and outter region
		Mat inner = faces.clone(), outter = faces.clone();
		for (int r = 0; r < faces.rows; r++)
		{
			int shkx = cvRound(slack*(faces.at<int>(r, 1*2+0) - faces.at<int>(r, 0*2+0)+1));
			int shky = cvRound(slack*(faces.at<int>(r, 2*2+1) - faces.at<int>(r, 1*2+1)+1));
			inner.at<Point>(r, 0) += Point(shkx/2, 0);//Point(shkx/2, shky/2);
			inner.at<Point>(r, 1) += Point(-shkx/2, 0);//Point(-shkx/2, shky/2);
			inner.at<Point>(r, 2) += Point(-shkx/2, shky);
			inner.at<Point>(r, 3) += Point(-shkx, shky);
			inner.at<Point>(r, 4) += Point(-shkx, 0);
			inner.at<Point>(r, 5) += Point(shkx, 0);
			inner.at<Point>(r, 6) += Point(shkx, shky);
			inner.at<Point>(r, 7) += Point(shkx/2, shky);

			outter.at<Point>(r, 0) += Point(-shkx/2, -shky/2);
			outter.at<Point>(r, 1) += Point(shkx/2, -shky/2);
			outter.at<Point>(r, 2) += Point(shkx/2, -shky);
			outter.at<Point>(r, 3) += Point(shkx, -shky);
			outter.at<Point>(r, 4) += Point(shkx, shky);
			outter.at<Point>(r, 5) += Point(-shkx, shky);
			outter.at<Point>(r, 6) += Point(-shkx, -shky);
			outter.at<Point>(r, 7) += Point(-shkx/2, -shky);
		}
		for (int k = 0; k < num_points; k++)
		{
			Point2f pt = points[k].pt;
			int flag = -1;
			for (int r = 0; r < faces.rows; r++) {
				double dd = pointPolygonTest(faces.row(r), pt, true);
				if (pointPolygonTest(outter.row(r), pt, false)>=0) {
					flag = 0;
					if (pointPolygonTest(inner.row(r), pt, false)>=0) {
						flag = 1;
						break;
					}
				}
			}

			switch (flag)
			{
				case -1: labels[offset+k] = 0.f;	Cbk(offset+k)++;	break;
				case 1:	labels[offset+k] = 1.f;	Cfg(offset+k)++;	break;
				default: labels[offset+k] = 0.5f;	break;
			}
		}
	}

	//// pairs of match
	//for (int i = 0; i < _num_frames-1; i++)
	//{
	//	for (int j = i+1; j < _num_frames; j++)
	//	{
	//		int num_matched = __matches[i*_num_frames+j].matches.size();
	//		if (num_matched <= 0)
	//			continue;
	//		const vector<DMatch>& M = __matches[i*_num_frames+j].matches;
	//		int offset1 = __starts[i];
	//		int offset2 = __starts[j];
	//		for (int k = 0; k < num_matched; k++)
	//		{
	//			const DMatch& t = M[k];
	//			if (labels[offset1+t.queryIdx] <= _thre_valid_bk)
	//				Cbk(offset2+t.trainIdx)++;
	//			else if (labels[offset1+t.queryIdx] >= 1-_thre_valid_bk)
	//				Cfg(offset2+t.trainIdx)++;

	//			if (labels[offset2+t.trainIdx] <= _thre_valid_bk)
	//				Cbk(offset1+t.queryIdx)++;
	//			else if (labels[offset2+t.trainIdx] >= 1-_thre_valid_bk)
	//				Cfg(offset1+t.queryIdx)++;
	//		}
	//	}
	//}

	//// no match constraints
	//for (int i = 0; i < _num_variables; i++)
	//{
	//	//if (Cfg(i) >= _req_same_labels)
	//	if (Cfg(i)>0 && Cbk(i)<=0)
	//		labels(i) = 1;
	//	//else if (Cbk(i) >= _req_same_labels*0.5)
	//	else if (Cbk(i)>0 && Cfg(i)<=0)
	//		labels(i) = 0;
	//	else
	//		labels(i) = 0.5;
	//}
}

void Selfie::initializeLabelsFromMotion(VectorXf& labels)
{
	printf("initialize labels from motion\n");	
	bool exist_priors = (__labels.squaredNorm()-__labels.sum()+0.25*_num_variables > EPSILON);
	if (labels.size() != _num_variables)
		labels.resize(_num_variables);
	labels.setZero();
	VectorXi Cnt(_num_variables);
	Cnt.setZero();

	for (int i = 0; i < _num_frames-1; i++)
	{
		for (int j = i+1; j < _num_frames; j++)
		{
			int offset1 = __starts[i], offset2 = __starts[j];
			const detail::MatchesInfo& M = __matches[i*_num_frames+j];
			const GMMInfo& G = __motionMix[i*_num_frames+j];
			int num_matched = M.matches.size();
			if (G.probs.empty())
				continue;
			
			// check whether the first component is foreground
			int fg_component = 0;
			if (exist_priors)
			{
				int num_known[2] = {0,0};
				for (int k = 0; k < num_matched; k++)
				{
					const DMatch& t = M.matches[k];
					if (__labels[offset1+t.queryIdx] >= 1-_thre_valid_bk)
						num_known[G.labels.at<int>(k)]++;
					if (__labels[offset2+t.trainIdx] >= 1-_thre_valid_bk)
						num_known[G.labels.at<int>(k)]++;
				}
				fg_component = (num_known[0] < num_known[1]);
			}
			else
			{
				Mat centers = G.model.getMat("means");
				fg_component = (norm(centers.row(0)) >  norm(centers.row(1)));
			}

			// according to the motion classification results
			for (int k = 0; k < num_matched; k++)
			{
				const DMatch& t = M.matches[k];
				float p = G.probs.at<double>(k, fg_component) / (G.probs.at<double>(k, 0)+G.probs.at<double>(k, 1));
				labels(offset1+t.queryIdx) += p;
				labels(offset2+t.trainIdx) += p;
				Cnt(offset1+t.queryIdx)++;
				Cnt(offset2+t.trainIdx)++;
			}
		}
	}
	
	// average the results
	for (int i = 0; i < _num_variables; i++)
	{
		if (Cnt(i))
			labels(i) /= Cnt(i);
		else
			labels(i) = 0.5f;
	}
	
	for (int i = 0; i < _num_trajs; i++)
	{
		const tracking_trajectory& tt = __trajectories[i];
		float sp = 0;
		for (int j = 0; j < tt.length; j++)
			sp += labels(__starts[tt.first_frame+j]+tt.feature_idx[j]);
		sp /= tt.length;
		for (int j = 0; j < tt.length; j++)
			labels(__starts[tt.first_frame+j]+tt.feature_idx[j]) = sp;
	}
	//for (int i = 0; i < _num_variables; i++)
	//{
	//	if (labels(i)<=_thre_valid_bk)
	//		labels(i) = 0.f;
	//	else if (labels(i)>=1-_thre_valid_bk)
	//		labels(i) = 1.f;
	//	else 
	//		labels(i) = 0.5;
	//}
}

void Selfie::initializeLabelsFromLength(VectorXf& labels)
{
	printf("initialize labels from length\n");
	if (labels.size() != _num_variables)
		labels.resize(_num_variables);
	labels.setConstant(0.5f);
	double la0 = 1, la1 = 1, pla0 = 1, pla1 = 1;
	double pi0 = 0.5, pi1 = 0.5, ppi0 = 0.5, ppi1 = 0.5;
	double d = _num_frames-1;

	VectorXd samples(_num_trajs);
	for (int i = 0; i < _num_trajs; i++)
		samples(i) = __trajectories[i].length-1;
	VectorXd inv_fac(_num_frames);
	inv_fac.setConstant(1);
	for (int i = 1; i < _num_frames; i++)
		inv_fac(i) = inv_fac(i-1)/i;
	
	Matrix<double, Dynamic, Dynamic, RowMajor> alpha(2, _num_trajs);
	for (int iter = 0; iter < MaxIters; iter++)
	{
		// expectation
		for (int j = 0; j < _num_trajs; j++)
		{
			int x = samples(j);
			double phi0 = pow(la0, x)*exp(-la0)*inv_fac(x);
			double phi1 = pow(la1, -x+d)*exp(-la1)*inv_fac(-x+d);
			alpha(0, j) = pi0*phi0/(pi0*phi0+pi1*phi1);
			alpha(1, j) = pi1*phi1/(pi0*phi0+pi1*phi1);
		}
		// maxmization
		double sum_a0 = alpha.row(0).sum();
		double sum_a1 = alpha.row(1).sum();
		pi0 = sum_a0/_num_trajs;
		pi1 = sum_a1/_num_trajs;
		la0 = 1/sum_a0*alpha.row(0)*samples;
		la1 = 1/sum_a1*alpha.row(1)*(-samples+VectorXd::Ones(_num_trajs)*d);
		if (sqrt((la0-pla0)*(la0-pla0)+(la1-pla1)*(la1-pla1))<=EPSILON && sqrt((pi0-ppi0)*(pi0-ppi0)+(pi1-ppi1)*(pi1-ppi1))<=EPSILON)
			break;
		else
		{
			pla0 = la0;
			pla1 = la1;
			ppi0 = pi0;
			ppi1 = pi1;
		}
	}

	for (int i = 0; i < _num_trajs; i++)
	{
		const tracking_trajectory& tt = __trajectories[i];
		float p = alpha(1, i);
		//if (p<=_thre_valid_bk)
		//	p = 0.f;
		//else if (p>=1-_thre_valid_bk)
		//	p = 1.f;
		//else
		//	continue;
		for (int j = 0; j < tt.length; j++)
			labels(__starts[tt.first_frame+j]+tt.feature_idx[j]) = p;
	}
}

void Selfie::updateEachFrame()
{
	printf("update each frame independantly\n");
	bool do_geodesic = false;
	float sigma_times2 = -log(0.1f);
	float same_residual = 0.4f;
	float sparse_degree = 0.1f;
	float lambda;

	for (int i = 0; i < _num_frames; i++)
	{
		printf("    %d of %d: ", i, _num_frames);
		int offset = __starts[i];
		int num_points = __features[i].keypoints.size();		
		SparseMatrix<float, RowMajor> sD = __sparseD.block(offset, offset, num_points, num_points);
		Matrix<float, Dynamic, Dynamic, RowMajor> Z(num_points, num_points);
		Z.setZero();
		float *z_data = Z.data();
		float tmpd;

		// do nothing or use the geodesic distance
		int num_edges = 0;
		if (do_geodesic) 
		{
			Matrix<float, Dynamic, Dynamic, RowMajor> D(num_points, num_points);
			float *d_data = D.data();
			Dijkstra dijk;
			dijk.ShortestPath(d_data, sD.valuePtr(), sD.innerIndexPtr(), sD.outerIndexPtr(), num_points);
			for (int m = 0; m < num_points; m++) {
				for (int n = m+1; n < num_points; n++) {
					if (d_data[m*num_points+n] <= sigma_times2) {
						z_data[m*num_points+n] = exp(-d_data[m*num_points+n]);
						z_data[n*num_points+m] = z_data[m*num_points+n];
						num_edges++;
					} else {
						z_data[m*num_points+n] = 0;
						z_data[n*num_points+m] = 0;
					}
				}
			}
		} 
		else 
		{
			for (int m = 0, n; m < num_points; m++) {
				for (SparseMatrix<float, RowMajor>::InnerIterator it(sD,m); it; ++it) {
					n = it.col();
					tmpd = it.value();
					if (n<=m)	
						continue;
					if (tmpd <= sigma_times2) {
						z_data[m*num_points+n] = exp(-tmpd);
						z_data[n*num_points+m] = z_data[m*num_points+n];
						num_edges++;
					}
				}
			}
		}
		for (int k = 0; k < num_points; k++)
			z_data[k*num_points+k] = 1.f;
		printf("%d nodes, %d edges, ", num_points, num_edges);

		// data term and weight matrix
		VectorXf G = __labels.block(offset, 0, num_points, 1), W(num_points);
		W.setOnes();
		for (int k= 0; k < num_points; k++)	{
			if (G[k] > _thre_valid_bk && G[k] < 1-_thre_valid_bk) 
				W[k] = 0;
			else {
				VectorXf cur(num_points);
				cur.setConstant(G[k]);
				VectorXf diff = G - cur;
				diff.cwiseAbs();
				float sumd = Z.row(k)*diff;
				float suma = Z.row(k).sum();
				if (sumd/suma > same_residual)
					W(k) = 0;
			}
		}
		// solve the linear equation
		lambda = W.sum()/W.rows();
		Matrix<float, Dynamic, Dynamic, RowMajor> L = -Z;
		for (int k = 0; k < num_points; k++)
			L(k, k) += Z.row(k).sum() + 1.f/lambda*Z.row(k)*W;
		VectorXf R = 1.f/lambda*Z*W.asDiagonal()*G;

		VectorXf X;
		if (float(num_edges*2+num_points)/(num_points*num_points) <= sparse_degree) {// sparse coefficients
			printf("CG solver\n");
			SparseMatrix<float, RowMajor> sL = L.sparseView();
			// the left matrix mush be processed with .prune()
			// function .sparseView() contains the prune operation
			ConjugateGradient<SparseMatrix<float, RowMajor> > cgSolver(sL);	
			X = cgSolver.solveWithGuess(R, G);
		} else { // dense coefficients
			printf("LDLT solver\n");
			LDLT<Matrix<float, Dynamic, Dynamic, RowMajor>> ldltSolver(L);
			X = ldltSolver.solve(R);
		}

		for (int k = 0; k < num_points; k++)
		{
			if (X[k]<=_thre_valid_bk)
				__labels[offset+k] = 0;
			else if (X[k]>=1-_thre_valid_bk)
				__labels[offset+k] = 1.0;
			else
				__labels[offset+k] = X[k];
		}
		//for (int k = 0; k < num_points; k++)
		//	__labels[offset+k] = min(max(X[k], 0.f), 1.f);
	}
}

void Selfie::updateAllFrames()
{
	printf("update all frames\n");
	bool do_geodesic = false;
	float sigma_times1 = -log(0.8f);
	float sigma_times2 = -log(0.1f);
	float same_residual = 0.4f;
	float lambda;

	vector<Triplet<float>> triplet;
	//triplet.reserve(_num_variables*_num_variables*0.01);
	_num_edges = 0;

	for (int i = 0; i < _num_frames; i++) 
	{
		int num_points = __features[i].keypoints.size();
		int offset = __starts[i], pos;
		float tmpd, tmps;
		Matrix<char,Dynamic,Dynamic,RowMajor> indicator(num_points, _num_variables);
		indicator.setZero();
		char *ind_data = indicator.data();

		// the original edges
		for (int m = 0, n; m < num_points; m++) {
			for (SparseMatrix<float, RowMajor>::InnerIterator it(__sparseD, m+offset); it; ++it) {
				n = it.col();
				tmpd = it.value();
				if (n <= m+offset)		
					continue;
				if (tmpd > sigma_times2)
					continue;
				tmps = exp(-tmpd);
				triplet.push_back(Triplet<float>(m+offset, n, tmps));
				triplet.push_back(Triplet<float>(n, m+offset, tmps));
				ind_data[m*_num_variables+n] = 1;
				_num_edges++;
			}
		}
		// use the geodesic distance
		if (do_geodesic) 
		{
			printf("    geodesic: %d of %d\r", i, _num_frames);
			Matrix<float, Dynamic, Dynamic, RowMajor> D(num_points, _num_variables);
			float* d_data = D.data();
			Dijkstra dijk;
			dijk.ShortestPath(d_data, __sparseD.valuePtr(), __sparseD.innerIndexPtr(), __sparseD.outerIndexPtr(), _num_variables, offset, offset+num_points);

			for (int m = 0, n; m < num_points; m++) {
				for (n = offset+num_points; n < _num_variables; n++) {
					pos = m*_num_variables+n;
					tmpd = d_data[m*_num_variables+n];
					if (!ind_data[pos] && tmpd <= sigma_times1) {
						tmps = exp(-tmpd);
						triplet.push_back(Triplet<float>(m+offset, n, tmps));
						triplet.push_back(Triplet<float>(n, m+offset, tmps));
						_num_edges++;
					}
				}
			}
		} 
	}
	for (int k = 0; k < _num_variables; k++)
		triplet.push_back(Triplet<float>(k, k, 1.f));
	printf("    %d nodes, %d edges, ", _num_variables, _num_edges);

	SparseMatrix<float, RowMajor> affinities(_num_variables, _num_variables);
	affinities.setFromTriplets(triplet.begin(), triplet.end());
	int num_triplet = triplet.size();

	// data term and weight matrix
	VectorXf W(_num_variables);
	W.setOnes();
	for (int k= 0; k < _num_variables; k++) {
		if (__labels[k] > _thre_valid_bk && __labels[k] < 1-_thre_valid_bk) 
			W(k) = 0;
		else {
			VectorXf cur(_num_variables);
			cur.setConstant(__labels[k]);
			VectorXf diff = __labels - cur;
			diff.cwiseAbs();
			VectorXf a = affinities.row(k);
			float sumd = a.transpose()*diff;
			float suma = a.sum();
			if (sumd/suma > same_residual)
				W(k) = 0;
		}
	}
	// solve the linear equation
	lambda = W.sum()/W.rows();
	for (int k = 0; k < num_triplet; k++) {
		Triplet<float> t = triplet[k];
		triplet[k] = Triplet<float>(t.row(), t.col(), -t.value());
	}
	SparseMatrix<float, RowMajor> L(_num_variables, _num_variables);// = -affinities;
	for (int k = 0; k < _num_variables; k++) {
		VectorXf a = affinities.row(k);
		Triplet<float> t = triplet[num_triplet-_num_variables+k];
		float v = a.sum() + 1.f/lambda*a.transpose()*W;
		triplet[num_triplet-_num_variables+k] = Triplet<float>(t.row(), t.col(), t.value()+v);
		//L.coeffRef(k, k) += a.sum() + 1.f/lambda*a.transpose()*W;
	}
	L.setFromTriplets(triplet.begin(), triplet.end());	
	VectorXf R = 1.f/lambda*affinities*(W.asDiagonal()*__labels);

	printf("CG solver\n");
	// !!! this is important !!!
	// the left matrix mush be processed with .prune()
	// function .sparseView() contains the prune operation
	L.prune(0, NumTraits<float>::dummy_precision());
	ConjugateGradient<SparseMatrix<float, RowMajor>> cgSolver(L);
	__labels = cgSolver.solveWithGuess(R, __labels);
	//printf("LDLT solver\n");
	//LDLT<Matrix<float, Dynamic, Dynamic, RowMajor>> ldltSolver(L);
	//__labels = ldltSolver.solve(R);

	for (int k = 0; k < _num_variables; k++)
	{
		if (__labels[k]<=_thre_valid_bk)
			__labels[k] = 0;
		else if (__labels[k]>=1-_thre_valid_bk)
			__labels[k] = 1.0;
		else
			__labels[k] = 0.5;
	}
	//for (int k = 0; k < _num_variables; k++)
	//	__labels[k] = min(max(__labels[k], 0.f), 1.f);
}

void Selfie::anPropagation(string outDir)
{
	printf("sparse to dense propagation via An's method\n");
	float sigma_rgb = 25, inv_sig_rgb = 1.f/sigma_rgb;
	float sigma_vis = 25, inv_sig_vis = 1.f/sigma_vis;
	float sigma_xy = 9, inv_sig_xy = 1.f/sigma_xy;
	int num_samples = 300;
	int psize = 1;
	_mkdir(outDir.c_str());
	__mattes.clear();
	__mattes.resize(_num_frames);
	RNG rng = theRNG();

	for (int i = 0; i < _num_frames; i++)
	{
		printf("    %d of %d\n", i, _num_frames);
		const vector<KeyPoint>& points = __features[i].keypoints;
		int offset = __starts[i];
		int num_points = points.size();
		int num_pixels = _width*_height;
		uchar *im_data = __imgs[i].data;
		
		Mat visual = imread(__visFiles[i]);
		uchar *vis_data = visual.data;

		// random sampling the image
		int fdim = _channels+3;
		Matrix<float, Dynamic, Dynamic, RowMajor> sfea(num_samples, fdim);
		Matrix<int, Dynamic, Dynamic, RowMajor> flag(_height, _width);
		flag.setConstant(-1);
		for (int k = 0; k < num_samples;)
		{
			int xx = rng(_width), yy = rng(_height);
			int &id = flag(yy, xx);
			if (id < 0) {				
				//sfea(k, 0) = xx;
				//sfea(k, 1) = yy;
				for (int c = 0; c < _channels; c++)
					sfea(k, c) = im_data[(yy*_width+xx)*_channels+c];
				for (int c = 0; c < 3; c++)
					sfea(k, c+_channels) = vis_data[(yy*_width+xx)*3+c];
				id = k++;
			}
		}
		int cnt = num_samples;
		for (int m = 0; m < _height; m++) {
			for (int n = 0; n < _width; n++)	{
				if (flag(m, n)<0)
					flag(m, n) = cnt++;
			}
		}

		// user input edits and corresponding mask
		VectorXf G(num_pixels);		
		G.setConstant(0.5);
		DiagonalMatrix<float, Dynamic> W(num_pixels);
		W.setZero();
		for (int k = 0; k < num_points; k++)
		{
			const Point2f& pt = points[k].pt;
			int xx = cvRound(pt.x), yy = cvRound(pt.y);
			int id = flag(yy, xx);
			float ll = __labels[offset+k];
			if (W.diagonal()(id) <= 0) {
				if ( ll<=_thre_valid_bk || ll >= 1-_thre_valid_bk) 	{
					for (int m = max(yy-psize,0); m <= min(yy+psize,_height-1); m++) {
						for (int n = max(xx-psize,0); n <= min(xx+psize,_width-1); n++) {
							W.diagonal()(flag(m,n)) = 1.0;
							G(flag(m,n)) = ll;
						}
					}
				}
			} else if (ll+G(id)<=2*_thre_valid_bk || ll+G(id)>=2*(1-_thre_valid_bk)){
				for (int m = max(yy-psize,0); m <= min(yy+psize,_height-1); m++) {
					for (int n = max(xx-psize,0); n <= min(xx+psize,_width-1); n++) {
						W.diagonal()(flag(m,n)) = 1.0;
						G(flag(m,n)) = (G(id)+ll)*0.5;
					}
				}
			}
		}

		Matrix<float, Dynamic, Dynamic, RowMajor> U(num_pixels, num_samples), A, invA;
		U.setZero();
		for (int m = 0; m < _height; m++)
		{
			for (int n = 0; n < _width; n++)
			{
				int pos = m*_width+n;
				int id = flag(m, n);
				for (int k = 0; k < num_samples; k++)
				{
					float dd = 0;
					//float dd = ((sfea(k,0)-n)*(sfea(k,0)-n)+(sfea(k,1)-m)*(sfea(k,1)-m))*0.5*inv_sig_xy*inv_sig_xy;
					for (int c = 0; c < _channels; c++)
						dd += ((sfea(k,c)-im_data[pos*_channels+c])*(sfea(k,c)-im_data[pos*_channels+c]))*0.5*inv_sig_rgb*inv_sig_rgb;
					for (int c = 0; c < 3; c++)
						dd += ((sfea(k,c+_channels)-vis_data[pos*3+c])*(sfea(k,c+_channels)-vis_data[pos*3+c]))*0.5*inv_sig_vis*inv_sig_vis;
					U(id, k) = exp(-dd);
				}
			}
		}

		float lambda = W.diagonal().sum()/W.rows();
		A = U.block(0, 0, num_samples, num_samples);
		invA = A.inverse();
		VectorXf d = U*invA*(U.transpose().rowwise().sum()) + 1.f/lambda*U*invA*(U.transpose()*W.diagonal());
		DiagonalMatrix<float, Dynamic> invD = d.asDiagonal().inverse();
		VectorXf R = 1.f/lambda*U*invA*(U.transpose()*(W*G));
		Matrix<float, Dynamic, Dynamic, RowMajor> tmp = -A+U.transpose()*(invD*U);		
		VectorXf X = invD*R - invD*U*tmp.inverse()*(U.transpose()*(invD*R));

		__mattes[i].resize(num_pixels);
		float *matte_data = __mattes[i].data();
		for (int m = 0; m < _height; m++)
			for (int n = 0; n < _width; n++)
				matte_data[m*_width+n] = X[flag(m,n)];
		saveFloatImage(outDir+"/"+__imNames[i]+".png", matte_data, _width, _height, 1);
	}
}

void Selfie::xuPropagation(string outDir)
{
	printf("sparse to dense propagation via Xu's method\n");
	float lambda = 1e-4;
	float sigma_rgb = 25, inv_sig_rgb = 1.0/sigma_rgb;
	float sigma_vis = 25, inv_sig_vis = 1.0/sigma_vis;
	float sigma_xy = 36, inv_sig_xy = 1.0/sigma_xy;
	float sigma_edit = 0.10, inv_sig_edit = 1.0/sigma_edit;
	float thre_edge = 50;
	int max_iters = 3;
	int psize = 1;
	_mkdir(outDir.c_str());
	__mattes.clear();
	__mattes.resize(_num_frames);

	for (int i = 0; i < _num_frames; i++)
	{
		printf("    %d of %d\n", i, _num_frames);
		const vector<KeyPoint>& points = __features[i].keypoints;
		int offset = __starts[i];
		int num_points = points.size();
		int num_pixels = _width*_height;
		uchar *im_data = __imgs[i].data;

		// compute the L matrix
		vector<Triplet<float>> triplet;
		triplet.reserve(num_pixels*10);
		VectorXf dL(num_pixels);
		dL.setZero();
		for (int y = 0; y < _height-1; y++)
		{
			for (int x = 0; x < _width-1; x++)
			{
				int pos = y*_width+x, os = pos*_channels;
				float dd = 0.5*inv_sig_xy*inv_sig_xy, d1 = dd, d2 = dd;
				for (int c = 0; c < _channels; c++) {
					d1 += (im_data[os+c]-im_data[os+_channels+c])*(im_data[os+c]-im_data[os+_channels+c])*0.5*inv_sig_rgb*inv_sig_rgb;
					d2 += (im_data[os+c]-im_data[os+_width*_channels+c])*(im_data[os+c]-im_data[os+_width*_channels+c])*0.5*inv_sig_rgb*inv_sig_rgb;
				}
				float s1 = exp(-d1);
				float s2 = exp(-d2);
				triplet.push_back(Triplet<float>(pos, pos+1, -s1));
				triplet.push_back(Triplet<float>(pos+1, pos, -s1));
				triplet.push_back(Triplet<float>(pos, pos+_width, -s2));
				triplet.push_back(Triplet<float>(pos+_width, pos, -s2));
				dL(pos) += (s1+s2);
				dL(pos+1) += s1;
				dL(pos+_width) += s2;
			}
		}

		// the user edited samples g, and its mask m
		TImage g(1,  _width, _height, 1), m(1, _width, _height, 1);	
		float *g_data = g.data;
		float *m_data = m.data;		
		for (int k = 0; k < num_pixels; k++) {
			g_data[k] = 0.5;
			m_data[k] = 0;
		}
		for (int k = 0; k < num_points; k++)
		{
			const Point2f& pt = points[k].pt;
			int xx = cvRound(pt.x), yy = cvRound(pt.y), pos = yy*_width+xx;
			float ll = __labels[offset+k];
			if (m_data[pos]<=0)	{
				if (ll<=_thre_valid_bk || ll>=1-_thre_valid_bk) {
					for (int y = max(yy-psize,0); y <= min(yy+psize,_height-1); y++) {
						for (int x = max(xx-psize,0); x<= min(xx+psize,_width-1); x++) {
							g_data[y*_width+x] = ll;
							m_data[y*_width+x] = 1;	
						}
					}
				}
			} else if (g_data[pos]+ll<=2*_thre_valid_bk || g_data[pos]+ll>=2*(1-_thre_valid_bk)){
				for (int y = max(yy-psize,0); y <= min(yy+psize,_height-1); y++)
					for (int x = max(xx-psize,0); x <= min(xx+psize,_width-1); x++) 
						g_data[y*_width+x] = (g_data[pos]+ll)/2;
			}
		}
		Mat visual = imread(__visFiles[i]);
		uchar *vis_data = visual.data;
		// construct the new feature for Gaussian filtering
		int fdim = _channels+3+1;
		TImage newfea(1, _width, _height, fdim);
		float *new_data = newfea.data;
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				int pos = y*_width+x;
				//new_data[pos*fdim+0] = x*inv_sig_xy;
				//new_data[pos*fdim+1] = y*inv_sig_xy;
				for (int c = 0; c < _channels; c++)
					new_data[pos*fdim+c] = im_data[pos*_channels+c]*inv_sig_rgb;
				for (int c = 0; c < _channels; c++)
					new_data[pos*fdim+_channels+c] = vis_data[pos*3+c]*inv_sig_vis;
			}
		}

		// fixed point iteration
		VectorXf s(num_pixels);
		s.setConstant(0.5);
		for (int iter = 0; iter < max_iters; iter++)
		{		
			for (int y = 0; y < _height; y++)
				for (int x = 0; x < _width; x++)
					new_data[(y*_width+x)*fdim+fdim-1] = s[y*_width+x]*inv_sig_edit;

			TImage gbar = GKDTree::filter(g, newfea, 0.5);
			TImage mbar = GKDTree::filter(m, newfea, 0.5);
			float *gbar_data = gbar.data;
			float *mbar_data = mbar.data;

			Mat gbarIm(_height, _width, CV_8U), edgeIm(_height, _width, CV_8U);
			uchar *gim_data = gbarIm.data;
			for (int k = 0; k < num_pixels; k++)
				gim_data[k] = uchar(min(max(gbar_data[k]*255.0, 0), 255));
			Canny(gbarIm, edgeIm, thre_edge, thre_edge*3, 3);
			uchar *edge_data = edgeIm.data;
			for (int k = 0; k < num_pixels; k++) 
			{
				if (edge_data[k]==0 && mbar_data[k]>EPSILON)
					edge_data[k] = 255;
				else
					edge_data[k] = 0;
			}
			Mat erosion = getStructuringElement(MORPH_RECT, Size(1,1)), confIm;
			morphologyEx(edgeIm, confIm, MORPH_ERODE, erosion);
			uchar *conf_data = confIm.data;

			SparseMatrix<float, RowMajor> L(num_pixels, num_pixels);
			L.setFromTriplets(triplet.begin(), triplet.end());
			L *= lambda;
			VectorXf R(num_pixels);
			for (int k = 0; k < num_pixels; k++) 
			{
				if (conf_data[k] >= 128) {
					L.insert(k,k) = 1+lambda*dL(k);
					R[k] = gbar_data[k] / mbar_data[k];
				} else {
					L.insert(k,k) = lambda*dL(k);
					R[k] = 0;
				}
			}

			// !!! this is important !!!
			// the left matrix mush be processed with .prune()
			// function .sparseView() contains the prune operation
			L.prune(0, NumTraits<float>::dummy_precision());
			ConjugateGradient<SparseMatrix<float,RowMajor>> solver(L);
			//s = solver.solveWithGuess(R, s);
			s = solver.solve(R);
			
			__mattes[i] = s;
			char filename[512];
			sprintf(filename, "%s/%s_iter%02d.png", outDir.c_str(), __imNames[i].c_str(), iter);
			saveFloatImage(filename, s.data(), _width, _height, 1);
		}
	}
}

void Selfie::myPropagation(string outDir)
{
	printf("sparse to dense propagation via my approach\n");
	float lambda = 1.f;
	float sigma_xy = sqrt(0.05*0.05*((_width-1)*(_width-1)+(_height-1)*(_height-1))); //36
	float sigma_rgb = sqrt(0.05*0.05*255*255*_channels); // 22
	float sigma_vis = sigma_rgb*3; //66
	float thre_edge = 48;
	float thre_length = min(_width, _height)*0.20;
	int ksize = cvRound(min(_width,_height)*0.02)/2*2+1;
	int thre_int_bk = cvRound(_thre_valid_bk*255);
	float inv_sig_xy2 = 0.5/(sigma_xy*sigma_xy);
	float inv_sig_rgb2 = 0.5/(sigma_rgb*sigma_rgb);
	float inv_sig_vis2 = 0.5/(sigma_vis*sigma_vis);

	_mkdir(outDir.c_str());
	__mattes.clear();
	__mattes.resize(_num_frames);
	char filename[512];
	vector<string> paths, names;
	if (	ScanDirectory(outDir, "png", paths, names)) {
		for (int k = 0; k < paths.size(); k++)
			remove(paths[k].c_str());
	}

	for (int i = 0; i < _num_frames; i++)
	{
		printf("    %d of %d\n", i, _num_frames);
		const vector<KeyPoint>& points = __features[i].keypoints;		
		int offset = __starts[i];
		int num_points = points.size();
		int num_pixels = _width*_height;
		Mat visual = imread(__visFiles[i]);
		uchar *vis_data = visual.data;
		uchar *im_data = __imgs[i].data;
		Mat edgeIm;
		getImgEdge(edgeIm, __imgs[i], thre_edge, 0);

		// construct the descriptors
		int fdim = 2+_channels+3+1;
		Matrix<float, Dynamic, Dynamic, RowMajor> sfea(num_points, fdim);
		float *sfea_data = sfea.data();
		int cnt_samples = 0;
		for (int k = 0; k < num_points; k++)
		{
			const Point2f& pt = points[k].pt;
			float ll = __labels[offset+k];
			if ( ll<=_thre_valid_bk || ll >= 1-_thre_valid_bk) 
			{
				sfea(cnt_samples, 0) = pt.x;
				sfea(cnt_samples, 1) = pt.y;
				bilinearInterpolate(sfea_data+cnt_samples*fdim+2, pt, im_data, _width, _height, _channels);
				bilinearInterpolate(sfea_data+cnt_samples*fdim+2+_channels, pt, vis_data, _width, _height, 3);
				sfea(cnt_samples, fdim-1) = ll;
				cnt_samples++;
			}
		}

		// compute the data term
		VectorXf G(num_pixels), W(num_pixels);		
		G.setConstant(0.5);
		W.setOnes();
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				float mindist = FLT_MAX, sumg = 0, sumw = 0, dd, ss;
				int minid = -1, pos = y*_width+x;
				float &gg = G(pos);
				for (int k = 0; k < cnt_samples; k++)
				{
					dd = ((x-sfea_data[k*fdim])*(x-sfea_data[k*fdim])+(y-sfea_data[k*fdim+1])*(y-sfea_data[k*fdim+1]))*inv_sig_xy2;
					if (dd > 2)
						continue;
					dd = 0;
					for (int c = 0; c < _channels; c++)
						dd += (im_data[pos*_channels+c]-sfea_data[k*fdim+c+2])*(im_data[pos*_channels+c]-sfea_data[k*fdim+c+2])*inv_sig_rgb2;
					for (int c = 0; c < 3; c++)
						dd += (vis_data[pos*3+c]-sfea_data[k*fdim+c+2+_channels])*(vis_data[pos*3+c]-sfea_data[k*fdim+c+2+_channels])*inv_sig_vis2;
					//if (dd < mindist) {
					//	mindist = dd;
					//	minid = k;
					//}
					ss = exp(-dd);
					sumw += ss;
					sumg += ss*sfea(k, fdim-1);
				}
				//if (minid>=0)
				//	gg = sfea(minid, fdim-1);
				if (sumw)
					gg = sumg/sumw;
			}
		}
		
		// user input as dataterm	
		for (int k = 0; k < num_pixels; k++) {
			if (G(k)<=_thre_valid_bk)
				G(k) = 0;
			else if (G(k)>=1-_thre_valid_bk) 
				G(k) = 1;
		}
		Mat mt, user, contUser, edgeUser;
		saveFloatImage(mt, G.data(), _width, _height, 1);
		threshold(mt, user, 255-thre_int_bk-0.5, 255, THRESH_BINARY);
		sprintf(filename, "%s/%s_0_dataterm.png", outDir.c_str(), __imNames[i].c_str());
		imwrite(filename, mt);

		// the image edges are maintained in the segmentation result
		//getImgEdge(edgeUser, user, thre_edge, ksize);
		getImgContour(contUser, user, thre_edge, thre_length, ksize);
		Mat validEdges = edgeIm & contUser;	
		uchar *edge_data = validEdges.data;
		sprintf(filename, "%s/%s_1_edge.png", outDir.c_str(), __imNames[i].c_str());
		imwrite(filename, validEdges);

		// weight matrix
		W.setOnes();
		getImgEdge(edgeUser, user, thre_edge, ksize);
		uchar *euser_data = edgeUser.data;
		for (int k = 0; k < num_pixels; k++) {
			if (euser_data[k] && G(k)>_thre_valid_bk && G(k)<1-_thre_valid_bk)
				W(k) = 0;
		}
		sprintf(filename, "%s/%s_2_weight.png", outDir.c_str(), __imNames[i].c_str());
		saveFloatImage(filename, W.data(), _width, _height, 1);

		// compute the Laplacian matrix
		vector<Triplet<float>> triplet;
		//triplet.reserve(num_pixels*10);
		VectorXf dL(num_pixels);
		dL.setZero();
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				int pos = y*_width+x, os = pos*_channels, pix;
				float dd, ss;
				int pixels[4] = {pos+1, pos+_width+1, pos+_width, pos+_width-1};
				bool inrange[4] = {false, false, false, false};
				if (x<_width-1 && edge_data[pixels[0]]<=0)
					inrange[0] = true;
				if (x<_width-1 && y<_height-1 && edge_data[pixels[1]]<=0)
					inrange[1] = true;
				if (y<_height-1 && edge_data[pixels[2]]<=0)
					inrange[2] = true;
				if (x>0 && y<_height-1 && edge_data[pixels[3]]<=0)
					inrange[3] = true;

				for (int k = 0; k < 4; k++)
				{
					if ( ! inrange[k] )
						continue;
					pix = pixels[k];
					dd = 0;
					for (int c = 0; c < _channels; c++)
						dd += (im_data[os+c]-im_data[pix*_channels+c])*(im_data[os+c]-im_data[pix*_channels+c])*inv_sig_rgb2;
					ss = exp(-dd);
					triplet.push_back(Triplet<float>(pos, pix, -ss));
					triplet.push_back(Triplet<float>(pix, pos, -ss));
					dL(pos) += ss;
					dL(pix) += ss;
				}
			}
		}
		for (int k = 0; k < num_pixels; k++)
			triplet.push_back(Triplet<float>(k, k, dL(k)));

		// Ax = b
		//lambda = W.sum() / W.rows();
		SparseMatrix<float, RowMajor> L(num_pixels, num_pixels);
		L.setFromTriplets(triplet.begin(), triplet.end());
		L *= lambda;
		for (int k = 0; k < num_pixels; k++)
			L.coeffRef(k, k) += W(k);
		VectorXf R = W.asDiagonal()*G;

		// !!! this is important !!!
		// the left matrix mush be processed with .prune()
		// function .sparseView() contains the prune operation
		L.prune(0, NumTraits<float>::dummy_precision());
		ConjugateGradient<SparseMatrix<float,RowMajor>> solver(L);
		solver.setMaxIterations(1e4);
		__mattes[i] = solver.solveWithGuess(R, G);
		
		sprintf(filename, "%s/%s_3_optimal.png", outDir.c_str(), __imNames[i].c_str());
		saveFloatImage(filename, __mattes[i].data(), _width, _height, 1);

		// image matting
		//Mat Trimap(_height, _width, CV_8U, Scalar(128,128,128));
		//uchar *trimap_data = Trimap.data;
		//for (int k = 0; k < num_pixels; k++){
		//	if (W(k)) {
		//		trimap_data[k] = (user_data[k]<=thre_int_bk) ? 0 : 255;
		//		G(k) = (G(k)<=_thre_valid_bk) ? 0 : 1;
		//	} else 
		//		G(k) = 0.5;
		//}
		//sprintf(filename, "%s/%s_iter%02d_5_trimap.png", outDir.c_str(), __imNames[i].c_str(), iter);
		//imwrite(filename, Trimap);
		//Matrix<double,Dynamic,Dynamic,RowMajor> M = SolveRobustMatting(__imgs[i], Mat(_height, _width, CV_8U), Trimap);
		//sprintf(filename, "%s/%s_iter%02d_6_matting.png", framedir, __imNames[i].c_str(), iter);
		//saveFloatImage(filename, M.data(), _width, _height, 1);
	}
}


void Selfie::myPropagation2(string outDir)
{
	printf("sparse to dense propagation via AllPairs\n");
	float sigma_rgb = 10, inv_sig_rgb = 1.f/sigma_rgb;
	float sigma_xy = 9, inv_sig_xy = 1.f/sigma_xy;
	int num_samples = 300;
	int psize = 1;
	_mkdir(outDir.c_str());
	__mattes.clear();
	__mattes.resize(_num_frames);
	RNG rng(getTickCount());

	for (int i = 0; i < _num_frames; i++)
	{
		printf("    %d of %d\n", i, _num_frames);
		const vector<KeyPoint>& points = __features[i].keypoints;
		int offset = __starts[i];
		int num_points = points.size();
		int num_pixels = _width*_height;
		uchar *im_data = __imgs[i].data;
		
		// random sampling the image
		Matrix<float, Dynamic, Dynamic, RowMajor> srgb(num_samples, _channels), sxy(num_samples, 2);
		Matrix<int, Dynamic, Dynamic, RowMajor> flag(_height, _width);
		flag.setConstant(-1);
		for (int k = 0; k < num_samples;)
		{
			int xx = rng(_width), yy = rng(_height);
			int &id = flag(yy, xx);
			if (id < 0) {
				for (int c = 0; c < _channels; c++)
					srgb(k, c) = im_data[(yy*_width+xx)*_channels+c];
				sxy(k, 0) = xx;
				sxy(k, 1) = yy;
				id = k++;
			}
		}
		int cnt = num_samples;
		for (int m = 0; m < _height; m++) {
			for (int n = 0; n < _width; n++)	{
				if (flag(m, n)<0)
					flag(m, n) = cnt++;
			}
		}

		// user input edits and corresponding mask
		VectorXf G(num_pixels);		
		G.setConstant(0.5);
		DiagonalMatrix<float, Dynamic> W(num_pixels);
		W.setZero();
		for (int k = 0; k < num_points; k++)
		{
			const Point2f& pt = points[k].pt;
			int xx = cvRound(pt.x), yy = cvRound(pt.y);
			int id = flag(yy, xx);
			float ll = __labels[offset+k];
			if (W.diagonal()(id) <= 0) {
				if ( ll<=_thre_valid_bk || ll >= 1-_thre_valid_bk) 	{
					for (int m = max(yy-psize,0); m <= min(yy+psize,_height-1); m++) {
						for (int n = max(xx-psize,0); n <= min(xx+psize,_width-1); n++) {
							W.diagonal()(flag(m,n)) = 1.0;
							G(flag(m,n)) = ll;
						}
					}
				}
			} else if (ll+G(id)<=2*_thre_valid_bk || ll+G(id)>=2*(1-_thre_valid_bk)){
				for (int m = max(yy-psize,0); m <= min(yy+psize,_height-1); m++) {
					for (int n = max(xx-psize,0); n <= min(xx+psize,_width-1); n++) {
						W.diagonal()(flag(m,n)) = 1.0;
						G(flag(m,n)) = (G(id)+ll)*0.5;
					}
				}
			}
		}

		Matrix<float, Dynamic, Dynamic, RowMajor> U(num_pixels, num_samples), A, invA;
		U.setZero();
		for (int m = 0; m < _height; m++)
		{
			for (int n = 0; n < _width; n++)
			{
				int os = (m*_width+n)*_channels;
				int id = flag(m, n);
				for (int k = 0; k < num_samples; k++)
				{
					float dd = ((sxy(k,0)-n)*(sxy(k,0)-n)+(sxy(k,1)-m)*(sxy(k,1)-m))*0.5*inv_sig_xy*inv_sig_xy;
					for (int c = 0; c < _channels; c++)
						dd += ((srgb(k,c)-im_data[os+c])*(srgb(k,c)-im_data[os+c]))*0.5*inv_sig_rgb*inv_sig_rgb;
					U(id, k) = exp(-dd);
				}
			}
		}

		float lambda = W.diagonal().sum()/W.rows();
		A = U.block(0, 0, num_samples, num_samples);
		invA = A.inverse();
		VectorXf d = U*invA*(U.transpose().rowwise().sum()) + 1.f/lambda*W.diagonal();
		DiagonalMatrix<float, Dynamic> invD = d.asDiagonal().inverse();
		VectorXf R = 1.f/lambda*W*G;
		Matrix<float, Dynamic, Dynamic, RowMajor> tmp = -A+U.transpose()*(invD*U);		
		VectorXf X = invD*R - invD*U*tmp.inverse()*(U.transpose()*(invD*R));

		__mattes[i] = X;
		saveFloatImage(outDir+"/"+__imNames[i]+".png", X.data(), _width, _height, 1);
	}
}

void Selfie::refineMattes(string outDir, const vector<string>& mt_files)
{
	printf("refine the dense mattes\n");
	_mkdir(outDir.c_str());
	float lambda = 1;
	float sigma_rgb = sqrt(0.05*0.05*255*255*_channels); // 22
	float inv_sig_rgb2 = 0.5/(sigma_rgb*sigma_rgb);
	float thre_edge = 48;
	float thre_length = min(_width, _height)*0.20;
	int ksize = cvRound(min(_width, _height)*0.02);
	int thre_int_bk = cvRound(_thre_valid_bk*255);
	int num_pixels = _width*_height;
	int num_total = _num_frames*num_pixels;
	char filename[512];
	vector<string> paths, names;
	if (	ScanDirectory(outDir, "png", paths, names)) {
		for (int k = 0; k < paths.size(); k++)
			remove(paths[k].c_str());
	}

	VectorXf G(num_total), W(num_total);
	W.setOnes();
	vector<Triplet<float>> triplet;
	//triplet.reserve(num_total*20);
	SparseMatrix<float, RowMajor> L(num_total, num_total);
	VectorXf dL(num_total);
	dL.setZero();
	for (int i = 0; i < _num_frames; i++)
	{
		printf("    load %d of %d\r", i, _num_frames);
		int offset = i*num_pixels;
		// read the mattes files
		Mat mt = imread(mt_files[i], CV_LOAD_IMAGE_GRAYSCALE), user;
		uchar *mt_data = mt.data;
		for (int k = 0; k < num_pixels; k++) {
			if (mt_data[k] <= thre_int_bk)
				G(offset+k) = 0;
			else if (mt_data[k]>=255-thre_int_bk)
				G(offset+k) = 1;
			else 
				G(offset+k) = mt_data[k]/255.f;
		}
		threshold(mt, user, 255-thre_int_bk-0.5, 255, THRESH_BINARY);

		// the image edges are maintained in the segmentation result
		Mat edgeIm, edgeUser, cont;
		getImgEdge(edgeIm, __imgs[i], thre_edge, 0);
		getImgContour(cont, user, thre_edge, thre_length, ksize);
		Mat validEdges = edgeIm & cont;	
		uchar *edge_data = validEdges.data;

		// weight matrix
		W.setOnes();
		getImgEdge(edgeUser, user, thre_edge, ksize);
		uchar *euser_data = edgeUser.data;
		for (int k = 0; k < num_pixels; k++) {
			if (euser_data[k] && G(k)>_thre_valid_bk && G(k)<1-_thre_valid_bk)
				W(k) = 0;
		}

		// read the optical flow
		DImage fflow, bflow;
		double *fflow_data = NULL, *bflow_data = NULL;
		uchar *im_data = __imgs[i].data, *im0_data = NULL, *im1_data = NULL;
		if (i < _num_frames-1) {
			readOpticalFlow(fflow, __flowFiles[i*2+0]);
			fflow_data = fflow.data();
			im1_data = __imgs[i+1].data;
		}
		if (i > 0) {
			readOpticalFlow(bflow, __flowFiles[(i-1)*2+1]);
			bflow_data = bflow.data();
			im0_data = __imgs[i-1].data;
		}
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				int pos = y*_width+x, pos_c = pos*_channels, pos_p = offset+pos, px, py;
				int pixels[4] = {offset+pos+1, offset+pos+_width};
				bool inrange[4] = {false, false, false, false};
				int cc[12] = {0};
				float dd, ss;
				if (x<_width-1) {
					inrange[0] = true;
					for (int c = 0; c < _channels; c++)
						cc[c] = im_data[pos_c+c]-im_data[pos_c+_channels+c];
				}
				if (y<_height-1) {
					inrange[1] = true;
					for (int c = 0; c < _channels; c++)
						cc[1*_channels+c] = im_data[pos_c+c]-im_data[pos_c+_width*_channels+c];
				}
				if (fflow_data) {
					px = int(fflow_data[pos*2]+x);
					py = int(fflow_data[pos*2+1]+y);
					if (px>=0 && px<=_width-1 && py>=0 && py<=_height-1) {
						inrange[2] = true;
						pixels[2] = offset+num_pixels+(py*_width+px);
						for (int c = 0; c < _channels; c++)
							cc[2*_channels+c] = im_data[pos_c+c]-im1_data[(py*_width+px)*_channels+c];
					}
				}
				if (bflow_data) {
					px = int(bflow_data[pos*2]+x);
					py = int(bflow_data[pos*2+1]+y);
					if (px>=0 && px<=_width-1 && py>=0 && py<=_height-1) {
						inrange[3] = true;
						pixels[3] = offset-num_pixels+(py*_width+px);
						for (int c = 0; c < _channels; c++)
							cc[3*_channels+c] = im_data[pos_c+c]-im0_data[(py*_width+px)*_channels+c];
					}
				}

				for (int k = 0; k < 4; k++)
				{
					if ( ! inrange[k] )
						continue;
					dd = 0;
					for (int c = 0; c < _channels; c++)
						dd += cc[k*_channels+c]*cc[k*_channels+c]*inv_sig_rgb2;
					ss = exp(-dd);
					triplet.push_back(Triplet<float>(pos_p, pixels[k], -lambda*ss));
					triplet.push_back(Triplet<float>(pixels[k], pos_p, -lambda*ss));
					dL(pos_p) += lambda*ss;
					dL(pixels[k]) += lambda*ss;
				}
			}
		}
	}
	for (int k = 0; k < num_total; k++)
		triplet.push_back(Triplet<float>(k, k, dL(k)+W(k)));

	L.setFromTriplets(triplet.begin(), triplet.end());
	VectorXf R = W.asDiagonal()*G;

	// !!! this is important !!!
	// the left matrix mush be processed with .prune()
	// function .sparseView() contains the prune operation
	L.prune(0, NumTraits<float>::dummy_precision());
	ConjugateGradient<SparseMatrix<float,RowMajor>> solver(L);
	solver.setMaxIterations(1e4);
	VectorXf X = solver.solveWithGuess(R, G);

	for (int i = 0; i < _num_frames; i++)
	{
		sprintf(filename, "%s/%s_optimal.png", outDir.c_str(), __imNames[i].c_str());
		Mat mt, the, cont_f, cont_b2, cont, elem, bin;
		saveFloatImage(mt, X.data()+i*num_pixels, _width, _height, 1);
		imwrite(filename, mt);

		threshold(mt, the, 255-thre_int_bk-0.5, 255, THRESH_BINARY);
		sprintf(filename, "%s/%s_thre.png", outDir.c_str(), __imNames[i].c_str());
		imwrite(filename, the);
		
		getImgContour(cont_f, the, -1, thre_length, CV_FILLED);
		// to avoid the lines on the boundary
		Mat the2(the.rows+2, the.cols+2, CV_8U, Scalar(255,255,255));
		Mat reThe(the2, Rect(1,1,the.cols,the.rows));
		reThe = ~the;
		getImgContour(cont_b2, the2, -1, thre_length, CV_FILLED);
		Mat cont_b(cont_b2, Rect(1,1,the.cols,the.rows));
		cont = cont_f | (~cont_b);
		sprintf(filename, "%s/%s_cont.png", outDir.c_str(), __imNames[i].c_str());
		imwrite(filename, cont);

		elem = getStructuringElement(MORPH_RECT, Size(ksize,ksize));
		morphologyEx(cont, bin, MORPH_CLOSE, elem);

		sprintf(filename, "%s/%s_binary.png", outDir.c_str(), __imNames[i].c_str());
		imwrite(filename, bin);
	}

	//_mkdir(outDir.c_str());
	//printf("refine the dense mattes\n");
	//int gridx = 4, gridy = 4;
	//float lambda = 1e-4;
	//int thre_int_bk = cvRound(_thre_valid_bk*255);

	//// convert the mattes to img;
	//vector<Mat> invMattes(_num_frames);
	//for (int i = 0; i < _num_frames; i++) 
	//{
	//	//saveFloatImage(invMattes[i], __mattes[i].data(), _width, _height, 1);
	//	//invMattes[i] = ~invMattes[i];
	//	invMattes[i] = ~imread(mt_files[i], CV_LOAD_IMAGE_GRAYSCALE);
	//}

	//Warping warps(_width, _height, gridx, gridy, lambda);
	//for (int i = 0; i < _num_frames; i++)
	//{
	//	int offset1 = __starts[i];
	//	for (int j = 0; j < _num_frames; j++)
	//	{
	//		const detail::MatchesInfo& M = __matches[j*_num_frames+i];
	//		int num_matched = M.matches.size();
	//		if (num_matched <= 0)
	//			continue;
	//		Mat warpMt2, warpIm2;
	//		
	//		//Mat H = M.H;
	//		//warpPerspective(invMattes[j], warpMt2, H, Size(_width,_height));
	//		//warpPerspective(__imgs[j], warpIm2, H, Size(_width,_height));

	//		warps.solve(M, __features[j], __features[i]);
	//		warps.warp(warpIm2, __imgs[j]);
	//		warps.warp(warpMt2, invMattes[j]);

	//		char filename[512];
	//		sprintf(filename, "%s/%s-%s_warp mattes.png", outDir.c_str(), __imNames[j].c_str(), __imNames[i].c_str());
	//		imwrite(filename, warpMt2);
	//		sprintf(filename, "%s/%s-%s_warp.jpg", outDir.c_str(), __imNames[j].c_str(), __imNames[i].c_str());
	//		imwrite(filename, warpIm2);
	//	}
	//}
}

void Selfie::main(string outDir)
{	
	_mkdir(outDir.c_str());
	
	//// estimate the groundtruth homograph bewtween frames
	//estimateGroundtruthHomos();

	//// show the reprojection error histogram
	//showReprojHistogram(outDir+"/ReprojHist", true);
	//
	//// show the flow histogram
	//showFlowHistogram(outDir+"/FlowHist");

	//// show matches
	//showMatches(outDir+"/Matches");

	//// run face detection
	//faceDetection(outDir + "/Faces");
	//
	//// initialize the result through face detection results
	//initializeLabelsFromFace(__labels);
	//showFeatures(outDir+"/Labels_0_face", __labels);

	//// clustering the optical flow
	//clustering(outDir+"/Clusters");
	//
	//// initialize the labels through motion 
	//VectorXf L1;
	//initializeLabelsFromMotion(L1);
	//showFeatures(outDir+"/Labels_0_motion", L1);

	//// initialize the labels through length
	//VectorXf L2;
	//initializeLabelsFromLength(L2);
	//showFeatures(outDir+"/Labels_0_length", L2);

	//// average different initilization results
	//__labels = 1.0/3*(__labels+L1+L2);
	//showFeatures(outDir+"/Labels_0", __labels);
	//writeMatrixToTXT(outDir+"/labels_0.txt", __labels);

	//// calculate the affinities between features
	//affinities();
	////showAffinities(outDir+"/Affinities");

	//// update each frame indenpendantly
	//updateEachFrame();
	//showFeatures(outDir+"/Labels_1", __labels);	
	//writeMatrixToTXT(outDir+"/labels_1.txt", __labels);
	//__labels = readMatrixFromTXT(outDir+"/labels_1.txt");

	//// update all frames
	//updateAllFrames();
	//showFeatures(outDir+"/Labels_2", __labels);
	//writeMatrixToTXT(outDir+"/labels_2.txt", __labels);
	//__labels = readMatrixFromTXT(outDir+"/labels_2.txt");

	// sparse-to-dense propagation
	//myPropagation(outDir+"/Mattes_1");

	vector<string> mt_files, mt_names;
	//ScanDirectory(outDir+"/Mattes_1", "optimal.png", mt_files, mt_names);

	// refine the mattes temporally 
	//refineMattes(outDir+"/Mattes_2", mt_files);

	// refresh the homography matries
	//refreshHomosAfterLabeling();
	//showMatches(outDir+"/Matches_2", false);

	//ScanDirectory(outDir+"/Mattes_2", "binary.png", mt_files, mt_names);

	int ksize = 31;
	char maskdir[512], filename[512];
	Mat elem = getStructuringElement(MORPH_RECT, Size(abs(ksize),abs(ksize))), changed;
	if (ksize>0)
		sprintf(maskdir, "%s/Dilate%02d", outDir.c_str(), abs(ksize));
	else
		sprintf(maskdir, "%s/Erode%02d", outDir.c_str(), abs(ksize));
	_mkdir(maskdir);
	for (int i = 0; i < __masks.size(); i++)
	{
		if (ksize>0)
			morphologyEx(__masks[i], changed, MORPH_DILATE, elem);
		else
			morphologyEx(__masks[i], changed, MORPH_ERODE, elem);
		char filename[512];
		sprintf(filename, "%s/%s.png", maskdir, __imNames[i].c_str());
		imwrite(filename, changed);
	}

	ScanDirectory(maskdir, "png", mt_files, mt_names);

	vector<string> im_files, im_names;
	ScanDirectory(__indir, "origin.png", im_files, im_names);

	// composite the final image
	vector<detail::ImageFeatures> features;
	vector<detail::MatchesInfo> matches;
	Composite mix(im_files, mt_files, /*features, matches);*/__features, __matches);
	vector<int> selects;
	//selects.push_back(0);
	mix.compose(outDir+"/pano.png", outDir+"/mask.png",selects);
}