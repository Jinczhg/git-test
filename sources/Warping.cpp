
#include "Warping.h"

Warping::Warping(Size size1, Size size2, int xquads, int yquads, float lambda)
{
	_src_size = size1;
	_dst_size = size2;
	_xquads = xquads;//x方向_xquads个块，_xquads+1个顶点
	_yquads = yquads;
	_lambda = lambda;//平滑项的权重
	_quadWidth = float(_src_size.width)/_xquads;
	_quadHeight = float(_src_size.height)/_yquads;

	__vertices1.resize((_yquads+1)*(_xquads+1)*2);//(_yquads+1)*(_xquads+1)个顶点，每个顶点二维坐标(x,y)
	__vertices2.resize((_yquads+1)*(_xquads+1)*2);
	for (int i = 0; i <= _yquads; i++) {
		for (int j = 0; j <= _xquads; j++) {
			__vertices1[(i*(_xquads+1)+j)*2+0] = j*_quadWidth;//min(j*_quadWidth, _src_size.width-1);
			__vertices1[(i*(_xquads+1)+j)*2+1] = i*_quadHeight;//min(i*_quadHeight, _src_size.height-1);
		}
	}
	__homos.resize(_yquads*_xquads);
}

Warping::~Warping()
{

}

//双线性插值系数
Vec4f Warping::calcQuadCoordinates(const Point2f& v00, const Point2f& v01, const Point2f& v10, const Point2f& v11, const Point2f& pt)
{
	float a_x = v00.x - v01.x - v10.x + v11.x;
	float b_x = -v00.x + v01.x;
	float c_x = -v00.x + v10.x;
	float d_x = v00.x - pt.x;
            
	float a_y = v00.y - v01.y - v10.y + v11.y;
	float b_y = -v00.y + v01.y;
	float c_y = -v00.y + v10.y;
	float d_y = v00.y - pt.y;
            
	float bigA = -a_y*b_x + b_y*a_x;
	float bigB = -a_y*d_x - c_y*b_x + d_y*a_x +b_y*c_x;
	float bigC = -c_y*d_x + d_y*c_x;
            
	float tmp1 = -1, tmp2 = -1, tmp3 = -1, tmp4 = -1, k1 = -1, k2 = -1;
	if (bigB*bigB - 4*bigA*bigC >= 0.0)
	{
		if (abs(bigA) >= 0.000001)	{
			tmp1 = ( -bigB + sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
			tmp2 = ( -bigB - sqrt(bigB*bigB - 4*bigA*bigC) ) / ( 2*bigA );
		} else
			tmp1 = -bigC/bigB;
                
		if (tmp1 >= -0.999999 && tmp1 <= 1.000001) {
			tmp3 = -(b_y*tmp1 + d_y) / (a_y*tmp1 + c_y);
			tmp4 = -(b_x*tmp1 + d_x) / (a_x*tmp1 + c_x);
			if (tmp3 >= -0.999999 && tmp3 <= 1.000001) {
				k1 = tmp1;
				k2 = tmp3;
			} else if (tmp4 >= -0.999999 && tmp4 <= 1.000001) {
				k1 = tmp1;
				k2 = tmp4;
			}
		}

		if ( tmp2 >= -0.999999 && tmp2 <= 1.000001) {
			if (tmp3 >= -0.999999 && tmp3 <= 1.000001) {
				k1 = tmp2;
				k2 = tmp3;
			} else if (tmp4 >= -0.999999 && tmp4 <= 1.000001) {
				k1 = tmp2;
				k2 = tmp4;
			}
		}
	}
	
	Vec4f coef(-10,-10, -10, -10);
	if (k1>=-0.999999 && k1<=1.000001 && k2>=-0.999999 && k2<=1.000001)
	{            
		coef[0] = (1.0-k1)*(1.0-k2);
		coef[1] = k1*(1.0-k2);
		coef[2] = (1.0-k1)*k2;
		coef[3] = k1*k2;
	}
	return coef;
}

// v1
// v2 v3, v1 is the target vertex
//
Vec2f Warping::calcLocalCoordinates(const Point2f& v1, const Point2f& v2, const Point2f& v3)
{
	float d1 = sqrt((v1.x - v2.x)*(v1.x - v2.x) + (v1.y - v2.y)*(v1.y - v2.y));
	float d3 = sqrt((v3.x - v2.x)*(v3.x - v2.x) + (v3.y - v2.y)*(v3.y - v2.y));     
	float cosin = ((v1.x - v2.x)*(v3.x - v2.x)+(v1.y - v2.y)*(v3.y - v2.y))/(d1*d3);
	float u_dis = cosin*d1;
	float v_dis = sqrt(d1*d1 - u_dis*u_dis);
	float u = u_dis/d3;
	float v = v_dis/d3;
	return Vec2f(u, v);
}

// R90 = [0 1;-1 0] for v1
//                             v2 v3, v1 is the target
// R90 = [0 -1;1 0] for v3
//                             v2 v1, v1 is the target
void Warping::addSmoothConstraints(MatrixXf& A, int& cnt_eqns, int id1, int id2, int id3, float u, float v, float w, bool clockwise)
{
	int r = 1;
	if (clockwise)
		r = -1;
	// with respect to v1
	A(cnt_eqns, id1) = -w;
	A(cnt_eqns, id2) = (1-u)*w;
	A(cnt_eqns, id2+1) = -r*v*w;  
	A(cnt_eqns, id3) = u*w;
	A(cnt_eqns, id3+1) = r*v*w;
	cnt_eqns++;
	A(cnt_eqns, id1+1) = -w;
	A(cnt_eqns, id2+1) = (1-u)*w;
	A(cnt_eqns, id2) = r*v*w;
	A(cnt_eqns, id3+1) = u*w;
	A(cnt_eqns, id3) = -r*v*w;
	cnt_eqns++;
}

void Warping::solve(const detail::MatchesInfo& m, const detail::ImageFeatures& f1, const detail::ImageFeatures& f2, const Mat& weight/* = Mat()*/)
{
	int num_matched = m.matches.size();
	int yctrls = _yquads+1;//y轴顶点数
	int xctrls = _xquads+1;//x轴顶点数
	int num_variables = yctrls*xctrls*2;
	float ransac_reject = min(_dst_size.width, _dst_size.height)*0.05;

	// weight coefficient for each 90
	Mat W(_yquads*2, _xquads*2, CV_32F);
	if (weight.empty() || weight.size() != W.size())
		W.setTo(Scalar(_lambda,_lambda,_lambda));
	else
		W = weight;
	float *w = (float *)W.data;

	//// re-calculate the homography
	//vector<Point2f> src_points, dst_points;
	//for (int k = 0; k < num_matched; k++)
	//{
	//	const DMatch& t = m.matches[k];
	//	if (m.inliers_mask[k])
	//	{
	//		src_points.push_back(f1.keypoints[t.queryIdx].pt);
	//		dst_points.push_back(f2.keypoints[t.trainIdx].pt);
	//	}
	//}
	//Mat H = findHomography(src_points, dst_points, CV_RANSAC, ransac_reject);
	//Mat iH = H.inv(cv::DECOMP_SVD);
	//double* h = (double *)H.data;
	//double* ih = (double *)iH.data;

	// allocate the data
	int num_reserved = num_matched*2+_yquads*_xquads*16, cnt_eqns = 0;//等式最大数，以及等式计数器
	MatrixXf A(num_reserved, num_variables);
	VectorXf R(num_reserved);
	A.setZero();
	R.setZero();
	Point2f* V1= (Point2f *)__vertices1.data();
	Point2f* V2 = (Point2f *)__vertices2.data();

	// data term
	for (int k = 0; k < num_matched; k++)
	{		
		//if (!m.inliers_mask[k])
		//	continue;
		const DMatch& t = m.matches[k];
		Point2f p1 = f1.keypoints[t.queryIdx].pt;
		if (p1.x<0 || p1.x>_src_size.width-1 || p1.y<0 || p1.y>_src_size.height-1)
			continue;
		Point2f p2 = f2.keypoints[t.trainIdx].pt;
		//Point2f p2 = warpPoint(f2.keypoints[t.trainIdx].pt, ih);
		int gx = int(p1.x / _quadWidth);//特征点p1的x轴索引（以_quadWidth为步长）
		int gy = int(p1.y / _quadHeight);//特征点p1的y轴索引（以_quadHeight为步长）

		int a[4] = {gy*xctrls+gx, gy*xctrls+gx+1, (gy+1)*xctrls+gx, (gy+1)*xctrls+gx+1};//包含特征点p1的4个顶点的索引号
		Vec4f coef = calcQuadCoordinates(V1[a[0]], V1[a[1]], V1[a[2]], V1[a[3]], p1);
		if (coef[0] <=-10)
			continue;
		
		for (int c = 0; c < 4; c++)
		{
			A(cnt_eqns, a[c]*2) = coef[c];
			A(cnt_eqns+1, a[c]*2+1) = coef[c];
		}
		R(cnt_eqns) = p2.x;
		R(cnt_eqns+1) = p2.y;
		cnt_eqns += 2;
	}
	A = float(_yquads*_xquads*16)/cnt_eqns*A;
	R = float(_yquads*_xquads*16)/cnt_eqns*R;

	// shape-preserving term
	for (int i = 0; i < _yquads; i++)
	{
		for (int j = 0; j < _xquads; j++)
		{
			int a[4] = {i*xctrls+j, i*xctrls+j+1, (i+1)*xctrls+j, (i+1)*xctrls+j+1};
			int b[4] = {i*_xquads*4+j*2, i*_xquads*4+j*2+1, (2*i+1)*_xquads*2+j*2, (2*i+1)*_xquads*2+j*2+1};
			Vec2f uv;
			// 0 
			// 2 3, 0 is target
			uv = calcLocalCoordinates(V1[a[0]], V1[a[2]], V1[a[3]]);
			addSmoothConstraints(A, cnt_eqns, a[0]*2, a[2]*2, a[3]*2, uv[0], uv[1], w[b[2]], false);
			// 0
			// 2 3, 3 is target
			uv = calcLocalCoordinates(V1[a[3]], V1[a[2]], V1[a[0]]);
			addSmoothConstraints(A, cnt_eqns, a[3]*2, a[2]*2, a[0]*2, uv[0], uv[1], w[b[2]], true);
			// 0 1
			//   3, 0 is target
			uv = calcLocalCoordinates(V1[a[0]], V1[a[1]], V1[a[3]]);
			addSmoothConstraints(A, cnt_eqns, a[0]*2, a[1]*2, a[3]*2, uv[0], uv[1], w[b[1]], true);
			// 0 1
			//   3, 3 is target
			uv = calcLocalCoordinates(V1[a[3]], V1[a[1]], V1[a[0]]);
			addSmoothConstraints(A, cnt_eqns, a[3]*2, a[1]*2, a[0]*2, uv[0], uv[1], w[b[1]], false);
			// 0 1
			// 2  , 1 is target
			uv = calcLocalCoordinates(V1[a[1]], V1[a[0]], V1[a[2]]);
			addSmoothConstraints(A, cnt_eqns, a[1]*2, a[0]*2, a[2]*2, uv[0], uv[1], w[b[0]], false);
			// 0 1
			// 2  , 2 is target
			uv = calcLocalCoordinates(V1[a[2]], V1[a[0]], V1[a[1]]);
			addSmoothConstraints(A, cnt_eqns, a[2]*2, a[0]*2, a[1]*2, uv[0], uv[1], w[b[0]], true);
			//   1
			// 2 3, 1 is target
			uv = calcLocalCoordinates(V1[a[1]], V1[a[3]], V1[a[2]]);
			addSmoothConstraints(A, cnt_eqns, a[1]*2, a[3]*2, a[2]*2, uv[0], uv[1], w[b[3]], true);
			//   1
			// 2 3, 2 is target
			uv = calcLocalCoordinates(V1[a[2]], V1[a[3]], V1[a[1]]);
			addSmoothConstraints(A, cnt_eqns, a[2]*2, a[3]*2, a[1]*2, uv[0], uv[1], w[b[3]], false);
		}
	}
	if (A.rows() != cnt_eqns)
	{
		A.conservativeResize(cnt_eqns, NoChange);
		R.conservativeResize(cnt_eqns);
	}

	// sovle the vertices
	SparseMatrix<float> sA = A.sparseView();
	SparseQR<SparseMatrix<float>, COLAMDOrdering<int>> solver(sA);
	__vertices2 = solver.solve(R);
	//__vertices2 = A.colPivHouseholderQr().solve(R);

	//// calculate the homos 
	//for (int i = 0; i < yctrls; i++) 
	//{
	//	for (int j = 0; j < xctrls; j++) 
	//		V2[i*xctrls+j] = warpPoint(V2[i*xctrls+j], h);
	//}
	for (int i = 0; i < _yquads; i++)
	{
		for (int j = 0; j < _xquads; j++)
		{
			vector<Point2f> src_points, dst_points;
			src_points.push_back(V2[i*xctrls+j]);
			src_points.push_back(V2[i*xctrls+j+1]);
			src_points.push_back(V2[(i+1)*xctrls+j+1]);
			src_points.push_back(V2[(i+1)*xctrls+j]);
			dst_points.push_back(V1[i*xctrls+j]);
			dst_points.push_back(V1[i*xctrls+j+1]);
			dst_points.push_back(V1[(i+1)*xctrls+j+1]);
			dst_points.push_back(V1[(i+1)*xctrls+j]);
			Mat tmph = findHomography(src_points, dst_points);
			double *th = (double *)tmph.data;
			//if (abs(th[6])>0.01 || abs(th[7])>0.01)
			//	__homos[i*_xquads+j] = Mat();
			//else
				__homos[i*_xquads+j] = tmph;
		}
	}
}

// size of mask = _src_size, 1 for background, 0 for foreground
void Warping::solve(Ptr<detail::RotationWarper> warper, const Mat& K, const Mat& R, const detail::ImageFeatures& feature, const Mat& mask, float smoother)
{
	if (mask.size() != _src_size)
	{
		printf("size not matched\n");
		return;
	}
	float scalex = 1, scaley = 1;
	if (_src_size != feature.img_size)
	{
		scalex = float(_src_size.width)/feature.img_size.width;
		scaley = float(_src_size.height)/feature.img_size.height;
	}
	detail::ImageFeatures F1 = feature;
	detail::ImageFeatures F2 = feature;
	
	// warp the frame rectangle
	Rect roi = warper->warpRoi(mask.size(), K, R);
	Point2f corner = roi.tl();

	// concturct the scaled and warped features, and the matches
	int num_points = feature.keypoints.size();
	detail::MatchesInfo M;
	M.src_img_idx = F1.img_idx;
	M.dst_img_idx = F2.img_idx;
	M.num_inliers = num_points;
	float c;
	//ofstream fp("points.txt");
	for (int k = 0; k < num_points; k++)
	{
		Point2f& p0 = F1.keypoints[k].pt;
		Point2f& p1 = F2.keypoints[k].pt;
		p0.x *= scalex;
		p0.y *= scaley;
		p1 = warper->warpPoint(p0, K, R)-corner;

		//fp << p0.x << " " << p0.y << " " << p1.x << " " << p1.y << endl;
		M.matches.push_back(DMatch(k, k, -1, 0));
		bilinearInterpolate(&c, p0, mask.data, mask.cols, mask.rows, 1);
		if (c>EPSILON)
			M.inliers_mask.push_back(1);
		else
			M.inliers_mask.push_back(0);
	}
	M.confidence = num_points / (8+0.3*num_points);
	//fp.close();

	// solve the mesh warping
	Mat W(_yquads*2, _xquads*2, CV_32F, Scalar(_lambda));
	Point2f* V1= (Point2f *)__vertices1.data();

	for (int i = 0; i < _yquads; i++)
	{
		for (int j = 0; j < _xquads; j++)
		{
			Point2f tl = V1[i*(_xquads+1)+j];
			Point2f br = V1[(i+1)*(_xquads+1)+j+1];
			Rect box(tl, br);
			Mat quad(mask, box);
			float ratio = float(countNonZero(quad))/box.area();
			if (ratio<0.99)
			{
				float cf[4];
				//if (mask.at<uchar>(box.y, box.x)<=EPSILON)
					W.at<float>(i*2, j*2) = smoother;
				//else
				//	W.at<float>(i*2, j*2) = sqrt(_lambda*smoother);
				//if (mask.at<uchar>(box.y, box.x+box.width-1)<=EPSILON)
					W.at<float>(i*2, j*2+1) = smoother;
				//else
				//	W.at<float>(i*2, j*2+1) = sqrt(_lambda*smoother);
				//if (mask.at<uchar>(box.y+box.height-1, box.x)<=EPSILON)
					W.at<float>(i*2+1, j*2) = smoother;
				//else
				//	W.at<float>(i*2+1, j*2) = sqrt(_lambda*smoother);
				//if (mask.at<uchar>(box.y+box.height-1, box.x+box.width-1)<=EPSILON)
					W.at<float>(i*2+1, j*2+1) = smoother;
				//else
				//	W.at<float>(i*2+1, j*2+1) = sqrt(_lambda*smoother);
			}
			// the boundary shape should be similar
		}
	}
	////for (int i = 0; i < _yquads*2; i++)
	////{
	////	W.at<float>(i, 0) = less_ratio;
	////	W.at<float>(i, _xquads*2-1) = less_ratio;
	////}
	////for (int j = 0; j < _xquads*2; j++)
	////{
	////	W.at<float>(0, j) = less_ratio;
	////	W.at<float>(_yquads*2-1, j) = less_ratio;
	////}
	solve(M, F1, F2, W);
}

void Warping::warp(Mat& result, const Mat& img)
{	
	int width1 = _src_size.width, height1 = _src_size.height;
	int width2 = _dst_size.width, height2 = _dst_size.height;
	if (img.cols != width1 || img.rows != height1)
		return;
	if (result.cols != width2 || result.rows != height2 || result.type()!=img.type())
		result.create(height2, width2, img.type());
	result.setTo(Scalar(0,0,0));

	int channels = img.channels();	
	int yctrls = _yquads+1;//y轴方向顶点个数
	int xctrls = _xquads+1;//x轴方向顶点个数
	Point2f* V2 = (Point2f *)__vertices2.data();
	uchar* im_data = img.data;
	uchar* re_data = result.data;
	float *rgb = new float[channels];
	Point2f p1, p2;
	Mat ind(height2, width2, CV_8U, Scalar(0,0,0));
	uchar* ind_data = ind.data;

	for (int i = 0; i < _yquads; i++)
	{
		for (int j = 0; j < _xquads; j++)
		{
			if (__homos[i*_xquads+j].empty())
				continue;
			double *h = (double *)__homos[i*_xquads+j].data;
			vector<Point2f> vpts;//该quad四个顶点（float型坐标）
			vpts.push_back(V2[i*xctrls+j]);
			vpts.push_back(V2[i*xctrls+j+1]);
			vpts.push_back(V2[(i+1)*xctrls+j+1]);
			vpts.push_back(V2[(i+1)*xctrls+j]);
			//求四个顶点的包络（int型坐标）
			int minx = cvFloor(min(min(min(vpts[0].x, vpts[1].x), vpts[2].x), vpts[3].x));
			int miny = cvFloor(min(min(min(vpts[0].y, vpts[1].y), vpts[2].y), vpts[3].y));
			int maxx = cvCeil(max(max(max(vpts[0].x, vpts[1].x), vpts[2].x), vpts[3].x));
			int maxy = cvCeil(max(max(max(vpts[0].y, vpts[1].y), vpts[2].y), vpts[3].y));
			minx = min(max(minx, 0), width2-1);
			miny = min(max(miny, 0), height2-1);
			maxx = min(max(maxx, 0), width2-1);
			maxy = min(max(maxy, 0), height2-1);

			//result图中包络内像素点赋值
			for (int y = miny; y <= maxy; y++)//像素点h反变换到img上，双线性插值得到该像素点的RGB值
			{
				for (int x = minx; x <= maxx; x++)
				{
					if (ind_data[y*width2+x])//是否已经赋值
						continue;

					p2 = Point2f(x,y);
					if (pointPolygonTest(vpts, p2, false)<0)//检查p2是否在该quad外部
						continue;
					p1 = warpPoint(p2, h);
					if (p1.x<0 || p1.y<0 || p1.x>width1-1 || p1.y>height1-1)
						continue;

					ind_data[y*width2+x] = 255;
					bilinearInterpolate(rgb, p1, im_data, width1, height1, channels);
					for (int c = 0; c < channels; c++)
						re_data[(y*width2+x)*channels+c] = uchar(min(max(cvRound(rgb[c]), 0), 255));
				}
			}
		}
	}
	//for (int i = 0; i < _yquads; i++)
	//{
	//	for (int j = 0; j < _xquads; j++)
	//	{
	//		vector<Point2f> vpts;
	//		vpts.push_back(V2[i*xctrls+j]);
	//		vpts.push_back(V2[i*xctrls+j+1]);
	//		vpts.push_back(V2[(i+1)*xctrls+j]);
	//		circle(result, vpts[0], 3, Scalar(0,0,255), CV_FILLED, CV_AA);
	//		circle(result, vpts[1], 3, Scalar(0,0,255), CV_FILLED, CV_AA);
	//		circle(result, vpts[2], 3, Scalar(0,0,255), CV_FILLED, CV_AA);
	//		line(result, vpts[0], vpts[1], Scalar(255,255,255), 1, CV_AA);
	//		line(result, vpts[0], vpts[2], Scalar(255,255,255), 1, CV_AA);
	//	}
	//}
	delete[] rgb;
}