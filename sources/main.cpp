#include "stitching.h"


int main()
{
	string img0_path = "input/test3/IMG_0.png";
	string img1_path = "input/test3/IMG_1.png";
	string simg0_path = "input/test3/IMG_0_smooth.png";
	string simg1_path = "input/test3/IMG_1_smooth.png";
	//string fea_path = "input/test3/features.txt";
	//string mat_path = "input/test3/matches.txt";
	string out_path = "output/test3";

	_mkdir("output");
	_mkdir(out_path.c_str());
	Stitching s(img0_path, img1_path, simg0_path, simg1_path);
	s.Stitch(out_path);

	return 0;
}
