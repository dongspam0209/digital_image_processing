#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더

using namespace cv;
using namespace std;

void exCvMeansShift() {
	Mat img = imread("C:\\images\\fruits.png");
	if (img.empty()) exit(-1);
	cout << "-----exCvMeanShift()-----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src_opencv", img);
	//imwrite("exCvMeanShift_src.jpg", img);

	pyrMeanShiftFiltering(img, img, 8, 16);

	imshow("Dst_opencv", img);
	waitKey(0);

	
	//imwrite("exCvMeanShift_dst.jpg",img);

}

class Point5D {
	// Mean shift 구현을 위한 전용 포인트(픽셀) 클래스
public:
	float x, y, l, u, v; //포인트의 좌표와 LUV 값

	Point5D();
	~Point5D();

	void accumPt(Point5D); //포인트 축적
	void copyPt(Point5D); //포인트 복사
	float getColorDist(Point5D); //색상 거리 계산
	float getSpatialDist(Point5D);
	void scalePt(float);
	void setPt(float, float, float, float, float);
	void printPt();

};
Point5D::Point5D() {
	x = -1;
	y = -1;
}
Point5D::~Point5D() {

}

void Point5D::accumPt(Point5D Pt) {
	x += Pt.x;
	y += Pt.y;
	l += Pt.l;
	u += Pt.u;
	v += Pt.v;
}

void Point5D::copyPt(Point5D Pt) {
	x = Pt.x;
	y = Pt.y;
	l = Pt.l;
	u = Pt.u;
	v = Pt.v;
}
float Point5D::getColorDist(Point5D Pt) {
	return sqrt(pow(l - Pt.l, 2) +
				pow(u - Pt.u, 2) +
				pow(v - Pt.v, 2));
}

float Point5D::getSpatialDist(Point5D Pt) {
	return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}//Euclidean distance

void Point5D::scalePt(float scale) {
	x *= scale;
	y *= scale;
	l *= scale;
	u *= scale;
	v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb) {
	x = px;
	y = py;
	l = pl;
	u = pa;
	v = pb;
}

void Point5D::printPt() {
	cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

class MeanShift
{
public:
	float bw_spatial = 8; //spatial bandwidth
	float bw_color = 16; //color bandwidth
	float min_shift_color = 0.1; //최소 컬러변화
	float min_shift_spatial = 0.1; //최소 위치변화
	int max_steps = 10;
	vector<Mat> img_split;
	MeanShift(float, float, float, float, int);
	void doFiltering(Mat&);
};

MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms)
{
	//생성자
	bw_spatial = bs;
	bw_color = bc;
	max_steps = ms;
	min_shift_color = msc;
	min_shift_spatial = mss;
}

void MeanShift::doFiltering(Mat& img) {
	int height = img.rows;
	int width = img.cols;
	split(img, img_split);

	Point5D pt, pt_prev, pt_cur, pt_sum;

	int pad_left, pad_right, pad_top, pad_bottom;
	size_t n_pt, step;

	for (int row = 0; row < height; row++)
	{
		for (int col = 0;col < width;col++) {
			pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
			pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
			pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
			pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;

			// 현재픽셀 세팅
			pt_cur.setPt(row, col,
				(float)img_split[0].at<uchar>(row, col),
				(float)img_split[1].at<uchar>(row, col),
				(float)img_split[2].at<uchar>(row, col));
			// 주변픽셀 세팅
			step = 0;
			do {
				pt_prev.copyPt(pt_cur);
				pt_sum.setPt(0, 0, 0, 0, 0);
				n_pt = 0;
				for (int hx = pad_top; hx < pad_bottom; hx++)
				{
					for (int hy = pad_left; hy < pad_right; hy++)
					{
						pt.setPt(hx, hy,
							(float)img_split[0].at<uchar>(hx, hy),
							(float)img_split[1].at<uchar>(hx, hy),
							(float)img_split[2].at<uchar>(hx, hy));
						// color bandwidth 안에서 축적
						if (pt.getColorDist(pt_cur) < bw_color)
						{
							pt_sum.accumPt(pt);
							n_pt++;
						}
					}
				}
				pt_sum.scalePt(1.0 / n_pt);
				pt_cur.copyPt(pt_sum);
				step++;
			} while ((pt_cur.getColorDist(pt_prev) > min_shift_color) &&
				(pt_cur.getSpatialDist(pt_prev) > min_shift_spatial) &&
				(step < max_steps));
				//변화량 최소조건을 만족할 때 까지 반복
				//최대 반복횟수 조건도 포함
			img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
		}
	}
}
void exMyMeanShift() {
	Mat img = imread("C:\\images\\fruits.png");
	if (img.empty())exit(-1);
	cout << "-----exMyMeanShift() -----" << endl;

	resize(img, img, Size(256, 256), 0, 0, CV_INTER_AREA);
	imshow("Src_mymean", img);
	//imwrite("exMyMeanShift_src.jpg",img);

	cvtColor(img, img, CV_RGB2Luv);

	MeanShift MSProc(8, 16, 0.1, 0.1, 10);
	MSProc.doFiltering(img);

	cvtColor(img, img, CV_Luv2RGB);

	imshow("Dst_mymean", img);
	waitKey(0);
	//imwrite("exMyMeanShift_dst.jpg",img);


}


int main() {
	exMyMeanShift();
	exCvMeansShift();

	destroyWindow("Dst_opencv");
	destroyWindow("Src_opencv");

	destroyWindow("Src_mymean");
	destroyWindow("Dst_mymean");

	//////#1 Grabcut 잘 되지않는 이미지
	Mat result_1, bg_model_1, fg_model_1;
	Mat img_1;//src_img
	img_1 = imread("C:\\images\\srcimg.jpg", 1);
	
	Rect rect(Point(160, 150), Point(460, 719));
	grabCut(img_1, result_1,
		rect, bg_model_1, fg_model_1,
		5,
		GC_INIT_WITH_RECT);

	compare(result_1, GC_PR_FGD, result_1, CMP_EQ);
	//GC_PR_FGD: Grabcut class forground 픽셀
	//CMP_EQ: compare 옵션(equal)

	Mat mask(img_1.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img_1.copyTo(mask, result_1);

	imshow("grabcut_srcimg_1", img_1);
	waitKey(0);
	imshow("grabcut_result_1", mask);
	waitKey(0);
	imshow("grabcut_mask_1", result_1);
	waitKey(0);
	
	destroyWindow("grabcut_srcimg_1");
	destroyWindow("grabcut_result_1");
	destroyWindow("grabcut_mask_1");

	//#2 Grabcut 잘 되는 이미지_1
	Mat result_2, bg_model_2, fg_model_2;
	Mat img_2; //src_img
	img_2 = imread("C:\\images\\grabcutex.jpg",1);
	Rect rect_2(Point(60, 65), Point(195, 206));
	grabCut(img_2, result_2,
		rect_2, bg_model_2, fg_model_2,
		5,
		GC_INIT_WITH_RECT);

	compare(result_2, GC_PR_FGD, result_2, CMP_EQ);
	Mat mask_2(img_2.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img_2.copyTo(mask_2, result_2);

	imshow("grabcut_srcimg_2", img_2);
	waitKey(0);
	imshow("grabcut_result_2", mask_2);
	waitKey(0);
	imshow("grabcut_mask_2", result_2);
	waitKey(0);

	destroyWindow("grabcut_srcimg_2");
	destroyWindow("grabcut_result_2");
	destroyWindow("grabcut_mask_2");

	///#3 Grabcut 잘 되는 이미지_2
	Mat result_3, bg_model_3, fg_model_3;
	Mat img_3;
	img_3 = imread("C:\\images\\diving.jpg", 1);
	Rect rect_3(Point(30, 40),Point(490, 300));
	grabCut(img_3, result_3,
		rect_3, bg_model_3, fg_model_3,
		5,
		GC_INIT_WITH_RECT);

	compare(result_3, GC_PR_FGD, result_3, CMP_EQ);
	Mat mask_3(img_3.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	img_3.copyTo(mask_3, result_3);

	imshow("grabcut_srcimg_3", img_3);
	waitKey(0);
	imshow("grabcut_result_3", mask_3);
	waitKey(0);
	imshow("grabcut_mask_3", result_3);
	waitKey(0);

	destroyWindow("grabcut_srcimg_3");
	destroyWindow("grabcut_result_3");
	destroyWindow("grabcut_mask_3");

}