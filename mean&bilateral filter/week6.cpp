#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <ctime>
using namespace cv;
using namespace std;

//동적할당 + 포인터 기반 커널 컨볼루션
void myKernelConv(const Mat& src_img, Mat& dst_img, const Mat& kn) {
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);
	int wd = src_img.cols; int hg = src_img.rows;
	int kwd = kn.cols; int khg = kn.rows;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	float* kn_data = (float*)kn.data;
	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float wei, tmp, sum;

	// 픽셀 인덱싱(가장자리 제외)
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			sum = 0.f;
			// 커널 인덱싱
			for (int kc = -rad_w; kc <= rad_w; kc++) {
				for (int kr = -rad_h; kr <= rad_h; kr++) {
					wei = (float)kn_data[(kr + rad_h) * kwd + (kc + rad_w)];
					tmp += wei * (float)src_data[(r + kr) * wd + (c + kc)];
					sum += wei;
				}
			}
			if (sum != 0.f) tmp = abs(tmp) / sum; // 정구화 및 overflow 방지
			else tmp = abs(tmp);


			if (tmp > 255.f) tmp = 255.f;// overflow 방지

			dst_data[r * wd + c] = (uchar)tmp;
		}
	}
}

double gaussian2D(float c, float r, double sigma) {
	return exp(-(pow(c, 2) + pow(r, 2)) / (2 * pow(sigma, 2)))
		/ (2 * CV_PI * pow(sigma, 2));
}

void myGaussian(const Mat& src_img, Mat& dst_img, Size size) {
	// 커널 생성
	Mat kn = Mat::zeros(size, CV_32FC1);
	double sigma = 1.0;
	float* kn_data = (float*)kn.data;
	for (int c = 0; c < kn.cols; c++) {
		for (int r = 0; r < kn.rows; r++) {
			kn_data[r * kn.cols + c] =
				(float)gaussian2D((float)(c - kn.cols / 2),
				(float)(r - kn.rows / 2), sigma);
		}
	}
	myKernelConv(src_img, dst_img, kn);
}

//Median 필터를 이용한 Salt and pepper noise 제거 빈칸있음



void myMedian(const Mat& src_img, Mat& dst_img, const Size& kn_size) {

	// 임시 저장해서 테이블을 정렬
	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	int wd = src_img.cols;
	int hg = src_img.rows;
	int kwd = kn_size.width; int khg = kn_size.height;
	int rad_w = kwd / 2; int rad_h = khg / 2;

	uchar* src_data = (uchar*)src_img.data;
	uchar* dst_data = (uchar*)dst_img.data;

	float* table = new float[kwd * khg](); //커널 테이블 동적할당
	float tmp;

	//픽셀 인덱싱(가장자리 제외)
	for (int c = rad_w + 1; c < wd - rad_w; c++) {
		for (int r = rad_h + 1; r < hg - rad_h; r++) {
			tmp = 0.f;
			for (int kc = -rad_w; kc <= rad_w; kc++)
			{
				for (int kr = -rad_h;kr <= rad_h;kr++) {
					tmp = (float)src_data[(r + kr) * wd + (c + kc)];
					table[(kr + rad_h) * kwd + (kc + rad_w)] = tmp;
				}
			}
			sort(table, table + kwd * khg);
			dst_data[r * wd + c] = (uchar)table[(kwd * khg) / 2];

		}
	}
	delete table;
}


void doMedianEx(int kernel_size) {
	cout << "--- doMedianEx() --- \n" << endl;

	// 입력
	Mat src_img = imread("C:\\images\\salt_pepper2.png", 0);
	if (!src_img.data) printf("No image data \n");

	//Median 필터링 수행
	Mat dst_img;
#if USE_OPENCV
	medianBlur(src_img, dst_img, 3);
#else 
	myMedian(src_img, dst_img, Size(kernel_size, kernel_size));
#endif
	//출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doMedianEx()", result_img);
	waitKey(0);
}

// Bilateral Filter이용한 edge aware smooothing
double gaussian(float x, double sigma) {
	return exp(-(pow(x, 2)) / (2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

float distance(int x, int y, int i, int j) {
	return float(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

void bilateral(const Mat& src_img, Mat& dst_img, int c, int r, int diameter, double sig_r, double sig_s) {

	int radius = diameter / 2;

	double gr, gs, wei;
	double tmp = 0.;
	double sum = 0.;

	//커널 인덱실
	for (int kc = -radius; kc <= radius; kc++) {
		for (int kr = -radius; kr <= radius; kr++) {
			//range calc
			gr = gaussian((float)src_img.at<uchar>(c + kc, r + kr) - (float)src_img.at<uchar>(c, r), sig_r);
			//spatial calc
			gs = gaussian(distance(c, r, c + kc, r + kr), sig_s);
			wei = gr * gs;
			tmp += src_img.at<uchar>(c + kc, r + kr) * wei;
			sum += wei;
		}
	}
	dst_img.at<double>(c, r) = tmp / sum; //정규화
}

void myBilateral(const Mat& src_img, Mat& dst_img, int diameter, double sig_r, double sig_s) {
	//가우시안은 픽셀간 거리만 신경쓰면 bilateral은 강도에 따라서

	dst_img = Mat::zeros(src_img.size(), CV_8UC1);

	Mat guide_img = Mat::zeros(src_img.size(), CV_64F);
	int wh = src_img.cols; int hg = src_img.rows;
	int radius = diameter / 2;

	// 픽셀 인덱싱
	for (int c = radius + 1; c < hg - radius; c++) {
		for (int r = radius + 1; r < wh - radius; r++) {
			bilateral(src_img, guide_img, c, r, diameter, sig_r, sig_s);
			//화소별 bilateral 수행
		}
	}
	guide_img.convertTo(dst_img, CV_8UC1); //Mat type 변환
}

void doBilateralEx(double sig_r, double sig_s) {
	cout << "--- doBilateralEx() --- \n" << endl;

	//입력
	Mat src_img = imread("C:\\images\\rock.png", 0);
	Mat dst_img;
	if (!src_img.data) printf("No image Data\n");

	//bilateral 필터링 수행
#if USE_OPENCV
	bilateralFilter(src_img, dst_img, 5, 25.0, 50.0);
#else
	myBilateral(src_img, dst_img, 10, sig_r, sig_s);
#endif
	//출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doBilateralEx()", result_img);
	waitKey(0);
}
//Gradient 계산하는 필터 통한 edge detection
void doEdgeEx() {
	cout << "--- doEdgeEx() --- \n" << endl;
	//입력
	Mat src_img = imread("rock_png", 0);
	if (!src_img.data) printf("No image data \n");

	//잡음제거
	Mat blur_img;
	myGaussian(src_img, blur_img, Size(5, 5));

	//커널 생성
	float kn_data[] = { 1.f, 0.f, -1.f,
	   1.f, 0.f, -1.f,
	   1.f, 0.f, -1.f };
	Mat kn(Size(3, 3), CV_32FC1, kn_data);
	cout << "Edge Kernel: \n" << kn << endl;

	//커널 컨볼루션 수행
	Mat dst_img;
	myKernelConv(blur_img, dst_img, kn);

	//출력 
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doEdgeEx()", result_img);
	waitKey(0);
}

//Canny edge detection 알고리즘
void followEdges(int x, int y, Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	edges.at<float>(y, x) = 255;

	// <이웃 픽셀 인덱싱>

	for (int i = -1; i < 2; i++)
	{
		for (int j = -1;j < 2;j++) {
			if ((i != 0) && (j != 0) && (x + i >= 0) && (y + j >= 0) && (x + i < magnitude.cols) && (y + j < magnitude.rows)) {
				if ((magnitude.at<float>(y + j, x + i) > tLower) && (edges.at<float>(y + j, x + i) != 255))
				{
					followEdges(x + i, y + j, magnitude, tUpper, tLower, edges);
				}
			}
		}
	}
}

void edgeDetect(Mat& magnitude, int tUpper, int tLower, Mat& edges) {
	int rows = magnitude.rows;
	int cols = magnitude.cols;

	edges = Mat(magnitude.size(), CV_32F, 0.0);

	// <픽셀 인덱싱>
	for (int x = 0; x < cols; x++)
	{
		for (int y = 0;y < rows;y++) {
			if (magnitude.at<float>(y, x) >= tUpper)
			{
				followEdges(x, y, magnitude, tUpper, tLower, edges);
				//edge가 확실하면 이와 연결된 불확실한 edge를 탐색
			}
		}
	}
}

void nonMaximumSuppreession(Mat& magnitudeImage, Mat& directionImage) {
	Mat checkImage = Mat(magnitudeImage.rows, magnitudeImage.cols, CV_8U);

	MatIterator_<float>itMag = magnitudeImage.begin<float>();
	MatIterator_<float>itDirection = directionImage.begin<float>();
	MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char>();
	MatIterator_<float>itEnd = magnitudeImage.end<float>();

	for (; itMag != itEnd;++itDirection, ++itRet, ++itMag)
	{
		const Point pos = itRet.pos();
		float currentDirection = atan(*itDirection) * (180 / 3.142);
		while (currentDirection < 0)currentDirection += 180;

		*itDirection = currentDirection;

		if (currentDirection > 22.5 && currentDirection <= 67.5)
		{
			if (pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 67.5 && currentDirection <= 112.5)
		{
			if (pos.y > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else if (currentDirection > 112.5 && currentDirection <= 157.5)
		{
			if (pos.y > 0 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows - 1 && pos.x >0 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
		else
		{
			if (pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x - 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x + 1)) {
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
	}
}

void doCannyEx(int threshold_A, int threshold_B) {
	cout << "--- doCannyEx() --- \n" << endl;
	//입력
	Mat src_img = imread("C:\\images\\rock.png", 0);
	if (!src_img.data) printf("No image data\n");
	Mat dst_img;
#if USE_OPENCV
	//Canny edge 탐색 수행
	Canny(src_img, dst_img, 180, 240);
#else

	clock_t start, end;
	start = clock();

	//가우시안 필터 기반 노이즈 제거
	Mat blur_img;
	GaussianBlur(src_img, blur_img, Size(3, 3), 1.5);

	//소벨 엣지 detection
	Mat magX = Mat(src_img.rows, src_img.cols, CV_32F);
	Mat magY = Mat(src_img.rows, src_img.cols, CV_32F);
	Sobel(blur_img, magX, CV_32F, 1, 0, 3);
	Sobel(blur_img, magY, CV_32F, 0, 1, 3);

	Mat sum = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodX = Mat(src_img.rows, src_img.cols, CV_64F);
	Mat prodY = Mat(src_img.rows, src_img.cols, CV_64F);
	multiply(magX, magX, prodX);
	multiply(magY, magY, prodY);
	sum = prodX + prodY;
	sqrt(sum, sum);

	Mat magnitude = sum.clone();

	// Non-maximum suppression
	Mat slopes = Mat(src_img.rows, src_img.cols, CV_32F);
	divide(magY, magX, slopes);

	//gradient의 방향 계산
	nonMaximumSuppreession(magnitude, slopes);

	//Edge tracking by hysteresis
	edgeDetect(magnitude, threshold_A, threshold_B, dst_img);
	dst_img.convertTo(dst_img, CV_8UC1);

	end = clock();


#endif
	//출력
	Mat result_img;
	hconcat(src_img, dst_img, result_img);
	imshow("doCannyEx()", result_img);
	cout << threshold_A << ", " << threshold_B <<" "<< end - start << "ms\n";
	waitKey(0);
}

int main() {
	doMedianEx(3);
	doMedianEx(5);
	doBilateralEx(25.0,50.0);
	doBilateralEx(62.5,50.0);
	doBilateralEx(100.0,50.0);
	doBilateralEx(25.0,150.0);
	doBilateralEx(62.5,150.0);
	doBilateralEx(100.0,150.0);
	doBilateralEx(25.0,450.0);
	doBilateralEx(62.5,450.0);
	doBilateralEx(100.0,450.0);

	doCannyEx(50, 240);
	doCannyEx(100, 240);
	doCannyEx(150, 240);



}