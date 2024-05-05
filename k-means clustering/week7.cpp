#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더

using namespace cv;
using namespace std;

Mat MyBgr2Hsv(Mat src_img) {
	double b, g, r, h, s, v;
	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++)
	{
		for (int x = 0; x < src_img.cols; x++)
		{
			b = (double)src_img.at<Vec3b>(y, x)[0];
			g = (double)src_img.at<Vec3b>(y, x)[1];
			r = (double)src_img.at<Vec3b>(y, x)[2];

			vector<double>vec = { r,g,b };							//R,G,B 성분을 vector에 저장한다.
			double min = *min_element(vec.begin(), vec.end());		//vector에서 최소값을 대입
			double max = *max_element(vec.begin(), vec.end());		//vecotr에서 최댓값을 대입

			v = max;												
			if (v == 0) { s = 0; }									//v=0 이면 s=0이다.
			else { s = (max - min) / max; }							//s=(max-min)/max이다.

			if (max == r) { h = 0 + (g - b) / (max - min); }		//0+(G-B)/(max-min)
			else if (max == g) { h = 2 + (b - r) / (max - min); }	//2+(B-R)/(max-min)
			else { h = 4 + (r - g) / (max - min); }					//4+(R-G)/(max-min)
			h *= 60;												//마지막으로 60 곱해야한다.

			if (h < 0) { h += 360; }		// h<0 이면 h+360
			h /= 2;
			s *= 255;
			h = h > 255.0 ? 255.0 : h < 0 ? 0 : h;
			s = s > 255.0 ? 255.0 : s < 0 ? 0 : s;
			v = v > 255.0 ? 255.0 : v < 0 ? 0 : v;

			dst_img.at<Vec3b>(y, x)[0] = (uchar)h;
			dst_img.at<Vec3b>(y, x)[1] = (uchar)s;
			dst_img.at<Vec3b>(y, x)[2] = (uchar)v;
		}

	}
	return dst_img;
}

void CvColorModels(Mat bgr_img) {
	Mat gray_img, rgb_img, hsv_img, yuv_img, xyz_img;

	cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
	cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
	cvtColor(bgr_img, hsv_img, cv::COLOR_BGR2HSV);
	cvtColor(bgr_img, yuv_img, cv::COLOR_BGR2YCrCb);
	cvtColor(bgr_img, xyz_img, cv::COLOR_BGR2XYZ);

	Mat print_img;
	bgr_img.copyTo(print_img);
	cvtColor(gray_img, gray_img, cv::COLOR_GRAY2BGR);

	imshow("hsv_img using opencv", hsv_img);
	waitKey(0);
}

Mat myinRange(Mat hsv_img, Scalar lower, Scalar upper) {
	Mat mask = Mat::zeros(hsv_img.size(), CV_8UC3);
	//3channel 검정 배경 영상 생성
	double lower_h = lower[0];
	double lower_s = lower[1];
	double lower_v = lower[2];

	double upper_h = upper[0];
	double upper_s = upper[1];
	double upper_v = upper[2];

	double h, s, v;

	for (int y = 0; y < hsv_img.rows; y++)
	{
		for (int x = 0; x < hsv_img.cols;x++) {
			h = hsv_img.at<Vec3b>(y, x)[0];
			s = hsv_img.at<Vec3b>(y, x)[1];
			v = hsv_img.at<Vec3b>(y, x)[2];

			if (lower_h <= h && h <= upper_h
				&& lower_s <= s && s <= upper_s
				&& lower_v <= v && v <= upper_v) {
				mask.at<Vec3b>(y, x)[0] = 255;
				mask.at<Vec3b>(y, x)[1] = 255;
				mask.at<Vec3b>(y, x)[2] = 255;
			}
		}
	}
	return mask;
}

void printColor(Mat src_img) {
	Mat hsv_img;
	hsv_img = MyBgr2Hsv(src_img);

	double h;

	int red = 0, orange = 0, yellow = 0, green = 0, blue = 0, purple = 0;

	for (int y = 0; y < hsv_img.rows; y++)
	{
		for (int x = 0; x < hsv_img.cols;x++) {
			h = hsv_img.at<Vec3b>(y, x)[0];

			if ((0 < h && h <= 10) || (170 <= h && h <= 180)) { red++; }
			else if (11 < h && h <= 25) { orange++; }
			else if (26 < h && h <= 35) { yellow++; }
			else if (36 < h && h <= 77) { green++; }
			else if (78 < h && h <= 99) { blue++; }
			else if (125 < h && h <= 155) { purple++; }
		}
	}
	int max_count = max(max(max(max(max(red, orange), yellow), green), blue), purple);

	if (max_count = red) { cout << "Red" << endl; }
	else if (max_count = orange) { cout << "Orange" << endl; }
	else if (max_count = yellow) { cout << "Yellow" << endl; }
	else if (max_count = green) { cout << "Green" << endl; }
	else if (max_count = blue) { cout << "Blue" << endl; }
	else if (max_count = purple) { cout << "Purple" << endl; }

}
Mat CvKMeans(Mat src_img, int k) {
	//2차원 영상 -> 1차원 벡터
	Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
	for (int y = 0; y < src_img.rows; y++)
	{
		for (int x = 0;x < src_img.cols;x++) {
			if (src_img.channels() == 3)
			{
				for (int z = 0;z < src_img.channels();z++) {
					samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
				}
			}
			else
			{
				samples.at<float>(y + x * src_img.rows) = (float)src_img.at<uchar>(y, x);
			}
		}
	}

	// OpenCv K-means 수행

	Mat labels;
	Mat centers;
	int attempts = 5;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);

	Mat dst_img(src_img.size(), src_img.type());
	for (int y = 0; y < src_img.rows; y++)
	{
		for (int x = 0;x < src_img.cols;x++) {
			int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
			if (src_img.channels()==3)
			{
				for (int z = 0;z < src_img.channels();z++) {
					dst_img.at<Vec3b>(y, x)[z] = (uchar)centers.at<float>(cluster_idx, z);
				}
			}
			else
			{
				dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
			}
		}
	}
	return dst_img;
}

int main() {
	Mat src_img,src_img_2, dst_img, mask, dst_img_2 , dst_img_3,dst_img_4;
	src_img = imread("C:\\images\\apple.png", 1);
	src_img_2 = imread("C:\\images\\beach.jpg", 1);
	dst_img = MyBgr2Hsv(src_img);
	imshow("hsv_img", dst_img);
	CvColorModels(src_img);
	waitKey(0);
	destroyWindow("hsv_img");
	destroyWindow("hsv_img using opencv");
	printColor(src_img);
	mask = myinRange(dst_img, cv::Scalar(170, 50, 50), cv::Scalar(180, 255, 255));
	imshow("mask", mask);
	waitKey(0);
	bitwise_and(src_img, mask, dst_img_2);
	imshow("dst_img", dst_img_2);
	waitKey(0);
	dst_img_4 = CvKMeans(src_img_2, 5);
	imshow("dst_img_4", dst_img_4);
	waitKey(0);
	dst_img_3=CvKMeans(src_img, 5);
	imshow("dst_img_3", dst_img_3);
	waitKey(0);
}