#include <iostream>
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
using namespace cv;
using namespace std;


Mat Gethistogram(Mat& src) {

	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0,255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	calcHist(&src, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / number_bins);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < number_bins; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(histogram.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(histogram.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}


void SpreadSalts_B(Mat img, int num) {

	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols; //x�� �̹��� �� ���� ���� 
		int y = rand() % img.rows; //y�� �̹��� ���� ���� ����
		/*
		�������� ������ ���� ���� �� �����Ƿ� ������ ��ġ��
		�̹����� ũ�⸦ ����� �ʵ��� �����ϴ� ������ ��
		*/
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			img.at<Vec3b>(y, x)[0] = 255; //Blue ä�� ����
			img.at<Vec3b>(y, x)[1] = 0;
			img.at<Vec3b>(y, x)[2] = 0;
		}
	}
}
void SpreadSalts_G(Mat img, int num) {

	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;
		/*
		�������� ������ ���� ���� �� �����Ƿ� ������ ��ġ��
		�̹����� ũ�⸦ ����� �ʵ��� �����ϴ� ������ ��
		*/

		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[1] = 255; //Green ä�� ����
			img.at<Vec3b>(y, x)[2] = 0;
		}
	}


}
void SpreadSalts_R(Mat img, int num) {

	for (int n = 0; n < num; n++) {
		int x = rand() % img.cols;
		int y = rand() % img.rows;
		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			img.at<Vec3b>(y, x)[0] = 0;
			img.at<Vec3b>(y, x)[1] = 0;
			img.at<Vec3b>(y, x)[2] = 255; //Red ä�� ����
		}
	}
}

void count(Mat img) {

	int B = 0; int G = 0; int R = 0;
	for (int x = 0; x < img.cols; x++) {
		for (int y = 0; y < img.rows; y++) {
			if (img.at<Vec3b>(y, x)[0] == 255 && img.at<Vec3b>(y, x)[1] == 0 && img.at<Vec3b>(y, x)[2] == 0) {
				B++;  
			}//255,0,0 blue
			else if (img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[1] == 255 && img.at<Vec3b>(y, x)[2] == 0) {
				G++;
			} //0,255,0 green
			else if (img.at<Vec3b>(y, x)[0] == 0 && img.at<Vec3b>(y, x)[1] == 0 && img.at<Vec3b>(y, x)[2] == 255) {
				R++;
			} //0,0,255 red
		}
	}
	cout << "�Ķ����� ����: " << B << "�ʷ����� ����: " << G << "�������� ����" << R;
}

void Grad(Mat img) {

	Mat grad1 = img.clone();
	Mat grad2 = img.clone();

	//���� ������ ���� ��ο�
	for (int i = 0; i < grad1.rows; i++)
	{
		for (int j = 0;j < grad1.cols;j++) {
			int a = grad1.at<uchar>(i, j);
			int darkness = (grad1.rows-i)*255/grad1.rows;
	
			if (a-darkness > 0)
			{
				grad1.at<uchar>(i, j) = a - darkness;
			}
			else
			{
				grad1.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("#2_1", grad1);
	Mat grad1_his = Gethistogram(grad1);
	imshow("#2_1_his", grad1_his);
	waitKey(0);

	//�Ʒ��� ������ ���� ��ο�
	for (int i = 0; i < grad2.rows; i++)
	{
		for (int j = 0;j < grad2.cols;j++) {
			int a = grad2.at<uchar>(i, j);
			int darkness = i * 255/grad2.rows;


			if (a - darkness > 0)
			{
				grad2.at<uchar>(i, j) = a - darkness;
			}
			else
			{
				grad2.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("#2_2", grad2);
	Mat grad2_his=Gethistogram(grad2);
	imshow("#2_2_his", grad2_his);
	waitKey(0);
}

int main() {
	Mat src_img1 = imread("C:\\images\\img1.jpg", 1);
	Mat src_img2 = imread("C:\\images\\img1.jpg", 0);


	SpreadSalts_B(src_img1, 50);
	SpreadSalts_G(src_img1, 50);
	SpreadSalts_R(src_img1, 50);

	count(src_img1);
	imshow("#1", src_img1);
	waitKey(0);

	Mat image_his = Gethistogram(src_img2);
	imshow("#2.1", src_img2);
	waitKey(0);

	imshow("#2.1_his", image_his);
	waitKey(0);
	/*
	2�� ����
	*/
	Grad(src_img2);
	/*
	3�� ����
	*/
	Mat imgA = imread("C:\\images\\img3.jpg", 1); //���ּ� �׸�
	Mat imgB = imread("C:\\images\\img4.jpg", 1); //��� �׸�
	Mat logo = imread("C:\\images\\img5.jpg", 1); // spacex �ΰ�

	resize(imgB, imgB, Size(imgA.cols, imgA.rows));
	Mat dst1; //dst1: ���ּ��� ��ϱ׸� �ռ�
	subtract(imgA, imgB, dst1);

	Mat roi; //���ɿ���
	roi = dst1(Rect(Point(350, 300), Point(950, 500)));


	Mat gray_img;
	cvtColor(logo, gray_img, CV_BGR2GRAY);
	resize(gray_img, gray_img, Size(600, 200));

	Mat blackmask;
	Mat whitemask;
	Mat blackmaskc3;
	Mat whitemaskc3;


	threshold(gray_img, blackmask, 220, 255, THRESH_BINARY);
	bitwise_not(blackmask, whitemask); //whitemask�� �۾��� ��� ����� �������� sapceX �ΰ�


	cvtColor(whitemask, whitemaskc3, COLOR_GRAY2BGR);

	Mat roiblack;
	Mat roiwhite;

	roi.copyTo(roiblack, blackmask);
	roiwhite = whitemaskc3 + roiblack;

	resize(logo, logo, Size(roiwhite.cols, roiwhite.rows));

	logo.copyTo(roi, whitemask);



	imshow("#3", dst1);

	waitKey(0);

	destroyWindow("#3"); // �̹��� ���â ����

	return 0;
}