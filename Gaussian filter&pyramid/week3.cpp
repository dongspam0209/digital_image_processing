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


int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	for (int j = -1; j <= 1; j++)
	{
		for (int i = -1;i <= 1;i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) {
		return sum / sumKernel; //���� 1�� ����ȭ�ǵ��� �ؼ� ������ ��� ��ȭ�� �����Ѵ�.
	}
	else
	{
		return sum;
	}
}


int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height, int ch, int p = 0) {
	int sum = 0;
	int sumKernel = 0;

	for (int j = -4; j <= 4; j++)
	{
		for (int i = -4;i <= 4;i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				//���� �����ڸ����� ���� ���� ȭ�Ҹ� ���� �ʵ��� �ϴ� ���ǹ�
				if (ch == 1)
					//1ä���϶�
					sum += arr[(y + j) * width + (x + j)] * kernel[i + 4][j + 4];
				else if (ch == 3)
					//3ä���϶�
					sum += arr[(y + j) * width * 3 + (x + i) * 3 + p] * kernel[i + 4][j + 4];
				sumKernel += kernel[i + 4][j + 4];
			}
		}
	}
	if (sumKernel != 0) {
		return sum / sumKernel; //���� 1�� ����ȭ�ǵ��� �ؼ� ������ ��� ��ȭ�� �����Ѵ�.
	}
	else
	{
		return sum;
	}
}


Mat myGaussianFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1,3,5,7,10,7,5,3,1,
						3,13,16,19,22,19,16,13,3,
						5,16,25,28,31,28,25,16,5,
						7,19,28,34,37,34,28,19,7,
						10,22,31,37,40,37,31,22,10,
						7,19,28,34,37,34,28,19,7,
						5,16,25,28,31,28,25,16,5,
						3,13,16,19,22,19,16,13,3,
						1,3,5,7,10,7,5,3,1 };

	Mat dstImg(srcImg.size(), CV_8UC1); //1ä�ο�

	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	
		for (int y = 0;y < height;y++) {
			for (int x = 0; x < width; x++)
			{
				dstData[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height, 1);
			}
		}
	
		return dstImg;

}

Mat myGaussianColorFilter(Mat srcImg) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1,3,5,7,10,7,5,3,1,
						3,13,16,19,22,19,16,13,3,
						5,16,25,28,31,28,25,16,5,
						7,19,28,34,37,34,28,19,7,
						10,22,31,37,40,37,31,22,10,
						7,19,28,34,37,34,28,19,7,
						5,16,25,28,31,28,25,16,5,
						3,13,16,19,22,19,16,13,3,
						1,3,5,7,10,7,5,3,1 };
	
	Mat dstImg(srcImg.size(), CV_8UC3); //3ä�ο�

	uchar* srcData = srcImg.data;

	uchar* dstData = dstImg.data;


	for (int y = 0;y < height;y++) {
		for (int x = 0; x < width; x++)
		{
			dstData[y * width * 3 + x * 3] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 0);
			dstData[y * width * 3 + x * 3 + 1] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 1);
			dstData[y * width * 3 + x * 3 + 2] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 2);
		}
	}

	return dstImg;
}

Mat mySobelFilter(Mat srcImg, int sel) {
	int kernelX[3][3] = {-2,-2,0,
						 -2,0,2,
						 0,2,2 };

	int kernelY[3][3] = { 0,-2,-2,
						 2,0,-2,
						 2,2,0 };



	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	//45�� Ŀ��
	if (sel == 0)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dstData[y * width + x] = abs(myKernelConv3x3(srcData, kernelX, x, y, width, height));
			}
		}
	}
	//135�� Ŀ��
	if (sel == 1) {
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dstData[y * width + x] = abs(myKernelConv3x3(srcData, kernelY, x, y, width, height));
			}
		}
	}
	//45�� 135�� Ŀ�� �����Ѱ��� ��ģ��.
	if (sel == 2)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
					abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
			}
		}
	}
	return dstImg;

}

Mat mySampling_color(Mat srcImg) {
	int width = srcImg.cols / 2;
	int height = srcImg.rows / 2;
	Mat dstImg(height, width, CV_8UC3);

	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width;x++) {
			dstData[y * width * 3 + x * 3] = srcData[(y * 2) * (width * 6) + (x * 6)];
			dstData[y * width * 3 + x * 3 + 1] = srcData[(y * 2) * (width * 6) + (x * 6) + 1];
			dstData[y * width * 3 + x * 3 + 2] = srcData[(y * 2) * (width * 6) + (x * 6) + 2];
		}
	}
	return dstImg;

}


vector<Mat> myGaussianPyramid_color(Mat srcImg) {
	vector<Mat> Vec;

	Vec.push_back(srcImg);
	for (int i = 0; i < 4; i++)
	{	srcImg = mySampling_color(srcImg);
		srcImg = myGaussianColorFilter(srcImg);

		Vec.push_back(srcImg);
	}
	return Vec;

}

vector<Mat> myLaplacianPyramid_color(Mat srcImg) {
	vector<Mat> Vec;

	for (int i = 0; i < 4; i++) 
	{
		if (i != 3) {
			Mat highImg = srcImg;

			srcImg = mySampling_color(srcImg);
			srcImg = myGaussianColorFilter(srcImg);
			Mat lowImg = srcImg;

			resize(lowImg, lowImg, highImg.size());
			//�۾��� ������ ����� ������ ũ��� Ȯ��
			Vec.push_back(highImg - lowImg + 128);
			//�� ������ ���Ϳ� ����
			//�� ������ �����÷ο츦 �����ϱ� ���ؼ� 128 ���Ѵ�.
		}
		else
		{
			Vec.push_back(srcImg);
		}
	}
	return Vec;
}

void SpreadSalts(Mat img, int num) {

	for (int n = 0; n < num; n++)
	{
		int x = rand() % img.cols;
		int y = rand() % img.rows;

		if (img.channels() == 1) {
			img.at<uchar>(y, x) = 255;
		}
		else {
			img.at<Vec3b>(y, x)[0] = 255;
			img.at<Vec3b>(y, x)[1] = 255;
			img.at<Vec3b>(y, x)[2] = 255;
		}
	}
}


int main() {
	Mat src_img, dst_img;

	src_img = imread("C:\\images\\gear.jpg", 0);
	Mat src_his = Gethistogram(src_img);

	dst_img = myGaussianFilter(src_img);
	Mat dst_his = Gethistogram(dst_img);
	
	//1) 9x9 ���콺 ����
	imshow("src_img", src_img);
	imshow("dst_img", dst_img);
	waitKey(0);

	//2) 9x9 ���콺 ���� ����� ������׷�
	imshow("src_hist", src_his);
	imshow("dst_hist", dst_his);
	waitKey(0);

	destroyWindow("src_img");
	destroyWindow("dst_img");
	destroyWindow("src_hist");
	destroyWindow("dst_hist");
	waitKey(0);


	//3) Salt and pepper noise �ְ� 9x9 ���콺 ���� ����
	SpreadSalts(src_img, 1000);
	imshow("src_img_salts", src_img);
	waitKey(0);
	dst_img = myGaussianFilter(src_img);
	imshow("dst_img_salts", dst_img);
	waitKey(0);

	destroyWindow("src_img_salts");
	destroyWindow("dst_img_salts");
	waitKey(0);

	//4) 45���� 135���� �밢 ������ �����ϴ� Sobel filter ����
	src_img = imread("C:\\images\\gear.jpg", 0);
	dst_img = mySobelFilter(src_img,0);
	Mat dst_img2 = mySobelFilter(src_img,1);
	Mat dst_img3 = mySobelFilter(src_img,2);

	imshow("45", dst_img);
	imshow("135", dst_img2);
	imshow("45+135", dst_img3);
	waitKey(0);

	destroyWindow("45");
	destroyWindow("135");
	destroyWindow("45+135");
	waitKey(0);



	//5) �÷����� ���� ���콺 �Ƕ�̵� ����
	src_img = imread("C:\\images\\gear.jpg", 1);
	vector<Mat>Vec_Gau = myGaussianPyramid_color(src_img);
	for (int i = 0; i < Vec_Gau.size(); i++)
	{
		imshow("Gaussian pyramid", Vec_Gau[i]);
		waitKey(0);
	}

	destroyWindow("Gaussian pyramid");

	//6) �÷����� ���� ���ö�þ� �Ƕ�̵� ����, ���� ���� ���
	src_img = imread("C:\\images\\gear.jpg", 1);
	vector<Mat>Vec_Lap = myLaplacianPyramid_color(src_img);

	reverse(Vec_Lap.begin(), Vec_Lap.end());
	//���� ������� ó���ϱ� ���ؼ� vector �� ������ �ݴ��
	for (int i = 0; i < Vec_Lap.size(); i++)
	{
		imshow("Laplacian pyramid", Vec_Lap[i]);
		waitKey(0);
	}
	destroyWindow("Laplacian pyramid");

	for (int i = 0; i < Vec_Lap.size(); i++)
	{
		if (i==0)
		{
			dst_img = Vec_Lap[i];
			//���� ���� ������ �� ������ �ƴϱ� ������ �ٷ� �ҷ��´�.
		}
		else
		{
			resize(dst_img, dst_img, Vec_Lap[i].size());
			dst_img = dst_img + Vec_Lap[i] - 128;
			// �� ������ �ٽ� ���ؼ� ū ������ �����Ѵ�.
			// �����÷ο� ���������� ������ 128�� �ٽ� ����.
		}
		string fname = "lap_pyr" + to_string(i) + ".png";
		imwrite(fname, dst_img);
		imshow("Laplacian recovery", dst_img);
		waitKey(0);
		destroyWindow("Laplacian recovery");
	}

}