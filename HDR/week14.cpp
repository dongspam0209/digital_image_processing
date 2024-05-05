#include <iostream>
#include "opencv2/core/core.hpp" // matclass와 각종 데이터구조 및 산술 루틴 포함 헤더
#include "opencv2/highgui/highgui.hpp" //gui와 관련된 요소를 포함하는 헤더 ( imshow)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat getHistogram(const Mat& image) {
	Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY); // grayscale 이미지로 변환한다.

	Mat histogram;
	const int* channel_numbers = { 0 };
	float channel_range[] = { 0.0,255.0 };
	const float* channel_ranges = channel_range;
	int number_bins = 255;

	calcHist(&grayImage, 1, channel_numbers, Mat(), histogram, 1, &number_bins, &channel_ranges);

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

	// Display the histogram
	return histImage;
}


void readImagesAndTimes(vector<Mat>& images, vector<float>& times) {
	int numImages = 4;
	static const float timesArray[] = { 1 / 30.0f, 0.25f, 2.5f, 15.0f };
	times.assign(timesArray, timesArray + numImages);
	static const char* filenames[] = { "C:\\images\\week14_1.jpg", "C:\\images\\week14_2.jpg", "C:\\images\\week14_3.jpg", "C:\\images\\week14_4.jpg" };
	for (int i = 0; i < numImages; i++) {
		Mat im = imread(filenames[i]);
		images.push_back(im);
		imshow("histogram", getHistogram(im));
		waitKey(0);
	}
}


int main() {
	//영상, 노출시간 불러오기
	cout << "Reading images and exposure times .." << endl;
	vector<Mat> images;
	vector<float> times;
	readImagesAndTimes(images, times);
	cout << "finished" << endl;

	//영상 정렬
	cout << "Aligning images .. " << endl;
	Ptr<AlignMTB>alignMTB = createAlignMTB();
	alignMTB->process(images, images);

	// camera response function 복원
	cout << "Calculating Camera Response Function ..." << endl;
	Mat responseDebevec;
	Ptr<CalibrateDebevec> calibrateDebevec = createCalibrateDebevec();
	calibrateDebevec->process(images, responseDebevec, times);
	
	//24bit 표현 범위로 이미지 병합
	cout << "Merging images into one HDR image ... " << endl;
	Mat hdrDebevec;
	Ptr<MergeDebevec> mergeDebevec = createMergeDebevec();
	mergeDebevec->process(images, hdrDebevec, times, responseDebevec);
	imwrite("hdrDebevec.hdr", hdrDebevec);
	cout << "saved hdrDebevec.hdr" << endl;


	//drago 톤맵
	cout << "Tonemaping using Drago's method ... ";
	Mat IdrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0f, 0.7f, 0.85f);
	tonemapDrago->process(hdrDebevec, IdrDrago);
	IdrDrago = 3 * IdrDrago;
	imwrite("Idr-Drago.jpg", IdrDrago * 255);
	cout << "saved Idr-Drago.jpg" << endl;
	imshow("histogram", getHistogram(IdrDrago*255));
	waitKey(0);

	//reinhard 톤맵
	cout << " Tonemaping using Reinhard's method";
	Mat IdrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5f, 0, 0, 0);
	tonemapReinhard->process(hdrDebevec, IdrReinhard);
	imwrite("Idr-Reinhard.jpg", IdrReinhard * 255);
	cout << "saved Idr-Reinhard.jpg" << endl;
	imshow("histogram", getHistogram(IdrReinhard*255));
	waitKey(0);

	//Mantiuk 톤맵
	cout << " Tonemaping using Mantiuk's method";
	Mat IdrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2f, 0.85f, 1.2f);
	tonemapMantiuk->process(hdrDebevec, IdrMantiuk);
	IdrMantiuk = 3 * IdrMantiuk;
	imwrite("Idr-Mantiuk.jpg", IdrMantiuk * 255);
	cout << "saved Idr-Mantiuk.jpg" << endl;
	imshow("histogram", getHistogram(IdrMantiuk*255));
	waitKey(0);
}

