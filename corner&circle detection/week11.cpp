#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

int cvBlobDetection(Mat img) {
	
	// <Set params>
	SimpleBlobDetector::Params params;
	params.minThreshold = 10;
	params.maxThreshold = 300;
	params.filterByArea = true;
	params.minArea = 100;
	params.maxArea = 10000;
	params.filterByCircularity = true;
	params.minCircularity = 0.5;
	params.filterByConvexity = true;
	params.minConvexity = 0.9;
	params.filterByInertia = true;
	params.minInertiaRatio = 0.01;

	// <Set blob detector>
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// <Detect blobs>
	std::vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);

	// <Draw blobs>
	Mat result;
	drawKeypoints(img, keypoints, result,
		Scalar(0, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	imshow("keypoints", result);
	waitKey(0);
	destroyWindow("keypoints");
	return keypoints.size();
}



Mat cvHarrisCorner(Mat img) {
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	// < Do Harris corner detection >
	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// < Get abs for Harris visualization >
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	// < Print corners >
	int thresh = 125;
	Mat result = img.clone();
	for (int y = 0; y < harr.rows; y+=1)
	{
		for (int x = 0;x < harr.cols;x+=1) {
			if ((int)harr.at<float>(y, x) > thresh)
				circle(result, Point(x, y), 7, Scalar(255, 0, 255), -1, 4, 0);
		}
	}
	imshow("Source image", img);
	imshow("Harris image", harr_abs);
	imshow("Target image", result);
	waitKey(0);
	destroyWindow("Source image");
	destroyWindow("Harris image");
	destroyWindow("Target image");
	return result;
}

Mat warpPers(Mat src) {
	Mat dst;
	Point2f src_p[4], dst_p[4];

	src_p[0] = Point2f(0, 0);
	src_p[1] = Point2f(1200, 0);
	src_p[2] = Point2f(0, 800);
	src_p[3] = Point2f(1200, 800);

	dst_p[0] = Point2f(0, 0);
	dst_p[1] = Point2f(1200, 0);
	dst_p[2] = Point2f(0, 800);
	dst_p[3] = Point2f(1000, 600);

	Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
	warpPerspective(src, dst, pers_mat, Size(1200, 800));
	return dst;
}


Mat cvFeatureSIFT(Mat img) {

	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);

	Ptr<cv::SiftFeatureDetector>detector = SiftFeatureDetector::create();
	std::vector<KeyPoint> keypoints;
	detector->detect(gray, keypoints);

	Mat result;
	drawKeypoints(img, keypoints, result);
	imwrite("sift_result.jpg", result);
	return result;
}

int main() {
	
	//#1
	Mat img_1 = imread("C:\\images\\coin.png", IMREAD_COLOR);
	cout << "Coins :" << cvBlobDetection(img_1) << endl;



	//#2
	Mat img_2 = imread("C:\\images\\3.png");
	Mat dst_2 = cvHarrisCorner(img_2);
	cout << cvBlobDetection(dst_2) << "각형" << endl;
	Mat img_3 = imread("C:\\images\\4.png");
	Mat dst_3 = cvHarrisCorner(img_3);
	cout << cvBlobDetection(dst_3) << "각형" << endl;
	Mat img_4 = imread("C:\\images\\5.png");
	Mat dst_4 = cvHarrisCorner(img_4);
	cout << cvBlobDetection(dst_4) << "각형" << endl;
	Mat img_5 = imread("C:\\images\\6.png");
	Mat dst_5 = cvHarrisCorner(img_5);
	cout << cvBlobDetection(dst_5) << "각형" << endl;

	//#3
	Mat img_6 = imread("C:\\images\\church.jpg",1);
	Mat dst_6,warp_img;
	add(img_6, Scalar(100, 100, 100), dst_6); //밝기 변화
	warp_img = warpPers(dst_6);
	Mat src_sift = cvFeatureSIFT(img_6);
	Mat dst_sift = cvFeatureSIFT(warp_img);

	imshow("src_sift", src_sift);
	imshow("dst_sift", dst_sift);
	waitKey(0);
	destroyWindow("src_sift");
	destroyWindow("dst_sift");


}