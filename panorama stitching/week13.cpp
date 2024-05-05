#include <iostream> 
#include "opencv2/core/core.hpp" // Mat class�� ���� data structure �� ��� ��ƾ�� �����ϴ� ���
#include "opencv2/highgui/highgui.hpp" // GUI�� ���õ� ��Ҹ� �����ϴ� ���(imshow ��)
#include "opencv2/imgproc/imgproc.hpp" // ���� �̹��� ó�� �Լ��� �����ϴ� ���
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>
#include<opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches);


void ex_panorama() {
	Mat matImage1 = imread("C:\\images\\center.jpg", IMREAD_COLOR);
	Mat matImage2 = imread("C:\\images\\left.jpg", IMREAD_COLOR);
	Mat matImage3 = imread("C:\\images\\right.jpg", IMREAD_COLOR);
	if (matImage1.empty() || matImage2.empty() || matImage3.empty()) exit(-1);

	Mat result;
	flip(matImage1, matImage1, 1);
	flip(matImage2, matImage2, 1);
	result = makePanorama(matImage1, matImage2, 3, 60);
	flip(result, result, 1);
	result = makePanorama(result, matImage3, 3, 60);

	imshow("ex_panorama_result", result);
	waitKey();
}


void ex_panorama_simple() {
	Mat img;
	vector<Mat> imgs;
	img = imread("C:\\images\\left.jpg", IMREAD_COLOR);
	resize(img, img, Size(512, 512), INTER_AREA);
	imgs.push_back(img);
	img = imread("C:\\images\\center.jpg", IMREAD_COLOR);
	resize(img, img, Size(512, 512), INTER_AREA);
	imgs.push_back(img);
	img = imread("C:\\images\\right.jpg", IMREAD_COLOR);
	resize(img, img, Size(512, 512), INTER_AREA);
	imgs.push_back(img);

	Mat result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false);
	Stitcher::Status status = stitcher->stitch(imgs, result);
	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code = " << int(status) << endl;
		exit(-1);
	}
	imshow("ex_panorama_simple_result", result);
	waitKey(0);
}

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	//<grayscale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<Ư¡��(KEY POINT) ����>
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_l.png", img_kpts_r);

	Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches
		, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);

	//<��Ī ��� ����>
	//��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f \n", dist_max);
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good,
		Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);

	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
	}

	//<��Ī ����κ��� homography ����� ����>
	Mat mat_homo = findHomography(scene, obj, RANSAC);
	//�̻�ġ ���Ÿ� ���� �߰�

	//<homograpy ����� �̿��� ���� ����ȯ>
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo,
		Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);
	//������ �߸��� ���� �����ϱ� ���� ���������� �ο�

	//<���� ����� ����ȯ�� ���� ���� ��ü>
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);

	//<���� ���� �߶󳻱�>
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++) {
		for (int x = 0; x < img_pano.cols; x++) {
			if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
				img_pano.at<Vec3b>(y, x)[1] == 0 &&
				img_pano.at<Vec3b>(y, x)[2] == 0) {
				continue;
			}
			if (cut_x < x) cut_x = x;
			if (cut_y < y)cut_y = y;
		}
	}
	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
	imwrite("img_pano_cut.png", img_pano_cut);

	return img_pano_cut;
}


void contour(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	//<grayscale�� ��ȯ>
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	//<Ư¡��(KEY POINT) ����> SIFT
	Ptr<SiftFeatureDetector> Detector = SIFT::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	//<Ư¡�� �ð�ȭ>
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	
	//����� ���� SIFT
	Ptr<SiftDescriptorExtractor> Extractor = SIFT::create(100, 4, 3, false, true);

	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	//<����ڸ� �̿��� Ư¡�� ��Ī>
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	//<��Ī ��� �ð�ȭ>
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches
		, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("mathces", img_matches);
	waitKey();

	//<��Ī ��� ����>
	//��Ī �Ÿ��� ���� ����� ��Ī ����� �����ϴ� ����
	//�ּ� ��Ī �Ÿ��� 3�� �Ǵ� ����� ��Ī ��� 60�̻� ���� ����
	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min) dist_min = dist;
		if (dist > dist_max) dist_max = dist;
	}
	printf("max_dist : %f \n", dist_max);
	printf("min_dist : %f \n", dist_min);

	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	//<����� ��Ī ��� �ð�ȭ>
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good,
		Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("good mathces", img_matches_good);
	waitKey();

	//<��Ī ��� ��ǥ ����>
	vector<Point2f> obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
	}

	//<��Ī ����κ��� homography ����� ����>

	Mat mat_homo = findHomography(obj,scene, RANSAC);
	//�̻�ġ ���Ÿ� ���� �߰�


	// corner point ����
	vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img_l.cols, 0);
	obj_corners[2] = Point(img_l.cols, img_l.rows);
	obj_corners[3] = Point(0, img_l.rows);


	//scene corners
	vector<Point2f> scene_corners(4);

	// < Ÿ�� ���� ���� >
	Mat img_object = img_l.clone();

	// < homography ����� ���Ͽ� �̹��� warping>
	perspectiveTransform(obj_corners, scene_corners, mat_homo);

	// < scene���� mapping�� object�� �ڳ� ������ ���� �׸��� >
	line(img_matches_good, scene_corners[0] + Point2f(img_object.cols, 0),
		scene_corners[1] + Point2f(img_object.cols, 0), Scalar(255,0 , 0), 3);
	line(img_matches_good, scene_corners[1] + Point2f(img_object.cols, 0),
		scene_corners[2] + Point2f(img_object.cols, 0), Scalar(255, 0, 0), 3);
	line(img_matches_good, scene_corners[2] + Point2f(img_object.cols, 0),
		scene_corners[3] + Point2f(img_object.cols, 0), Scalar(255, 0, 0), 3);
	line(img_matches_good, scene_corners[3] + Point2f(img_object.cols, 0),
		scene_corners[0] + Point2f(img_object.cols, 0), Scalar(255, 0, 0), 3);

	// show detected object
	imshow("Object detection", img_matches_good);
	waitKey();
	destroyAllWindows();

}
int main() {
	//#1
	ex_panorama();
	ex_panorama_simple();

	//#2
	Mat src_img1 = imread("C:\\images\\Book1.jpg", IMREAD_COLOR);
	Mat src_img2 = imread("C:\\images\\Book2.jpg", IMREAD_COLOR);
	Mat src_img3 = imread("C:\\images\\Book3.jpg", IMREAD_COLOR);
	resize(src_img1, src_img1, src_img1.size() / 2, INTER_AREA);
	resize(src_img2, src_img2, src_img2.size() / 2, INTER_AREA);
	resize(src_img3, src_img3, src_img3.size() / 2, INTER_AREA);
	Mat scene_img = imread("C:\\images\\Scene.jpg", IMREAD_COLOR);
	resize(scene_img, scene_img, scene_img.size() / 2, INTER_AREA);

	Mat result;
	flip(src_img1, src_img1, 1);
	flip(src_img2, src_img2, 1);
	flip(src_img3, src_img3, 1);
	flip(scene_img, scene_img, 1);

	contour(src_img1, scene_img, 2, 80); // �Ӱ谪 ����
	contour(src_img2, scene_img, 3, 60);
	contour(src_img3, scene_img, 3, 60);
	waitKey(0);
}
