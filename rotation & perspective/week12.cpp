#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

Mat getMyRotationMatrix(Point center, double angle) {
	double a = cos(angle * CV_PI / 180);
	double b = sin(angle * CV_PI / 180);
	Mat matrix = (
		Mat_<double>(2, 3) <<
		a, b, (1 - a) * center.x - b * center.y,
		-b, a, b * center.x + (1 - a) * center.y
		);

	return matrix;
}

void cvRotation() {
	Mat src = imread("C:\\images\\Lenna.png", 1);
	Mat dst,dst2, matrix;

	Point center = Point(src.cols / 2, src.rows / 2);

	//CV
	matrix = getRotationMatrix2D(center, 45.0, 1.0);
	warpAffine(src, dst, matrix, src.size());
	//Rotation
	matrix = getMyRotationMatrix(center, 45.0);
	warpAffine(src, dst2, matrix, src.size());



	imshow("nonrot.jpg", src);
	imshow("rot", dst);
	imshow("my_rot", dst2);
	waitKey(0);

	destroyAllWindows();

}

void cvPerspective(Mat src,vector <Point2f> xy) {
	Mat dst, matrix;

	Point2f srcQuad[4];
	srcQuad[0] = xy[0];
	srcQuad[1] = xy[1];
	srcQuad[2] = xy[3];
	srcQuad[3] = xy[2];

	Point2f dstQuad[4];
	dstQuad[0] = Point2f(xy[3].x, xy[0].y);
	dstQuad[1] = Point2f(xy[2].x, xy[0].y);
	dstQuad[2] = Point2f(xy[3].x, xy[3].y);
	dstQuad[3] = Point2f(xy[2].x, xy[3].y);

	matrix = getPerspectiveTransform(srcQuad, dstQuad);
	warpPerspective(src, dst, matrix, src.size());

	imshow("nonper", src);
	imshow("per", dst);
	waitKey(0);
	destroyAllWindows();

}

vector <Point2f> cvHarrisCorner(Mat img) {
	Mat gray = img.clone();

	// < Do Harris corner detection >
	Mat harr;
	cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
	normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());

	// < Get abs for Harris visualization >
	Mat harr_abs;
	convertScaleAbs(harr, harr_abs);

	// < Print corners >
	int thresh = 100;
	int min_pixel = 0;

	vector <Point2f>xy;
	Mat result = img.clone();

	for (int y = 0; y < harr.rows; y++) {
		for (int x = 0; x < harr.rows; x++)
		{
			if ((int)harr.at<float>(y, x) > thresh) {

				circle(result, Point(x, y), 7, Scalar(255, 0, 255), 0, 4, 0);
				if (xy.size() == 0) {
					xy.push_back(Point(x, y));
				}
				else
				{
					if (x != xy[min_pixel].x && y != xy[min_pixel].y) {
						if ((x<xy[min_pixel].x - 5 || x > xy[min_pixel].x + 5) &&
							(x<xy[min_pixel].y - 5 || x > xy[min_pixel].y + 5))
						{
							xy.push_back(Point(x, y));
							min_pixel++;
						}
					}
				}

			}
		}
	}

	imshow("Target image", result);
	waitKey(0);
	cout << xy[0]<<endl;
	cout << xy[1]<<endl;
	cout << xy[2]<<endl;
	cout << xy[3]<<endl;
	return xy;
}



int main() {


	cvRotation();

	Mat src,grabcut_result,dst,bg_model,fg_model;
	src = imread("C:\\images\\card_per.png", 1);

	Rect rect = Rect(Point(55, 105), Point(450, 370));
	grabCut(src, grabcut_result,
		rect,
		bg_model,
		fg_model,
		5,
		GC_INIT_WITH_RECT);

	compare(grabcut_result, GC_PR_FGD, grabcut_result, CMP_EQ);

	Mat mask(src.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	src.copyTo(mask, grabcut_result);

	cvPerspective(src, cvHarrisCorner(grabcut_result));

	
}