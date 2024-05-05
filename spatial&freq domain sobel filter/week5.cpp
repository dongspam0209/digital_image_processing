#include <iostream>
#include "opencv2/core/core.hpp" // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
using namespace cv;
using namespace std;

Mat padding(Mat img) {
	int dftRows = getOptimalDFTSize(img.rows);
	int dftCols = getOptimalDFTSize(img.cols);

	Mat padded;

	copyMakeBorder(img, padded, 0, dftRows - img.rows, 0, dftCols - img.cols, BORDER_CONSTANT, Scalar::all(0));

	return padded;
}
//이미지 주위를 0으로 둘러주는 과정이다.


Mat getMagnitude(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리

	Mat magImg;
	magnitude(planes[0], planes[1], magImg);
	magImg += Scalar::all(1);
	log(magImg, magImg);
	// magnitude 취득
	// log(1+sqrt(RE(DFT(I))^2+IM(DFT(I))^2))

	return magImg;
}

Mat myNormalize(Mat src) {
	Mat dst;
	src.copyTo(dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);

	return dst;
}

Mat getPhase(Mat complexImg) {
	Mat planes[2];
	split(complexImg, planes);
	//실수부, 허수부 분리

	Mat phaImg;
	phase(planes[0], planes[1], phaImg);
	//phase 취득

	return phaImg;
}

Mat centralize(Mat complex) {
	Mat planes[2];
	split(complex, planes);
	int cx = planes[0].cols / 2;
	int cy = planes[1].rows / 2;

	Mat q0Re(planes[0], Rect(0, 0, cx, cy));
	Mat q1Re(planes[0], Rect(cx, 0, cx, cy));
	Mat q2Re(planes[0], Rect(0, cy, cx, cy));
	Mat q3Re(planes[0], Rect(cx, cy, cx, cy));

	Mat tmp;
	q0Re.copyTo(tmp);
	q3Re.copyTo(q0Re);
	tmp.copyTo(q3Re);
	q1Re.copyTo(tmp);
	q2Re.copyTo(q1Re);
	tmp.copyTo(q2Re);

	Mat q0Im(planes[1], Rect(0, 0, cx, cy));
	Mat q1Im(planes[1], Rect(cx, 0, cx, cy));
	Mat q2Im(planes[1], Rect(0, cy, cx, cy));
	Mat q3Im(planes[1], Rect(cx, cy, cx, cy));

	q0Im.copyTo(tmp);
	q3Im.copyTo(q0Im);
	tmp.copyTo(q3Im);
	q1Im.copyTo(tmp);
	q2Im.copyTo(q1Im);
	tmp.copyTo(q2Im);

	Mat centerComplex;
	merge(planes, 2, centerComplex);

	return centerComplex;
}

Mat setComplex(Mat magImg, Mat phaImg) {
	exp(magImg, magImg);
	magImg -= Scalar::all(1);

	// magnitude 계산을 반대로 수행

	Mat planes[2];
	polarToCart(magImg, phaImg, planes[0], planes[1]);
	// 극 좌표계 -> 직교 좌표계 (각도와 크기로부터 2차원 좌표);

	Mat complexImg;
	merge(planes, 2, complexImg);

	// 실수부 허수부 합체

	return complexImg;
}

Mat doIdft(Mat complexImg) {
	Mat idftcvt;
	idft(complexImg, idftcvt);
	//IDFT를 이용한 원본 영상 취득

	Mat planes[2];
	split(idftcvt, planes);

	Mat dstImg;
	magnitude(planes[0], planes[1], dstImg);
	normalize(dstImg, dstImg, 255, 0, NORM_MINMAX);
	dstImg.convertTo(dstImg, CV_8UC1);
	// 일반 영상의 type과 표현 범위로 변환
	return dstImg;
}

Mat doDft(Mat srcImg) {
	Mat floatImg;
	srcImg.convertTo(floatImg, CV_32F);

	Mat complexImg;
	dft(floatImg, complexImg, DFT_COMPLEX_OUTPUT);

	return complexImg;
}

Mat doLPF(Mat srcImg) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 20, Scalar::all(1), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);


}

Mat doHPF(Mat srcImg) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//<LPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	circle(maskImg, Point(maskImg.cols / 2, maskImg.rows / 2), 50, Scalar::all(0), -1, -1, 0);

	Mat magImg2;
	multiply(magImg, maskImg, magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}

Mat doBPF(Mat srcImg) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);
	Mat normImg = myNormalize(magImg);


	//<BPF>
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg_band = Mat::zeros(magImg.size(), CV_32F);
	circle(maskImg_band, Point(maskImg_band.cols / 2, maskImg_band.rows / 2), 100, Scalar::all(1), -1, -1, 0); //LPF
	circle(maskImg_band, Point(maskImg_band.cols / 2, maskImg_band.rows / 2), 20, Scalar::all(0), -1, -1, 0); //HPF
	Mat magImg2;

	multiply(maskImg_band, magImg, magImg2);
	imshow("magImg1", magImg);
	imshow("magImg2", magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}


int myKernelConv9x9(uchar* arr, int kernel[][9], int x, int y, int width, int height, int ch, int p = 0) { //채널과, 3채널 일때 픽셀접근을 위해 p 선언
	int sum = 0;
	int sumKernel = 0;

	for (int j = -4; j <= 4; j++) {
		for (int i = -4; i <= 4; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				// 마스크 가장자리 연산 안하기
				if (ch == 1)  // 1채널 일때
					sum += arr[(y + j) * width + (x + i)] * kernel[i + 4][j + 4];
				else if (ch == 3) // 3채널 일때
					sum += arr[(y + j) * width * 3 + (x + i) * 3 + p] * kernel[i + 4][j + 4];
				sumKernel += kernel[i + 4][j + 4];
			}
		}
	}
	if (sumKernel != 0) { return sum / sumKernel; }// 합이 1로 정규화 되도록 해 영상의 밝기변화 방지
	else { return sum; }
}

Mat myGaussianFilter(Mat srcImg, int ch) {
	int width = srcImg.cols;
	int height = srcImg.rows;
	int kernel[9][9] = { 1,3,5,7,10,7,5,3,1,             //직접 구현한 9x9 필터 근사값들을 배치하였다.
						3,13,16,19,22,19,16,13,3,
						5,16,25,28,31,28,25,16,5,
						7,19,28,34,37,34,28,19,7,
						10,22,31,37,40,37,31,22,10,
						7,19,28,34,37,34,28,19,7,
						5,16,25,28,31,28,25,16,5,
						3,13,16,19,22,19,16,13,3,
						1,3,5,7,10,7,5,3,1 };
	Mat dstImg_1(srcImg.size(), CV_8UC1); //1채널용 변수 영상의 size와 1 채널 정보를 저장한다.
	Mat dstImg_3(srcImg.size(), CV_8UC3); //3채널용 변수 영상의 size와 3 채널 정보를 저장한다.
	// Mat객체를 가리키는 포인터
	uchar* srcData = srcImg.data;
	uchar* dstData_1 = dstImg_1.data;
	uchar* dstData_3 = dstImg_3.data;
	if (ch == 1) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dstData_1[y * width + x] = myKernelConv9x9(srcData, kernel, x, y, width, height, 1);
				// 앞서 구현한 convolution 에 마스크 배열을 입력해 사용
			}
		}
	}
	else if (ch == 3) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dstData_3[y * width * 3 + x * 3] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 0);
				dstData_3[y * width * 3 + x * 3 + 1] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 1);
				dstData_3[y * width * 3 + x * 3 + 2] = myKernelConv9x9(srcData, kernel, x, y, width, height, 3, 2);
				// 앞서 구현한 convolution 에 마스크 배열을 입력해 사용
			}
		}
	}
	if (ch == 1) { return dstImg_1; }
	else if (ch == 3) { return dstImg_3; }
}


Mat doFDF(Mat srcImg) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//doFDF
	double minVal, maxVal;
	Point minLoc, maxLoc;
	minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
	normalize(magImg, magImg, 0, 1, NORM_MINMAX);

	Mat maskImg = Mat::ones(magImg.size(), CV_32F);
	line(maskImg, Point(maskImg.cols / 2, 0), Point(maskImg.cols / 2, (maskImg.rows / 2) - 15), Scalar::all(0), 25);
	line(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2) + 15), Point(maskImg.cols / 2, maskImg.rows), Scalar::all(0), 25);


	Mat magImg2;
	multiply(magImg, maskImg, magImg2);
	imshow("magImg2", magImg2);

	//<IDFT>
	normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
	Mat complexImg2 = setComplex(magImg2, phaImg);
	Mat dstImg = doIdft(complexImg2);

	return myNormalize(dstImg);

}
int myKernelConv3x3(uchar* arr, int kernel[][3], int x, int y, int width, int height) {
	int sum = 0;
	int sumKernel = 0;

	// 특정 화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
	for (int j = -1; j <= 1; j++) {
		for (int i = -1; i <= 1; i++) {
			if ((y + j) >= 0 && (y + j) < height && (x + i) >= 0 && (x + i) < width) {
				sum += arr[(y + j) * width + (x + i)] * kernel[i + 1][j + 1];
				sumKernel += kernel[i + 1][j + 1];
			}
		}
	}
	if (sumKernel != 0) { return sum / sumKernel; }// 합이 1로 정규화 되도록 해 영상의 밝기변화 방지
	else { return sum; }
	//color channel indexing 계산식 다르고 채널별 각 수행 
}

//spatial domain에서 sobelfilter구현
Mat mySobelFilter(Mat srcImg, int sel) {
	//horizontal filter
	int kernelX[3][3] = { -1, 0, 1,
						 -2, 0, 2,
						 -1, 0, 1 };
	//vertical filter
	int kernelY[3][3] = { 1, 2, 1,
						  0, 0, 0,
						-1, -2, -1 };

	// 마스크 합이 0이 되므로 1로 정규화하는 과정은 필요 없음
	Mat dstImg(srcImg.size(), CV_8UC1);
	uchar* srcData = srcImg.data;
	uchar* dstData = dstImg.data;
	int width = srcImg.cols;
	int height = srcImg.rows;

	//horizontal filter
	if (sel == 0) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dstData[y * width + x] = abs(myKernelConv3x3(srcData, kernelX, x, y, width, height));
			}
		}
	}
	//vertical filter
	if (sel == 1) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dstData[y * width + x] = abs(myKernelConv3x3(srcData, kernelY, x, y, width, height));
			}
		}
	}

	if (sel == 2) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dstData[y * width + x] = (abs(myKernelConv3x3(srcData, kernelX, x, y, width, height)) +
					abs(myKernelConv3x3(srcData, kernelY, x, y, width, height))) / 2;
			}
		}
	}
	return dstImg;
}

Mat mySobelFIlter_freq(Mat srcImg, int sel) {
	//<DFT>
	Mat padImg = padding(srcImg);
	Mat complexImg = doDft(padImg);
	Mat centerComplexImg = centralize(complexImg);
	Mat magImg = getMagnitude(centerComplexImg);
	Mat phaImg = getPhase(centerComplexImg);

	//horizontal filter
	if (sel == 0) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
		normalize(magImg, magImg, 0, 1, NORM_MINMAX);

		Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
		line(maskImg, Point(maskImg.cols / 2,0), Point(maskImg.cols / 2, (maskImg.rows/2) -50), Scalar::all(1),50);
		line(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2)+50), Point(maskImg.cols / 2, maskImg.rows ), Scalar::all(1),50);

		Mat magImg2;
		multiply(magImg, maskImg, magImg2);
		imshow("magImg2_horizontal", magImg2);

		//<IDFT>
		normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
		Mat complexImg2 = setComplex(magImg2, phaImg);
		Mat dstImg = doIdft(complexImg2);

		return myNormalize(dstImg);
	}
	//vertical filter
	if (sel == 1) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
		normalize(magImg, magImg, 0, 1, NORM_MINMAX);

		Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
		line(maskImg, Point(0, maskImg.rows / 2), Point((maskImg.cols / 2) - 70, maskImg.rows / 2), Scalar::all(1), 50);
		line(maskImg, Point((maskImg.cols / 2) + 70, maskImg.rows / 2), Point(maskImg.cols, maskImg.rows / 2), Scalar::all(1), 50);

		Mat magImg2;
		multiply(magImg, maskImg, magImg2);
		imshow("magImg2_vertical", magImg2);

		//<IDFT>
		normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
		Mat complexImg2 = setComplex(magImg2, phaImg);
		Mat dstImg = doIdft(complexImg2);

		return myNormalize(dstImg);
	}

	if (sel == 2) {
		double minVal, maxVal;
		Point minLoc, maxLoc;
		minMaxLoc(magImg, &minVal, &maxVal, &minLoc, &maxLoc);
		normalize(magImg, magImg, 0, 1, NORM_MINMAX);

		Mat maskImg = Mat::zeros(magImg.size(), CV_32F);
		line(maskImg, Point(maskImg.cols / 2, 0), Point(maskImg.cols / 2, (maskImg.rows / 2) - 50), Scalar::all(1), 50);
		line(maskImg, Point(maskImg.cols / 2, (maskImg.rows / 2) + 50), Point(maskImg.cols / 2, maskImg.rows), Scalar::all(1), 50);
		line(maskImg, Point(0, maskImg.rows / 2), Point((maskImg.cols / 2) - 70, maskImg.rows / 2), Scalar::all(1), 50);
		line(maskImg, Point((maskImg.cols / 2) + 70, maskImg.rows / 2), Point(maskImg.cols, maskImg.rows / 2), Scalar::all(1), 50);

		Mat magImg2;
		multiply(magImg, maskImg, magImg2);
		imshow("magImg2_sobel", magImg2);

		//<IDFT>
		normalize(magImg2, magImg2, (float)minVal, (float)maxVal, NORM_MINMAX);
		Mat complexImg2 = setComplex(magImg2, phaImg);
		Mat dstImg = doIdft(complexImg2);

		return myNormalize(dstImg);
	}

}


int main() {
	
	//#1
	Mat srcImg, dstImg;
	srcImg = imread("C:\\images\\img1week5.jpg", 0);
	dstImg = doBPF(srcImg);
	imshow("srcimg", srcImg);
	imshow("BPFimg", dstImg);
	waitKey(0);


	destroyWindow("srcimg");
	destroyWindow("BPFimg");
	destroyWindow("magImg1");
	destroyWindow("magImg2");


	//#2
	srcImg = imread("C:\\images\\img2week5.jpg", 0);
	//spatial domain
	Mat dstImg_hor, dstImg_ver;
	dstImg_hor = mySobelFilter(srcImg, 0); //horizontal
	dstImg_ver = mySobelFilter(srcImg, 1); //vertical
	dstImg = mySobelFilter(srcImg, 2); //sobel

	imshow("horizontal_spatial", dstImg_hor);
	imshow("vertical_spatial", dstImg_ver);
	imshow("sobel_spatial", dstImg);
	waitKey(0);
	destroyWindow("horizontal_spatial");
	destroyWindow("vertical_spatial");
	destroyWindow("sobel_spatial");


	//frequency domain
	dstImg_hor = mySobelFIlter_freq(srcImg, 0);
	dstImg_ver = mySobelFIlter_freq(srcImg, 1);
	dstImg = mySobelFIlter_freq(srcImg, 2);


	imshow("horizontal_freq", dstImg_hor);
	imshow("vertical_freq", dstImg_ver);
	imshow("sobel_freq", dstImg);
	waitKey(0);
	destroyWindow("horizontal_freq");
	destroyWindow("vertical_freq");
	destroyWindow("sobel_freq");
	destroyWindow("magImg2_vertical");
	destroyWindow("magImg2_horizontal");
	destroyWindow("magImg2_sobel");


	//#3
	Mat dstImg2;
	srcImg = imread("C:\\images\\img3week5.jpg", 0);
	dstImg = myGaussianFilter(srcImg, 1);
	dstImg2 = doFDF(dstImg);
	imshow("flickering", dstImg2);
	waitKey(0);
}
