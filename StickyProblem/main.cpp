/*
	Uses Utilities.h and .cpp for drawHistogram()
	and Images.cpp for invertImage()
	both from the book "A Practical Introduction to Computer Vision with OpenCV"
	by Kenneth Dawson-Howe
*/
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime> //for timing the function

#define DEBUG false
//set to true to see intermediate pictures

using namespace std;

struct PageCoords{
	Point left;
	Point right;
	Point top;
	Point bottom;
};

void readImages(vector<Mat> &vector, string filename, int size);
Mat getHistogram(Mat& originalImage, Mat& mask, int numBins, bool showHist);
Mat getBackProjImage(Mat &inputImage, Mat &referenceImage);
Mat getPageMask(Mat& inputImage);
double getStdDev(Mat image, Mat mask);
PageCoords getPageCoords(Mat &input);
void drawCornerCircles(Mat &input, PageCoords &coords);
void warpPage(Mat &input, Mat &output, PageCoords &coords);
Mat getPageFromImage(Mat &input, Mat &blueReference);
int recognisePage(vector<Mat> &templates, Mat &image);

//See note above definition for following function
void rot90(cv::Mat &matImage, int rotflag);

int main(int argc, const char** argv)
{

	int groundTruth[25] = {1,2,3,4,5,6,7,8,9,10,11,12,13,2,3,5,4,7,9,8,7,11,13,12,2};

	//time program execution
	time_t initTime = clock();
	time_t sectionTime;
	time_t elapsed;

	//The Code
	/* --- Read Images --- */
	const int numPages = 13;
	vector<Mat> pagesVector;
	readImages(pagesVector, "Given/Page", numPages);

	const int numInputImages = 25;
	vector<Mat> inputImages;
	readImages(inputImages, "Given/BookView", numInputImages);

	elapsed = clock()-initTime;
	cout<<"Images read in\t" << (float)elapsed/CLOCKS_PER_SEC<<endl;

	
	Mat blueReference = imread("Given/BlueReference.png");
	Mat whiteReference = imread("Given/WhiteReference.png");
	cvtColor(blueReference, blueReference, CV_RGB2HSV);
	cvtColor(whiteReference, whiteReference, CV_RGB2HSV);
	
	sectionTime = clock();
	vector<Mat> templates = vector<Mat>(numPages);
	for(int i = 0; i < numPages; i++){
		resize(pagesVector[i], templates[i],Size(103,103));
		//getPageFromImage(templates[i], blueReference);
	}
	elapsed = clock() - sectionTime;
	cout << "resizing templates\t" << (float)elapsed/CLOCKS_PER_SEC<<endl;
	/*--- optimising ---*/
	vector<Mat> smallerInputs;
	for(int i = 0; i < inputImages.size(); i++){
		Mat temp;
		resize(inputImages[i], temp, Size(800,600));
		smallerInputs.push_back(temp);
	}
	sectionTime = clock();
	vector<Mat> resultPages;
	for(int i = 0; i < numInputImages; i++){
		resultPages.push_back(getPageFromImage(smallerInputs[i], blueReference));
	}
	elapsed = clock() - sectionTime;
	cout << "getting pages\t" << (float)elapsed/CLOCKS_PER_SEC<<endl;
	
	sectionTime = clock();
	vector<Mat> greyscaleTemplates;
	for(int i = 0; i<numPages; i++){
		Mat temp;
		cvtColor(templates[i], temp, CV_BGR2GRAY);
		threshold(temp, temp, 128, 255, THRESH_OTSU);
		greyscaleTemplates.push_back(temp);
	}

	vector<Mat> greyscalePages;
	for(int i = 0; i < resultPages.size(); i++){
		Mat temp;
		cvtColor(resultPages[i], temp, CV_BGR2GRAY);
		threshold(temp, temp, 128, 255, THRESH_OTSU);
		greyscalePages.push_back(temp);
	}
	elapsed = clock() - sectionTime;
	cout << "converting templates and pages\t" << (float)elapsed/CLOCKS_PER_SEC<<endl;

	int TP = 0;
	int TN = 0;
	int FP = 0;
	int FN = 0;

	sectionTime = clock();
	for(int i = 0; i < greyscalePages.size(); i++){
		if(recognisePage(greyscaleTemplates, greyscalePages[i])+1 == groundTruth[i]){
			TP++;
		}
	}
	elapsed = clock() - sectionTime;
	cout << "recognising pages\t" << (float)elapsed/CLOCKS_PER_SEC<<endl;

	cout<<TP<<"/25"<<endl;
	
	//Calculate time elapsed
	time_t elapsedTicks =  clock() - initTime;
	float elapsedTime = ((float)elapsedTicks)/CLOCKS_PER_SEC;
	// Calculate key metrics
	

	//Key Metric Calcs


	/*printf("TP : %i \nTN : %i \nFP : %i \nFN : %i \n", TP, TN, FP, FN);
	float recall = TP/(TP+FN);
	float precision = TP/(TP+FP);
	float accuracy = (TP+TN)/30;
	float specificity = TN/(FP+TN);
	float F1 = 2*precision*recall/(precision+recall);
	printf("Recall : %.4f \nPrecision : %.4f \nAccuracy : %.4f \nSpecificity : %.4f \nF1 : %.4f\n", recall, precision, accuracy, specificity, F1);
	printf("Elapsed time: %.4f\n", elapsedTime);
	//waitKey() doesn't work without an imshow()
	*/
	printf("Elapsed time: %.4f\n", elapsedTime);
	getchar();
}

void readImages(vector<Mat> &vector, string filename, int size){
	char file[50];
	for(int i = 0; i < size; i++){
		if(i<9) 
			sprintf(file, (filename+"0%i.jpg").c_str(), i+1);
		else
			sprintf(file, (filename+"%i.jpg").c_str(), i+1);

		if(DEBUG) printf("filename: %s\n", file);
		vector.push_back(imread(file));
		if(vector[i].data == NULL){
			printf("Error reading file %s, press enter to exit", file);
			getchar();
			exit(-1);
		}
	}
}

Mat getHistogram(Mat& originalImage, Mat& mask, int numBins, bool showHist){
	Mat hist;
	bool uniform = true;
	bool accumulate = false;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };

	calcHist(&originalImage, 1, 0, mask, hist, 1, &numBins, &histRange, uniform, accumulate);

	return hist;
}

Mat getBackProjImage(Mat &inputImage, Mat &referenceImage){
	MatND hist;
	MatND backproj;

	int hueBins = 16;
	int satBins = 16;
	int histSize[] = {hueBins, satBins};

	float hueRange[] = {0, 179};
	float satRange[] = {0, 255};
	const float* ranges[] = {hueRange, satRange};
	int channels[] = {0, 1};

	calcHist(&referenceImage, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());

	calcBackProject(&inputImage, 1, channels, hist, backproj, ranges, 1, true);

	return backproj;
}

Mat getPageMask(Mat &inputImage){
	Mat thresholdedSaturation;

	threshold(inputImage, thresholdedSaturation, 0.7*255, 255, THRESH_OTSU);
	if(DEBUG){
		imshow("thresholded saturation", thresholdedSaturation);
		waitKey();
	}

	dilate(thresholdedSaturation, thresholdedSaturation, Mat(), Point(-1,-1), 1);
	erode(thresholdedSaturation, thresholdedSaturation, Mat(), Point(-1,-1), 7);
	if(DEBUG){
		imshow("opened", thresholdedSaturation);
		waitKey();
	}

	//Mat scaledDown;
	//resize(inputImg1, scaledDown, Size(600, 400), 0, 0, CV_INTER_AREA);
	//imshow("scaled", scaledDown);

	Mat inverted;
	invertImage(thresholdedSaturation, inverted);
	return inverted;

}

PageCoords getPageCoords(Mat &input){
	PageCoords result;
	result.left = Point(input.cols, 0);
	result.right = Point(0, 0);
	result.top = Point(0, input.rows);
	result.bottom = Point(0, 0);

	for(int y = 1; y<input.rows;y++){
		for(int x = 1; x<input.cols;x++){
			//printf("x: %i, y: %i \n", x, y);
			if(input.at< unsigned char>(y, x) > 0){
				/*unsigned char foo = input.at<char>(y, x);
				int val = foo;
				cout << "input: " << val << endl;*/
				if(x < result.left.x) result.left = Point(x, y);
				if(x > result.right.x) result.right = Point(x, y);
				if(y < result.top.y) result.top = Point(x, y);
				if(y > result.bottom.y) result.bottom = Point(x, y);
			}
		}
	}

	return result;
}

void drawCornerCircles(Mat &input, PageCoords &coords){
	RotatedRect boundingLeft = RotatedRect(coords.left, Size2f(30, 30), 0);
	RotatedRect boundingRight = RotatedRect(coords.right, Size2f(30, 30), 0);
	RotatedRect boundingTop = RotatedRect(coords.top, Size2f(30, 30), 0);
	RotatedRect boundingBottom = RotatedRect(coords.bottom, Size2f(30, 30), 0);
	
	ellipse(input, boundingLeft, Scalar(255, 0, 0));
	ellipse(input, boundingRight, Scalar(255, 0, 0));
	ellipse(input, boundingTop, Scalar(255, 0, 0));
	ellipse(input, boundingBottom, Scalar(255, 0, 0));
}

void warpPage(Mat &input, Mat &output, PageCoords &coords){
	Point2f source[4] = {coords.left, coords.right, coords.top, coords.bottom};

	Point2f destination[4] = {
								Point2f(0, input.rows), 
								Point2f(input.cols, 0), 
								Point2f(0,0), 
								Point2f(input.cols, input.rows)
							};
	Mat perspTransform = getPerspectiveTransform(source, destination);
	
	warpPerspective(input, output, perspTransform, input.size());

}

Mat getPageFromImage(Mat &input, Mat &blueReference){
	Mat inputImg1;
	cvtColor(input, inputImg1, CV_RGB2HSV);

	Mat HSVChannels[3];
	split(inputImg1, HSVChannels);
	
	Mat pageMask = getPageMask(HSVChannels[2]);
	//Mat pageMask = getPageMask(inputImg1);

	Mat backProjected = getBackProjImage(inputImg1, blueReference);
	if(DEBUG){
		namedWindow("backprojected", CV_WINDOW_NORMAL);
		imshow("backprojected", backProjected);
		waitKey();
	}

	Mat maskedBlue = backProjected - pageMask;
	if(DEBUG){
		imshow("MaskedBlue", maskedBlue);
		waitKey();
	}

	threshold(maskedBlue, maskedBlue, 0.35*255, 255, THRESH_OTSU);
	if(DEBUG){
		imshow("thresholded", maskedBlue);
		waitKey();
	}
	
	PageCoords coords = getPageCoords(maskedBlue);
	
	Mat resized;

	if(DEBUG){
		drawCornerCircles(maskedBlue, coords);
		resize(maskedBlue, resized, Size(1000, 600));
		imshow("Corners", resized);
		waitKey();
	}

	Mat warped;
	warpPage(input, warped, coords);
	
	resize(warped, resized, Size(100, 100));
	if(DEBUG){
		imshow("warped", resized);
		waitKey();
	}
	return resized;
}

int recognisePage(vector<Mat> &templates, Mat &image){
	int mostProbablePage = -1;
	float highestLikelihood = 0.0;
	for(int j = 0; j < 4; j++){
		rot90(image, j);
		for(int i = 0; i < templates.size(); i++){
			Mat matching_space;
			if(DEBUG){
				imshow("input", image);
				imshow("reference", templates[i]);
				waitKey();
			}
			matching_space.create(Size(templates[i].cols - image.cols + 1, templates[i].rows - image.rows+1), CV_32FC1 );
			matchTemplate(templates[i], image, matching_space, CV_TM_CCORR_NORMED );
			float largestValue = 0.0;
			for(int x = 0; x < matching_space.cols; x++){
				for(int y = 0; y < matching_space.rows; y++){
					if(matching_space.at<float>(y, x) > largestValue) largestValue = matching_space.at<float>(y, x);
				}
			}
			if(largestValue > highestLikelihood){
				highestLikelihood = largestValue;
				mostProbablePage = i;
			}
			//cout << "template page:" << i << "\tlargest value: " << largestValue << endl;
		}
		
	}
	if(DEBUG) cout << "Highest Likelihood: " << highestLikelihood << endl;
	return highestLikelihood > 0.8 ? mostProbablePage : -1;
}

/*
	The following function taken directly from an answer by TimZaman on:
	http://stackoverflow.com/questions/15043152/rotate-opencv-matrix-by-90-180-270-degrees

*/
void rot90(cv::Mat &matImage, int rotflag){
  //1=CW, 2=CCW, 3=180
  if (rotflag == 1){
    transpose(matImage, matImage);  
    flip(matImage, matImage,1); //transpose+flip(1)=CW
  } else if (rotflag == 2) {
    transpose(matImage, matImage);  
    flip(matImage, matImage,0); //transpose+flip(0)=CCW     
  } else if (rotflag ==3){
    flip(matImage, matImage,-1);    //flip(-1)=180          
  } else if (rotflag != 0){ //if not 0,1,2,3:
    cout  << "Unknown rotation flag(" << rotflag << ")" << endl;
  }
}