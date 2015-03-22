#include "imagejoint.h"
#include <iostream>
#include <sstream>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
struct ImageInfo{
	Mat colorImg;
	Mat img;
	Mat descriptors;
	vector<KeyPoint> keypoints;
};
vector<DMatch> calcMatches(const Mat &descriptors1, const Mat &descriptors2)
{
	const int firstNumbers = 50;
	BFMatcher matcher(NORM_L2, true);
	vector<DMatch> matches, submatches;

	matcher.match(descriptors1, descriptors2, matches);
	cout << "sift features numbers:" << matches.size() << endl;
	sort(matches.begin(), matches.end(), [](const DMatch &m1, const DMatch &m2){
		return m1.distance < m2.distance;
	});
	submatches = vector<DMatch>(matches.cbegin(), matches.cbegin() + firstNumbers);
	return submatches;
}

double calcDistanceSum(const vector<DMatch>& matches)
{
	double sum = 0;
	for (auto it = matches.cbegin(); it != matches.cend(); ++it)
		sum += it->distance;
	return sum;
}

Mat imageJoint(const ImageInfo& img1, const ImageInfo& img2, const vector<DMatch> matches)
{
	vector<Point2f> points1, points2;
	for (auto it = matches.cbegin(); it != matches.cend(); ++it)
	{
		points1.push_back(img1.keypoints[it->queryIdx].pt);
		points2.push_back(img2.keypoints[it->trainIdx].pt);
	}
	Mat homography = findHomography(points1, points2, CV_RANSAC);
	Mat transformed;
	cout << homography << endl;
	vector<Point2f> corners(4);
	corners[0] = Point2f(0, 0);
	corners[1] = Point2f(0, img1.img.rows);
	corners[2] = Point2f(img1.img.cols, img1.img.rows);
	corners[3] = Point2f(img1.img.cols, 0);
	perspectiveTransform(corners, corners, homography);
	corners.push_back(Point2f(0, 0));
	corners.push_back(Point2f(0, img2.img.rows));
	corners.push_back(Point2f(img2.img.cols, img2.img.rows));
	corners.push_back(Point2f(img2.img.cols, 0));
	float minx = 0, miny = 0, maxx = 0, maxy = 0;
	for (auto it = corners.cbegin(); it != corners.cend(); ++it)
	{
		if (it->x < minx) minx = it->x;
		if (it->y < miny) miny = it->y;
		if (it->x > maxx) maxx = it->x;
		if (it->y > maxy) maxy = it->y;
	}
	cout << "miny:" << miny << endl << "maxy:" << maxy << endl
		<< "minx:" << minx << endl << "maxx" << maxx << endl;
	Mat translate = (Mat_<double>(3, 3) << 1, 0, -minx, 0, 1, -miny, 0, 0, 1);
	warpPerspective(img1.colorImg, transformed, translate * homography, Size2f(maxx - minx, maxy - miny));

	for (int i = 0; i != img2.img.rows; ++i)
		for (int j = 0; j != img2.img.cols; ++j)
			if (img2.colorImg.at<Vec3b>(i, j) != Vec3b(0, 0, 0))
				transformed.at<Vec3b>(i - miny, j - minx) = img2.colorImg.at<Vec3b>(i, j);
	return transformed;
}
int main(int argc, char **args)
{
	list<ImageInfo> imgs;
	SIFT sift;
	for (int i = 1; i != argc; ++i)
	{
		ImageInfo info;
		info.colorImg = imread(args[i]);
		cvtColor(info.colorImg, info.img, CV_RGB2GRAY);
		sift(info.img, Mat(0, 0, CV_8UC1), info.keypoints, info.descriptors);
		cout << args[i] << ":" << info.keypoints.size() << endl;

		imgs.push_back(info);
	}
	cout << "initialize ends" << endl;

	int i = 0;
	while (imgs.size() > 1)
	{
		ImageInfo info = imgs.front();
		imgs.pop_front();
		double min = 0xffff;
		auto cur = imgs.begin();
		vector<DMatch> bestMatches;
		for (auto it = imgs.begin(); it != imgs.end(); ++it)
		{
			vector<DMatch> matches = calcMatches(info.descriptors, it->descriptors);
			Mat matchImg;/*
			namedWindow("img1");
			imshow("img1", info.colorImg);
			namedWindow("img2");
			imshow("img2", it->colorImg);
			namedWindow("features");
			drawMatches(info.colorImg, info.keypoints, it->colorImg, it->keypoints, matches, matchImg);
			imshow("features", matchImg);
			waitKey();*/
			double sum = calcDistanceSum(matches);
			if (min > sum)
			{
				cur = it;
				min = sum;
				bestMatches = matches;
			}
		}
		ImageInfo temp;
		temp.colorImg = imageJoint(info, *cur, bestMatches);
		cvtColor(temp.colorImg, temp.img, CV_RGB2GRAY);
		namedWindow("joint");
		imshow("joint", temp.colorImg);
		stringstream strio;
		strio << "output" << i << ".jpeg";
		i++;
		imwrite(strio.str(), temp.colorImg);
		waitKey();
		sift(temp.img, Mat(0, 0, CV_8UC1), temp.keypoints, temp.descriptors);
		imgs.erase(cur);
		imgs.push_back(temp);
	}
	
	destroyAllWindows();
}