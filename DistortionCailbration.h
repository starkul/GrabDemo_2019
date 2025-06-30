#ifndef DISTORTIONCAILBRATION_H
#define DISTORTIONCAILBRATION_H
#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>
#include <tchar.h>
#include <cassert>
#include <filesystem>
#include <numeric>
#include <random>
#include <algorithm>
//using namespace cv;
//using namespace std;




class DistortionCailbration
{
public:
	DistortionCailbration();
	std::vector<std::vector<cv::Mat>> Rev;
	std::vector<std::vector<cv::Mat>> Tev;



	cv::Size shape_inner_corner_;
	float size_grid_;
	cv::Mat DistortionCailbration::process(cv::Mat src);



	std::vector<cv::Mat> intrinsic_matrices_;
	std::vector<cv::Mat> new_camera_matrix_;
	std::vector<cv::Mat> dist_coeffs_;
	std::vector<cv::Mat> map1_;
	std::vector<cv::Mat> map2_;
	std::vector<std::vector<cv::Mat>> sub_aperture_images_;
	std::vector<std::vector<std::vector<cv::Point3f>>> world_points_;
	std::vector<std::vector<std::vector<cv::Point2f>>> pixel_points_;
	std::vector<std::vector<int>> flags;
	std::vector<std::vector<cv::Mat>> rvecs_;
	std::vector<std::vector<cv::Mat>> tvecs_;

};


#endif

