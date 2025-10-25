#include "stdafx.h"
#include "DistortionCailbration.h"


DistortionCailbration::DistortionCailbration() {
	std::vector<int> x_coords = { 0, 203, 444, 640 }; // x-axis splits
	std::vector<int> y_coords = { 0, 170, 353, 512 }; // y-axis splits
	std::string filename = "calibration_data_real.yml";
	std::vector<cv::Mat> intrinsic_matrices;
	std::vector<cv::Mat> dist_coeffs;
	cv::Size shape_inner_corner(6, 4); // Set the pattern size
	float size_grid = 40;              // Size of the grid in mm
	shape_inner_corner_ = shape_inner_corner;
	size_grid_ = size_grid;
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	cv::FileNode intrinsicNode = fs["intrinsic_matrices"];
	cv::FileNode distNode = fs["dist_coeffs"];

	for (auto it = intrinsicNode.begin(); it != intrinsicNode.end(); ++it) {
		cv::Mat intrinsic;
		*it >> intrinsic;
		intrinsic_matrices.push_back(intrinsic);
	}

	for (auto it = distNode.begin(); it != distNode.end(); ++it) {
		cv::Mat coeff;
		*it >> coeff;
		dist_coeffs.push_back(coeff);
	}
	fs.release();
	intrinsic_matrices_ = intrinsic_matrices;
	dist_coeffs_ = dist_coeffs;
	int index = 0;
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			// Get the current sub-image
			//cv::Mat img = src;
			cv::Mat intrinsic_matrix = intrinsic_matrices_[index]; // Assume intrinsic_matrices_ is initialized correctly
			cv::Mat dist_coeff = dist_coeffs_[index]; // Assume dist_coeffs_ is initialized correctly
			cv::Rect roi(x_coords[j], y_coords[i], x_coords[j + 1] - x_coords[j], y_coords[i + 1] - y_coords[i]);
			cv::Size img_size = roi.size();
			// cv::Size img_size = cv::Size(640, 512);
			cv::Mat map1, map2;
			cv::Mat new_camera_matrix = cv::getOptimalNewCameraMatrix(intrinsic_matrix, dist_coeff, img_size, 0, img_size);
			initUndistortRectifyMap(intrinsic_matrix, dist_coeff, cv::Mat(), new_camera_matrix, img_size, CV_16SC2, map1, map2);
			map1_.push_back(map1);
			map2_.push_back(map2);
			new_camera_matrix_.push_back(new_camera_matrix);
			++index;
		}
	}

}


cv::Mat DistortionCailbration::process(cv::Mat img) {
	// Define the split positions based on your specified coordinates
	std::vector<int> x_coords = { 0, 203, 444, 640 }; // x-axis splits
	std::vector<int> y_coords = { 0, 170, 353, 512 }; // y-axis splits

													  // Create a 3x3 grid to store the sub-images (3 rows and 3 columns)
	std::vector<std::vector<cv::Mat>> rows(3, std::vector<cv::Mat>(3)); // 3 rows and 3 columns

	int index = 0;

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			// Get the current sub-image
			cv::Rect roi(x_coords[j], y_coords[i], x_coords[j + 1] - x_coords[j], y_coords[i + 1] - y_coords[i]);
			cv::Mat sub_img = img(roi);

			cv::Mat intrinsic_matrix = intrinsic_matrices_[index]; // Assume intrinsic_matrices_ is initialized correctly
			cv::Mat dist_coeff = dist_coeffs_[index]; // Assume dist_coeffs_ is initialized correctly

			cv::Size img_size = sub_img.size();
			cv::Mat new_camera_matrix = new_camera_matrix_[index]; // Assume new_camera_matrix_ is initialized correctly

			//if (index == 0 || index == 2 || index == 5 || index == 6 || index == 7 || index == 8) {
				// Undistort the image
				// cv::undistort(sub_img, dst, intrinsic_matrix, dist_coeff, new_camera_matrix);
				cv::Mat dst;
				remap(sub_img, dst, map1_[index], map2_[index], cv::INTER_LINEAR);
				//Place the undistorted image in the correct position based on the sub-image grid
				rows[i][j] = dst; // Store the undistorted image in the corresponding row and column
			//}
			//else {
			//	rows[i][j] = sub_img;
			//}
			
			++index; // Move to the next sub-image
		}
	}

	// Merge each row of images horizontally
	std::vector<cv::Mat> merged_rows(3);
	for (int i = 0; i < 3; ++i) {
		cv::hconcat(rows[i], merged_rows[i]); // Horizontally concatenate each row
	}

	// Merge all rows vertically to form the final image
	cv::Mat merged_image;
	cv::vconcat(merged_rows, merged_image); // Vertically concatenate the 3 rows

											// Save the final merged image
	return merged_image;
}

