#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <string>

struct TrackingResult {
    bool success = false;
    cv::Rect bbox;
    std::vector<cv::Point> trajectory;
};

class TrackerUtils {
public:
    TrackerUtils();
    ~TrackerUtils();

    bool initTracker(const cv::Mat& frame, const cv::Rect& detBox);
    TrackingResult updateTracker(const cv::Mat& frame);
    void reset();
    bool isTracking() const { return m_isTracking; }

private:
    cv::Ptr<cv::TrackerCSRT> m_tracker;  // <-- ¸ÄÎª TrackerKCF Ö¸Õë
    cv::Rect m_bbox;
    bool m_isTracking = false;
    std::vector<cv::Point> m_trajectory;
    int m_lostCount = 0;
    int m_maxLost = 5;
};
