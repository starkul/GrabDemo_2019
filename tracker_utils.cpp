#include "tracker_utils.hpp"
#include <iostream>
#include "stdafx.h"  // 如果使用预编译头文件
// 或者
#include <afxwin.h>   // MFC 核心头文件
#include <afxext.h>   // MFC 扩展头文件
TrackerUtils::TrackerUtils() {}

TrackerUtils::~TrackerUtils() {
    reset();
}

bool TrackerUtils::initTracker(const cv::Mat& frame, const cv::Rect& detBox)
{
    reset();

    if (frame.empty() || detBox.width <= 0 || detBox.height <= 0)
        return false;

    // 确保检测框在图像范围内
    cv::Rect validBox = detBox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (validBox.width < 10 || validBox.height < 10) {
        return false;
    }

    m_tracker = cv::TrackerCSRT::create();

    // OpenCV 4.x init 返回 void，所以不能赋值给 bool
    m_tracker->init(frame, detBox);

    m_isTracking = true;  // 手动标记初始化成功
    m_bbox = detBox;
    m_trajectory.clear();
    m_trajectory.push_back(cv::Point(detBox.x + detBox.width / 2, detBox.y + detBox.height / 2));

    std::cout << "[Tracker] Initialized successfully. Box = " << detBox << std::endl;
    return true;
}


TrackingResult TrackerUtils::updateTracker(const cv::Mat& frame)
{
    TrackingResult result;
    result.success = false;

    if (!m_isTracking || frame.empty() || !m_tracker)
        return result;

    try {
        cv::Rect newBox;
        bool ok = m_tracker->update(frame, newBox);
        // 更严格的边界框验证
        bool isValidBox = ok &&
            newBox.width > 4 &&
            newBox.height > 4 &&
            newBox.x >= 0 &&
            newBox.y >= 0 &&
            newBox.x + newBox.width <= frame.cols &&
            newBox.y + newBox.height <= frame.rows &&
            newBox.area() > 90; // 最小面积限制
        if (isValidBox) {
            // 跟踪成功
            m_bbox = newBox;
            result.success = true;
            result.bbox = m_bbox;
            m_lostCount = 0;

            // 更新轨迹
            cv::Point center(m_bbox.x + m_bbox.width / 2, m_bbox.y + m_bbox.height / 2);
            m_trajectory.push_back(center);

            // 限制轨迹长度，避免无限增长
            if (m_trajectory.size() > 20) {
                m_trajectory.erase(m_trajectory.begin());
            }

            result.trajectory = m_trajectory;
        //if (ok && newBox.width > 0 && newBox.height > 0) {
        //    // 跟踪成功
        //    m_bbox = newBox;
        //    result.success = true;
        //    result.bbox = m_bbox;
        //    m_lostCount = 0;
        //    // 更新轨迹
        //    cv::Point center(m_bbox.x + m_bbox.width / 2, m_bbox.y + m_bbox.height / 2);
        //    m_trajectory.push_back(center);

        //    // 限制轨迹长度，避免无限增长
        //    if (m_trajectory.size() > 30) {
        //        m_trajectory.erase(m_trajectory.begin());
        //    }
        //   
        //    result.trajectory = m_trajectory;
        }
        else {
            // 跟踪失败 - 立即处理
            //AfxMessageBox(_T("[Tracker] update() failed or invalid bounding box."));
            m_lostCount++;

            if (m_lostCount >= m_maxLost) {
                //AfxMessageBox(_T("[Tracker] Lost target too long, resetting."));
                reset();
            }
            else {
                // 单次跟踪失败，但还未达到重置阈值
                // 返回失败结果，但保持跟踪器状态
                result.success = false;
            }
        }
    }
    catch (const cv::Exception& e) {
        //CString msg;
        //msg.Format(_T("[Tracker] OpenCV exception during update: %s"), CString(e.what()));
        //AfxMessageBox(msg);
        reset();
    }
    return result;
}

void TrackerUtils::reset()
{
    m_tracker.release();
    m_isTracking = false;
    m_trajectory.clear();
    m_lostCount = 0;
    m_tracker = cv::TrackerCSRT::create();
    std::cout << "[Tracker] Reset." << std::endl;
}
//cv::TrackerKCF::create()
//cv::TrackerCSRT::create()
//cv::TrackerMIL::create()
//cv::TrackerMOSSE::create()
//cv::TrackerBoosting::create()