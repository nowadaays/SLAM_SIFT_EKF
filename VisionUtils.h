#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

namespace VisionUtils {

    // ѕровер€ет принадлежность точки изображению.
    bool pointInsideImage(
        const cv::Point2f& point,
        const cv::Mat& image);

    // ѕовышает локальный контраст и подавл€ет шум.
    cv::Mat preprocessImage(
        const cv::Mat& gray,
        const cv::Ptr<cv::CLAHE>& clahe);

    // ”величивает маленький кадр перед применением SIFT.
    cv::Mat upscaleSmallFrame(
        const cv::Mat& frame,
        double& scale);

    // ¬ыполн€ет взаимную проверку совпадений дескрипторов.
    std::vector<cv::DMatch> findMutualMatches(
        const cv::Mat& mapDescriptors,
        const cv::Mat& frameDescriptors,
        cv::BFMatcher& matcher);

    // ќпредел€ет положение центра кадра на карте.
    bool estimateMapPosition(
        const std::vector<cv::KeyPoint>& mapKeypoints,
        const std::vector<cv::KeyPoint>& frameKeypoints,
        const std::vector<cv::DMatch>& matches,
        const cv::Size& frameSize,
        cv::Point2f& mapPosition,
        int& inlierCount,
        double& inlierRatio,
        std::string& failureReason);

}