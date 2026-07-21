#include "VisionUtils.h"

#include "ApplicationConfig.h"

#include <algorithm>
#include <cmath>

namespace VisionUtils {

    bool pointInsideImage(
        const cv::Point2f& point,
        const cv::Mat& image)
    {
        return std::isfinite(point.x) &&
            std::isfinite(point.y) &&
            point.x >= 0.0f &&
            point.x <
            static_cast<float>(image.cols) &&
            point.y >= 0.0f &&
            point.y <
            static_cast<float>(image.rows);
    }

    cv::Mat preprocessImage(
        const cv::Mat& gray,
        const cv::Ptr<cv::CLAHE>& clahe)
    {
        cv::Mat result;

        clahe->apply(
            gray,
            result);

        cv::GaussianBlur(
            result,
            result,
            cv::Size(3, 3),
            0.4);

        return result;
    }

    cv::Mat upscaleSmallFrame(
        const cv::Mat& frame,
        double& scale)
    {
        int minimumSide =
            std::min(
                frame.cols,
                frame.rows);

        scale = 1.0;

        if (minimumSide < 360) {
            scale =
                360.0 /
                static_cast<double>(
                    minimumSide);

            scale =
                std::min(
                    scale,
                    4.0);
        }

        if (scale <= 1.0) {
            return frame.clone();
        }

        cv::Mat enlarged;

        cv::resize(
            frame,
            enlarged,
            cv::Size(),
            scale,
            scale,
            cv::INTER_CUBIC);

        return enlarged;
    }

    std::vector<cv::DMatch> findMutualMatches(
        const cv::Mat& mapDescriptors,
        const cv::Mat& frameDescriptors,
        cv::BFMatcher& matcher)
    {
        std::vector<std::vector<cv::DMatch>>
            mapToFrame;

        std::vector<std::vector<cv::DMatch>>
            frameToMap;

        matcher.knnMatch(
            mapDescriptors,
            frameDescriptors,
            mapToFrame,
            2);

        matcher.knnMatch(
            frameDescriptors,
            mapDescriptors,
            frameToMap,
            2);

        std::vector<cv::DMatch> matches;

        for (std::size_t mapIndex = 0;
            mapIndex < mapToFrame.size();
            mapIndex++) {

            const std::vector<cv::DMatch>& forward =
                mapToFrame[mapIndex];

            if (forward.size() < 2) {
                continue;
            }

            if (forward[0].distance >=
                AppConfig::LOWE_RATIO *
                forward[1].distance) {
                continue;
            }

            const cv::DMatch& bestForward =
                forward[0];

            int frameIndex =
                bestForward.trainIdx;

            if (frameIndex < 0 ||
                frameIndex >=
                static_cast<int>(
                    frameToMap.size())) {
                continue;
            }

            const std::vector<cv::DMatch>& reverse =
                frameToMap[frameIndex];

            if (reverse.size() < 2) {
                continue;
            }

            if (reverse[0].distance >=
                AppConfig::LOWE_RATIO *
                reverse[1].distance) {
                continue;
            }

            if (reverse[0].trainIdx !=
                bestForward.queryIdx) {
                continue;
            }

            matches.push_back(
                bestForward);
        }

        std::sort(
            matches.begin(),
            matches.end(),
            [](const cv::DMatch& left,
                const cv::DMatch& right) {
                    return left.distance <
                        right.distance;
            });

        return matches;
    }

    bool estimateMapPosition(
        const std::vector<cv::KeyPoint>& mapKeypoints,
        const std::vector<cv::KeyPoint>& frameKeypoints,
        const std::vector<cv::DMatch>& matches,
        const cv::Size& frameSize,
        cv::Point2f& mapPosition,
        int& inlierCount,
        double& inlierRatio,
        std::string& failureReason)
    {
        inlierCount = 0;
        inlierRatio = 0.0;

        if (matches.size() <
            static_cast<std::size_t>(
                AppConfig::MIN_MATCHES)) {

            failureReason =
                "Not enough mutual matches";

            return false;
        }

        std::vector<cv::Point2f> mapPoints;
        std::vector<cv::Point2f> framePoints;

        mapPoints.reserve(
            matches.size());

        framePoints.reserve(
            matches.size());

        for (const cv::DMatch& match :
            matches) {

            mapPoints.push_back(
                mapKeypoints[
                    match.queryIdx].pt);

            framePoints.push_back(
                frameKeypoints[
                    match.trainIdx].pt);
        }

        cv::Mat inlierMask;

        cv::Mat affineTransform =
            cv::estimateAffinePartial2D(
                framePoints,
                mapPoints,
                inlierMask,
                cv::RANSAC,
                3.0,
                3000,
                0.995,
                10);

        if (affineTransform.empty() ||
            inlierMask.empty()) {

            failureReason =
                "Affine RANSAC failed";

            return false;
        }

        inlierCount =
            cv::countNonZero(
                inlierMask);

        inlierRatio =
            static_cast<double>(
                inlierCount) /
            static_cast<double>(
                matches.size());

        if (inlierCount <
            AppConfig::MIN_INLIERS) {

            failureReason =
                "Not enough RANSAC inliers";

            return false;
        }

        if (inlierRatio <
            AppConfig::MIN_INLIER_RATIO) {

            failureReason =
                "Low RANSAC inlier ratio";

            return false;
        }

        std::vector<cv::Point2f>
            inlierFramePoints;

        for (int index = 0;
            index <
            static_cast<int>(
                framePoints.size());
            index++) {

            if (inlierMask.at<unsigned char>(
                index) != 0) {

                inlierFramePoints.push_back(
                    framePoints[index]);
            }
        }

        if (inlierFramePoints.size() <
            static_cast<std::size_t>(
                AppConfig::MIN_INLIERS)) {

            failureReason =
                "Invalid inlier points";

            return false;
        }

        cv::Rect pointBounds =
            cv::boundingRect(
                inlierFramePoints);

        double spreadX =
            static_cast<double>(
                pointBounds.width) /
            static_cast<double>(
                frameSize.width);

        double spreadY =
            static_cast<double>(
                pointBounds.height) /
            static_cast<double>(
                frameSize.height);

        if (spreadX <
            AppConfig::MIN_POINT_SPREAD ||
            spreadY <
            AppConfig::MIN_POINT_SPREAD) {

            failureReason =
                "RANSAC points are too clustered";

            return false;
        }

        std::vector<cv::Point2f> frameCenter = {
            cv::Point2f(
                frameSize.width * 0.5f,
                frameSize.height * 0.5f)
        };

        std::vector<cv::Point2f>
            transformedCenter;

        cv::transform(
            frameCenter,
            transformedCenter,
            affineTransform);

        if (transformedCenter.empty()) {
            failureReason =
                "Center transformation failed";

            return false;
        }

        mapPosition =
            transformedCenter[0];

        failureReason = "NONE";

        return
            std::isfinite(mapPosition.x) &&
            std::isfinite(mapPosition.y);
    }

}