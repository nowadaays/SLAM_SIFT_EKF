#pragma once

#include <opencv2/core.hpp>
#include <vector>

struct TrajectoryConfig {
    // Заглушка масштаба до появления симуляции.
    double pixelsPerMeter = 3.0;

    // Доля коррекции SIFT при наличии движения БИНС.
    double siftCorrectionGain = 0.75;

    double minimumPointDistance = 0.5;
    std::size_t maximumHistorySize = 300;
};

struct TrajectoryMotionData {
    bool available = false;

    double velocityNorth = 0.0;
    double velocityUp = 0.0;
    double velocityEast = 0.0;

    double dt = 0.0;
};

class TrajectoryCalculator {
public:
    explicit TrajectoryCalculator(
        const TrajectoryConfig& config =
        TrajectoryConfig());

    void reset();

    // Вызывается перед SIFT.
    void predict(
        const TrajectoryMotionData& motion);

    // Вызывается после SIFT.
    void correct(
        bool siftValid,
        const cv::Point2f& siftMeasurement);

    bool isInitialized() const;
    bool hasMotionData() const;

    const cv::Point2f& getBinsPosition() const;
    const cv::Point2f& getSiftPosition() const;
    const cv::Point2f& getFusedPosition() const;

    double getLastCorrectionPixels() const;
    double getDistancePixels() const;
    double getDistanceMeters() const;

    const std::vector<cv::Point2f>&
        getSiftTrajectory() const;

    const std::vector<cv::Point2f>&
        getBinsTrajectory() const;

    const std::vector<cv::Point2f>&
        getFusedTrajectory() const;

private:
    void addPoint(
        std::vector<cv::Point2f>& trajectory,
        const cv::Point2f& point);

    TrajectoryConfig cfg;

    bool initialized = false;
    bool motionAvailable = false;

    cv::Point2f binsPosition;
    cv::Point2f siftPosition;
    cv::Point2f fusedPosition;
    cv::Point2f frameStartPosition;

    double lastCorrectionPixels = 0.0;
    double distancePixels = 0.0;

    std::vector<cv::Point2f> siftTrajectory;
    std::vector<cv::Point2f> binsTrajectory;
    std::vector<cv::Point2f> fusedTrajectory;
};