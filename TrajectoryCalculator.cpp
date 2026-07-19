#include "TrajectoryCalculator.h"

#include <algorithm>

TrajectoryCalculator::TrajectoryCalculator(
    const TrajectoryConfig& config)
    : cfg(config)
{
    reset();
}

void TrajectoryCalculator::reset()
{
    initialized = false;
    motionAvailable = false;

    binsPosition =
        cv::Point2f(0.0f, 0.0f);

    siftPosition =
        cv::Point2f(0.0f, 0.0f);

    fusedPosition =
        cv::Point2f(0.0f, 0.0f);

    frameStartPosition =
        cv::Point2f(0.0f, 0.0f);

    lastCorrectionPixels = 0.0;
    distancePixels = 0.0;

    siftTrajectory.clear();
    binsTrajectory.clear();
    fusedTrajectory.clear();
}

void TrajectoryCalculator::addPoint(
    std::vector<cv::Point2f>& trajectory,
    const cv::Point2f& point)
{
    if (!trajectory.empty()) {
        double movement =
            cv::norm(
                point -
                trajectory.back());

        if (movement <
            cfg.minimumPointDistance) {
            return;
        }
    }

    trajectory.push_back(point);

    if (trajectory.size() >
        cfg.maximumHistorySize) {
        trajectory.erase(
            trajectory.begin());
    }
}

void TrajectoryCalculator::predict(
    const TrajectoryMotionData& motion)
{
    motionAvailable =
        motion.available;

    if (!initialized) {
        return;
    }

    frameStartPosition =
        fusedPosition;

    // Если симуляции пока нет, ничего не двигаем.
    if (!motion.available ||
        motion.dt <= 0.0) {
        return;
    }

    // Восток соответствует положительному X карты.
    binsPosition.x +=
        static_cast<float>(
            motion.velocityEast *
            cfg.pixelsPerMeter *
            motion.dt);

    // Север направлен вверх, а Y изображения — вниз.
    binsPosition.y -=
        static_cast<float>(
            motion.velocityNorth *
            cfg.pixelsPerMeter *
            motion.dt);

    fusedPosition =
        binsPosition;

    addPoint(
        binsTrajectory,
        binsPosition);
}

void TrajectoryCalculator::correct(
    bool siftValid,
    const cv::Point2f& siftMeasurement)
{
    lastCorrectionPixels = 0.0;

    if (!initialized) {
        if (!siftValid) {
            return;
        }

        initialized = true;

        siftPosition =
            siftMeasurement;

        binsPosition =
            siftMeasurement;

        fusedPosition =
            siftMeasurement;

        frameStartPosition =
            siftMeasurement;

        addPoint(
            siftTrajectory,
            siftPosition);

        addPoint(
            fusedTrajectory,
            fusedPosition);

        return;
    }

    if (siftValid) {
        siftPosition =
            siftMeasurement;

        cv::Point2f correction =
            siftPosition -
            binsPosition;

        lastCorrectionPixels =
            cv::norm(correction);

        // Пока симуляции нет, gain = 1.
        // Поэтому результат полностью совпадает с текущим SIFT.
        double gain =
            motionAvailable
            ? cfg.siftCorrectionGain
            : 1.0;

        fusedPosition =
            binsPosition +
            correction *
            static_cast<float>(gain);

        binsPosition =
            fusedPosition;

        addPoint(
            siftTrajectory,
            siftPosition);
    }
    else {
        // При наличии БИНС позиция уже была рассчитана
        // в predict(). Без симуляции позиция не меняется.
        fusedPosition =
            binsPosition;
    }

    distancePixels +=
        cv::norm(
            fusedPosition -
            frameStartPosition);

    addPoint(
        fusedTrajectory,
        fusedPosition);
}

bool TrajectoryCalculator::isInitialized() const
{
    return initialized;
}

bool TrajectoryCalculator::hasMotionData() const
{
    return motionAvailable;
}

const cv::Point2f&
TrajectoryCalculator::getBinsPosition() const
{
    return binsPosition;
}

const cv::Point2f&
TrajectoryCalculator::getSiftPosition() const
{
    return siftPosition;
}

const cv::Point2f&
TrajectoryCalculator::getFusedPosition() const
{
    return fusedPosition;
}

double
TrajectoryCalculator::getLastCorrectionPixels() const
{
    return lastCorrectionPixels;
}

double TrajectoryCalculator::getDistancePixels() const
{
    return distancePixels;
}

double TrajectoryCalculator::getDistanceMeters() const
{
    if (cfg.pixelsPerMeter <= 0.0) {
        return 0.0;
    }

    return
        distancePixels /
        cfg.pixelsPerMeter;
}

const std::vector<cv::Point2f>&
TrajectoryCalculator::getSiftTrajectory() const
{
    return siftTrajectory;
}

const std::vector<cv::Point2f>&
TrajectoryCalculator::getBinsTrajectory() const
{
    return binsTrajectory;
}

const std::vector<cv::Point2f>&
TrajectoryCalculator::getFusedTrajectory() const
{
    return fusedTrajectory;
}