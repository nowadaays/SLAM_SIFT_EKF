#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ApplicationConfig.h"
#include "FlightSimulationProvider.h"
#include "INS.h"
#include "TrajectoryCalculator.h"
#include "VisionUtils.h"

using namespace cv;
using namespace std;
using namespace AppConfig;
using namespace VisionUtils;

int main()
{
    cout << "========================================" << endl;
    cout << " Drone Navigation System (BINS + SIFT)" << endl;
    cout << "========================================" << endl;
    cout << endl;

    INSConfig config;
    INS ins(config);

    TrajectoryConfig trajectoryConfig;

    trajectoryConfig.pixelsPerMeter =
        PIXELS_PER_METER;

    trajectoryConfig.siftCorrectionGain =
        SIFT_CORRECTION_GAIN;

    trajectoryConfig.minimumPointDistance =
        MINIMUM_TRAJECTORY_POINT_DISTANCE;

    trajectoryConfig.maximumHistorySize =
        MAXIMUM_TRAJECTORY_HISTORY_SIZE;

    TrajectoryCalculator trajectoryCalculator(
        trajectoryConfig);

    double dt = 0.033;
    double currentTime = 0.0;

    Vector3d acceleration(
        0.0, 0.0, 9.81);

    Vector3d angularVelocity(
        0.0, 0.0, 0.0);

    cout << "Loading map..." << endl;

    Mat mapOriginal =
        imread(
            MAP_PATH,
            IMREAD_GRAYSCALE);

    if (mapOriginal.empty()) {
        cout << "ERROR: Failed to load map!"
            << endl;

        cout << MAP_PATH << endl;
        return -1;
    }

    cout << "Map loaded. Size: "
        << mapOriginal.cols << "x"
        << mapOriginal.rows << endl;

    cout << "Loading video..." << endl;

    VideoCapture cap(
        VIDEO_PATH);

    if (!cap.isOpened()) {
        cout << "ERROR: Failed to open video!"
            << endl;

        cout << VIDEO_PATH << endl;
        return -1;
    }

    double videoFPS =
        cap.get(CAP_PROP_FPS);

    if (videoFPS > 0.0) {
        dt = 1.0 / videoFPS;
    }

    cout << "Video loaded. FPS: "
        << videoFPS << endl;

    Ptr<CLAHE> clahe =
        createCLAHE(
            2.0,
            Size(8, 8));

    Mat processedMap =
        preprocessImage(
            mapOriginal,
            clahe);

    Ptr<SIFT> sift =
        SIFT::create(
            SIFT_FEATURE_COUNT,
            3,
            SIFT_CONTRAST_THRESHOLD,
            SIFT_EDGE_THRESHOLD,
            1.6);

    BFMatcher matcher(
        NORM_L2);

    // Глобальные признаки используются только для
    // определения начальной позиции.
    vector<KeyPoint> globalMapKeypoints;
    Mat globalMapDescriptors;

    sift->detectAndCompute(
        processedMap,
        noArray(),
        globalMapKeypoints,
        globalMapDescriptors);

    cout << "Global map keypoints: "
        << globalMapKeypoints.size()
        << endl;

    if (globalMapDescriptors.empty()) {
        cout << "ERROR: Map has no SIFT descriptors."
            << endl;

        return -1;
    }

    bool initialPositionFound = false;
    bool siftValid = false;

    int frameID = 0;
    int lostFrames = 0;

    Point2f siftPosition(
        0.0f, 0.0f);

    Point2f binsPosition(
        0.0f, 0.0f);

    Point2f fusedPosition(
        0.0f, 0.0f);

    double binsErrorRadius =
        BINS_INITIAL_RADIUS_PX;

    int goodMatchesCount = 0;
    int inlierCount = 0;
    double inlierRatio = 0.0;

    string failureReason =
        "Initial localization required";

    vector<Point2f> trajectory;

    double processedFrames = 0.0;
    double successfulFrames = 0.0;

    ofstream trajectoryFile(
        TRAJECTORY_PATH);

    if (!trajectoryFile.is_open()) {
        cout << "ERROR: Cannot create trajectory file."
            << endl;

        return -1;
    }

    trajectoryFile
        << "frame\ttime\t"
        << "SIFT_x\tSIFT_y\t"
        << "BINS_x\tBINS_y\t"
        << "radius\tmatches\tinliers\t"
        << "inlier_ratio\tvalid\tmode\t"
        << "Fused_x\tFused_y\t"
        << "motion_available\t"
        << "velocity_n\tvelocity_e\tvelocity_u\t"
        << "distance_px\tcorrection_px\n";

    cout << endl;
    cout << "Starting processing..." << endl;
    cout << endl;

    while (true) {
        Mat frame;

        cap >> frame;

        if (frame.empty()) {
            cout << "End of video." << endl;
            break;
        }

        frameID++;
        processedFrames++;
        currentTime += dt;

        FlightSimulationData simulationData =
            getFlightSimulationData(
                currentTime,
                dt);

        Vector3d currentAcceleration =
            simulationData.available
            ? simulationData.acceleration
            : acceleration;

        Vector3d currentAngularVelocity =
            simulationData.available
            ? simulationData.angularVelocity
            : angularVelocity;

        double currentAltitude =
            simulationData.available
            ? simulationData.altitude
            : 0.0;

        double currentVerticalVelocity =
            simulationData.available
            ? simulationData.verticalVelocity
            : 0.0;

        ins.update(
            currentAcceleration,
            currentAngularVelocity,
            dt,
            currentTime,
            currentAltitude,
            currentVerticalVelocity);

        double binsVelocityNorth = 0.0;
        double binsVelocityUp = 0.0;
        double binsVelocityEast = 0.0;

        ins.getVelocity(
            binsVelocityNorth,
            binsVelocityUp,
            binsVelocityEast);

        TrajectoryMotionData motionData;

        motionData.available =
            simulationData.available;

        motionData.velocityNorth =
            binsVelocityNorth;

        motionData.velocityUp =
            binsVelocityUp;

        motionData.velocityEast =
            binsVelocityEast;

        motionData.dt = dt;

        trajectoryCalculator.predict(
            motionData);

        if (trajectoryCalculator.isInitialized()) {
            binsPosition =
                trajectoryCalculator
                .getBinsPosition();

            fusedPosition =
                trajectoryCalculator
                .getFusedPosition();
        }

        Mat gray;

        cvtColor(
            frame,
            gray,
            COLOR_BGR2GRAY);

        double frameScale = 1.0;

        Mat enlargedFrame =
            upscaleSmallFrame(
                gray,
                frameScale);

        Mat processedFrame =
            preprocessImage(
                enlargedFrame,
                clahe);

        vector<KeyPoint> frameKeypoints;
        Mat frameDescriptors;

        sift->detectAndCompute(
            processedFrame,
            noArray(),
            frameKeypoints,
            frameDescriptors);

        bool globalInitializationMode =
            !initialPositionFound;

        vector<KeyPoint> activeMapKeypoints;
        Mat activeMapDescriptors;

        Mat binsMask =
            Mat::zeros(
                processedMap.size(),
                CV_8UC1);

        double currentSearchRadius =
            binsErrorRadius;

        if (globalInitializationMode) {
            activeMapKeypoints =
                globalMapKeypoints;

            activeMapDescriptors =
                globalMapDescriptors;
        }
        else {
            circle(
                binsMask,
                binsPosition,
                cvRound(
                    currentSearchRadius),
                Scalar(255),
                FILLED,
                LINE_AA);

            sift->detectAndCompute(
                processedMap,
                binsMask,
                activeMapKeypoints,
                activeMapDescriptors);
        }

        siftValid = false;
        goodMatchesCount = 0;
        inlierCount = 0;
        inlierRatio = 0.0;

        failureReason = "Unknown";

        vector<DMatch> goodMatches;

        if (frameDescriptors.empty()) {
            failureReason =
                "Frame descriptors are empty";
        }
        else if (activeMapDescriptors.empty()) {
            failureReason =
                "Map descriptors are empty";
        }
        else {
            goodMatches =
                findMutualMatches(
                    activeMapDescriptors,
                    frameDescriptors,
                    matcher);

            goodMatchesCount =
                static_cast<int>(
                    goodMatches.size());

            Point2f candidatePosition;

            bool positionEstimated =
                estimateMapPosition(
                    activeMapKeypoints,
                    frameKeypoints,
                    goodMatches,
                    processedFrame.size(),
                    candidatePosition,
                    inlierCount,
                    inlierRatio,
                    failureReason);

            if (positionEstimated &&
                !pointInsideImage(
                    candidatePosition,
                    processedMap)) {

                positionEstimated = false;
                failureReason =
                    "Position outside map";
            }

            if (positionEstimated &&
                initialPositionFound) {

                double distanceToBINS =
                    norm(
                        candidatePosition -
                        binsPosition);

                if (distanceToBINS >
                    currentSearchRadius) {

                    positionEstimated = false;
                    failureReason =
                        "Position outside BINS area";
                }
            }

            if (positionEstimated) {
                siftPosition =
                    candidatePosition;

                trajectoryCalculator.correct(
                    true,
                    siftPosition);

                binsPosition =
                    trajectoryCalculator
                    .getBinsPosition();

                fusedPosition =
                    trajectoryCalculator
                    .getFusedPosition();

                siftValid = true;
                successfulFrames++;
                lostFrames = 0;

                binsErrorRadius =
                    BINS_MIN_RADIUS_PX;

                if (!initialPositionFound) {
                    initialPositionFound = true;

                    cout << endl;

                    cout << "INITIAL POSITION FOUND: ("
                        << siftPosition.x << ", "
                        << siftPosition.y << ")"
                        << endl;

                    cout << "Matches: "
                        << goodMatchesCount
                        << ", inliers: "
                        << inlierCount
                        << endl;

                    cout << endl;
                }
            }
        }

        if (!siftValid) {
            trajectoryCalculator.correct(
                false,
                Point2f(0.0f, 0.0f));

            if (trajectoryCalculator.isInitialized()) {
                binsPosition =
                    trajectoryCalculator
                    .getBinsPosition();

                fusedPosition =
                    trajectoryCalculator
                    .getFusedPosition();
            }
        }

        if (!siftValid) {
            lostFrames++;

            if (initialPositionFound) {
                binsErrorRadius +=
                    BINS_RADIUS_GROWTH_PX_PER_SEC *
                    dt;

                if (activeMapKeypoints.size() <
                    MIN_MAP_KEYPOINTS) {

                    binsErrorRadius +=
                        LOW_FEATURE_EXPANSION_PX;
                }

                binsErrorRadius =
                    std::min(
                        BINS_MAX_RADIUS_PX,
                        binsErrorRadius);
            }
        }

        trajectory =
            trajectoryCalculator
            .getFusedTrajectory();

        string mode =
            globalInitializationMode
            ? "INITIAL_GLOBAL_SEARCH"
            : "BINS_AREA_ONLY";

        cout << "----------------------------------------"
            << endl;

        cout << "Frame:             "
            << frameID << endl;

        cout << "Mode:              "
            << mode << endl;

        cout << "SIFT:              "
            << (siftValid
                ? "ACTIVE"
                : "LOST")
            << endl;

        cout << "Reason:            "
            << failureReason << endl;

        cout << "Frame size:        "
            << gray.cols << "x"
            << gray.rows << endl;

        cout << "SIFT frame size:   "
            << processedFrame.cols << "x"
            << processedFrame.rows << endl;

        cout << "Frame keypoints:   "
            << frameKeypoints.size() << endl;

        cout << "Map keypoints:     "
            << activeMapKeypoints.size() << endl;

        cout << "Mutual matches:    "
            << goodMatchesCount << endl;

        cout << "RANSAC inliers:    "
            << inlierCount << endl;

        cout << "Inlier ratio:      "
            << fixed << setprecision(2)
            << inlierRatio << endl;

        cout << "BINS radius:       "
            << binsErrorRadius << " px"
            << endl;

        cout << "Motion data:       "
            << (simulationData.available
                ? "AVAILABLE"
                : "STUB / NOT AVAILABLE")
            << endl;

        cout << "BINS velocity N:   "
            << binsVelocityNorth
            << " m/s" << endl;

        cout << "BINS velocity E:   "
            << binsVelocityEast
            << " m/s" << endl;

        cout << "BINS velocity U:   "
            << binsVelocityUp
            << " m/s" << endl;

        cout << "Flight distance:   "
            << fixed << setprecision(1)
            << trajectoryCalculator
            .getDistancePixels()
            << " px"
            << endl;

        cout << "SIFT correction:   "
            << trajectoryCalculator
            .getLastCorrectionPixels()
            << " px"
            << endl;

        if (initialPositionFound) {
            cout << "Position:          ("
                << cvRound(fusedPosition.x)
                << ", "
                << cvRound(fusedPosition.y)
                << ")"
                << endl;
        }

        cout << endl;

        trajectoryFile
            << frameID << "\t"
            << currentTime << "\t"

            << (siftValid
                ? siftPosition.x
                : -1.0f) << "\t"

            << (siftValid
                ? siftPosition.y
                : -1.0f) << "\t"

            << (initialPositionFound
                ? binsPosition.x
                : -1.0f) << "\t"

            << (initialPositionFound
                ? binsPosition.y
                : -1.0f) << "\t"

            << binsErrorRadius << "\t"
            << goodMatchesCount << "\t"
            << inlierCount << "\t"
            << inlierRatio << "\t"
            << siftValid << "\t"
            << mode << "\t"

            << (trajectoryCalculator.isInitialized()
                ? fusedPosition.x
                : -1.0f) << "\t"

            << (trajectoryCalculator.isInitialized()
                ? fusedPosition.y
                : -1.0f) << "\t"

            << simulationData.available << "\t"
            << binsVelocityNorth << "\t"
            << binsVelocityEast << "\t"
            << binsVelocityUp << "\t"

            << trajectoryCalculator
            .getDistancePixels() << "\t"

            << trajectoryCalculator
            .getLastCorrectionPixels()

            << "\n";

        Mat mapColor;
        Mat frameColor;

        cvtColor(
            processedMap,
            mapColor,
            COLOR_GRAY2BGR);

        cvtColor(
            processedFrame,
            frameColor,
            COLOR_GRAY2BGR);

        if (initialPositionFound) {
            Mat overlay =
                mapColor.clone();

            circle(
                overlay,
                binsPosition,
                cvRound(
                    currentSearchRadius),
                Scalar(0, 0, 255),
                FILLED,
                LINE_AA);

            addWeighted(
                overlay,
                0.22,
                mapColor,
                0.78,
                0.0,
                mapColor);

            circle(
                mapColor,
                binsPosition,
                cvRound(
                    currentSearchRadius),
                Scalar(0, 0, 255),
                2,
                LINE_AA);
        }

        int mapWidth =
            mapColor.cols;

        int frameWidth =
            frameColor.cols;

        int contentHeight =
            std::max(
                mapColor.rows,
                frameColor.rows);

        constexpr int PANEL_HEIGHT = 140;

        Mat result = Mat::zeros(
            Size(
                mapWidth + frameWidth,
                contentHeight +
                PANEL_HEIGHT),
            CV_8UC3);

        mapColor.copyTo(
            result(
                Rect(
                    0,
                    0,
                    mapColor.cols,
                    mapColor.rows)));

        frameColor.copyTo(
            result(
                Rect(
                    mapWidth,
                    0,
                    frameColor.cols,
                    frameColor.rows)));

        size_t matchesToDraw =
            std::min(
                static_cast<size_t>(40),
                goodMatches.size());

        for (size_t index = 0;
            index < matchesToDraw;
            index++) {

            const DMatch& match =
                goodMatches[index];

            Point2f mapPoint =
                activeMapKeypoints[
                    match.queryIdx].pt;

            Point2f framePoint =
                frameKeypoints[
                    match.trainIdx].pt;

            framePoint.x +=
                static_cast<float>(
                    mapWidth);

            Scalar color =
                siftValid
                ? Scalar(0, 255, 0)
                : Scalar(0, 100, 255);

            line(
                result,
                mapPoint,
                framePoint,
                color,
                1,
                LINE_AA);

            circle(
                result,
                mapPoint,
                3,
                color,
                FILLED,
                LINE_AA);
        }

        for (size_t index = 1;
            index < trajectory.size();
            index++) {

            line(
                result,
                trajectory[index - 1],
                trajectory[index],
                Scalar(255, 255, 0),
                2,
                LINE_AA);
        }

        const vector<Point2f>& binsTrajectory =
            trajectoryCalculator
            .getBinsTrajectory();

        for (size_t index = 1;
            index < binsTrajectory.size();
            index++) {

            line(
                result,
                binsTrajectory[index - 1],
                binsTrajectory[index],
                Scalar(255, 0, 255),
                1,
                LINE_AA);
        }

        if (initialPositionFound) {
            circle(
                result,
                binsPosition,
                12,
                Scalar(255, 0, 255),
                2,
                LINE_AA);

            putText(
                result,
                "BINS",
                Point(
                    cvRound(
                        binsPosition.x - 18),
                    cvRound(
                        binsPosition.y - 16)),
                FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar(255, 0, 255),
                2,
                LINE_AA);
        }

        if (siftValid) {
            circle(
                result,
                siftPosition,
                6,
                Scalar(0, 255, 255),
                FILLED,
                LINE_AA);

            putText(
                result,
                "SIFT",
                Point(
                    cvRound(
                        siftPosition.x + 8),
                    cvRound(
                        siftPosition.y + 5)),
                FONT_HERSHEY_SIMPLEX,
                0.45,
                Scalar(0, 255, 255),
                1,
                LINE_AA);
        }

        rectangle(
            result,
            Point(0, contentHeight),
            Point(
                result.cols,
                result.rows),
            Scalar(35, 35, 35),
            FILLED);

        int textY =
            contentHeight + 25;

        char text[500];

        sprintf_s(
            text,
            sizeof(text),
            "Frame: %d | Mode: %s | SIFT: %s",
            frameID,
            mode.c_str(),
            siftValid
            ? "ACTIVE"
            : "LOST");

        putText(
            result,
            text,
            Point(10, textY),
            FONT_HERSHEY_SIMPLEX,
            0.52,
            siftValid
            ? Scalar(0, 255, 0)
            : Scalar(0, 100, 255),
            2,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "Reason: %s",
            failureReason.c_str());

        putText(
            result,
            text,
            Point(10, textY + 28),
            FONT_HERSHEY_SIMPLEX,
            0.47,
            Scalar(255, 255, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "Frame points: %zu | Map points: %zu | Matches: %d | Inliers: %d (%.2f)",
            frameKeypoints.size(),
            activeMapKeypoints.size(),
            goodMatchesCount,
            inlierCount,
            inlierRatio);

        putText(
            result,
            text,
            Point(10, textY + 56),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(200, 200, 200),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "BINS radius: %.1f px | Lost: %d | Scale: %.2f | Motion: %s",
            binsErrorRadius,
            lostFrames,
            frameScale,
            simulationData.available
            ? "BINS+SIFT"
            : "SIFT ONLY");

        putText(
            result,
            text,
            Point(10, textY + 84),
            FONT_HERSHEY_SIMPLEX,
            0.45,
            Scalar(200, 200, 255),
            1,
            LINE_AA);

        if (initialPositionFound) {
            sprintf_s(
                text,
                sizeof(text),
                "Position: (%.1f, %.1f) px | Distance: %.1f px | Correction: %.1f px",
                fusedPosition.x,
                fusedPosition.y,
                trajectoryCalculator
                .getDistancePixels(),
                trajectoryCalculator
                .getLastCorrectionPixels());

            putText(
                result,
                text,
                Point(10, textY + 112),
                FONT_HERSHEY_SIMPLEX,
                0.47,
                Scalar(0, 255, 255),
                1,
                LINE_AA);
        }

        imshow(
            "Drone Navigation - Reliable BINS + SIFT",
            result);

        int key =
            waitKey(1);

        if (key == 27) {
            break;
        }

        if (key == 's' ||
            key == 'S') {

            string screenshotPath =
                string(SCREENSHOT_DIRECTORY) +
                "screenshot_" +
                to_string(frameID) +
                ".png";

            imwrite(
                screenshotPath,
                result);
        }
    }

    double successPercent =
        processedFrames > 0.0
        ? successfulFrames /
        processedFrames *
        100.0
        : 0.0;

    cout << endl;
    cout << "========================================" << endl;
    cout << "FINAL STATISTICS" << endl;
    cout << "========================================" << endl;

    cout << "Total frames:    "
        << processedFrames << endl;

    cout << "Successful SIFT: "
        << successfulFrames << endl;

    cout << "Success:         "
        << fixed << setprecision(1)
        << successPercent << "%"
        << endl;

    cout << "Flight distance: "
        << fixed << setprecision(1)
        << trajectoryCalculator
        .getDistancePixels()
        << " px"
        << endl;

    cout << "Motion data:     "
        << (trajectoryCalculator.hasMotionData()
            ? "AVAILABLE"
            : "NOT AVAILABLE / SIFT ONLY")
        << endl;

    cout << "========================================" << endl;

    trajectoryFile.close();
    cap.release();
    destroyAllWindows();

    return 0;
}