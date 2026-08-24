#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
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

    // Истинная траектория симулятора отображается на карте
    // относительно точки первой успешной локализации SIFT.
    bool simulationMapAnchorInitialized = false;

    Point2f simulationMapAnchor(
        0.0f, 0.0f);

    SimulatedFlightState simulationTrueAnchor;

    Point2f trueSimulationMapPosition(
        0.0f, 0.0f);

    vector<Point2f> trueSimulationTrajectory;

    FlightSimulationData lastSimulationData;

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
        << "distance_px\tcorrection_px\t"
        << "sim_phase\t"
        << "true_x_m\ttrue_y_m\ttrue_z_m\t"
        << "true_vx\ttrue_vy\ttrue_vz\t"
        << "sim_bins_x_m\tsim_bins_y_m\tsim_bins_z_m\t"
        << "sim_error_m\theading_error_deg\n";

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

        lastSimulationData =
            simulationData;

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

        // При активной симуляции для расчёта траектории
        // используются скорости модельной БИНС. Сам класс INS
        // и его исходные файлы при этом не изменяются.
        if (simulationData.available) {
            binsVelocityNorth =
                simulationData.binsState.velocityX;

            binsVelocityEast =
                simulationData.binsState.velocityY;

            binsVelocityUp =
                simulationData.binsState.velocityZ;
        }

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

        double simulationHorizontalErrorMeters = 0.0;
        double simulationPositionErrorMeters = 0.0;
        double simulationHeadingErrorDegrees = 0.0;

        if (simulationData.available) {
            const double errorX =
                simulationData.binsState.x -
                simulationData.trueState.x;

            const double errorY =
                simulationData.binsState.y -
                simulationData.trueState.y;

            const double errorZ =
                simulationData.binsState.z -
                simulationData.trueState.z;

            simulationHorizontalErrorMeters =
                std::sqrt(
                    errorX * errorX +
                    errorY * errorY);

            simulationPositionErrorMeters =
                std::sqrt(
                    errorX * errorX +
                    errorY * errorY +
                    errorZ * errorZ);

            const double headingDifference =
                std::atan2(
                    std::sin(
                        simulationData.binsState.heading -
                        simulationData.trueState.heading),
                    std::cos(
                        simulationData.binsState.heading -
                        simulationData.trueState.heading));

            simulationHeadingErrorDegrees =
                headingDifference *
                180.0 /
                3.14159265358979323846;
        }

        if (simulationData.available &&
            initialPositionFound) {

            if (!simulationMapAnchorInitialized) {
                simulationMapAnchorInitialized = true;
                simulationMapAnchor = fusedPosition;
                simulationTrueAnchor =
                    simulationData.trueState;

                trueSimulationTrajectory.clear();
            }

            // Восток модели соответствует X изображения,
            // север — отрицательному Y изображения.
            trueSimulationMapPosition.x =
                simulationMapAnchor.x +
                static_cast<float>(
                    (simulationData.trueState.y -
                        simulationTrueAnchor.y) *
                    PIXELS_PER_METER);

            trueSimulationMapPosition.y =
                simulationMapAnchor.y -
                static_cast<float>(
                    (simulationData.trueState.x -
                        simulationTrueAnchor.x) *
                    PIXELS_PER_METER);

            if (trueSimulationTrajectory.empty() ||
                norm(
                    trueSimulationMapPosition -
                    trueSimulationTrajectory.back()) >=
                MINIMUM_TRAJECTORY_POINT_DISTANCE) {

                trueSimulationTrajectory.push_back(
                    trueSimulationMapPosition);

                if (trueSimulationTrajectory.size() >
                    static_cast<size_t>(
                        MAXIMUM_TRAJECTORY_HISTORY_SIZE)) {

                    trueSimulationTrajectory.erase(
                        trueSimulationTrajectory.begin());
                }
            }
        }

        string mode =
            globalInitializationMode
            ? "INITIAL_GLOBAL_SEARCH"
            : "BINS_AREA_ONLY";

        const double trueHorizontalSpeed =
            std::sqrt(
                simulationData.trueState.velocityX *
                simulationData.trueState.velocityX +
                simulationData.trueState.velocityY *
                simulationData.trueState.velocityY);

        cout << "----------------------------------------"
            << endl;

        cout << fixed << setprecision(2);

        cout << "Frame / time:       "
            << frameID << " / "
            << currentTime << " s" << endl;

        cout << "Processing mode:    "
            << mode << endl;

        cout << "Simulation:         "
            << (simulationData.available
                ? "ACTIVE"
                : "OFF")
            << endl;

        cout << "Flight phase:       "
            << simulationData.phase << endl;

        cout << "True position N/E/H: "
            << simulationData.trueState.x << " / "
            << simulationData.trueState.y << " / "
            << simulationData.trueState.z
            << " m" << endl;

        cout << "True velocity N/E/U: "
            << simulationData.trueState.velocityX << " / "
            << simulationData.trueState.velocityY << " / "
            << simulationData.trueState.velocityZ
            << " m/s" << endl;

        cout << "True ground speed:  "
            << trueHorizontalSpeed
            << " m/s" << endl;

        cout << "BINS position N/E/H: "
            << simulationData.binsState.x << " / "
            << simulationData.binsState.y << " / "
            << simulationData.binsState.z
            << " m" << endl;

        cout << "BINS velocity N/E/U: "
            << binsVelocityNorth << " / "
            << binsVelocityEast << " / "
            << binsVelocityUp
            << " m/s" << endl;

        cout << "BINS error H / 3D: "
            << simulationHorizontalErrorMeters << " / "
            << simulationPositionErrorMeters
            << " m" << endl;

        cout << "Heading error:      "
            << simulationHeadingErrorDegrees
            << " deg" << endl;

        cout << "True accel X/Y/Z:   "
            << simulationData.trueAcceleration.x << " / "
            << simulationData.trueAcceleration.y << " / "
            << simulationData.trueAcceleration.z
            << " m/s^2" << endl;

        cout << "IMU accel X/Y/Z:    "
            << simulationData.acceleration.x << " / "
            << simulationData.acceleration.y << " / "
            << simulationData.acceleration.z
            << " m/s^2" << endl;

        cout << "True gyro X/Y/Z:    "
            << simulationData.trueAngularVelocity.x << " / "
            << simulationData.trueAngularVelocity.y << " / "
            << simulationData.trueAngularVelocity.z
            << " rad/s" << endl;

        cout << "IMU gyro X/Y/Z:     "
            << simulationData.angularVelocity.x << " / "
            << simulationData.angularVelocity.y << " / "
            << simulationData.angularVelocity.z
            << " rad/s" << endl;

        cout << "SIFT:               "
            << (siftValid
                ? "ACTIVE"
                : "LOST")
            << endl;

        if (!siftValid) {
            cout << "SIFT reason:        "
                << failureReason << endl;
        }

        cout << "Matches / inliers:  "
            << goodMatchesCount << " / "
            << inlierCount
            << " (" << inlierRatio << ")"
            << endl;

        cout << "BINS search radius: "
            << binsErrorRadius << " px"
            << endl;

        cout << "Map position:       ("
            << fusedPosition.x << ", "
            << fusedPosition.y << ") px"
            << endl;

        cout << "SIFT correction:    "
            << trajectoryCalculator
            .getLastCorrectionPixels()
            << " px" << endl;

        cout << "Map distance:       "
            << trajectoryCalculator
            .getDistancePixels()
            << " px" << endl;

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
            << "\t"

            << simulationData.phase << "\t"
            << simulationData.trueState.x << "\t"
            << simulationData.trueState.y << "\t"
            << simulationData.trueState.z << "\t"
            << simulationData.trueState.velocityX << "\t"
            << simulationData.trueState.velocityY << "\t"
            << simulationData.trueState.velocityZ << "\t"
            << simulationData.binsState.x << "\t"
            << simulationData.binsState.y << "\t"
            << simulationData.binsState.z << "\t"
            << simulationPositionErrorMeters << "\t"
            << simulationHeadingErrorDegrees
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

        constexpr int PANEL_HEIGHT = 270;
        constexpr int MINIMUM_INTERFACE_WIDTH = 1050;

        int interfaceWidth =
            std::max(
                mapWidth + frameWidth,
                MINIMUM_INTERFACE_WIDTH);

        Mat result = Mat::zeros(
            Size(
                interfaceWidth,
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

        // Зелёная линия — эталонная траектория симулятора.
        for (size_t index = 1;
            index < trueSimulationTrajectory.size();
            index++) {

            line(
                result,
                trueSimulationTrajectory[index - 1],
                trueSimulationTrajectory[index],
                Scalar(0, 220, 0),
                2,
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

        if (simulationMapAnchorInitialized) {
            circle(
                result,
                trueSimulationMapPosition,
                7,
                Scalar(0, 220, 0),
                FILLED,
                LINE_AA);

            putText(
                result,
                "TRUE",
                Point(
                    cvRound(
                        trueSimulationMapPosition.x + 9),
                    cvRound(
                        trueSimulationMapPosition.y - 7)),
                FONT_HERSHEY_SIMPLEX,
                0.48,
                Scalar(0, 220, 0),
                2,
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
            contentHeight + 28;

        char text[500];

        sprintf_s(
            text,
            sizeof(text),
            "SIMULATION: %s | Phase: %s | Time: %.1f s | Frame: %d",
            simulationData.available
            ? "ACTIVE"
            : "OFF",
            simulationData.phase.c_str(),
            currentTime,
            frameID);

        putText(
            result,
            text,
            Point(12, textY),
            FONT_HERSHEY_SIMPLEX,
            0.58,
            simulationData.available
            ? Scalar(0, 255, 0)
            : Scalar(0, 0, 255),
            2,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "TRUE  position N/E/H: %7.2f  %7.2f  %6.2f m | heading: %6.1f deg",
            simulationData.trueState.x,
            simulationData.trueState.y,
            simulationData.trueState.z,
            simulationData.trueState.heading *
            180.0 /
            3.14159265358979323846);

        putText(
            result,
            text,
            Point(12, textY + 29),
            FONT_HERSHEY_SIMPLEX,
            0.48,
            Scalar(0, 220, 0),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "TRUE  velocity N/E/U: %6.2f  %6.2f  %6.2f m/s | ground speed: %.2f m/s",
            simulationData.trueState.velocityX,
            simulationData.trueState.velocityY,
            simulationData.trueState.velocityZ,
            trueHorizontalSpeed);

        putText(
            result,
            text,
            Point(12, textY + 58),
            FONT_HERSHEY_SIMPLEX,
            0.48,
            Scalar(0, 220, 0),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "BINS  pos N/E/H: %7.2f  %7.2f  %6.2f m | error H/3D: %.3f/%.3f m | heading: %.2f deg",
            simulationData.binsState.x,
            simulationData.binsState.y,
            simulationData.binsState.z,
            simulationHorizontalErrorMeters,
            simulationPositionErrorMeters,
            simulationHeadingErrorDegrees);

        putText(
            result,
            text,
            Point(12, textY + 87),
            FONT_HERSHEY_SIMPLEX,
            0.48,
            Scalar(255, 0, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "BINS  velocity N/E/U: %6.2f  %6.2f  %6.2f m/s",
            binsVelocityNorth,
            binsVelocityEast,
            binsVelocityUp);

        putText(
            result,
            text,
            Point(12, textY + 116),
            FONT_HERSHEY_SIMPLEX,
            0.48,
            Scalar(255, 0, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "SENSORS true A:[%.2f %.2f %.2f] | IMU A:[%.2f %.2f %.2f] m/s2 | gyro Z true/IMU: %.4f/%.4f",
            simulationData.trueAcceleration.x,
            simulationData.trueAcceleration.y,
            simulationData.trueAcceleration.z,
            simulationData.acceleration.x,
            simulationData.acceleration.y,
            simulationData.acceleration.z,
            simulationData.trueAngularVelocity.z,
            simulationData.angularVelocity.z);

        putText(
            result,
            text,
            Point(12, textY + 145),
            FONT_HERSHEY_SIMPLEX,
            0.46,
            Scalar(210, 210, 210),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "SIFT: %s | matches/inliers: %d/%d | ratio: %.2f | mode: %s",
            siftValid
            ? "ACTIVE"
            : "LOST",
            goodMatchesCount,
            inlierCount,
            inlierRatio,
            mode.c_str());

        putText(
            result,
            text,
            Point(12, textY + 174),
            FONT_HERSHEY_SIMPLEX,
            0.48,
            siftValid
            ? Scalar(0, 255, 255)
            : Scalar(0, 100, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "NAV   map position: (%.1f, %.1f) px | search radius: %.1f px | correction: %.1f px",
            fusedPosition.x,
            fusedPosition.y,
            binsErrorRadius,
            trajectoryCalculator
            .getLastCorrectionPixels());

        putText(
            result,
            text,
            Point(12, textY + 203),
            FONT_HERSHEY_SIMPLEX,
            0.46,
            Scalar(255, 255, 255),
            1,
            LINE_AA);

        sprintf_s(
            text,
            sizeof(text),
            "Legend: TRUE=green | BINS=magenta | FUSED=cyan | SIFT=yellow | ESC=exit | S=screenshot");

        putText(
            result,
            text,
            Point(12, textY + 232),
            FONT_HERSHEY_SIMPLEX,
            0.44,
            Scalar(180, 180, 180),
            1,
            LINE_AA);

        if (!siftValid) {
            sprintf_s(
                text,
                sizeof(text),
                "SIFT reason: %s",
                failureReason.c_str());

            putText(
                result,
                text,
                Point(550, textY + 174),
                FONT_HERSHEY_SIMPLEX,
                0.43,
                Scalar(0, 130, 255),
                1,
                LINE_AA);
        }

        imshow(
            "Drone Navigation - Simulation + BINS + SIFT",
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

    cout << "Map distance:    "
        << fixed << setprecision(1)
        << trajectoryCalculator
        .getDistancePixels()
        << " px"
        << endl;

    cout << "Simulation:      "
        << (lastSimulationData.available
            ? "ACTIVE"
            : "OFF")
        << endl;

    if (lastSimulationData.available) {
        const double finalErrorX =
            lastSimulationData.binsState.x -
            lastSimulationData.trueState.x;

        const double finalErrorY =
            lastSimulationData.binsState.y -
            lastSimulationData.trueState.y;

        const double finalErrorZ =
            lastSimulationData.binsState.z -
            lastSimulationData.trueState.z;

        const double finalSimulationError =
            std::sqrt(
                finalErrorX * finalErrorX +
                finalErrorY * finalErrorY +
                finalErrorZ * finalErrorZ);

        cout << "Final phase:     "
            << lastSimulationData.phase << endl;

        cout << "True position:   ("
            << lastSimulationData.trueState.x << ", "
            << lastSimulationData.trueState.y << ", "
            << lastSimulationData.trueState.z << ") m"
            << endl;

        cout << "BINS position:   ("
            << lastSimulationData.binsState.x << ", "
            << lastSimulationData.binsState.y << ", "
            << lastSimulationData.binsState.z << ") m"
            << endl;

        cout << "BINS 3D error:   "
            << finalSimulationError << " m"
            << endl;
    }

    cout << "Detailed log:    "
        << TRAJECTORY_PATH << endl;

    cout << "========================================" << endl;

    trajectoryFile.close();
    cap.release();
    destroyAllWindows();

    return 0;
}