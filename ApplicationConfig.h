#pragma once

namespace AppConfig {

    // Пути к файлам.
    constexpr const char* MAP_PATH =
        "C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\map.jpg";

    constexpr const char* VIDEO_PATH =
        "C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\video.mp4";

    constexpr const char* TRAJECTORY_PATH =
        "C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\trajectory.txt";

    constexpr const char* SCREENSHOT_DIRECTORY =
        "C:\\Diplom\\OpenCV_SIFT\\x64\\Debug\\";

    // Параметры SIFT.
    constexpr int SIFT_FEATURE_COUNT = 5000;
    constexpr double SIFT_CONTRAST_THRESHOLD = 0.015;
    constexpr double SIFT_EDGE_THRESHOLD = 15.0;
    constexpr float LOWE_RATIO = 0.82f;

    constexpr int MIN_MATCHES = 5;
    constexpr int MIN_INLIERS = 4;
    constexpr double MIN_INLIER_RATIO = 0.50;
    constexpr double MIN_POINT_SPREAD = 0.12;

    // Параметры области неопределённости БИНС.
    constexpr double BINS_INITIAL_RADIUS_PX = 160.0;
    constexpr double BINS_MIN_RADIUS_PX = 100.0;
    constexpr double BINS_MAX_RADIUS_PX = 700.0;
    constexpr double BINS_RADIUS_GROWTH_PX_PER_SEC = 70.0;
    constexpr double LOW_FEATURE_EXPANSION_PX = 10.0;

    constexpr int MIN_MAP_KEYPOINTS = 15;

    // Параметры расчёта траектории.
    constexpr double PIXELS_PER_METER = 3.0;
    constexpr double SIFT_CORRECTION_GAIN = 0.75;
    constexpr double MINIMUM_TRAJECTORY_POINT_DISTANCE = 0.5;
    constexpr int MAXIMUM_TRAJECTORY_HISTORY_SIZE = 300;

}