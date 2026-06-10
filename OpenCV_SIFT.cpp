#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <windows.h>
#include "INS.h"
#include <fstream>
#include <iomanip>

using namespace cv;
using namespace std;

int main() {
    // NO setlocale - remove Russian encoding completely

    cout << "========================================" << endl;
    cout << "  Drone Navigation System (INS + SIFT)" << endl;
    cout << "========================================" << endl << endl;

    // ========== INS INITIALIZATION ==========
    INSConfig config;
    INS ins(config);

    // Time parameters for INS
    double dt = 0.033;  // approximately 30 FPS
    double time = 0.0;

    bool initial_position_set = false;

    // ========== LOAD MAP ==========
    cout << "Loading map..." << endl;
    Mat map = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg", IMREAD_GRAYSCALE);

    if (map.empty()) {
        cout << "ERROR: Failed to load map!" << endl;
        cout << "Check path: C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg" << endl;
        return -1;
    }

    cout << "Map loaded! Size: " << map.cols << "x" << map.rows << endl;

    // Map enhancement
    equalizeHist(map, map);
    GaussianBlur(map, map, Size(3, 3), 0);

    // ========== LOAD VIDEO ==========
    cout << "Loading video..." << endl;
    VideoCapture cap("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\video.mp4");

    if (!cap.isOpened()) {
        cout << "ERROR: Failed to open video!" << endl;
        cout << "Check path: C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\video.mp4" << endl;
        return -1;
    }

    cout << "Video loaded! FPS: " << cap.get(CAP_PROP_FPS) << endl;

    // ========== SIFT INITIALIZATION ==========
    cout << "Initializing SIFT..." << endl;
    Ptr<SIFT> sift = SIFT::create(2000);

    // Map keypoints
    vector<KeyPoint> kp_map;
    Mat des_map;
    sift->detectAndCompute(map, noArray(), kp_map, des_map);

    cout << "Map keypoints: " << kp_map.size() << endl;

    // Matcher
    BFMatcher matcher(NORM_L2);

    // ========== TRACKING VARIABLES ==========
    int frame_id = 0;
    Point2f sift_position(0, 0);
    Point2f ins_position(0, 0);
    Point2f fused_position(0, 0);
    bool sift_valid = false;
    int good_matches_count = 0;
    const int MIN_MATCHES = 10;

    vector<Point2f> position_history;
    const int HISTORY_SIZE = 5;

    // Trajectory for drawing
    vector<Point2f> trajectory;

    // Statistics
    double total_sift_found = 0;
    double total_frames_processed = 0;
    double min_matches = 999999;
    double max_matches = 0;
    double avg_matches = 0;

    // File for trajectory recording
    ofstream traj_file("trajectory.txt");
    traj_file << "frame\ttime\tSIFT_x\tSIFT_y\tINS_x\tINS_y\tFused_x\tFused_y\tlat\tlon\talt\tspeed\tSIFT_valid\tmatches\n";

    // ========== IMU DATA (in reality get from drone) ==========
    Vector3d A(0.0, 0.0, 9.81);
    Vector3d W(0.0, 0.0, 0.0);

    // ========== MAIN LOOP ==========
    cout << endl << "Starting video processing..." << endl << endl;
    cout << "Match threshold: " << MIN_MATCHES << " (SIFT ACTIVE if >= " << MIN_MATCHES << ")" << endl << endl;

    while (true) {
        Mat frame, gray;
        cap >> frame;

        if (frame.empty()) {
            cout << "End of video!" << endl;
            break;
        }

        frame_id++;
        time += dt;
        total_frames_processed++;

        // ===== 1. INS UPDATE =====
        ins.update(A, W, dt, time, 0.0, 0.0);

        double lat, lon, alt;
        double vn, vh, ve;
        ins.getPosition(lat, lon, alt);
        ins.getVelocity(vn, vh, ve);

        double scale = 100000;
        ins_position.x = lon * scale;
        ins_position.y = lat * scale;

        // ===== 2. FRAME PROCESSING (SIFT) =====
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        GaussianBlur(gray, gray, Size(3, 3), 0);

        vector<KeyPoint> kp_frame;
        Mat des_frame;
        sift->detectAndCompute(gray, noArray(), kp_frame, des_frame);

        sift_valid = false;
        good_matches_count = 0;
        vector<DMatch> good_matches;

        if (!des_frame.empty() && kp_map.size() > 0) {
            vector<vector<DMatch>> knn_matches;
            matcher.knnMatch(des_map, des_frame, knn_matches, 2);

            for (size_t i = 0; i < knn_matches.size(); i++) {
                if (knn_matches[i].size() >= 2) {
                    if (knn_matches[i][0].distance < 0.7 * knn_matches[i][1].distance) {
                        good_matches.push_back(knn_matches[i][0]);
                    }
                }
            }

            good_matches_count = good_matches.size();

            if (good_matches_count > 0) {
                if (good_matches_count < min_matches) min_matches = good_matches_count;
                if (good_matches_count > max_matches) max_matches = good_matches_count;
                avg_matches = (avg_matches * (total_sift_found)+good_matches_count) / (total_sift_found + 1);
            }

            if (good_matches_count >= MIN_MATCHES) {
                vector<Point2f> pts_map, pts_frame;
                for (auto& m : good_matches) {
                    pts_map.push_back(kp_map[m.queryIdx].pt);
                    pts_frame.push_back(kp_frame[m.trainIdx].pt);
                }

                Mat H = findHomography(pts_frame, pts_map, RANSAC, 3.0);
                if (!H.empty()) {
                    vector<Point2f> frame_center(1, Point2f(gray.cols / 2, gray.rows / 2));
                    vector<Point2f> map_center;
                    perspectiveTransform(frame_center, map_center, H);
                    sift_position = map_center[0];
                    sift_valid = true;
                    total_sift_found++;

                    if (!initial_position_set) {
                        ins.resetPosition(sift_position.y / scale, sift_position.x / scale, 0.0);
                        initial_position_set = true;
                        cout << "\n>>> INITIAL POSITION SET: (" << sift_position.x << ", " << sift_position.y << ")" << endl;
                        cout << "    (found " << good_matches_count << " matches, threshold " << MIN_MATCHES << ")" << endl << endl;
                    }
                    else {
                        double alpha = 0.3;
                        double target_lat = sift_position.y / scale;
                        double target_lon = sift_position.x / scale;
                        double current_lat = lat;
                        double current_lon = lon;
                        double new_lat = current_lat * (1 - alpha) + target_lat * alpha;
                        double new_lon = current_lon * (1 - alpha) + target_lon * alpha;
                        ins.resetPosition(new_lat, new_lon, 0.0);
                    }
                }
            }
        }

        // ===== 3. DATA FUSION =====
        if (sift_valid) {
            fused_position.x = 0.7 * sift_position.x + 0.3 * ins_position.x;
            fused_position.y = 0.7 * sift_position.y + 0.3 * ins_position.y;
        }
        else {
            fused_position = ins_position;
        }

        // Smoothing trajectory
        position_history.push_back(fused_position);
        if (position_history.size() > HISTORY_SIZE) {
            position_history.erase(position_history.begin());
        }

        Point2f smoothed_position(0, 0);
        for (const auto& pos : position_history) {
            smoothed_position.x += pos.x;
            smoothed_position.y += pos.y;
        }
        if (position_history.size() > 0) {
            smoothed_position.x /= position_history.size();
            smoothed_position.y /= position_history.size();
        }

        trajectory.push_back(smoothed_position);
        if (trajectory.size() > 200) {
            trajectory.erase(trajectory.begin());
        }

        // ===== 4. CONSOLE OUTPUT =====
        double speed = sqrt(vn * vn + ve * ve);
        string mode = sift_valid ? "FUSION" : "INS_ONLY";
        string sift_status = sift_valid ? "YES" : "NO";

        cout << "--------------------------------------------------" << endl;
        cout << "Frame #" << frame_id << " | Time: " << fixed << setprecision(1) << time << "s" << endl;
        cout << "--------------------------------------------------" << endl;
        cout << "  SIFT Status:      " << sift_status << endl;
        cout << "  Matches:          " << good_matches_count << " / " << MIN_MATCHES << endl;
        cout << "  Position (X,Y):   (" << (int)smoothed_position.x << ", " << (int)smoothed_position.y << ") px" << endl;
        cout << "  Mode:             " << mode << endl;
        cout << "--------------------------------------------------" << endl;
        cout << "  INS Latitude:     " << fixed << setprecision(6) << lat << " deg" << endl;
        cout << "  INS Longitude:    " << lon << " deg" << endl;
        cout << "  INS Altitude:     " << fixed << setprecision(1) << alt << " m" << endl;
        cout << "  INS Speed:        " << (int)speed << " m/s" << endl;
        cout << "--------------------------------------------------" << endl;

        // Warnings
        if (!sift_valid && good_matches_count > 0 && good_matches_count < MIN_MATCHES) {
            cout << "[WARNING] Not enough matches! Need " << MIN_MATCHES
                << ", got " << good_matches_count << endl;
        }

        if (sift_valid && initial_position_set) {
            double error_x = abs(sift_position.x - ins_position.x);
            double error_y = abs(sift_position.y - ins_position.y);
            if (error_x > 50 || error_y > 50) {
                cout << "[CORRECTION] INS deviation (" << (int)error_x << ", " << (int)error_y << ") px" << endl;
            }
        }

        cout << endl;

        // Write to file
        traj_file << frame_id << "\t"
            << time << "\t"
            << (sift_valid ? sift_position.x : -1) << "\t"
            << (sift_valid ? sift_position.y : -1) << "\t"
            << ins_position.x << "\t" << ins_position.y << "\t"
            << fused_position.x << "\t" << fused_position.y << "\t"
            << lat << "\t" << lon << "\t" << alt << "\t"
            << speed << "\t"
            << sift_valid << "\t"
            << good_matches_count << "\n";

        // ===== 5. VISUALIZATION =====
        Mat map_color, gray_color;
        cvtColor(map, map_color, COLOR_GRAY2BGR);
        cvtColor(gray, gray_color, COLOR_GRAY2BGR);

        int map_w = map_color.cols;
        int frame_w = gray_color.cols;
        int h = max(map_color.rows, gray_color.rows);

        Mat result = Mat::zeros(Size(map_w + frame_w, h + 100), map_color.type());
        map_color.copyTo(result(Rect(0, 0, map_w, map_color.rows)));
        gray_color.copyTo(result(Rect(map_w, 0, frame_w, gray_color.rows)));

        // Draw SIFT matches
        if (sift_valid && good_matches.size() > 0) {
            for (size_t i = 0; i < min((size_t)50, good_matches.size()); i++) {
                Point2f pt_map = kp_map[good_matches[i].queryIdx].pt;
                Point2f pt_frame = kp_frame[good_matches[i].trainIdx].pt;
                pt_frame.x += map_w;
                line(result, pt_map, pt_frame, Scalar(0, 255, 0), 1);
                circle(result, pt_map, 3, Scalar(255, 0, 0), -1);
                circle(result, pt_frame, 3, Scalar(0, 255, 0), -1);
            }
        }

        // Draw trajectory
        for (size_t i = 1; i < trajectory.size(); i++) {
            line(result, trajectory[i - 1], trajectory[i], Scalar(255, 255, 0), 2);
        }

        // Draw positions
        if (sift_valid) {
            circle(result, sift_position, 12, Scalar(0, 255, 255), -1);
            putText(result, "SIFT", Point(sift_position.x - 20, sift_position.y - 15),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 2);
        }

        circle(result, ins_position, 12, Scalar(255, 0, 255), -1);
        putText(result, "INS", Point(ins_position.x - 15, ins_position.y - 15),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2);

        circle(result, smoothed_position, 12, Scalar(0, 255, 255), 2);
        circle(result, smoothed_position, 5, Scalar(0, 100, 255), -1);
        putText(result, "FUSED", Point(smoothed_position.x - 22, smoothed_position.y - 18),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);

        // ===== 6. INFO PANEL =====
        int y_offset = h + 20;
        rectangle(result, Point(0, h), Point(result.cols, result.rows), Scalar(40, 40, 40), -1);

        char info_text[200];
        sprintf_s(info_text, sizeof(info_text), "Frame: %d | Time: %.1f s | SIFT: %s | Matches: %d/%d | Mode: %s",
            frame_id, time, sift_valid ? "ACTIVE" : "LOST", good_matches_count, MIN_MATCHES, mode.c_str());
        putText(result, info_text, Point(10, y_offset), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        sprintf_s(info_text, sizeof(info_text), "Position: (%.0f, %.0f) px",
            smoothed_position.x, smoothed_position.y);
        putText(result, info_text, Point(10, y_offset + 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        sprintf_s(info_text, sizeof(info_text), "INS: Lat=%.6f Lon=%.6f Alt=%.1f m", lat, lon, alt);
        putText(result, info_text, Point(10, y_offset + 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);

        // Statistics
        sprintf_s(info_text, sizeof(info_text), "Stats: Found: %.0f%% | Matches: min/avg/max = %d/%.0f/%d",
            (total_sift_found / total_frames_processed) * 100,
            (int)min_matches, avg_matches, (int)max_matches);
        putText(result, info_text, Point(result.cols - 480, y_offset + 30),
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);

        // Status
        if (sift_valid) {
            putText(result, "STATUS: FUSION MODE", Point(result.cols - 250, y_offset),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
        }
        else {
            putText(result, "STATUS: INS ONLY", Point(result.cols - 250, y_offset),
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }

        // Legend
        putText(result, "Yellow: SIFT | Magenta: INS | Orange: FUSED | Cyan line: Trajectory",
            Point(10, y_offset + 90), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(200, 200, 200), 1);

        imshow("Drone Navigation - INS + SIFT", result);

        char key = waitKey(1);
        if (key == 27) break;
        if (key == 's' || key == 'S') {
            string filename = "screenshot_" + to_string(frame_id) + ".png";
            imwrite(filename, result);
            cout << "Screenshot saved: " << filename << endl;
        }
    }

    // ========== FINAL STATISTICS ==========
    cout << endl << endl;
    cout << "========================================" << endl;
    cout << "         FINAL STATISTICS" << endl;
    cout << "========================================" << endl;
    cout << "  Match threshold:      " << MIN_MATCHES << endl;
    cout << "  Total frames:         " << frame_id << endl;
    cout << "  Successful SIFT:      " << total_sift_found << " (" << fixed << setprecision(1)
        << (total_sift_found / total_frames_processed) * 100 << "%)" << endl;
    cout << "  Lost SIFT:            " << total_frames_processed - total_sift_found << " ("
        << fixed << setprecision(1) << ((total_frames_processed - total_sift_found) / total_frames_processed) * 100 << "%)" << endl;
    cout << "  Min matches:          " << (min_matches == 999999 ? 0 : (int)min_matches) << endl;
    cout << "  Max matches:          " << (int)max_matches << endl;
    cout << "  Avg matches:          " << (int)avg_matches << endl;
    cout << "========================================" << endl;
    cout << "  Trajectory file:      trajectory.txt" << endl;
    cout << "========================================" << endl;

    traj_file.close();
    cap.release();
    destroyAllWindows();

    return 0;
}