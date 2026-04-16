#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
    setlocale(LC_ALL, "Russian");

    cout << "Запуск программы..." << endl;

    // Загрузка карты
    Mat map = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg", IMREAD_GRAYSCALE);

    if (map.empty()) {
        cout << "Ошибка загрузки карты!" << endl;
        return -1;
    }

    cout << "Карта загружена!" << endl;

    // Улучшение карты
    equalizeHist(map, map);
    GaussianBlur(map, map, Size(3, 3), 0);

    // Видео (камера)
    VideoCapture cap("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\video.mp4");

    if (!cap.isOpened()) {
        cout << "Ошибка открытия видео!" << endl;
        return -1;
    }

    cout << "Видео запущено!" << endl;

    // SIFT (увеличили количество точек)
    Ptr<SIFT> sift = SIFT::create(800);

    // Ключевые точки карты
    vector<KeyPoint> kp_map;
    Mat des_map;

    sift->detectAndCompute(map, noArray(), kp_map, des_map);

    cout << "Ключевых точек (map): " << kp_map.size() << endl;

    BFMatcher matcher(NORM_L2);

    int frame_id = 0;

    // Последняя позиция
    Point2f last_position;
    bool has_position = false;

    while (true) {
        Mat frame, gray;
        cap >> frame;

        if (frame.empty()) break;

        frame_id++;

        // Пропуск кадров
        if (frame_id % 5 != 0) continue;

        // В серый
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Улучшение кадра
        equalizeHist(gray, gray);
        GaussianBlur(gray, gray, Size(3, 3), 0);

        // SIFT для кадра
        vector<KeyPoint> kp_frame;
        Mat des_frame;

        sift->detectAndCompute(gray, noArray(), kp_frame, des_frame);

        if (des_frame.empty()) continue;

        // Сопоставление
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(des_map, des_frame, knn_matches, 2);

        vector<DMatch> good_matches;

        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() < 2) continue;

            if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        cout << "Кадр: " << frame_id
            << " | точек: " << kp_frame.size()
            << " | совпадений: " << good_matches.size();

        Point2f current_position;
        bool valid_position = false;

        // Проверка через гомографию
        if (good_matches.size() >= 10) {
            vector<Point2f> pts_map, pts_frame;

            for (auto& m : good_matches) {
                pts_map.push_back(kp_map[m.queryIdx].pt);
                pts_frame.push_back(kp_frame[m.trainIdx].pt);
            }

            Mat H = findHomography(pts_frame, pts_map, RANSAC);

            if (!H.empty()) {
                // Центр кадра
                vector<Point2f> frame_center(1, Point2f(gray.cols / 2, gray.rows / 2));
                vector<Point2f> map_center;

                perspectiveTransform(frame_center, map_center, H);

                current_position = map_center[0];
                valid_position = true;

                cout << " | Гомография OK";
            }
            else {
                cout << " | Гомография FAIL";
            }
        }

        // Логика принятия решения
        if (valid_position) {
            if (has_position) {
                // Сглаживание
                current_position = 0.8 * last_position + 0.2 * current_position;
            }

            last_position = current_position;
            has_position = true;

            cout << " | Позиция: (" << current_position.x << ", " << current_position.y << ")";
        }
        else if (has_position) {
            cout << " | Используем прошлую позицию: ("
                << last_position.x << ", " << last_position.y << ")";
        }
        else {
            cout << " | Позиция не определена";
        }

        cout << endl;

        // Отрисовка
        Mat result;
        drawMatches(map, kp_map, gray, kp_frame, good_matches, result);

        // Рисуем точку на карте
        if (has_position) {
            circle(result, last_position, 10, Scalar(0, 255, 0), -1);
        }

        imshow("SIFT навигация", result);

        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}