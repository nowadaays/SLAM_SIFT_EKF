#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

int main() {
    // Русская консоль
    setlocale(LC_ALL, "Russian");

    cout << "Запуск программы..." << endl;

    //Загрузка карты
    Mat map = imread("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\map.jpg", IMREAD_GRAYSCALE);

    if (map.empty()) {
        cout << "Ошибка загрузки карты!" << endl;
        return -1;
    }

    cout << "Карта загружена!" << endl;

    //Камера
    VideoCapture cap("C:\\Users\\pshen\\source\\repos\\OpenCV_SIFT\\x64\\Debug\\video.mp4");

    if (!cap.isOpened()) {
        cout << "Ошибка открытия камеры!" << endl;
        return -1;
    }

    cout << "Камера запущена!" << endl;

    //SIFT
    Ptr<SIFT> sift = SIFT::create(300); // можно менять 300–1000

    //Ключевые точки карты
    vector<KeyPoint> kp_map;
    Mat des_map;

    sift->detectAndCompute(map, noArray(), kp_map, des_map);

    cout << "Ключевых точек (map): " << kp_map.size() << endl;

    //Matcher
    BFMatcher matcher(NORM_L2);

    int frame_id = 0;

    while (true) {
        Mat frame, gray;
        cap >> frame;

        if (frame.empty()) break;

        frame_id++;

        //Пропуск кадров(берём не каждый кадр видео, а с определённой(заданной) периодичностью)
        if (frame_id % 5 != 0) continue;

        //Перводим изображение в серую картинку
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        //SIFT для кадра
        vector<KeyPoint> kp_frame;
        Mat des_frame;

        sift->detectAndCompute(gray, noArray(), kp_frame, des_frame);

        if (des_frame.empty()) continue;

        //Сравнение карты с кадрами камеры
        vector<vector<DMatch>> knn_matches;
        matcher.knnMatch(des_map, des_frame, knn_matches, 2);

        //Фильтр Лоу (сглаживаем картинку)
        vector<DMatch> good_matches;

        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() < 2) continue;

            if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
                good_matches.push_back(knn_matches[i][0]);
            }
        }

        cout << "Кадр: " << frame_id
            << " | точек: " << kp_frame.size()
            << " | совпадений: " << good_matches.size() << endl;

        //Окно приложения
        Mat result;
        drawMatches(map, kp_map, gray, kp_frame, good_matches, result);

        imshow("SIFT навигация", result);

        // выход по ESC
        if (waitKey(1) == 27) break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
