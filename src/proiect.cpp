#include <iostream>
#include <opencv2/opencv.hpp>
#include "proiect.h"
#include <fstream>
using namespace std;
using namespace cv;

image_channels_bgr break_channels(Mat source){

    int rows=source.rows, cols=source.cols;
    Mat B, G, R;
    image_channels_bgr bgr_channels;

    B=Mat(rows, cols, CV_8UC1);
    G=Mat(rows, cols, CV_8UC1);
    R=Mat(rows, cols, CV_8UC1);
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            Vec3b pixel=source.at<Vec3b>(i,j);
            B.at<uchar>(i,j)=pixel[0];
            G.at<uchar>(i,j)=pixel[1];
            R.at<uchar>(i,j)=pixel[2];
        }
    }
    bgr_channels.B=B;
    bgr_channels.G=G;
    bgr_channels.R=R;


    return bgr_channels;

}

void display_channels(image_channels_bgr bgr_channels){

    Mat B= bgr_channels.B;
    Mat G= bgr_channels.G;
    Mat R= bgr_channels.R;

    imshow("channel B", B);
    imshow("channel G", G);
    imshow("channel R", R);
}

float maxim(float a, float b) {
    return (a>b)?a:b;
}

float minim(float a, float b) {
    return (a<b)?a:b;
}

image_channels_hsv bgr_2_hsv(image_channels_bgr bgr_channels){

int rows=bgr_channels.B.rows, cols=bgr_channels.B.cols;
    Mat H, S, V;
    image_channels_hsv hsv_channels;
    H=Mat(rows, cols, CV_32FC1);
    S=Mat(rows, cols, CV_32FC1);
    V=Mat(rows, cols, CV_32FC1);
    uchar pixel_b,pixel_g,pixel_r;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uchar pixel_b = bgr_channels.B.at<uchar>(i, j);
            uchar pixel_g = bgr_channels.G.at<uchar>(i, j);
            uchar pixel_r = bgr_channels.R.at<uchar>(i, j);

            float red = pixel_r / 255.0f;
            float green = pixel_g / 255.0f;
            float blue = pixel_b / 255.0f;

            float maxim1 = maxim(maxim(red, green), blue);
            float minim1 = minim(minim(red, green), blue);
            float C = maxim1 - minim1;

            float H1 = 0.0f;
            float S1 = (maxim1 != 0.0f) ? (C / maxim1) : 0.0f;
            float V1 = maxim1;

            if (C != 0.0f) {
                if (maxim1 == red) {
                    H1 = 60.0f * (green - blue) / C;
                } else if (maxim1 == green) {
                    H1 = 120.0f + 60.0f * (blue - red) / C;
                } else if (maxim1 == blue) {
                    H1 = 240.0f + 60.0f * (red - green) / C;
                }

                if (H1 < 0.0f) {
                    H1 += 360.0f;
                }
            }

            H.at<float>(i, j) = H1;
            S.at<float>(i, j) = S1;
            V.at<float>(i, j) = V1;
        }
    }


    hsv_channels.H = H;
    hsv_channels.S = S;
    hsv_channels.V = V;
    return hsv_channels;
}


Mat color_labels(labels labels_str){


    int rows=labels_str.labels.rows, cols=labels_str.labels.cols, no_labels=labels_str.no_labels;
    Mat labels, result;
    result=Mat(rows, cols, CV_8UC3);
    Vec3b* colors= new Vec3b[no_labels+1];
    srand(time(NULL));


    for (int i=0;i<no_labels;i++) {
        colors[i]=Vec3b(rand()%256, rand()%256, rand()%256);
    }
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            int label = labels_str.labels.at<int>(i, j);
            if (label > 0 && label < no_labels)
                result.at<Vec3b>(i, j) = colors[label];
            else
                result.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
        }
    }


    return result;

}

labels BFS_labeling(Mat source){


    int rows=source.rows, cols=source.cols, no_labels;
    int label0=0;
    Mat labels(rows, cols, CV_32SC1, Scalar(0));
    const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
    const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};
    int ni=0,nj=0;

    queue<Point> bfs_queue;
    for (int i=0;i<rows;i++) {
        for (int j=0;j<cols;j++) {
            if (source.at<uchar>(i,j)==0 && labels.at<int>(i,j)==0) {
                label0++;
                labels.at<int>(i,j) = label0;
                bfs_queue.push(Point(j,i));
                while (!bfs_queue.empty()) {
                    Point p=bfs_queue.front();
                    bfs_queue.pop();
                    for (int k = 0; k < 8; k++) {
                        ni = p.y + n8_di[k];
                        nj = p.x + n8_dj[k];

                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) { // verificare granițe
                            if (source.at<uchar>(ni, nj) == 0 && labels.at<int>(ni, nj) == 0) {
                                labels.at<int>(ni, nj) = label0;
                                bfs_queue.push(Point(nj, ni));
                            }
                        }
                    }

                }
            }
        }
    }


    return {labels, label0};
}

Mat saturation_binarization(image_channels_bgr bgr_channels, int threshold) {
    image_channels_hsv hsv_channels = bgr_2_hsv(bgr_channels); // HSV din RGB
    Mat S = hsv_channels.S;

    int rows = S.rows, cols = S.cols;
    Mat binary(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float s_value = S.at<float>(i, j);
            if (s_value > threshold / 255.0f)
                binary.at<uchar>(i, j) = 0;
            else
                binary.at<uchar>(i, j) = 255;
        }
    }

    return binary;
}
Mat correct_red_eye(Mat image, Point center) {
    int radius = 10;
    for (int i = center.y - radius; i <= center.y + radius; i++) {
        for (int j = center.x - radius; j <= center.x + radius; j++) {
            if (i >= 0 && i < image.rows && j >= 0 && j < image.cols) {
                int dx = j - center.x;
                int dy = i - center.y;
                if (dx*dx + dy*dy <= radius*radius) {
                    Vec3b pixel = image.at<Vec3b>(i, j);
                    int R = pixel[2];
                    int G = pixel[1];
                    int B = pixel[0];

                    if (R > G + 30 && R > B + 30) {
                        pixel[2] = (G + B) / 2;
                        image.at<Vec3b>(i, j) = pixel;
                    }
                }
            }
        }
    }
    return image;
}


Mat draw_symmetric_eyes(Mat image, labels label_data) {
    int min_area = 50;
    int max_area = 300;
    int rows = label_data.labels.rows;
    int cols = label_data.labels.cols;
    int no_labels = label_data.no_labels;

    vector<int> area(no_labels + 1, 0);
    vector<Point2f> center(no_labels + 1, Point2f(0, 0));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int label = label_data.labels.at<int>(i, j);
            if (label > 0) {
                area[label]++;
                center[label].x += j;
                center[label].y += i;
            }
        }
    }

    for (int i = 1; i <= no_labels; i++) {
        if (area[i] > 0)
            center[i] /= area[i];
    }

    float best_diff = 9999;
    Point2f best_c1, best_c2;

    for (int i = 1; i <= no_labels; i++) {
        if (area[i] >= min_area && area[i] <= max_area) {
            for (int j = i + 1; j <= no_labels; j++) {
                if (area[j] >= min_area && area[j] <= max_area) {

                    float dy = abs(center[i].y - center[j].y);
                    float dx = abs(center[i].x - center[j].x);

                    if (dy < 20 && dx > 30 && dx < 200) {
                        float simm = abs((center[i].x + center[j].x) / 2 - cols / 2);
                        if (simm < best_diff) {
                            best_diff = simm;
                            best_c1 = center[i];
                            best_c2 = center[j];
                        }
                    }
                }
            }
        }
    }

    if (best_diff < 9999) {
        image = correct_red_eye(image, best_c1);
        image = correct_red_eye(image, best_c2);
    }

    return image;
}






