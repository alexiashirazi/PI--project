#ifndef LAB7_H
#define LAB7_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
typedef struct {
    Mat B;
    Mat G;
    Mat R;
} image_channels_bgr;
typedef struct {
    Mat H;
    Mat S;
    Mat V;
} image_channels_hsv;


const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};

const int np_di[4] = { 0,-1,-1, -1};
const int np_dj[4] = { -1,-1, 0, 1};

typedef struct {
    Mat labels;
    int no_labels;
}labels;


image_channels_bgr break_channels(Mat source);


void display_channels(image_channels_bgr bgr_channels);
Mat color_labels(labels labels_str);
labels BFS_labeling(Mat source);
Mat saturation_binarization(image_channels_bgr bgr_channels, int threshold) ;
Mat draw_symmetric_eyes(Mat image, labels label_data);
Mat correct_red_eye(Mat  image, Point center);


#endif