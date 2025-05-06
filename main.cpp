#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/proiect.h"
using namespace std;
using namespace cv;


int main() {
    Mat source = imread("C:/Users/Alexia/Desktop/UTCN/an3 sem2/PI/Image_Processing_UTCN_Labs-Project/images/redeye.bmp",
                        IMREAD_COLOR);

    image_channels_bgr bgr_channels=break_channels(source);
    Mat B= bgr_channels.B;
    Mat G= bgr_channels.G;
    Mat R= bgr_channels.R;
    imshow("original picture",source);
     display_channels(bgr_channels);

    Mat binary_S = saturation_binarization(bgr_channels, 160);
    imshow("binar pe S", binary_S);

    labels result = BFS_labeling(binary_S);
    Mat colored = color_labels(result);
    imshow("etichete colorate", colored);
    Mat new_image = draw_symmetric_eyes(source, result);
    imshow("noua poza", new_image);



    waitKey();

    return 0;
}