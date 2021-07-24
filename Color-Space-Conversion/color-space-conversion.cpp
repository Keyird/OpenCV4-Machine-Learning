#include <iostream>
#include<opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("Path of Image", 1);
    imshow("Original", img);
  
    Mat img_gray,  img_hsv, img_hsl, img_lab, img_rgb;
    cvtColor(img, img_rgb, COLOR_BGR2RGB);   //BGR转RGB
    cvtColor(img, img_gray, COLOR_BGR2GRAY); //BGR转灰度图
    cvtColor(img, img_hsv, COLOR_BGR2HSV);   //BGR转HSV
    cvtColor(img, img_lab, COLOR_BGR2Lab);   //BGR转Lab
    cvtColor(img, img_hsl, COLOR_BGR2HLS);   //BGR转HSL

    imshow("rgb", img_rgb);
    imshow("gray", img_gray);
    imshow("hsv", img_hsv);
    imshow("lab", img_lab);
    imshow("hsl", img_hsl);

    waitKey(0);
    return 0;
}
