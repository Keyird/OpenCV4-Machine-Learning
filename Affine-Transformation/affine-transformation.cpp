#include <iostream>
#include <opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    Mat img = imread("D:\\CSDN\\lena.jpeg", 1);
    imshow("original", img);

  
    /************************* 尺度变换 *************************/
    // 方法1：resize
    Mat img_resize;
    resize(img, img_resize, cv::Size(0, 0), 0.5, 0.5, 1);
    imshow("resize", img_resize);

    ////方法2：仿射变换
    Mat img_warpAffine;
    Mat M = Mat::zeros(2, 3, CV_32F); //声明尺度变换M矩阵(2x3)
    float cx = 0.5, cy = 0.5;
    M.at<float>(0, 0) = cx;
    M.at<float>(1, 1) = cy;
    cv::Size size = img.size();
    size.width *= cx;
    size.height *= cy;

    warpAffine(img, img_warpAffine, M, size, INTER_LINEAR, 1); //仿射变换
    imshow("warpAffine", img_warpAffine);


    /************************* 平移变换 *************************/
    Mat img_trans;
    Mat M1 = cv::Mat::eye(2, 3, CV_32F); //声明平移变换矩阵M1（2x3），并初始化对角线元素为1
    float tx = 40, ty = 20; //设置平移变换参数
    M1.at<float>(0, 2) = tx;
    M1.at<float>(1, 2) = ty;

    //为了让平移后的图像能完整地显示，此处做扩边处理，将原图在上、下、左、右四个方向分别扩充相应的元素，并对扩充区域填充恒定灰度值200
    int top = 0, bottom = ty, left = 0, right = tx;
    cv::copyMakeBorder(img, img_trans, top, bottom, left, right, BORDER_CONSTANT, cv::Scalar(200)); //扩边
    cv::warpAffine(img_trans, img_trans, M1, img_trans.size(), INTER_LINEAR, BORDER_TRANSPARENT);
    imshow("translation", img_trans);


    /************************* 旋转变换 *************************/
    Mat img_rotate;
    double angle = 45; //设定旋转角度
    int border = 0.207 * img.cols;
    cv::copyMakeBorder(img, img_rotate, border, border, border, border, BORDER_CONSTANT, cv::Scalar(0)); //扩边
    cv::Point2f center(img_rotate.cols / 2., img_rotate.rows / 2.); //指定旋转中心
    cv::Mat M2 = cv::getRotationMatrix2D(center, angle, 1.0); //获取旋转变换矩阵M2
    cv::warpAffine(img_rotate, img_rotate, M2, img_rotate.size(), INTER_LINEAR, BORDER_REPLICATE);
    cv::imshow("rotate", img_rotate);


    /************************* 垂直剪切变换 *************************/
    Mat img_shear_vertical;
    int border = 40;
    cv::copyMakeBorder(img, img_shear_vertical, 10, 10, 10, 10+4* border, BORDER_CONSTANT, cv::Scalar(0)); //扩边
    Mat M3 = cv::Mat::eye(2, 3, CV_32F); //声明垂直剪切变换M3矩阵（2x3），初始化对角线为1
    float sv = 0.3; // 垂直剪切系数
    M3.at<float>(0, 1) = sv;
    cv::warpAffine(img_shear_vertical, img_shear_vertical, M3, img_shear_vertical.size(), INTER_LINEAR, BORDER_REPLICATE);
    cv::imshow("shear_vertical", img_shear_vertical);

    waitKey(0);
    return 0;
}
