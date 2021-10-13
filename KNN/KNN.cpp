#include "opencv.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

// 生成训练集与测试集的函数
void generateDataSet(Mat &img, Mat &trainData, Mat &testData, Mat &trainLabel, Mat &testLabel, int train_rows=4);

int main()
{
	// 1.读取原始数据
	Mat img = imread("data.png", 1); // 使用图片格式的MNIST数据集（部分）
	cvtColor(img, img, COLOR_BGR2GRAY);
	
	// 2.制作训练集
	int train_sample_count = 4000; // 设置训练集、测试集大小
	int test_sample_count = 1000;
	int train_rows = 4; // 每类用于训练的行数，4000/10类/100(样本/行)=4
	Mat trainData, testData; // 申明训练集与测试
	Mat trainLabel(train_sample_count, 1, CV_32FC1); // 申明训练集标签
	Mat testLabel(test_sample_count, 1, CV_32FC1);   // 申明测试集标签
	generateDataSet(img, trainData, testData, trainLabel, testLabel/*, train_rows*/); // 生成训练集、测试集与标签

	// 3.创建并初始化KNN模型
	cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create(); // 创建knn模型
	int K = 8; // 考察的最邻近样本个数
	knn->setDefaultK(K);
	knn->setIsClassifier(true); // 用于分类
	knn->setAlgorithmType(cv::ml::KNearest::BRUTE_FORCE);

	// 4.训练
	printf("开始训练...\n");
	knn->train(trainData, cv::ml::ROW_SAMPLE, trainLabel);
	printf("训练完成\n\n");

	// 5.测试
	printf("开始测试...\n");
	Mat result;
	knn->findNearest(testData, K, result);
	// 计算分类精度
	int count = 0;
	for (int i = 0; i < test_sample_count; i++)
	{
		int predict = int(result.at<float>(i));
		int actual = int(testLabel.at<float>(i));		
		if (predict == actual)
		{
			printf("label: %d, predict: %d\n", actual, predict);
			count++;			
		}
		else
			printf("label: %d, predict: %d ×\n", actual, predict);
	}
	printf("测试完成\n");

	double accuracy = double(count) / double(test_sample_count);
	printf("K = %d, accuracy = %.4f\n", K, accuracy);
	waitKey();
    return 0;
}


/*  @生成模型的训练集与测试集
	参数1：img       ，输入，灰度图像，由固定尺寸小图拼接成的大图，不同类别的小图像依次排列
	参数2：trainData ，输出，训练集，维度为：训练样本数 * 单个样本特征数，CV_32FC3类型
	参数3：testData  ，输出，测试集，维度为：测试样本数 * 单个样本特征数，CV_32FC3类型
	参数4：trainLabel，输出，训练集标签，维度为：训练样本数 * 1，CV_32FC1类型
	参数4：testLabel ，输出，测试集标签，维度为：测试样本数 * 1，CV_32FC1类型
	参数5：train_rows，输入，用于训练的样本所占行数，默认4行用于训练，1行用于测试
*/
void generateDataSet(Mat &img, Mat &trainData, Mat &testData, Mat &trainLabel, Mat &testLabel, int train_rows)
{
	// 初始化图像中切片图与其他参数
	int width_slice = 20;  // 单个数字切片图像的宽度
	int height_slice = 20; // 单个数字切片图像的高度
	int row_sample = 100;  // 每行样本数100幅小图
	int col_sample =  50;  // 每列样本数50幅小图
	int row_single_number = 5; // 单个数字占5行
	int test_rows = row_single_number - train_rows; // 测试样本所占行数

	Mat trainMat(train_rows * 20 *10, img.cols, CV_8UC1); // 存放所有训练图片
	trainMat = Scalar::all(0);
	Mat testMat(test_rows * 20 * 10, img.cols, CV_8UC1);  // 存放所有测试图片
	testMat = Scalar::all(0);

	// 生成测试、训练大图
	for (int i = 1; i <= 10 ; i++)
	{
		Mat tempTrainMat = img.rowRange((i - 1) * row_single_number * 20, (i * row_single_number - 1) * 20).clone();
		Mat tempTestMat  = img.rowRange((i * row_single_number - 1) * 20, (i * row_single_number) * 20).clone();
		//imshow("temptrain", tempTrainMat);
		//imshow("temptest",  tempTestMat);

		cv::Mat roi_train = trainMat(Rect(0, (i - 1) * train_rows * 20, tempTrainMat.cols, tempTrainMat.rows));
		Mat mask_train(roi_train.rows, roi_train.cols, roi_train.depth(), Scalar(1));
		// test
		cv::Mat roi_test = testMat(Rect(0, (i - 1) * test_rows * 20, tempTestMat.cols, tempTestMat.rows));
		Mat mask_test(roi_test.rows, roi_test.cols, roi_test.depth(), Scalar(1));
		// 提取的训练测试行分别复制到训练图与测试图中
		tempTrainMat.copyTo(roi_train, mask_train);
		tempTestMat.copyTo(roi_test, mask_test);
		//显示效果图
		//imshow("trainMat", trainMat);
		//imshow("tesetMat", testMat);
		cv::waitKey(10);
	}
	// 存大图
	imwrite("trainMat.jpg", trainMat);
	imwrite("testMat.jpg", testMat);


	// 生成训练、测试数据
	printf("开始生成训练、测试数据...\n");
	Rect roi;
	for (int i = 1; i <= col_sample; i++) // 50行：1-50行数字图像
	{
		//printf("第%d行: \n", i);
		for (int j = 1; j <= row_sample; j++) // 100列：1-100列数字图像
		{
			// 第行为训练集
			Mat temp_single_num; // 读取一个数字图像
			// 关键步骤：当前切片数字的位置区域
			roi = Rect((j-1)*width_slice, (i-1)*height_slice, width_slice, height_slice); 
			temp_single_num = img(roi).clone(); // 注意此处需要使用深拷贝.clone()，后面才能改变切片图的形状，否则roi内存区域不连续
			//imshow("slice", temp_single_num);
			//waitKey(1);
			if (i % 5 != 0) 
			//{
				// 起始行记为1-4,6-9,11-14...46-49行为测试集
				// 将单个数字切片拉成向量连续放入Mat容器中
				trainData.push_back(temp_single_num.reshape(0, 1)); 
			//}
			else
			//{	// 起始行记为1，第5,10,15...50行为测试集
				testData.push_back(temp_single_num.reshape(0, 1));  
			//}	
		}
	}
	trainData.convertTo(trainData, CV_32FC1);
	testData.convertTo(testData, CV_32FC1);
	printf("训练、测试数据已生成\n\n");

	// 生成标签
	printf("开始生成标签数据...\n");
	for (int i = 1; i <= 10; i++)
	{		
		// train label
		Mat tmep_label_train = Mat::ones(train_rows * row_sample, 1, CV_32FC1); // 临时存放当前标签的矩阵
		tmep_label_train = tmep_label_train * (i - 1); // 标签从0开始
		Mat temp = trainLabel.rowRange((i - 1)* train_rows * row_sample, i * train_rows * row_sample);
		tmep_label_train.copyTo(temp); // 将临时标签复制到trainLabel对应区域，因为浅拷贝，改变temp即改变trainLabel

		// test label
		Mat tmep_label_test = Mat::ones(test_rows * row_sample, 1, CV_32FC1);
		tmep_label_test = tmep_label_test * (i - 1);
		temp = testLabel.rowRange((i - 1)* test_rows * row_sample, i * test_rows * row_sample);
		tmep_label_test.copyTo(temp);
	}
	printf("标签数据已生成\n\n");
}