#include<iostream>
#include<opencv.hpp>
using namespace std;
using namespace cv;

int main() {
	const int MAX_CLUSTERS = 5; //最大类别数
	Scalar colorTab[] = {   //绘图颜色
						 Scalar(0, 0, 255),
						 Scalar(0, 255, 0),
						 Scalar(255, 100, 100),
						 Scalar(255, 0, 255),
						 Scalar(0, 255, 255)
						};

	Mat img(500, 500, CV_8UC3); //新建画布
	img = Scalar::all(255); //将画布设置为白色
	RNG rng(35345); //随机数产生器

	//初始化类别数
	int clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
	//在指定区间，随机生成一个整数,样本数
	int sampleCount = rng.uniform(1, 1001);
	//输入样本矩阵：sampleCount行x1列, 浮点型，2通道
	Mat points(sampleCount, 1, CV_32FC2);
	Mat labels; 
	//聚类类别数 < 样本数
	clusterCount = MIN(clusterCount, sampleCount); 

	//聚类结果索引矩阵
	vector<Point2f> centers;

	//随机生成多高斯分布的样本
	//for (int k = 0; k < clusterCount; k++) {
	Point center;
	center.x = rng.uniform(0, img.cols);
	center.y = rng.uniform(0, img.rows);

	//对样本points指定进行赋值
	Mat pointChunk = points.rowRange(0, sampleCount / clusterCount);
			
	//以center为中心，产生高斯分布的随机点,把坐标点保存在 pointChunk 中
	rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
	//打乱points中的值
	randShuffle(points, 1, &rng);

	//执行k-means
	double compactness = kmeans(points,  //样本
								clusterCount, //类别数
								labels,  //输出整数数组，用于存储每个样本的聚类类别索引
								TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),  //算法终止条件：即最大迭代次数或所需精度
								3, //用于指定使用不同初始标记执行算法的次数
								KMEANS_PP_CENTERS, //初始化均值点的方法
								centers); //聚类中心的输出矩阵，每个聚类中心占一行
			
	//绘制或输出聚类结果
	for (int i = 0; i < sampleCount; i++) {
		int clusterIdx = labels.at<int>(i);

		Point ipt = points.at<Point2f>(i);
		circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
	}

	//以聚类中心为圆心绘制圆形
	for (int i = 0; i < (int)centers.size(); ++i) {
		Point2f c = centers[i];
		circle(img, c, 40, colorTab[i], 1, LINE_AA);
	}

	cout << "Compactness: " << compactness << endl;
	imshow("clusters", img);
	waitKey(0);

	return 0;
}
