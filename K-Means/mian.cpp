#include<iostream>
#include<opencv.hpp>
using namespace std;
using namespace cv;

int main() {
	const int MAX_CLUSTERS = 5; //��������
	Scalar colorTab[] = {   //��ͼ��ɫ
						 Scalar(0, 0, 255),
						 Scalar(0, 255, 0),
						 Scalar(255, 100, 100),
						 Scalar(255, 0, 255),
						 Scalar(0, 255, 255)
						};

	Mat img(500, 500, CV_8UC3); //�½�����
	img = Scalar::all(255); //����������Ϊ��ɫ
	RNG rng(35345); //�����������

	//��ʼ�������
	int clusterCount = rng.uniform(2, MAX_CLUSTERS + 1);
	//��ָ�����䣬�������һ������,������
	int sampleCount = rng.uniform(1, 1001);
	//������������sampleCount��x1��, �����ͣ�2ͨ��
	Mat points(sampleCount, 1, CV_32FC2);
	Mat labels; 
	//��������� < ������
	clusterCount = MIN(clusterCount, sampleCount); 

	//��������������
	vector<Point2f> centers;

	//������ɶ��˹�ֲ�������
	//for (int k = 0; k < clusterCount; k++) {
	Point center;
	center.x = rng.uniform(0, img.cols);
	center.y = rng.uniform(0, img.rows);

	//������pointsָ�����и�ֵ
	Mat pointChunk = points.rowRange(0, sampleCount / clusterCount);
			
	//��centerΪ���ģ�������˹�ֲ��������,������㱣���� pointChunk ��
	rng.fill(pointChunk, RNG::NORMAL, Scalar(center.x, center.y), Scalar(img.cols*0.05, img.rows*0.05));
	//����points�е�ֵ
	randShuffle(points, 1, &rng);

	//ִ��k-means
	double compactness = kmeans(points,  //����
								clusterCount, //�����
								labels,  //����������飬���ڴ洢ÿ�������ľ����������
								TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),  //�㷨��ֹ�����������������������辫��
								3, //����ָ��ʹ�ò�ͬ��ʼ���ִ���㷨�Ĵ���
								KMEANS_PP_CENTERS, //��ʼ����ֵ��ķ���
								centers); //�������ĵ��������ÿ����������ռһ��
			
	//���ƻ����������
	for (int i = 0; i < sampleCount; i++) {
		int clusterIdx = labels.at<int>(i);

		Point ipt = points.at<Point2f>(i);
		circle(img, ipt, 2, colorTab[clusterIdx], FILLED, LINE_AA);
	}

	//�Ծ�������ΪԲ�Ļ���Բ��
	for (int i = 0; i < (int)centers.size(); ++i) {
		Point2f c = centers[i];
		circle(img, c, 40, colorTab[i], 1, LINE_AA);
	}

	cout << "Compactness: " << compactness << endl;
	imshow("clusters", img);
	waitKey(0);

	return 0;
}
