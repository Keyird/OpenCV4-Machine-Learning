#include <opencv.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <iostream> 
#include <iomanip>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char *argv[]) {
	// 1.��ȡ����
	const char *csv_file_name = argc >= 2 ? argv[1] : "../mushroom/agaricus-lepiota.data";
	// 1.1 ��ȡCSV�����ļ�
	// �����÷�...
	cv::Ptr<TrainData> dataSet =
		TrainData::loadFromCSV(csv_file_name, // Input file name
			0, // �������ļ���ͷ����������
			0, // �����ı�ǩ�Ӵ��п�ʼ
			1, // �����������������Ӵ��п�ʼ
			"cat[0-22]" // All 23 columns are categorical
		);
	// Use defaults for delimeter (',') and missch ('?')ʹ��Ĭ�ϵġ�,���ָ�����

	// 1.2 ȷ��������������
	int n_samples = dataSet->getNSamples();
	cout << "��" << csv_file_name << "�У���ȡ��" << n_samples << "������" << endl;
	
	// 1.3 ����ѵ��������Լ�
	dataSet->setTrainTestSplitRatio(0.9, false); //��90%��10%�ı��������ݼ�Ϊѵ�����Ͳ��Լ�
	int n_train_samples = dataSet->getNTrainSamples();
	int n_test_samples = dataSet->getNTestSamples();
	cout << "Train Samples: " << n_train_samples << endl
		<< "Test  Samples: " << n_test_samples << endl;

	// 2.����������ģ��
	cv::Ptr<RTrees> dtree = RTrees::create();

	// 3.����ģ�Ͳ���
	// 3.1 ���������Ҫ������
	// Set up priors to penalize "poisonous" 10x as much as "edible"
	//float _priors[] = { 1.0, 10.0 };
	//cv::Mat priors(1, 2, CV_32F, _priors);
	dtree->setMaxDepth(10);//10
	dtree->setMinSampleCount(10);//10
	dtree->setRegressionAccuracy(0.01f);
	dtree->setUseSurrogates(false /* true */);
	dtree->setMaxCategories(15);
	dtree->setCVFolds(1 /*10*/); // nonzero causes core dump
	dtree->setUse1SERule(false/*true*/);
	dtree->setTruncatePrunedTree(true);
	//dtree->setPriors( priors );
	dtree->setPriors(cv::Mat()); // ignore priors for now...
								 // Now train the model
								 // NB: we are only using the "train" part of the data set

	// 4.ѵ��������
	cout << "start training..." << endl;
	dtree->train(dataSet);
	cout << "training success." << endl;

	// 5.����
	cv::Mat results_train, results_test;
	float train_error = dtree->calcError(dataSet, false, results_train);// use training data
	float test_error = dtree->calcError(dataSet, true, results_test); // use test data
	std::vector<cv::String> names;
	dataSet->getNames(names);
	Mat flags = dataSet->getVarSymbolFlags();
	cout << "[Decision Tree] Error on training data: " << train_error << "%" << endl;
	cout << "[Decision Tree] Error on test data: " << test_error << "%" << endl;

	// 6.ͳ��������
	cv::Mat expected_responses = dataSet->getTestResponses();
	int t = 0, f = 0, total = 0;
	for (int i = 0; i < dataSet->getNTestSamples(); ++i) {
		float responses = results_test.at<float>(i, 0);
		float expected = expected_responses.at<float>(i, 0);
		cv::String r_str = names[(int)responses];
		cv::String e_str = names[(int)expected];
		if (responses == expected)
		{
			t++;
			cout << "label: " << e_str << ", predict: " << r_str << endl;
		}
		else
		{
			f++;
			cout << "label: " << e_str << ", predict: " << r_str << " ��" << endl;
		}
		total++;
	}
	cout << "Correct answer    = " << t << endl;
	cout << "Incorrect answer  = " << f << endl;
	cout << "Total test sample = " << total << endl;
	cout << setiosflags(ios::fixed) << setprecision(2);
	cout << "[Decision Tree] Correct answers  : " << (float(t) / total) * 100 << "%" << endl;
	system("pause");
	return 0;
}
