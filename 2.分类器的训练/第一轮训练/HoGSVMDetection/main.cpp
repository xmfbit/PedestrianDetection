#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "MySVM.h"
#include "HogPedestrianDete.h"
using namespace std;
using namespace  cv;

const string NEG_IMAGE_LIST="E:\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\neg_64x128.lst" ;

const string POS_IMAGE_LIST="E:\\INRIAPerson\\INRIAPerson\\train_64x128_H96\\pos_64x128.lst";



const int pos_num = 2410;
const int neg_num = 12180;
int hard_num = 0;

#define TRAIN false
int main()
{
    //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
    //特征的维度
    int descriptorDim = 0;
    //训练SVM
    MySVM svmTrain;

    ofstream foutVector("Dim.txt");
	string img_path;
	bool find_hard= (hard_num==0) ;   //hard_num=0说明需要找难例

BEGIN:	if(TRAIN)
    {
        ifstream fin_pos(POS_IMAGE_LIST);
        ifstream fin_neg(NEG_IMAGE_LIST);

        Mat trainFeatureMat;
        Mat trainLabelMat;

		//读入正样本
        for(int i = 0;i<pos_num && getline(fin_pos,img_path);i++)
        {
            cout<<"Process: "<<img_path<<endl;
            Mat img = imread(img_path,1);  //read the positive img
            vector<float> descriptors;
            hog.compute(img,descriptors,Size(8,8));  //compute the descriptors

            //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
            if(i == 0)
            {
                descriptorDim = descriptors.size();//HOG描述子的维数
                foutVector<<descriptorDim<<endl;
                //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
                trainFeatureMat =
                      Mat::zeros(pos_num + neg_num + hard_num, descriptorDim, CV_32FC1);
                //初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
                trainLabelMat = Mat::zeros(pos_num + neg_num + hard_num, 1, CV_32FC1);\
            }
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int j = 0;j<descriptorDim;j++)
            {
                trainFeatureMat.at<float>(i,j) = descriptors [j];
            }

            trainLabelMat.at<float>(i,0) = 1;     //positive

        }   // 读入正样本结束
		fin_pos.close();

		// 读入负样本
        for(int i = 0;i<neg_num && getline(fin_neg,img_path);i++)
        {
            cout<<"Process: "<<img_path<<endl;
            Mat img = imread(img_path,1);  //read the positive img
            vector<float> descriptors;
            hog.compute(img,descriptors,Size(8,8));  //compute the descriptors
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int j = 0;j<descriptorDim;j++)
            {
                trainFeatureMat.at<float>(i+pos_num,j) = descriptors [j];
            }
            trainLabelMat.at<float>(i+pos_num,0) = -1;     //positive
        }     // 读入负样本结束
		fin_neg.close();


		// 读入难例
		if (hard_num>0)
		{
			find_hard=false;   // 不需要再找难例了

			ifstream fin_hard("E:/INRIAPerson/INRIAPerson/HardExample/hard.lst");
			for (int i=0;i<hard_num && getline(fin_hard,img_path);i++)
			{
				cout<<"Process: "<<img_path<<endl;
				Mat img=imread(img_path,1);
				vector<float> descriptors;
				hog.compute(img,descriptors,Size(8,8));  //compute the descriptors
				//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
				for(int j = 0;j<descriptorDim;j++)
				{
					trainFeatureMat.at<float>(i+pos_num+neg_num,j) = descriptors [j];
				}
				trainLabelMat.at<float>(i+pos_num+neg_num,0) = -1;     //
			}
			fin_hard.close();
			cout<<"Hard Example Complete!"<<endl;
		}


/*
        //输出样本的HOG特征向量矩阵到文件
        ofstream foutDescriptor("TrainFeatureMat.txt");
        for(int i=0; i<pos_num + neg_num; i++)
        {
          foutDescriptor<<i<<endl;
          for(int j=0; j<descriptorDim; j++)
              foutDescriptor<<trainFeatureMat.at<float>(i,j)<<"  ";
          foutDescriptor<<endl;
        }
        foutDescriptor.close();
*/
        foutVector.close();
        cout<<"*********************************"<<endl;
        cout<<"Feature Extraction Done!"<<endl;
        //训练SVM分类器
        //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
        //CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1E7, FLT_EPSILON);
        CvTermCriteria criteria=cvTermCriteria(CV_TERMCRIT_EPS,1000,1E-7);   //mark
        //SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
     //int svm_type,int kernel_type,double degree,double gamma,double coef0,
     //double CValue,double nu,double p,cvMat *class_wights,CvTermCriteria criteria
        CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);

        cout<<endl;
        cout<<"---------------------------------"<<endl;
        cout<<"Start Training....."<<endl;
        double t = (double)getTickCount();
        svmTrain.train(trainFeatureMat, trainLabelMat, Mat(), Mat(), param);//训练分类器
        t = (double)getTickCount() - t;
        cout<<"Train Complete!"<<endl;

        svmTrain.save("SVM_HOG.xml");//将训练好的SVM模型保存为xml文件
        printf("train time = %gms\n", t*1000./cv::getTickFrequency());
    }

    else    //load previous xml file如果不训练的话，直接加载已有的XML文件
    {
        const char svm_xml_name[128]="SVM_HOG_wuless_lou_more.xml";
        cout<<"Load the previous XML File: "<<svm_xml_name<<endl;
        svmTrain.load(svm_xml_name);
    }

	// 对训练出的SVM进行描述
    descriptorDim = svmTrain.get_var_count();
    cout<<"The dim of HOG descriptor is "<<descriptorDim<<endl;

    int supportVecNum = svmTrain.get_support_vector_count();
    cout<<"The count of support vector is "<<supportVecNum<<endl;

    //alpha向量，长度等于支持向量个数
    Mat alphaMat = Mat::zeros(1, supportVecNum, CV_32FC1);   //1x1
    //支持向量矩阵
    Mat supportVecMat = Mat::zeros(supportVecNum, descriptorDim, CV_32FC1);//1x3780
    //alpha向量乘以支持向量矩阵的结果
    Mat resultMat = Mat::zeros(1, descriptorDim, CV_32FC1); //1x3780

    //将支持向量的数据复制到supportVectorMat矩阵中
    for(int i=0; i<supportVecNum; i++)
    {
        const float * pSVData = svmTrain.get_support_vector(i);//返回第i个支持向量的数据指针
        for(int j=0; j<descriptorDim; j++)
        {
            supportVecMat.at<float>(i,j) = pSVData[j];
        }
    }

    //将alpha向量的数据复制到alphaMat中
    double * pAlphaData = svmTrain.get_alpha_vector();//返回SVM的决策函数中的alpha向量
    for(int i=0; i<supportVecNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }

    //计算-(alphaMat * supportVectorMat),结果放到resultMat中
    //gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
    resultMat = -1 * alphaMat * supportVecMat;

    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
    vector<float> myDetector;
    //将resultMat中的数据复制到数组myDetector中
    for(int i=0; i<descriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }
    //最后添加偏移量rho，得到检测子
    myDetector.push_back(svmTrain.get_rho());
    cout<<"the dim of mydetector: "<<myDetector.size()<<endl;

    //保存检测子参数到文件
    ofstream foutHOG("HOGDetectorForOpenCV0425.txt");
    for(size_t i=0; i<myDetector.size(); i++)
    {
        foutHOG<<myDetector[i]<<endl;
    }
    foutHOG.close();
	
    waitKey();
    return 0;
}
