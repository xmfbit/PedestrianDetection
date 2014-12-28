#include <iostream>  
#include <fstream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/ml/ml.hpp>  

using namespace std;
using namespace  cv;

int hardExampleCount = 0; //hard example计数  

int main()  
{  
	Mat src;  
	char saveName[256];//剪裁出来的hard example图片的文件名  
	string ImgName;  
	ifstream fin_detector("HOGDetectorForOpenCV_2400PosINRIA_12000Neg.txt");//打开自己训练的SVM检测器文件  
	ifstream fin_imgList("INRIANegativeImageList.txt");//打开原始负样本图片文件列表  
	//ifstream fin_imgList("subset.txt");  

	//从文件中读入自己训练的SVM参数  
	float temp;  
	vector<float> myDetector;//3781维的检测器参数  
	while(!fin_detector.eof())  
	{  
		fin_detector >> temp;  
		myDetector.push_back(temp);//放入检测器数组  
	}  

	cout<<"检测子维数："<<myDetector.size()<<endl;  

	//namedWindow("src",0);  
	HOGDescriptor hog;//HOG特征检测器  
	hog.setSVMDetector(myDetector);//设置检测器参数为自己训练的SVM参数  

	//一行一行读取文件列表  
	while(getline(fin_imgList,ImgName))  
	{  
		cout<<"处理："<<ImgName<<endl;  
		string fullName = "D:\\DataSet\\INRIAPerson\\INRIAPerson\\Train\\neg\\" + ImgName;//加上路径名  
		src = imread(fullName);//读取图片  
		Mat img = src.clone();//复制原图  

		vector<Rect> found;//矩形框数组  
		//对负样本原图进行多尺度检测，检测出的都是误报  
		hog.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);  

		//遍历从图像中检测出来的矩形框，得到hard example  
		for(int i=0; i < found.size(); i++)  
		{  
			//检测出来的很多矩形框都超出了图像边界，将这些矩形框都强制规范在图像边界内部  
			Rect r = found[i];  
			if(r.x < 0)  
				r.x = 0;  
			if(r.y < 0)  
				r.y = 0;  
			if(r.x + r.width > src.cols)  
				r.width = src.cols - r.x;  
			if(r.y + r.height > src.rows)  
				r.height = src.rows - r.y;  

			//将矩形框保存为图片，就是Hard Example  
			Mat hardExampleImg = src(r);//从原图上截取矩形框大小的图片  
			resize(hardExampleImg,hardExampleImg,Size(64,128));//将剪裁出来的图片缩放为64*128大小  
			sprintf(saveName,"hardexample%09d.jpg",hardExampleCount++);//生成hard example图片的文件名  
			imwrite(saveName, hardExampleImg);//保存文件  


			//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
			//r.x += cvRound(r.width*0.1);  
			//r.width = cvRound(r.width*0.8);  
			//r.y += cvRound(r.height*0.07);  
			//r.height = cvRound(r.height*0.8);  
			rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);  

		}  
		//imwrite(ImgName,img);  
		//imshow("src",src);  
		//waitKey(100);//注意：imshow之后一定要加waitKey，否则无法显示图像  

	}  

	system("pause");  
}  

//优先队列
/*
#include <iostream>
#include <queue>
#include <string>
using namespace std;

struct Human
{
	string name;
	int age;
	Human(string _name,int _age):name(_name),age(_age) {}
};

class compare_human
{
public:
	bool operator() (const Human& h1,const Human& h2)
	{
		return h1.age<h2.age;
	}
};

typedef priority_queue<Human,vector<Human>,compare_human>				  human_pariority_queue  ;
int main()
{
	string names[]={"xiao","messi","zhao","C","zhang","Lilu"};
	human_pariority_queue q;
	for(int i=0;i<6;i++)
	{
		q.push(Human(names[i],i*10));
	}
	for (int i=0;i<6;i++)
	{
		cout<<q.top().name<<"   "<<q.top().age<<endl;
		q.pop();
	}
	return 0;
}
*/