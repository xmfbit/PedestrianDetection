
#include <iostream>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace  cv;

struct annotation
{
	int _x;
	int _y;
	int _w;
	int _h;
	annotation(int x,int y,int w,int h):
		_x(x),_y(y),_w(w),_h(h){}
};
void readAnnotation(const char* filename,vector<annotation>& v)
{
	v.clear();
	int x,y,w,h;
	FILE* fp=fopen(filename,"r");
	if(fp==NULL)
	{
		printf("ERROR! Cant open file %s!\n",filename);
		return ;
	}
	while(fscanf(fp, "%d %d %d %d ", &x, &y, &w, &h) == 4)
	{
		char a[100]; 
		fscanf(fp, "%s\n", a); 
		if (!strcmp(a, "PEDESTRIAN-OBLIGATORY")) 
		{
			// create new annotation in your own structure with x, y, w and h
			annotation tmp(x,y,w,h);
			v.push_back(tmp);
		}
	}
	fclose(fp); 
}

void DrawRectangle(Mat& img,int x,int y,int w,int h)
{
	rectangle(img,Point(x-w/2,y-h/2),Point(x+w/2,y+h/2),Scalar(0,0,255),2);
}

void DrawRectangle(Mat& img,const annotation& annota)
{
	int x=annota._x,
		y=annota._y,
		w=annota._w,
		h=annota._h;
	DrawRectangle(img,x,y,w,h);
}

int main()
{
	const char annotation_file_path[]="I:/CVC-02-System01/DATASET-CVC-02/CVC-02-System/sequence-01/annotations/";
	const string image_save_path="E:/INRIAPerson/AnnotationImage/";
	const string image_name_txt="E:/INRIAPerson/color/img_path.txt";
	ifstream fin(image_name_txt);
	ofstream fout("E:/INRIAPerson/AnnotationImage/annotated_img_lst.txt");
	string img_file_name;   //图像的初始存储路径

	const int start=0;
	const int end=545;
	int count=0;
	vector<annotation> v;

	VideoWriter outvideo;
	outvideo.open("E:/INRIAPerson/AnnotationImage/annotated_image.avi",
		CV_FOURCC('M','J','P','G'),10,Size(640,480),true);
	if(!outvideo.isOpened())
	{
		cout<<"Can not open the video file!"<<endl;
		return -1;
	}

	while ( getline(fin,img_file_name) )
	{
		if (count<start)
		{
			count++;
			continue;
		}
		else if(count>end)
			break;
		cout<<"Process Image "<<count<<endl;
		/*得到annotatio文件的绝对路径*/
		char file_name[128];
		sprintf(file_name,"sequence-01-%06d.txt",count);
		char absolute_file_name[128];
		strcpy(absolute_file_name,annotation_file_path);
		strcat(absolute_file_name,file_name);
		puts(absolute_file_name);
		// Read the annotation txt !
		readAnnotation(absolute_file_name,v);
		// Annotate the image !
		Mat img=imread(img_file_name,1);
		for (size_t i=0;i<v.size();i++)
		{
			annotation tmp=v[i];
			DrawRectangle(img,tmp);
		}
		// show the image
		imshow("image",img);
		waitKey(10);

		// save the image
		char savename[128];
		sprintf(savename,"annotated_img_sequence-01-%06d.png",count);
		string img_name_tosave=image_save_path+string(savename);
		imwrite(img_name_tosave,img);

		fout<<img_name_tosave<<endl;
		cout<<"Complete Image "<<count<<endl;

		// generate videos
		outvideo<<img;
		count++;
	}
	cout<<"--------------------------------------"<<endl;
	cout<<"Complete!"<<endl;
	fin.clear();
	fout.close();

	return 0;
}