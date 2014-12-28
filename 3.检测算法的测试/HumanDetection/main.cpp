#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/video.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "HogPedestrianDete.h"

//#define PICTUREINPUT
#define PICTUREINPUT
using namespace std;
using namespace cv;

const char videoname[] = "E:/研究生课件/机器视觉/Project/Data/VID_20141225_094144.mp4";


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


const char annotation_file_path[]="I:/CVC-02-System01/DATASET-CVC-02/CVC-02-System/sequence-01/annotations/";

int main()
{
#ifdef VIDEOINPUT
	VideoCapture video(videoname);    // load the video from the disk
#endif
	
	VideoWriter videowriter;
	videowriter.open("humandetection_CCV_annotation.avi",CV_FOURCC('M','J','P','G'),
		5,Size(640,480),true);

	if(!videowriter.isOpened())
	{
		cout<<"Can not open the video file!"<<endl;
		return -1;
	}

	Mat frame;
	
	HOGDescriptor hog_des;     // 用来测试的HoG检测子
	hog_des.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector ());

	HogPedestrianDete human_detector(hog_des);

	Mat dst;

	namedWindow ("Detection Window");
	

#ifdef VIDEOINPUT
	if (!video.isOpened ())
	{
		cout<<"sdfas"<<endl;
		return -1;
	}
#endif // VIDEO

#ifdef PICTUREINPUT
	ifstream fin("E:/INRIAPerson/color/img_path.txt");
	string picname;
	
	int count = 0;
	vector<annotation> v;

#endif
	while(getline (fin, picname))
	{
		count ++;
#ifdef VIDEOINPUT
		video<<frame;
#endif
#ifdef PICTUREINPUT
		cout<<picname<<endl;
		frame  = imread (picname, 1);
#endif

		if (frame.empty ())
			break;

		dst = human_detector.HumanDetect (frame);

		imshow ("Detection Window",dst);
		
		/*得到annotatio文件的绝对路径*/
		char file_name[128];
		sprintf(file_name,"sequence-01-%06d.txt",count);
		char absolute_file_name[128];
		strcpy(absolute_file_name,annotation_file_path);
		strcat(absolute_file_name,file_name);
		puts(absolute_file_name);
		// Read the annotation txt !
		readAnnotation(absolute_file_name,v);
		for (size_t i=0;i<v.size();i++)
		{
			annotation tmp=v[i];
			DrawRectangle(dst, tmp);
		}
		
		videowriter << dst;
	}
#ifdef PICTUREINPUT
	fin.close ();
#endif // PICTUREINPUT

	waitKey ();
	return 0;
}

