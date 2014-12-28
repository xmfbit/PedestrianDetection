#ifndef HOGDETECTOR_H
#define HOGDETECTOR_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

class HogPedestrianDete
{
private:
    HOGDescriptor hog;
    Mat img_original;
public:
    HogPedestrianDete(const Mat& img)
    {
        this->img_original=img;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    }

    void setHOGDetector(const vector<float>& newdetector)
    {
        hog.setSVMDetector(newdetector);
    }

    void setImg(const Mat& img)
    {
        this->img_original=img;
    }
    Mat getImg() const
    {
        return this->img_original;
    }

    Mat HogDetection() const
    {
        if(this->img_original.data==NULL)
        {
            cout<<"Orinal image is null!"<<endl;
            return Mat();
        }

        Mat img_detection=this->img_original;
        vector<Rect> found, found_filtered;
        // run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        hog.detectMultiScale(img_original, found, 0, Size(8,8), Size(32,32), 1.05, 2);

        size_t i, j;

        for( i = 0; i < found.size(); i++ )
        {
            Rect r = found[i];
            for( j = 0; j < found.size(); j++ )
                if( j != i && (r & found[j]) == r)
                    break;
            if( j == found.size() )
                found_filtered.push_back(r);
        }
        for( i = 0; i < found_filtered.size(); i++ )
        {
            Rect r = found_filtered[i];
            // the HOG detector returns slightly larger rectangles than the real objects.
            // so we slightly shrink the rectangles to get a nicer output.
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
            rectangle(img_detection, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
        }
        return img_detection;
    }
};



#endif // HOGDETECTOR_H
