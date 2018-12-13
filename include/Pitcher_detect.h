#include<opencv2/dnn.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<chrono>
#include<random>
#include<set>
#include<cmath>
struct KeyPoint
{
    KeyPoint(cv::Point point,float probability)
    {
        this->id = -1;
        this->point = point;
        this->probability = probability;
    }
    int id;
    cv::Point point;
    float probability;
};

class Pitcher_detect
{
    public:
        std::string fname;
        const int nPoints = 18;
        cv::dnn::Net inputNet;
        cv::Mat Valid_frames;
        cv::Point2f Wrist_Cord;
        int frame = 0;
        int output_frame_no = 0;
        cv::Mat input_full;
        cv::Mat input_aROI, input;
        cv::Mat output_frame;
        cv::Rect ROI_player = cv::Rect(1100,700,1600,900);
        std::string keypointsMapping[18] = {"Nose", "Neck", "R-Sho", "R-Elb", "R-Wr","L-Sho", "L-Elb", "L-Wr", "R-Hip", "R-Knee", "R-Ank", "L-Hip", "L-Knee", "L-Ank", "R-Eye", "L-Eye", "R-Ear", "L-Ear"};
        Pitcher_detect(std::string filename);
        int getKeyPoints(cv::Mat& probMap,double threshold,std::vector<KeyPoint>& keyPoints);
        void splitNetOutputBlobToParts(cv::Mat& netOutputBlob,const cv::Size& targetSize,std::vector<cv::Mat>& netOutputParts);
        bool Detect_ball(cv::Point2f Wrist_Cord, cv::Mat Valid_frames);
        bool Process_frames();

};