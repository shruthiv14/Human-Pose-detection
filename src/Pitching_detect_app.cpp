#include "Pitcher_detect.h"

int main(int argc,char** argv)
{
    if(argc < 2 || argc > 3)
    {
        std::cout<<"Usage: ./pitching_detect <video_filename.mp4>";
    }
    Pitcher_detect detect_throw(argv[1]);
    bool val = detect_throw.Process_frames();
    if(val)
    {
        std::cout<<"Pitch detected in video "<<argv[1]<<" at frame number- "<<detect_throw.output_frame_no<<std::endl;
        cv::namedWindow("Pitch detect frame",0);
        cv::imshow("Pitch detect frame", detect_throw.output_frame);
        std::cout<<"Press any key to continue"<<std::endl;
        cv::waitKey(0);
    }
    else
    {
        std::cout<<"Pitch not detected in video "<<argv[1]<<std::endl;
    }
}