#include "Pitcher_detect.h"

int Pitcher_detect::getKeyPoints(cv::Mat& probMap,double threshold,std::vector<KeyPoint>& keyPoints)
{
	cv::Mat smoothProbMap;
	cv::GaussianBlur( probMap, smoothProbMap, cv::Size( 3, 3 ), 0, 0 );

	cv::Mat maskedProbMap;
	cv::threshold(smoothProbMap,maskedProbMap,threshold,255,cv::THRESH_BINARY);

	maskedProbMap.convertTo(maskedProbMap,CV_8U,1);

	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(maskedProbMap,contours,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);

	for(int i = 0; i < contours.size();++i){
		cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows,smoothProbMap.cols,smoothProbMap.type());

		cv::fillConvexPoly(blobMask,contours[i],cv::Scalar(1));

		double maxVal;
		cv::Point maxLoc;

		cv::minMaxLoc(smoothProbMap.mul(blobMask),0,&maxVal,0,&maxLoc);

		keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y,maxLoc.x)));
	}
	return keyPoints.size();
}

void Pitcher_detect::splitNetOutputBlobToParts(cv::Mat& netOutputBlob,const cv::Size& targetSize,std::vector<cv::Mat>& netOutputParts){
	int nParts = netOutputBlob.size[1];
	int h = netOutputBlob.size[2];
	int w = netOutputBlob.size[3];

	for(int i = 0; i< nParts;++i){
		cv::Mat part(h, w, CV_32F, netOutputBlob.ptr(0,i));

		cv::Mat resizedPart;

		cv::resize(part,resizedPart,targetSize);

		netOutputParts.push_back(resizedPart);
	}
}

Pitcher_detect::Pitcher_detect(std::string filename)
{
    fname = filename;
    inputNet = cv::dnn::readNetFromCaffe("../pose/coco/pose_deploy_linevec.prototxt","../pose/coco/pose_iter_440000.caffemodel");
    inputNet.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
}

bool Pitcher_detect::Detect_ball(cv::Point2f Wrist_Cord, cv::Mat Valid_frames)
{
		int tx = Wrist_Cord.x - 125 < 0? 0:Wrist_Cord.x - 125;
		int ty = Wrist_Cord.y - 125 < 0? 0:Wrist_Cord.y - 125;

		cv::Mat Wrist_ROI = Valid_frames(cv::Rect(tx,ty,150,150));
		cv::GaussianBlur( Wrist_ROI, Wrist_ROI, cv::Size(9, 9), 2, 2 );
		cv::medianBlur(Wrist_ROI,Wrist_ROI,13);

		cv::Mat gs_roi, img_edge, labels, centroids, img_color, stats;
		cv::cvtColor(Wrist_ROI,gs_roi,cv::COLOR_RGB2GRAY);
		cv::threshold(gs_roi, img_edge, 118, 255, cv::THRESH_BINARY);
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(9,9),cv::Point(0,0));
		cv::dilate(img_edge,img_edge,element); 	
		int i, nccomps = cv::connectedComponentsWithStats (	img_edge,labels,stats,centroids);
		std::vector<int> colors(nccomps+1);
 		colors[0] = 0;
		for( i = 1; i <= nccomps; i++ ) 
		{
			colors[i] = 255;
			//std::cout<<"Area -"<<stats.at<int>(i-1, cv::CC_STAT_AREA) <<std::endl;
			if( stats.at<int>(i, cv::CC_STAT_AREA) < 1500 )
    		colors[i] = 0;
		}
  		img_color = cv::Mat::zeros(gs_roi.size(), CV_8UC1);
		for( int y = 0; y < img_color.rows; y++ )
		{
			for( int x = 0; x < img_color.cols; x++ )
			{
			int label = labels.at<int>(y, x);
			img_color.at<char>(y, x) = colors[label];
			}
		}
		cv::Mat label2;
		int ncc2 = cv::connectedComponents(img_color,label2);
		if(nccomps > 2) //check area
		{
			//output_frame = Output_Frame_pool[t].clone();
			//output_frame_no = Valid_Fno;
			//cv::namedWindow("Frame with pitcher",0);
			//cv::imshow("Frame with pitcher",output_frame);
			//std::cout<<"Frame with pitcher-"<<output_frame_no<<std::endl;
			//cv::waitKey(0);
			return true;
		}
    return false;
}
bool Pitcher_detect::Process_frames()
{
    cv::VideoCapture cap(fname); 
    if(!cap.isOpened())
    {   
        std::cout<<"Unable to open video"<<std::endl;  
        return false;
    }
    for(;;)
    {
    	std::cout<<"Frame -"<<frame<<std::endl;
        cap >> input_full;
        input_aROI = input_full(ROI_player);
        input_aROI.copyTo(input);
        cv::Mat inputBlob = cv::dnn::blobFromImage(input,1.0/255.0,cv::Size((int)((168*input.cols)/input.rows),168),cv::Scalar(0,0,0),false,false);
        inputNet.setInput(inputBlob);
        cv::Mat netOutputBlob = inputNet.forward();
        std::vector<cv::Mat> netOutputParts; 	
        splitNetOutputBlobToParts(netOutputBlob,cv::Size(input.cols,input.rows),netOutputParts);
        std::vector<std::vector<KeyPoint>> detectedKeypoints;
        std::vector<int> valid_detect;
        for(int i = 0; i < nPoints;++i)
        {
            std::vector<KeyPoint> keyPoints;
            valid_detect.push_back(getKeyPoints(netOutputParts[i],0.1,keyPoints));
            detectedKeypoints.push_back(keyPoints);
        }
        if(!valid_detect[4] || !valid_detect[3] || !valid_detect[2]) 
        {
            frame++;
            continue;
        }
        cv::Point2f wrist, elbow, shoulder;
        wrist = detectedKeypoints[4][0].point;
        elbow = detectedKeypoints[3][0].point;
        shoulder = detectedKeypoints[2][0].point;
        if(wrist.x < elbow.x && elbow.x < shoulder.x)
        {
            if(wrist.y < elbow.y && elbow.y < shoulder.y)
            {
                cv::Mat inputBlob_aROI = cv::dnn::blobFromImage(input_aROI,1.0/255.0,cv::Size((int)((368*input_aROI.cols)/input_aROI.rows),368),cv::Scalar(0,0,0),false,false);
                inputNet.setInput(inputBlob_aROI);
                cv::Mat netOutputBlob_aROI = inputNet.forward();
                std::vector<cv::Mat> netOutputParts_aROI;
                splitNetOutputBlobToParts(netOutputBlob_aROI,cv::Size(input_aROI.cols,input_aROI.rows),netOutputParts_aROI);
                std::vector<std::vector<KeyPoint>> detectedKeypoints_aROI;
                std::vector<int> valid_detect_aROI;
                for(int i = 0; i < nPoints;++i)
                    {
                        std::vector<KeyPoint> keyPoints_aROI;
                        valid_detect_aROI.push_back(getKeyPoints(netOutputParts_aROI[i],0.1,keyPoints_aROI));
                        detectedKeypoints_aROI.push_back(keyPoints_aROI);
                    }
                cv::Point2f wrist_aROI, elbow_aROI, shoulder_aROI;
                wrist_aROI = detectedKeypoints_aROI[4][0].point;
                elbow_aROI = detectedKeypoints_aROI[3][0].point;
                shoulder_aROI = detectedKeypoints_aROI[2][0].point;
                float wx,wy,ex,ey,sx,sy;
                wx = wrist_aROI.x+ input_aROI.cols/2;
                ex = elbow_aROI.x+ input_aROI.cols/2;
                sx = shoulder_aROI.x+ input_aROI.cols/2;
                wy = input_aROI.rows-(wrist_aROI.y+ input_aROI.rows/2);
                ey = input_aROI.rows- (elbow_aROI.y+ input_aROI.rows/2);
                sy = input_aROI.rows - (shoulder_aROI.y+ input_aROI.rows/2);
                float slope_forearm = (wy - ey)/(wx-ex);
                float slope_bicep = (ey - sy)/(ex - sx);
                float angle =  180;
                if(slope_forearm != slope_bicep)
                {
                    angle = 180 - atan((slope_bicep-slope_forearm)/(1+slope_bicep*slope_forearm))*180/M_PI;
                }
                if(angle > 183) break;
                if(angle > 150 && angle < 183)
                {
                    bool val = Detect_ball(wrist_aROI, input_aROI);
                    if(val)
                    {
                        output_frame = input_full;
                        output_frame_no = frame;
                        return true;
                    }
                }
                
            }
        }
        frame++;
    }
    return false;
} 