MLB Pitch detector :-
Dependencies :- Opencv 3.4.1 or higher
		Cmake 2.8.12 or higher

Step 1:- Open terminal in the home folder (MLB) and run following script to get the Neural Network weights
	>> ./getModels.sh

Step 2:- Compiling the project
	>> cd build
	>> cmake ..
	>> make

Step 3:- Executing the project
	>> ./pitching_detect <full_path_to_video.mp4>
