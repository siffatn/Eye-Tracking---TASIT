# Eye-Tracking---TASIT
First remove the files which have missing data using "droppingmissingfile.m"

Add extra features to the data using "add_vertical_horizontal.m, add_disconjugate_eye.m,add_saccadic_velocity.m"

To find correlatied frames, use "find_correlation_3sec.m", the program has dependency on "read_data_fn_correlation.m, pointbiserial.m"

Then for creating feature for Video10, use "creating_feature_vector_for_classification.m", it has dependency on "read_data_video10_vectorized_r01.m"

Then for creating feature for Video13, use "creating_feature_vector_for_classification.m", it has dependency on "read_data_video13_vectorized_r01.m" (you have to make the changes in the code for running the program for particular video)

Changes should be made in line - 5 (num(:,238) for video10, and num(:,241) for video13), 
in line 16,18,20,24,37 (video10 or video13)

After feature vectors are generated for each videos in csv files, run the "TASIT_Classifier_Part3_Video10_TimeSeries_Without_Landmarks.py" and "TASIT_Classifier_Part3_Video13_TimeSeries_Without_Landmarks.py" for video10 and video13 respectively
