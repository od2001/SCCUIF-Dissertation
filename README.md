# SCCUIF
##  Clustering Scene Classification Using Features Extracted From Images

## Note

Due to the ethics committee, I could not upload the original  images or the mask. Since the system is not a user-based system, the code relies on the images which were present. Therefore any execution of the code will likely fail.

However if one was to put images into the images folder, the code may still successfully execute. However this was not tested.

Since the code is not the primary factor of marking, most files are uncommented. Those that are may contain variety of descriptive and testing comments.

Below is an explanation for each folder but for the purposes of supplementing the dissertation document, folders Algo Comparisons and FeatureExtraction provide the best insights into how the system functions.


## File and Folder explanations

### 'completed_images'

Contains some the images that were fully processed from base to comparison

### 'images'

This folder is where images have their features extracted 

### 'test_images'

A folder for containing images for testing purposes 

### 'scripts'

This folder contains all the scripts and notebooks for the projects final output and testing

#### 'algo'

This folder has the files associated with the creation of the custom algorithm. algorithm.py and algorithm_revised.py best showcase the code for the algorthim, with alog_fixxing being used to test each revison of the algorithm.

#### 'Comparisions'

These files are the comparison code for skeltal keypoints and colour comparision used in the algorithm

#### 'FeatureExtraction'

This collection of files is where features are extracted from each image. The main.py file is what is excuted to achieve this, depending on the arguments it is executed with 

#### 'LabExamples'

This folder contains the lab examples and many instance of testing, this folder is particularly disorganised

##### 'notebooks' 

This folder contains most of the testing notebooks used for the system and are also particularly disorganised

#### 'Pre-processing'

This folder contains the file test of the of the blurring  faces