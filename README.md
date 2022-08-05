8/4/2022

# Eye_movement_detection
Files to run trained model on EOG, EEG data to detect eye movement.

## Requirements
- pytorch 1.10.2
- list of python packages in [requirements.txt](https://github.com/smudge1872/Eye_movement_detection/blob/main/requirements.txt)

## Data
- Input file should be edf with file with sampling rate of 200Hz with channel names
  - F3-A2
  - A1-F4
  - A2-C3
  - A1-C4
  - A2-O1
  - A1-O2
  - A2-PG1  This is LOC
  - A1-PG2  This is ROC
  - PZ-CZ
  
## Generate h5 examples
A matlab script, [generate_h5_files_fromEDF.m](https://github.com/smudge1872/Eye_movement_detection/blob/main/code/exampleGeneration/generate_h5_files_fromEDF.m), opens the EDF file and does some signal smoothing using some filters and saves the data in h5 files for the pytorch program to read. Each h5 it saves out is one second. To configure for your data, change the <code> currEDFPath </code> string variable to the path of your edf file. Also change the <code> exampleSaveDir </code> string variable to the path where you want the output data saved. Set the <code> patientNum </code> to a number prefix for the subfolder that is created for the edf file (Typically a patient id number).  Set the <code>extractionOffsetStartHour</code> variable to how many hours past the beginning of the file to start extracting. Set this to 0 if want to start at the beginning of the file. Also set <code>howManyHoursToExtract</code> for how many hours you want to extract from that start point.

```matlab
  %generate_h5_files_fromEDF.m
  %Extracts from edf file and applies signal filtering and saves in h5 format
  %for pytorch so it can be run on ey movement detection model
  
  addpath(genpath('./edfread'));
  
  %Generated h5 files are saved here
  exampleSaveDir = '/users/sjschmug/coma/githubRepo/h5_examples/';
  
  %Prefix str num for all generated h5 files and sub folder name to create
  patientNum = 17; 
  
  %Path to EDF file
  currEDFPath = '/projects/mcshin_research/coma/data/sleep_study_50/data/17/EX6X7PHEZ99S3PGO.edf';
  
  %At what hour should extraction begin. 0 means beginning of file
  extractionOffsetStartHour = 0;
    
  %How many hours to extract
  howManyHoursToExtract = 1;
  
  ...
```

Run the script in matlab

## Running the detection algorithm
 The [sampleConfig.ini](https://github.com/smudge1872/Eye_movement_detection/blob/main/code/pytorchAlgorithm/sampleConfig.ini) specifies the input folder name that has the h5 files, model file name, and the output directory where the detection output is stored in .csv and pkl files. There are 5 model files files in the trainedModels folder corresponding to the learned models in 5 fold cross validation.
 ```ini
 [DEFAULT]
inputFolderName=/users/sjschmug/coma/githubRepo/h5_examples/17/	
modelFileName=/users/sjschmug/coma/githubRepo/Eye_movement_detection/trainedModels/fold_4_twoLayer_model_00029.pt
outputdir=/users/sjschmug/coma/githubRepo/sampleOutput/
#Should not need to change anything below this line
...
```
To run the algorithm, 
