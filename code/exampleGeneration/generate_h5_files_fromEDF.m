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

waveSampleRate = 200;  %Model is trained for 200 Hz 9 channels
numWaveFeatures = 9;  %9
whichFEATs = 'all';  %Can be 'all', 'LOCROC', 'ROC', 'LOC', 'other7'
seqLen = 1;  % 1 second sequence length


rng(1);

doFiltering = 1;
doAugNoise = 0;


%Filter Parameters
hzThresh = 5; %20;
sampleRateHz = waveSampleRate;
orderN = 2;
smoothWindowSize = 25; %25;

%For Noise augmentation
%sigmas = [0.5, 0.2, 0.1, 0.05, 0.02];
snrs = [6, 8, 10, 12, 14];
numExPerSNR = 2;

% -------------------------------------------------------------------------

mkdir(exampleSaveDir);


% Track number of examples created
iExample = 0;

       
[hdr, records] = edfread(currEDFPath);
    
% Convert feature names to standard format to keep data consistent
origFeatNames = string(hdr.label);
    
    
[rawFeatIndices, newFeatNames] = FeatureComparator.convertFeatureNames(origFeatNames,whichFEATs);
    
% Index the example being created with this
featIndices = nonzeros(rawFeatIndices);
% Index the records with this
logicalFeatIndices = logical(rawFeatIndices);
    
disp(featIndices);
disp(logicalFeatIndices);
   
patientStudyStartTime = datevec(strrep(hdr.starttime, '.', ':'));
% set month/day/year to 0
patientStudyStartTime(1:3) = 0;
    
patientGTStartTime = datevec(datetime(patientStudyStartTime) + hours(extractionOffsetStartHour) ); %datevec(gt.TimeStamp(patientGTEntryStart));
patientGTStartTime(1:3) = 0;
    
fprintf('Study start time:n');
disp(patientStudyStartTime);
fprintf('GT start time:\n');
disp(patientGTStartTime);
    
gtStartTimeInDateForm = datetime(patientGTStartTime);
currentTime = gtStartTimeInDateForm;
    
% Get difference between study start and GT start
startDiff = etime(patientGTStartTime, patientStudyStartTime);
% Account for passage of midnight between study start and GT start
if startDiff < 0
   startDiff = (24 * 60 * 60) + startDiff;
end
    
fprintf('Time between study start and gt start: %d seconds (%d minutes)\n', ...
startDiff, startDiff / 60);
    
startRecordOffset = round(waveSampleRate * startDiff);
if startRecordOffset == 0
    startRecordOffset = 1;
end

    
fprintf('Starting with offset of %d\n', startRecordOffset);
    
% This is where we start in the records 
currStudyRowStart = startRecordOffset;

    
currFinalStudyRowEnd = currStudyRowStart + ((3600*howManyHoursToExtract) * waveSampleRate); 
    
mkdir(fullfile(exampleSaveDir, string(patientNum)));
    
% Keep going until number of hours extracted
while (currStudyRowStart <= currFinalStudyRowEnd) 
        % Each iteration of this loop will produce 1 example (n timesteps)
        
        % Build example
        currExample = zeros(seqLen, numWaveFeatures, waveSampleRate);
        %currExample = zeros(seqLen, numWaveFeatures, waveSampleRate*3);
        %currExamplePrev = zeros(seqLen, numWaveFeatures, waveSampleRate);
        %currExampleAfter = zeros(seqLen, numWaveFeatures, waveSampleRate);
        currExampleRowStart = currStudyRowStart;
        for iStep = 1:seqLen
            currStudyRowEnd = currStudyRowStart + waveSampleRate - 1;
            currExample(iStep, featIndices, :) = records(logicalFeatIndices, currStudyRowStart:currStudyRowEnd);
            %currExample(iStep, featIndices, :) = records(logicalFeatIndices, currStudyRowStart-waveSampleRate:currStudyRowEnd+waveSampleRate);
            %currExamplePrev(iStep, featIndices, :) = records(logicalFeatIndices, currStudyRowStart-waveSampleRate:currStudyRowStart-1);
            %currExampleAfter(iStep, featIndices, :) = records(logicalFeatIndices, currStudyRowEnd+1:currStudyRowEnd+waveSampleRate);
            
            currStudyRowStart = currStudyRowEnd + 1;
        end
        
        % There is no GT so everything set to 0
        isEyeMovement = 0;
        eyeSpeed = 0;
        eyeDirection = 0;
        currGTVector = [isEyeMovement, eyeSpeed, eyeDirection];
        
        %Get timestamp for example
        %times = gt.TimeStamp(currStudyGTStart: currStudyGTStart + seqLen - 1);
        timesVec = [];
        for t=0:seqLen-1 
            tempVec = datevec(currentTime);
            timesVec = [timesVec; tempVec(4:end) ];
            currentTime = currentTime + seconds(1);
        end
        
        
        
           
        if doFiltering == 0
                writeExample(currExample, patientNum, currGTVector, timesVec, currExampleRowStart, currStudyRowEnd, 0, exampleSaveDir);
        else
                [outputExample] = createFilteredSample(currExample,seqLen, featIndices,waveSampleRate, hzThresh, sampleRateHz, orderN, smoothWindowSize);
                writeExample(outputExample, patientNum, currGTVector, timesVec, currExampleRowStart, currStudyRowEnd, 0, exampleSaveDir);
        end
            
        if doAugNoise == 1 && isEyeMovement > 0  %Augment only movement examples
               
            aggNum = 1;
                
            for si=snrs
                   
                    for exNum=1:numExPerSNR
                        [outputExample] = createNoiseSample(currExample,seqLen, featIndices,waveSampleRate, si);
                        
                        if doFiltering == 1
                            [outputExample] = createFilteredSample(outputExample,seqLen, featIndices,waveSampleRate, hzThresh, sampleRateHz, orderN, smoothWindowSize);
                        end
                        
                        writeExample(outputExample, patientNum, currGTVector, timesVec, currExampleRowStart, currStudyRowEnd, aggNum, exampleSaveDir);
                        aggNum = aggNum + 1;
                        
                    end
                    
                    
            end
                
                
        end
            
            
        
      
        iExample = iExample + 1;
 end
    

    


