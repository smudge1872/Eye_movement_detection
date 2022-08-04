function [] = writeExample(currExample, patientNum, currGTVector, timesVec, currExampleRowStart, currStudyRowEnd, sampNum, exampleSaveDir)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes he


 filePrefix = sprintf('%d/%d_%d_%d_s%d', patientNum, patientNum, currExampleRowStart, currStudyRowEnd,sampNum);
            exampleSaveFileName = [filePrefix, '.h5'];
            exampleSavePath = fullfile(exampleSaveDir, exampleSaveFileName);
            try
                h5create(exampleSavePath, '/example', size(currExample));
            catch
            end        
            h5write(exampleSavePath, '/example', currExample);

            try
                h5create(exampleSavePath, '/gt', size(currGTVector));
            catch
            end        
            h5write(exampleSavePath, '/gt', currGTVector);

            try
                h5create(exampleSavePath, '/time', size(timesVec));
            catch
            end        
            h5write(exampleSavePath, '/time', timesVec);

            try
                h5create(exampleSavePath, '/patientNum', size(patientNum));
            catch
            end        
            h5write(exampleSavePath, '/patientNum', patientNum);





end

