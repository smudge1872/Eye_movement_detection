function [outputExample] = createFilteredSample(currExample,seqLen, featIndices,waveSampleRate, hzThresh, sampleRateHz, orderN, smoothWindowSize)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

       
        outputExample = currExample;
        
        
        for iStep = 1:seqLen
            
            
            %Apply Filters on each channel
            for feaInd=featIndices'
                inputSignal = zeros(1,waveSampleRate);
                inputSignal(1,:) = currExample(iStep,feaInd,:);
                
                
                [buttSignal] = applyButtworthFilter(inputSignal, hzThresh, sampleRateHz, orderN);
                [meanSignal] = applyMeanFilter(buttSignal, smoothWindowSize);
                %minVal = min(meanSignal);
                %maxVal = max(meanSignal);
                %plotSignal(inputSignal, buttSignal, meanSignal);
                outputExample(iStep,feaInd,:) = meanSignal;
            end
            
        end




end
