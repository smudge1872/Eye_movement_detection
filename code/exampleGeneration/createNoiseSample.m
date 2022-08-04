function [outputExample] = createNoiseSample(currExample,seqLen, featIndices,waveSampleRate, snr)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

        %snr = -10 * log10(sigma^2);

        outputExample = currExample;
        
        
        for iStep = 1:seqLen
            
            
            %Apply Filters on each channel
            for feaInd=featIndices'
                inputSignal = zeros(1,waveSampleRate);
                inputSignal(1,:) = currExample(iStep,feaInd,:);
                
                noisedSignal = awgn(inputSignal, snr, 'measured' );
                %[buttSignal] = applyButtworthFilter(inputSignal, hzThresh, sampleRateHz, orderN);
                %[meanSignal] = applyMeanFilter(buttSignal, smoothWindowSize);
                %minVal = min(meanSignal);
                %maxVal = max(meanSignal);
                %plotSignal(inputSignal, buttSignal, meanSignal);
                %plotSignal(inputSignal, noisedSignal, noisedSignal);
                outputExample(iStep,feaInd,:) = noisedSignal; %meanSignal;
            end
            
        end




end

