function [outputSignal] = applyButtworthFilter(inputSignal, hzThresh, sampleRateHz, orderN)


%Following this example form Matlab's documentation, if you want the cutoff frequency to be at fc Hz at a sampling 
% frequency of fs Hz, you should use:

Wn = hzThresh/(sampleRateHz/2);

[b,a] = butter(orderN, Wn, 'low');

outputSignal = filtfilt(b,a, inputSignal);%filter(b,a, inputSignal);


end

