function [outputSignal] = applyMeanFilter(inputSignal, windowSize)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
kernel = ones(1, windowSize) ./ windowSize;

outputSignal = filtfilt(kernel,1, inputSignal); %filter(kernel, 1, inputSignal);



end

