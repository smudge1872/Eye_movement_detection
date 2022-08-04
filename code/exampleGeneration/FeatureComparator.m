classdef FeatureComparator
    %FEATURECOMPARATOR Determines which feature a column name is
    %   Detailed explanation goes here
    
    
    methods(Static)
        function [featIndices, newFeatureNames] = convertFeatureNames(featureNames, whichFEATs)
            
            % Return numeric index representation for each feature name
            
            % Also convert feature names to aliases if they exist to keep the
            % data consistent. (Not all data given to us had the same
            % labels)
            
            % Certain features simply have the first characters swapped
            % with the last ones
            
            newFeatureNames = strings(size(featureNames));
            
            for iFeature = 1:numel(featureNames)
                
                feature = featureNames(iFeature);
                switch feature
                    case "A2F3"
                        newFeature = "F3A2";
                    case "F4A1"
                        newFeature = "A1F4";
                    case "C3A2"
                        newFeature = "A2C3";
                    % Patient 3 has this anomaly
                    case "A2C4"
                        newFeature = "A2C3";
                    case "C4A1"
                        newFeature = "A1C4";
                    case "O1A2"
                        newFeature = "A2O1";
                    case "O2A1"
                        newFeature = "A1O2";
                    % Patient 24 has this anomaly
                    case "A1O1"
                        newFeature = "A1O2";
                    case "PG1A2"
                        newFeature = "A2PG1";
                    case "PG2A1"
                        newFeature = "A1PG2";
                    case "CZPZ"
                        newFeature = "PZCZ";
                    otherwise
                        newFeature = feature;
                end               
                newFeatureNames(iFeature) = newFeature;
            end
            
            if strcmp(whichFEATs, 'all')
                numExpectedFeat = 9;
            elseif strcmp(whichFEATs, 'LOCROC')
                numExpectedFeat = 2;
            elseif ( strcmp(whichFEATs, 'ROC') || strcmp(whichFEATs, 'LOC') )
                numExpectedFeat = 1;
            elseif( strcmp(whichFEATs, 'other7') )
                numExpectedFeat = 7;
            else
                fprintf(2,'ERROR Feature selection of %s not found !!!\n', whichFEATs);
                featIndices = [];
                newFeatureNames = [];
                return;
            end
            
            
            featIndices = zeros(size(featureNames));
            for iFeature = 1:numel(newFeatureNames)
                feature = newFeatureNames(iFeature);
                
                if strcmp(whichFEATs, 'all')
                    switch feature
                        case "F3A2"
                            newIndex = 1;
                        case "A1F4"
                            newIndex = 2;
                        case "A2C3"
                            newIndex = 3;
                        case "A1C4"
                            newIndex = 4;
                        case "A2O1"
                            newIndex = 5;
                        case "A1O2"
                            newIndex = 6;
                        case "A2PG1"
                            newIndex = 7;
                        case "A1PG2"
                            newIndex = 8;
                        case "PZCZ"
                            newIndex = 9;
                        otherwise
                            newIndex = 0;
                    end 
                  
                elseif strcmp(whichFEATs, 'LOCROC')
                        switch feature

                            case "A2PG1"  %LOC
                                newIndex = 1; %7;
                            case "A1PG2"  %ROC
                                newIndex = 2; %8;

                            otherwise
                                newIndex = 0;
                        end        
                elseif strcmp(whichFEATs, 'LOC')
                       switch feature

                         case "A2PG1"  %LOC
                                newIndex = 1; %7;
                   
                         otherwise
                             newIndex = 0;
                        end            
                    
                elseif strcmp(whichFEATs, 'ROC')
                    
                        switch feature
                            case "A1PG2"  %ROC
                                newIndex = 1; %8;

                            otherwise
                                newIndex = 0;
                        end          
                    
                elseif strcmp(whichFEATs, 'other7')
                   
                        switch feature
                         case "F3A2"
                            newIndex = 1;
                         case "A1F4"
                            newIndex = 2;
                         case "A2C3"
                             newIndex = 3;
                         case "A1C4"
                             newIndex = 4;
                         case "A2O1"
                             newIndex = 5;
                         case "A1O2"
                             newIndex = 6;
                         case "PZCZ"
                            newIndex = 7;
                         otherwise
                            newIndex = 0;
                         end       
                    
                    
                end
                
                    
                featIndices(iFeature) = newIndex;
            end
            
            numValidFeatures = sum(featIndices ~= 0);
            if numValidFeatures ~= numExpectedFeat
                error('Incorrect number of features provided: %d, expected %d', numValidFeatures, numExpectedFeat);
            end
            
        end
    end
end

