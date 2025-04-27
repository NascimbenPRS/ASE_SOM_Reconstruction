function fmi = fmi(trueClass, predictedClass)
    % Inputs:
    % trueClass  - vector of true labels (ground-truth)
    % predictedClass - vector of clustering labels
    % Output: the Fowlkes-Mallows Index of the clustering

    % Initialize the counts for TP, FP, and FN
    TP = 0;  % True Positives
    FP = 0;  % False Positives
    FN = 0;  % False Negatives
    
    % Number of samples
    m = length(trueClass);
    
    % Loop over all pairs of samples
    for i = 1:m-1
        for j = i+1:m
            if trueClass(i) == trueClass(j)  % Same ground-truth label
                if predictedClass(i) == predictedClass(j)  % Same predicted cluster
                    TP = TP + 1;  % True Positive
                else
                    FN = FN + 1;  % False Negative
                end
            else  % Different ground-truth labels
                if predictedClass(i) == predictedClass(j)  % Same predicted cluster
                    FP = FP + 1;  % False Positive
                end
            end
        end
    end
    
    % Compute the Fowlkes-Mallows Index
    fmi = TP / sqrt((TP + FP) * (TP + FN));
end
