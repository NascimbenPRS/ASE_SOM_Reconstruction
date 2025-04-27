function sqi = sqi(clusterClass_matrix)
% Compute a simple clustering quality index on a cluster-class matrix
% SQI (Simple Quality Index)
    [numClusters, numClasses]= size(clusterClass_matrix);
    rowMax= max(clusterClass_matrix,[],2); rowSum= sum(clusterClass_matrix,2);

    clusterScore = (rowMax ./ rowSum)./numClusters;
    
    clusterScore(isnan(clusterScore))=0;

    clusterScore = sum(clusterScore); % =1 if all members of a cluster come from same class

    colMax= max(clusterClass_matrix); colSum= sum(clusterClass_matrix);

    classScore= (colMax ./ colSum)./numClasses; 
    
    classScore(isnan(classScore))=0;
    
    classScore = sum(classScore); % =1 if all members of a class are assigned to the same cluster

    sqi= sqrt(clusterScore) * sqrt(classScore);
end