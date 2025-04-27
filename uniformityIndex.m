function unif = uniformityIndex(cluster_genotypes, set_genotypes)
% Mean genotype-uniformity across all clusters with >=2 genotypes
% cluster_genotypes: [numNeurons,numGenotypes] matrix
% set_genotypes: [numNeurons,2*numGenes] array, row k= genotype k 
g= size(set_genotypes,2)/2; % number of genes
numNeurons= size(cluster_genotypes,1);
nonMonoClusters= 0;
u= zeros(numNeurons,1); % u(i)= uniformity of cluster i
for i=1:numNeurons %for each cluster       
                u(i)=0;
                assignedGenos= find(cluster_genotypes(i,:) > 0); % genotypes featured in the cluster
                for j= 1:length(assignedGenos)
                    g1= assignedGenos(j);
                    for k= (j+1):length(assignedGenos)
                        g2= assignedGenos(k);
                        commonAlleles= length( intersect(set_genotypes(g1,:),set_genotypes(g2,:)) )/(g*2); %fraction of common alleles
                        u(i)= u(i) + commonAlleles;
                    end
                end
                j= length(assignedGenos);
                if j==1 
                    u(i)= 0; % only one cluster, don't count similarity
                end
                if (j >= 2)
                    nonMonoClusters= nonMonoClusters + 1;
                    u(i)= u(i)/nchoosek(j,2); % average uniformity in cluster i
                end
end
if nonMonoClusters>0 
    % there is at least one cluster with >=2 genotypes
    unif= sum(u)/nonMonoClusters; % average unif in population
else 
    % all clusters have at most 1 genotype
    unif=1;
end
end