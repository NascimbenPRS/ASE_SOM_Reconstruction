%{
Francesco Nascimben
PhD student in AI&CS, University of Udine, Italy
%}

% Generate a training population with multiple genes and a limited number of possible
% allele-pairs per gene (i.e. from a limited genoset), use it to train a SOM. 
% Generate a number of test populations P, whose genoset can be the one
% used for training, partially replaced or completely new.
% Turn  each P into a "corrupt" population P' by deleting a variable number of alleles and compare
% clustering results on P vs P'.

% Reconstruct P from P' by applying completion rules based on allele
% frequency (see paper)

            % --- To EVALUATE CLUSTERING on base and corrupted population, 3 quality indices:

            % 1. Simple Quality Index (SQI)= a simple measure of purity (created ad hoc)
            % 2. Fowlkes-Marrows Index (FMI)= measure of how main pairs of individuals
            % in the same cluster have the same genotype
            % 3. Normalized Mutual Information (NMI))= measure based on entropy

            % --- To EVALUATE SIMILARITY between base and corr. pop., we
            % use rand index

            % --- To EVALUATE ALLELE ACCURACY and READ

% To account for randomness, we train multiple networks on the same
% parameters (deleted alleles, replaced genotypes in the genoset) and
% compute min, mean, max and standard dev. of all indices

% Initialize values
g= 10; % number of genes
n = 10; % number of rows/alleles PER GENE
m = 5000; % number of columns/individuals for TRAINING population
m_test= 2000; % number indiv. for TEST populations
readsRange= [1,700];
num_genotypes = 100; % number of distinct genotypes in the population ACROSS ALL GENES for training
num_genotypes_test = num_genotypes; % number of distinct genotypes in the population ACROSS ALL GENES for testing

numPopulations= 100; % number of populations for each test

dimension1 = 10; % First dimension for the SOM
dimension2 = 10; % Second dimension for the SOM
numNeurons= dimension1*dimension2;
numNetworks= 15; % number of different networks
% Stats across networks (same number of deleted alleles)
sqiStats= zeros(numNetworks,4); % min, mean, max SQI (+ standard dev) for each population of the repeat
sqiStats_corr= zeros(numNetworks,4); % min, mean, max SQI (+ st. dev) for each corr. pop. of the rep.
fmiStats= zeros(numNetworks,4); fmiStats_corr = zeros(numNetworks,4);
nmiStats= zeros(numNetworks,4); nmiStats_corr= zeros(numNetworks,4);
similarityStats= zeros(numNetworks,4); % min, mean, max similarity (+ st. dev) for each rep.

accuracyStats= zeros(numNetworks,4); % min, mean, max allele accuracy (+ st. dev) for each rep.
readErrorStats= zeros(numNetworks,4); % min, mean, max read error (+ st. dev) for each rep.
% Stats across deleted alleles
allSQIStats= zeros(11,4); allSQIStats_corr= zeros(11,4);
allFMIStats= zeros(11,4); allFMIStats_corr= zeros(11,4);
allNMIStats= zeros(11,4); allNMIStats_corr= zeros(11,4);
allSimStats= zeros(11,4);

finalAccStats= zeros(11,4);
finalReadAccStats= zeros(11,4);

finalGuessedOnce= zeros(11,1);

for num_del= 0:2:18 % test the networks deleting a varying number of alleles
    guessedOnceNet= zeros(numNetworks,1); % (ntw) = fraction of populations where at least one allele was guessed by network ntw
    for netw=1:numNetworks % repeat the experiment w/ same parameters, but w/ different training-test sets (to account for training randomness)
        
        %                       ----- TRAINING
        
        % Generate a set of genotypes which will appear in the training population 
        set_genotypes = zeros(num_genotypes, 2*g);
        for i = 1:num_genotypes % generate a set of genotypes
            for k = 1:g % for each gene...
                set_genotypes(i, [1+2*(k-1),2+2*(k-1)]) = n*(k-1) + sort(randperm(n, 2),2); % ...randomly select 2 positions
            end
        end
        % Populate Reads matrix: each column has 2 random positions with values between 20 and 200
        [Reads,geno_train]= randomReadsMatrixGenerator(m,g,n,set_genotypes,readsRange);

        % Create a SOM network, train it on the Reads matrix, visualize it
        net = selforgmap([dimension1 dimension2]); % Create a SOM network
        net = train(net, Reads);

        allFrequency= zeros(g,n,numNeurons); % (i,j,t): frequency of j-th allele of gene i in subpopulation t
        pairFrequency= zeros(g,n*n,numNeurons); % (i,j,t): frequency of j-th allele-pair of gene i in subpopulation t
        % pair (Au,Ah): j= u*n + h
        allAvgExpr= zeros(g,n,numNeurons); % (i,j,t): avg expr of j-th allele of gene i in subpopulation t
        pairAvgExpr= zeros(g,n*n,numNeurons); % (i,j,t): avg expr of j-th allele-pair of gene i in subpopulation t
        % i.e. avg expr of allele u when paired with allele h

        output_train = sim(net, Reads); % run SOM on training population        
        % Populate cluster-genotypes matrices
        class_train= zeros(m,1); % store the cluster assigned to each (corr) individual
        cluster_genotypes_train = zeros(numNeurons,num_genotypes);
        for k = 1:m
            class_train(k)= find(output_train(:,k) ==1);
            cluster_genotypes_train(class_train(k),geno_train(k))= cluster_genotypes_train(class_train(k),geno_train(k))+1;
        end

        % Compute allele frequency and avg expr in each subpopulation of the
        % training set induced by the SOM clustering
        for k=1:m % for each k in training set...
            t = class_train(k); % t= cluster k is assigned to 
            for j= set_genotypes(geno_train(k),:)% idx-th allele of k
                i= floor((j-1)/n)+1; % i= gene of allele j
                js= mod(j,n); % e.g. js= x*n+4 -> 4th allele of gene x)
                if (js==0)
                    js=n; % case js= x*n+0 (last allele of gene x)
                end
                allAvgExpr(i,js,t)= allAvgExpr(i,js,t) + Reads(j,k);
                allFrequency(i,js,t)= allFrequency(i,js,t) + 1;
            end
        end
        for t=1:numNeurons % compute average
            allAvgExpr(:,:,t)= allAvgExpr(:,:,t) ./ allFrequency(:,:,t); 
            %allFrequency(:,:,t)= allFrequency(:,:,t)/(2 * length(class_train(class_train == t)) );
        end
        allAvgExpr(isnan(allAvgExpr))=0; % fix 0/0 cases
        %allFrequency(isnan(allFrequency))=0;

        % Compute pair frequency and avg expr in each subpopulation of trainset
        partitionSize= zeros(numNeurons,1);
        for k=1:m % for each k in training set
            t = class_train(k); % t= cluster k is assigned to
            partitionSize(t)= partitionSize(t) +1;
            for i= 1:g % for each gene
               j1= set_genotypes(geno_train(k),(i-1)*2 +1); % 1st allele in gene i of k
               j2= set_genotypes(geno_train(k),(i-1)*2 +2); % 2nd allele in gene i of k
               j1s= rem(j1,n); j1s(j1s==0)=n; j2s= rem(j2,n); j2s(j2s==0)=n;
               pairFrequency(i,(j1s-1)*n +j2s,t)= pairFrequency(i,(j1s-1)*n +j2s,t) + 1;
               pairFrequency(i,(j2s-1)*n +j1s,t)= pairFrequency(i,(j2s-1)*n +j1s,t) + 1; % update freqs. simmetrically
               pairAvgExpr(i,(j1s-1)*n +j2s,t)= pairAvgExpr(i,(j1s-1)*n +j2s,t) + Reads(j1,k); % increase reads of all. j1 when paired with all. j2
               pairAvgExpr(i,(j2s-1)*n +j1s,t)= pairAvgExpr(i,(j2s-1)*n +j1s,t) + Reads(j2,k); % increase reads of all. j2 when paired with all. j1
            end
        end
        for t=1:numNeurons % compute average
            pairAvgExpr(:,:,t)= pairAvgExpr(:,:,t) ./ pairFrequency(:,:,t); 
            pairAvgExpr(isnan(pairAvgExpr))=0; % fix 0/0 cases
        end

        [SQI, SQI_corr]= deal(zeros(numPopulations,1)); % Simple Quality Index for each (corrupted) test population
        [FMI, FMI_corr]= deal(zeros(numPopulations,1)); % Fowlkes-Mallows Index for each (corrupted) test population
        [NMI, NMI_corr]= deal(zeros(numPopulations,1)); % Normalized Mutual Information for each (corrupted) test population
        clusterSim= zeros(numPopulations,2); % clustering similarity for each test population (rand index)

        [readsAcc_Pop, allAcc_Pop]= deal(zeros(numPopulations,1)); % average reads and allele prediction accuracy for each (corrupted) test population
   
        % Test the net on a number of populations
        for p =1:numPopulations

            %    -----------           GENERATING TEST SETS

            set_genotypes_Test= set_genotypes; % same genoset as training

            % Populate Reads_Test matrix and corrupted matrix (annihilate num_del alleles per individual)
            [Reads_Test,geno]= randomReadsMatrixGenerator(m_test,g,n,set_genotypes_Test,readsRange);

            Reads_Test_Corrupted = Reads_Test;
            del= zeros(m_test,num_del); % save alleles deleted for each individual
            for k = 1:m_test % for each individual k
                j= 1;
                indexes= randperm(2*g,num_del); % at random...
                for idx=indexes
                    deletedAllele = set_genotypes_Test(geno(k),idx); % ...pick an allele...
                    del(k,j)= deletedAllele; %... memorize it...
                    Reads_Test_Corrupted(deletedAllele,k)= 0; %...and set it to 0
                    j= j+1;
                end
            end
            
            %                   ----    PHASE 1    ----
            %       cluster (corrupted) individuals and 
            %       compute quality/similarity
            
            % Run the trained SOM on the test population matrix
            output = sim(net, Reads_Test);
            output_Corrupted= sim(net, Reads_Test_Corrupted);
            
            % Compare clusters and genotypes
            [class, class_Corrupted]= deal(zeros(m_test,1)); % store the cluster assigned to each (corr) individual
            [cluster_genotypes, cluster_genotypes_Corrupted]= deal(zeros(numNeurons,num_genotypes_test)); % (i,j)= how many indiv. with genot. j in cluster i 
            
            % Populate cluster-genotypes matrices
            for k = 1:m_test
                class(k)= find(output(:,k) ==1);
                i= class(k);
                cluster_genotypes(i,geno(k))= cluster_genotypes(i,geno(k))+1;
                class_Corrupted(k)= find(output_Corrupted(:,k) ==1);
                i= class_Corrupted(k);
                cluster_genotypes_Corrupted(i,geno(k))= cluster_genotypes_Corrupted(i,geno(k))+1;
            end

            % Compute clustering quality indexes for
            % each population and its corrupted version
            SQI(p)= sqi(cluster_genotypes); SQI_corr(p)= sqi(cluster_genotypes_Corrupted);
            FMI(p)= fmi(class, geno); FMI_corr(p)= fmi(class_Corrupted, geno);
            NMI(p)= nmi(class, geno); NMI_corr(p)= nmi(class_Corrupted, geno);
            % Compute similarity between base and corrupted population
            clusterSim(p,:)= randindex(class, class_Corrupted);
            
            % Initialize accuracy arrays for 2nd phase
            allAccuracy_Ind= zeros(m_test,1); % allele prediction accuracy on each k. Ranges in [0,1]
            numGuessed= zeros(m_test,1); % num correctly guessed genes for each k
            readsAccuracy_Ind= zeros(m_test,1); % expr accuracy on each k (on correctly guessed genes)
            % readsAccuracy_Ind ranges in [0,+inf]. Perfect prediction when
            % readAcc==0



            %              ----     PHASE 2     ----
            %       Reconstruct corrupted populations and 
            %       evaluate allele/read accuracy
            for k= 1:m_test
                t = class_Corrupted(k); % t= cluster k is assigned to
                if (partitionSize(t)==0)
                    % no prediction rule for cluster t, can't reconstruct k

                else    
                    for i= 1:g % for each gene i
                        geneOffset= (i-1)*n;
                        alpha= find(Reads_Test_Corrupted( (geneOffset+1):(geneOffset+n) ,k) > 0); % find uncorr. alleles
                        l=length(alpha);
                        if l==0 % all gene-i alleles were deleted
                            % assign most frequent all. pair in subpopulation t
                            [~,pairIdx]= max(pairFrequency(i,:,t)); % find most frequent pair
                            alpha1= floor((pairIdx-1)/n) + 1;
                            alpha2= pairIdx - (alpha1-1)*n;
                            % expr alpha1[k]= avgExpr. alpha1 with alpha2 in t
                            Reads_Test_Corrupted(geneOffset+alpha1,k)= pairAvgExpr(i,n*(alpha1-1)+alpha2,t);
                            % expr alpha2[k]= avgExpr. alpha2 with alpha1 in t
                            Reads_Test_Corrupted(geneOffset+alpha2,k)= pairAvgExpr(i,n*(alpha2-1)+alpha1,t);
                        end % end case l=0
    
                        if (l==1) % one gene-i allele remains
                            alpha1= alpha;
                            alpha1Expr= Reads_Test(geneOffset+alpha1,k);
                            if (allFrequency(i,alpha1,t) == 0)
                                % All. alpha1 does not appear in subpop. t
                                % Assign most frequent allele in t
                                [~,alpha2]= max(allFrequency(i,:,t)); % find most frequent allele
                                % alpha2Expr= alpha1Expr * mean(alpha2/beta), beta in {alleles paired to alpha2 in t}
                                % -- Idea: avgRatio scales alpha1Expr to
                                % -- account for environmental effects on expr in k
                                startIdx= (alpha2-1)*n +1; % start of alpha2 block in pairAvgExpr
                                paired2= find(pairAvgExpr(i,startIdx:(startIdx+n-1),t) > 0);
                                avgRatio=0;
                                for beta=paired2 % compute ratio of alpha2-beta
                                    ratio2beta= pairAvgExpr(i,n*(alpha2-1)+beta,t)/pairAvgExpr(i,n*(beta-1)+alpha2,t);
                                    avgRatio= avgRatio + ratio2beta;
                                end
                                avgRatio= avgRatio/length(paired2);
                                alpha2Expr= avgRatio * alpha1Expr;
                                Reads_Test_Corrupted(geneOffset+alpha2,k)= alpha2Expr;
                            else
                                % All. alpha1 appears in subpop. t
                                % Assign all. most frequent with alpha1
                                startIdx= ((alpha1-1)*n+1); % start of alpha1 block in pairFrequency
                                [~,alpha2]= max(pairFrequency(i,startIdx:(startIdx+n-1),t)); % find most frequent allele with alpha1
                                % alpha2Expr= alpha1Expr * (alpha2/alpha1)
                                avgRatio= pairAvgExpr(i,n*(alpha2-1)+alpha1,t)/pairAvgExpr(i,n*(alpha1-1)+alpha2,t);
                                alpha2Expr= avgRatio*alpha1Expr;
                                Reads_Test_Corrupted(geneOffset+alpha2,k)= alpha2Expr;
    
                            end % end case l=1
                        else
                            % no deletion on gene-i, do nothing
                        end
                    end
                end % end check on partition size 
                
                % Compute allele and reads accuracy on k
                numGuessed(k)= 0;
                for deletedAllele=del(k,:)
                    r_corr= Reads_Test_Corrupted(deletedAllele,k);
                    if (r_corr > 0)
                        % allele was correctly guessed in the second phase
                        numGuessed(k)= numGuessed(k) +1;
                        r_og= Reads_Test(deletedAllele,k);
                        readsAccuracy_Ind(k)= readsAccuracy_Ind(k) + (abs(r_corr-r_og))/r_og;
                    end
                end
                allAccuracy_Ind(k)= numGuessed(k)/num_del;
                if (numGuessed(k) > 0) % at least one allele guessed
                    readsAccuracy_Ind(k)= readsAccuracy_Ind(k)/numGuessed(k); % average reads acc
                end
            end % end all/readAcc computation for each individual
            % Compute average allele and reads acc. on test population p
            allAcc_Pop(p)= mean(allAccuracy_Ind);
            if (not (isempty(numGuessed(numGuessed > 0))) ) % at least one allele was guessed in test pop p
                guessedOnceNet(netw)= guessedOnceNet(netw) +1;
                % compute mean readsAcc only on those k with at least one
                % guessed allele
                readsAcc_Pop(p)= mean( readsAccuracy_Ind(find(numGuessed > 0)) );
            end



        end % end for: test a single net on numPopulations populations
        guessedOnceNet(netw)= guessedOnceNet(netw) / numPopulations;

        % Compute min, mean, max and standard dev. for each index for both
        % base and corrupted populations
        sqiStats(netw,:)= [min(SQI),mean(SQI),max(SQI), std(SQI)]; sqiStats_corr(netw,:) = [min(SQI_corr),mean(SQI_corr),max(SQI_corr),std(SQI_corr)];
        fmiStats(netw,:)= [min(FMI),mean(FMI),max(FMI), std(FMI)]; fmiStats_corr(netw,:) = [min(FMI_corr),mean(FMI_corr),max(FMI_corr),std(FMI_corr)];
        nmiStats(netw,:)= [min(NMI),mean(NMI),max(NMI), std(NMI)]; nmiStats_corr(netw,:) = [min(NMI_corr),mean(NMI_corr),max(NMI_corr),std(NMI_corr)];

        similarityStats(netw,:) = [min(clusterSim(:,1)),mean(clusterSim(:,1)),max(clusterSim(:,1)), std(clusterSim(:,1))];

        accuracyStats(netw,:)=[min(allAcc_Pop),mean(allAcc_Pop),max(allAcc_Pop), std(allAcc_Pop)]; % allele accuracy stats on this network
        readErrorStats(netw,:)=[min(readsAcc_Pop),mean(readsAcc_Pop),max(readsAcc_Pop), std(readsAcc_Pop)]; % reads error stats on this network

    end % end for: repeats of the whole experiment on different networks, deleting a fixed n. of alleles

    % Compute statistics of each index for base and corr. populations (across numNetworks different repeats)
    % for the current number of deleted alleles (num_del)
    allSQIStats((num_del/2)+1, :)= [min(sqiStats(:,1)),mean(sqiStats(:,2)),max(sqiStats(:,3)),mean(sqiStats(:,4))];
    allSQIStats_corr((num_del/2)+1, :)= [min(sqiStats_corr(:,1)),mean(sqiStats_corr(:,2)),max(sqiStats_corr(:,3)),mean(sqiStats_corr(:,4))];
    allFMIStats((num_del/2)+1, :)= [min(fmiStats(:,1)),mean(fmiStats(:,2)),max(fmiStats(:,3)),mean(fmiStats(:,4))];
    allFMIStats_corr((num_del/2)+1, :)= [min(fmiStats_corr(:,1)),mean(fmiStats_corr(:,2)),max(fmiStats_corr(:,3)),mean(fmiStats_corr(:,4))];
    allNMIStats((num_del/2)+1, :)= [min(nmiStats(:,1)),mean(nmiStats(:,2)),max(nmiStats(:,3)),mean(nmiStats(:,4))];
    allNMIStats_corr((num_del/2)+1, :)= [min(nmiStats_corr(:,1)),mean(nmiStats_corr(:,2)),max(nmiStats_corr(:,3)),mean(nmiStats_corr(:,4))];
    allSimStats((num_del/2)+1, :)= [min(similarityStats(:,1)),mean(similarityStats(:,2)),max(similarityStats(:,3)),mean(similarityStats(:,4))];

    finalAccStats((num_del/2)+1, :)= [min(accuracyStats(:,1)),mean(accuracyStats(:,2)),max(accuracyStats(:,3)),mean(accuracyStats(:,4))];
    finalReadAccStats((num_del/2)+1, :)= [min(readErrorStats(:,1)),mean(readErrorStats(:,2)),max(readErrorStats(:,3)),mean(readErrorStats(:,4))];

    finalGuessedOnce((num_del/2)+1,:)= mean(guessedOnceNet);
    

end % end for: varying number of alleles
