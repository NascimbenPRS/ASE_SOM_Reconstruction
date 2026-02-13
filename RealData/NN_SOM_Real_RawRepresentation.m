%{
Francesco Nascimben
PhD student in AI&CS, University of Udine, Italy
VERSION 2.4 (FINAL): NN_SOM_Real_v2 + Robust Normalized Relevance
%}

% --- 1. LOAD & SETUP ---
clear; clc; close all;

%dataFileName = 'FullData_1148.mat';
dataFileName = 'FullData_95.mat'; % Real grapevine dataset
if exist(dataFileName, 'file')
    fprintf('Loading %s...\n', dataFileName);
    load(dataFileName);
else
    error('File %s not found.', dataFileName);
end

% Map variables
Reads = SynthCounts;
%TrueLabels = ResultsRealSynth.ClusterLabels; % extended dataset
TrueLabels = Results.ClusterLabels; % Real grapevine dataset
AlleleCounts = AlleleNumbersCh1;

% Pre-processing
g = length(AlleleCounts); % number of alleles
max_n = max(AlleleCounts); 
GeneOffsets = [0; cumsum(AlleleCounts)]; 

% Map creation
totalAlleles = sum(AlleleCounts);
AlleleToGeneMap = zeros(totalAlleles, 1);
for i = 1:g
    AlleleToGeneMap( (GeneOffsets(i)+1):GeneOffsets(i+1) ) = i;
end

% Filtering
minNumAllele = 700; 
col_idx = find(sum(Reads > 0, 1) < minNumAllele); 
Reads(:, col_idx) = [];
TrueLabels(col_idx,:) = []; 

m = size(Reads,2); 
m_test = m; 
fprintf('Individuals: %d | Genes: %d\n', m, g);

% Average expressed alleles
avgExpressed = mean(sum(Reads > 0, 1));
fprintf('Avg Expressed Alleles: %.2f\n', avgExpressed);

% Network Configuration
dimension1 = 4; dimension2 = 4;
numNeurons = dimension1 * dimension2; 
numNetworks = 15; 

% Stats Storage
step = 100;
maxDel = minNumAllele;
num_steps = floor(maxDel/step) + 1;

[allSQIStats, allSQIStats_corr, allFMIStats, allFMIStats_corr, ...
 allNMIStats, allNMIStats_corr, allSimStats, allSimStats_corr, ...
 finalAccStats, finalReadAccStats] = deal(zeros(num_steps, 4));
 
finalGuessedOnce = zeros(num_steps, 1);

% --- Feature Relevance Storage ---
GlobalRel_Error = zeros(num_steps, totalAlleles);    
GlobalRel_Discrim = zeros(num_steps, totalAlleles);  
Rank_Error = zeros(num_steps, totalAlleles);         
Rank_Discrim = zeros(num_steps, totalAlleles);       

% Store partition sizes
partitionSize = zeros(numNeurons, 1);

% --- MAIN LOOP ---
for num_del = 0:step:maxDel 
    fprintf('\n=== Corruption Level: %d / %d ===\n', num_del, maxDel);
    
    [sqiS, sqiS_c, fmiS, fmiS_c, nmiS, nmiS_c, simS, simS_c, accS, readS] = deal(zeros(numNetworks, 4));
    guessedOnceNet = zeros(numNetworks,1); 

    tempRel_R = zeros(numNetworks, totalAlleles);
    tempRel_D = zeros(numNetworks, totalAlleles);

    for netw = 1:numNetworks 
        fprintf('  Net %d: Train...', netw);
        
        % --- A. SOM TRAINING ---
        net = selforgmap([dimension1 dimension2]); 
        net.trainParam.showWindow = false; 
        net = train(net, Reads);
        output_train = net(Reads);   
        
        % Get Cluster Labels
        [~, labelsSOM_train] = max(output_train, [], 1);
        labelsSOM_train = labelsSOM_train'; 
        
        % Evaluation (Train)
        SOMvs2PP_train = zeros(numNeurons, max(TrueLabels));
        for k = 1:m
            c = labelsSOM_train(k);
            l = TrueLabels(k);
            SOMvs2PP_train(c, l) = SOMvs2PP_train(c, l) + 1;
        end
        
        % --- B. STATS COLLECTION ---
        fprintf(' Stats...');
        
        allFrequency = zeros(g, max_n, numNeurons);
        allAvgExpr = zeros(g, max_n, numNeurons);
        pairFrequency = zeros(g, max_n^2, numNeurons);
        pairAvgExpr = zeros(g, max_n^2, numNeurons);
        partitionSize(:) = 0; 
        
        for t = 1:numNeurons
            indivs_in_t = find(labelsSOM_train == t);
            partitionSize(t) = length(indivs_in_t);
            if isempty(indivs_in_t), continue; end
            
            Block = Reads(:, indivs_in_t);
            for i = 1:g
                r_start = GeneOffsets(i) + 1;
                r_end = GeneOffsets(i+1);
                n_alleles = r_end - r_start + 1;
                
                subBlock = Block(r_start:r_end, :);
                
                % Single Stats
                allFrequency(i, 1:n_alleles, t) = sum(subBlock > 0, 2);
                allAvgExpr(i, 1:n_alleles, t) = sum(subBlock, 2);
                
                % Pair Stats
                binBlock = subBlock > 0;
                validCols = find(sum(binBlock, 1) == 2);
                if ~isempty(validCols)
                    pairBlockVal = subBlock(:, validCols);
                    pairBlockBin = binBlock(:, validCols);
                    [rows, cols] = find(pairBlockBin); 
                    
                    r1 = rows(1:2:end); r2 = rows(2:2:end);
                    vals = pairBlockVal(sub2ind(size(pairBlockVal), rows, cols));
                    v1 = vals(1:2:end); v2 = vals(2:2:end);
                    
                    idx12 = (r1-1)*max_n + r2;
                    idx21 = (r2-1)*max_n + r1;
                    
                    pairFrequency(i, :, t) = pairFrequency(i, :, t) + ...
                        accumarray(idx12, 1, [max_n^2, 1])' + ...
                        accumarray(idx21, 1, [max_n^2, 1])';
                        
                    pairAvgExpr(i, :, t) = pairAvgExpr(i, :, t) + ...
                        accumarray(idx12, v1, [max_n^2, 1])' + ...
                        accumarray(idx21, v2, [max_n^2, 1])';
                end
            end
        end
        
        mask = allFrequency > 0;
        allAvgExpr(mask) = allAvgExpr(mask) ./ allFrequency(mask);
        mask_p = pairFrequency > 0;
        pairAvgExpr(mask_p) = pairAvgExpr(mask_p) ./ pairFrequency(mask_p);
        
        fprintf(' Test...');

        % --- C. TEST & CORRUPTION ---
        Reads_Test_Corrupted = Reads;
        del = zeros(m_test, num_del); 
        
        if num_del > 0
            for k = 1:m_test 
                non_null = find(Reads(:,k) > 0); 
                if length(non_null) >= num_del
                    perm_idx = randperm(length(non_null), num_del);
                    chosen = non_null(perm_idx);
                    del(k,:) = chosen;
                    Reads_Test_Corrupted(chosen,k) = 0;   
                end
            end
        end 
        
        % Clustering on Corrupted Data
        dists = dist(net.IW{1,1}, Reads_Test_Corrupted); 
        [min_vals, labelsSOM_Corrupted] = min(dists, [], 1);
        labelsSOM_Corrupted = labelsSOM_Corrupted';
        
        dists_train = dist(net.IW{1,1}, Reads);
        [~, labelsSOM] = min(dists_train, [], 1);
        labelsSOM = labelsSOM';

        % --- ROBUST LOCAL FEATURE RELEVANCE ---
        accum_R = zeros(totalAlleles, 1);
        accum_D = zeros(totalAlleles, 1);
        AllWeights = net.IW{1,1};
        
        for k = 1:m_test
            x = Reads_Test_Corrupted(:, k);
            
            % 1. Identify BMU and 2nd BMU
            dist_vec = dists(:, k);
            [~, sortedNeurons] = sort(dist_vec, 'ascend'); 
            bmu_idx = sortedNeurons(1);
            sbmu_idx = sortedNeurons(2);
            
            w_bmu = AllWeights(bmu_idx, :)';
            w_sbmu = AllWeights(sbmu_idx, :)';
            
            % 2. Squared Component Errors
            diff_bmu_sq = (x - w_bmu).^2;
            diff_sbmu_sq = (x - w_sbmu).^2;
            
            % 3. R Index Normalization (Total Error)
            total_error = sum(diff_bmu_sq);
            if total_error > 1e-10
                r_vec = diff_bmu_sq ./ total_error;
            else
                r_vec = zeros(totalAlleles, 1);
            end
            
            % 4. D Index Normalization (Sum of Absolute Margins)
            % D_raw = Error_RunnerUp - Error_Winner
            d_vec_raw = diff_sbmu_sq - diff_bmu_sq;
            total_abs_margin = sum(abs(d_vec_raw));
            
            if total_abs_margin > 1e-10
                % This ensures d_vec is strictly in [-1, 1]
                d_vec = d_vec_raw ./ total_abs_margin;
            else
                d_vec = zeros(totalAlleles, 1);
            end
            
            accum_R = accum_R + r_vec;
            accum_D = accum_D + d_vec;
        end
        
        tempRel_R(netw, :) = accum_R' / m_test;
        tempRel_D(netw, :) = accum_D' / m_test;

        % ------------------------------------------------
        
        % Metrics
        SOMvs2PP = zeros(numNeurons, max(TrueLabels));
        SOMvs2PP_Corr = zeros(numNeurons, max(TrueLabels));
        for k=1:m_test
           SOMvs2PP(labelsSOM(k), TrueLabels(k)) = SOMvs2PP(labelsSOM(k), TrueLabels(k)) + 1;
           SOMvs2PP_Corr(labelsSOM_Corrupted(k), TrueLabels(k)) = SOMvs2PP_Corr(labelsSOM_Corrupted(k), TrueLabels(k)) + 1;
        end
        
        SQI = sqi(SOMvs2PP); SQI_C = sqi(SOMvs2PP_Corr);
        FMI = fmi(labelsSOM', TrueLabels); FMI_C = fmi(labelsSOM_Corrupted', TrueLabels);
        NMI = nmi(labelsSOM', TrueLabels); NMI_C = nmi(labelsSOM_Corrupted', TrueLabels);
        Sim = randindex(labelsSOM', labelsSOM_Corrupted');
        Sim_C = randindex(labelsSOM_Corrupted', TrueLabels);

        sqiS(netw,:) = [SQI SQI SQI 0]; sqiS_c(netw,:) = [SQI_C SQI_C SQI_C 0];
        fmiS(netw,:) = [FMI FMI FMI 0]; fmiS_c(netw,:) = [FMI_C FMI_C FMI_C 0];
        nmiS(netw,:) = [NMI NMI NMI 0]; nmiS_c(netw,:) = [NMI_C NMI_C NMI_C 0];
        simS(netw,:) = [Sim Sim Sim 0]; simS_c(netw,:) = [Sim_C Sim_C Sim_C 0];

        % --- D. RECONSTRUCTION ---
        fprintf(' Rec...');
        allAcc_Ind = zeros(m_test, 1);
        readAcc_Ind = zeros(m_test, 1);
        hasGuessed = false(m_test, 1);
        
        for k = 1:m_test
            t = labelsSOM_Corrupted(k);
            if partitionSize(t) == 0, continue; end
            
            deleted_indices = del(k, :);
            deleted_indices = deleted_indices(deleted_indices > 0);
            
            if isempty(deleted_indices)
                allAcc_Ind(k) = NaN; 
                continue; 
            end
            
            affected_genes = unique(AlleleToGeneMap(deleted_indices));
            
            for i = affected_genes'
                g_start = GeneOffsets(i) + 1;
                g_end = GeneOffsets(i+1);
                gene_vals = Reads_Test_Corrupted(g_start:g_end, k);
                alpha = find(gene_vals > 0);
                l = length(alpha);
                
                if l == 0
                    pair_freqs = pairFrequency(i, :, t);
                    [maxVal, pairIdx] = max(pair_freqs);
                    if maxVal > 0
                        a1 = floor((pairIdx-1)/max_n) + 1;
                        a2 = pairIdx - (a1-1)*max_n;
                        Reads_Test_Corrupted(g_start + a1 - 1, k) = pairAvgExpr(i, (a1-1)*max_n + a2, t);
                        Reads_Test_Corrupted(g_start + a2 - 1, k) = pairAvgExpr(i, (a2-1)*max_n + a1, t);
                    end
                elseif l == 1
                    a1 = alpha(1);
                    expr1 = gene_vals(a1);
                    if allFrequency(i, a1, t) == 0
                        [~, a2] = max(allFrequency(i, :, t));
                        startIdx = (a2-1)*max_n + 1;
                        pairs_slice = pairAvgExpr(i, startIdx:(startIdx+max_n-1), t);
                        valid_betas = find(pairs_slice > 0);
                        if ~isempty(valid_betas)
                            nums = pairs_slice(valid_betas);
                            den_idx = (valid_betas-1)*max_n + a2;
                            dens = pairAvgExpr(i, den_idx, t);
                            if any(dens > 0)
                                avgRatio = mean(nums(dens>0) ./ dens(dens>0));
                                Reads_Test_Corrupted(g_start + a2 - 1, k) = avgRatio * expr1;
                            end
                        end
                    else
                        startIdx = (a1-1)*max_n + 1;
                        [~, a2] = max(pairFrequency(i, startIdx:(startIdx+max_n-1), t));
                        num = pairAvgExpr(i, (a2-1)*max_n + a1, t);
                        den = pairAvgExpr(i, (a1-1)*max_n + a2, t);
                        if den > 0
                            Reads_Test_Corrupted(g_start + a2 - 1, k) = (num/den) * expr1;
                        end
                    end
                end
            end 
            
            n_guessed = 0; sum_read_err = 0;
            for d_idx = deleted_indices
                pred = Reads_Test_Corrupted(d_idx, k);
                if pred > 0
                    n_guessed = n_guessed + 1;
                    actual = Reads(d_idx, k);
                    if actual > 0
                        sum_read_err = sum_read_err + abs(pred - actual)/actual;
                    end
                end
            end
            
            allAcc_Ind(k) = n_guessed / num_del;
            if n_guessed > 0
                readAcc_Ind(k) = sum_read_err / n_guessed;
                hasGuessed(k) = true;
            end
        end
        
        meanAcc = mean(allAcc_Ind, 'omitnan');
        if any(hasGuessed)
            meanRead = mean(readAcc_Ind(hasGuessed));
            guessedOnceNet(netw) = 1;
        else
            meanRead = 0;
        end
        
        accS(netw,:) = [meanAcc meanAcc meanAcc 0];
        readS(netw,:) = [meanRead meanRead meanRead 0];
        
        fprintf(' Done.\n');
    end 

    % Aggregate
    idx = floor(num_del/step) + 1;
    
    final_R_scores = mean(tempRel_R, 1);
    final_D_scores = mean(tempRel_D, 1);
    
    GlobalRel_Error(idx, :) = final_R_scores;
    GlobalRel_Discrim(idx, :) = final_D_scores;
    
    [~, idx_R] = sort(final_R_scores, 'descend');
    [~, idx_D] = sort(final_D_scores, 'descend');
    
    Rank_Error(idx, :) = idx_R; % relevance error
    Rank_Discrim(idx, :) = idx_D; % relevance discrimination index
    
    % Final stats for corruption level (row)
    allSQIStats(idx,:) = [min(sqiS(:,1)), mean(sqiS(:,1)), max(sqiS(:,1)), std(sqiS(:,1))];
    allSQIStats_corr(idx,:) = [min(sqiS_c(:,1)), mean(sqiS_c(:,1)), max(sqiS_c(:,1)), std(sqiS_c(:,1))];
    allFMIStats(idx,:) = [min(fmiS(:,1)), mean(fmiS(:,1)), max(fmiS(:,1)), std(fmiS(:,1))];
    allFMIStats_corr(idx,:) = [min(fmiS_c(:,1)), mean(fmiS_c(:,1)), max(fmiS_c(:,1)), std(fmiS_c(:,1))];
    allNMIStats(idx,:) = [min(nmiS(:,1)), mean(nmiS(:,1)), max(nmiS(:,1)), std(nmiS(:,1))];
    allNMIStats_corr(idx,:) = [min(nmiS_c(:,1)), mean(nmiS_c(:,1)), max(nmiS_c(:,1)), std(nmiS_c(:,1))];
    allSimStats(idx,:) = [min(simS(:,1)), mean(simS(:,1)), max(simS(:,1)), std(simS(:,1))];
    allSimStats_corr(idx,:) = [min(simS_c(:,1)), mean(simS_c(:,1)), max(simS_c(:,1)), std(simS_c(:,1))];
    finalAccStats(idx,:) = [min(accS(:,1)), mean(accS(:,1)), max(accS(:,1)), std(accS(:,1))];
    finalReadAccStats(idx,:) = [min(readS(:,1)), mean(readS(:,1)), max(readS(:,1)), std(readS(:,1))];
    
    finalGuessedOnce(idx) = mean(guessedOnceNet);
end

fprintf('\nExecution Complete.\n');