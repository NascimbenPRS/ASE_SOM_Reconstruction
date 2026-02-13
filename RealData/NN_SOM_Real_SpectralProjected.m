%{
Francesco Nascimben
PhD student in AI&CS, University of Udine, Italy
VERSION 4: Spectral Embedding + SOM + Nyström Projection (Script Version)
%}

% --- 1. LOAD & SETUP ---
clear; clc; close all;

%dataFileName = 'FullData_1148.mat'; % extended dataset
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
g = length(AlleleCounts); % number of genes
max_n = max(AlleleCounts); 
GeneOffsets = [0; cumsum(AlleleCounts)]; 

% Map creation
totalAlleles = sum(AlleleCounts);
AlleleToGeneMap = zeros(totalAlleles, 1);
for i = 1:g
    AlleleToGeneMap( (GeneOffsets(i)+1):GeneOffsets(i+1) ) = i;
end

% Filtering away indiv. with < 700 expressed alleles
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
numNeurons = dimension1 * dimension2; % K = 16
numNetworks = 15; 

% Stats Storage
step = 100;
maxDel = minNumAllele;
num_steps = floor(maxDel/step) + 1;

[allSQIStats, allSQIStats_corr, allFMIStats, allFMIStats_corr, ...
 allNMIStats, allNMIStats_corr, allSimStats, allSimStats_corr, ...
 finalAccStats, finalReadAccStats] = deal(zeros(num_steps, 4));
 
finalGuessedOnce = zeros(num_steps, 1);

% Store partition sizes for reference
partitionSize = zeros(numNeurons, 1);

% --- MAIN LOOP ---
for num_del = 0:step:maxDel 
    fprintf('\n=== Corruption Level: %d / %d ===\n', num_del, maxDel);
    
    [sqiS, sqiS_c, fmiS, fmiS_c, nmiS, nmiS_c, simS, simS_c, accS, readS] = deal(zeros(numNetworks, 4));
    guessedOnceNet = zeros(numNetworks,1); 

    for netw = 1:numNetworks 
        fprintf('  Net %d: SpecTrain...', netw);
        
        % --- A. SPECTRAL TRANSFORMATION (Training Data) ---
        % We capture the 'Model' (Vectors, Values, Degrees, BinaryRef)
        % to use for projection later.
        [SpectralData_Train, V_train, D_train, TrainBin, Deg_train] = ...
            computeSpectralMatrix(Reads, numNeurons);
        
        fprintf(' Train...');
        
        % --- B. SOM TRAINING ---
        net = selforgmap([dimension1 dimension2]); 
        net.trainParam.showWindow = false; 
        net = train(net, SpectralData_Train);
        output_train = sim(net, SpectralData_Train);   
        
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
        
        % --- C. STATS COLLECTION (Using RAW READS) ---
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

        % --- D. TEST & CORRUPTION ---
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
        
        % --- E. SPECTRAL PROJECTION (Corrupted Data) ---
        % Instead of re-computing eigenspace, we project corrupted data
        % into the existing Training Eigenspace (Nyström Extension).
        SpectralData_Corr = projectSpectralFeatures(Reads_Test_Corrupted, ...
                                TrainBin, V_train, D_train, Deg_train);
        
        % Clustering (SOM projection)
        output = sim(net, SpectralData_Train);
        output_Corrupted = sim(net, SpectralData_Corr);
        
        [~, labelsSOM] = max(output, [], 1);
        [~, labelsSOM_Corrupted] = max(output_Corrupted, [], 1);
        
        % Evaluation
        SOMvs2PP = zeros(numNeurons, max(TrueLabels));
        SOMvs2PP_Corr = zeros(numNeurons, max(TrueLabels));
        for k=1:m_test
           SOMvs2PP(labelsSOM(k), TrueLabels(k)) = SOMvs2PP(labelsSOM(k), TrueLabels(k)) + 1;
           SOMvs2PP_Corr(labelsSOM_Corrupted(k), TrueLabels(k)) = SOMvs2PP_Corr(labelsSOM_Corrupted(k), TrueLabels(k)) + 1;
        end
        
        % Metrics
        SQI = sqi(SOMvs2PP); SQI_C = sqi(SOMvs2PP_Corr);
        FMI = fmi(labelsSOM', TrueLabels); FMI_C = fmi(labelsSOM_Corrupted', TrueLabels);
        NMI = nmi(labelsSOM', TrueLabels); NMI_C = nmi(labelsSOM_Corrupted', TrueLabels);
        Sim = randindex(labelsSOM', labelsSOM_Corrupted');
        Sim_C = randindex(labelsSOM_Corrupted', TrueLabels);

        sqiS(netw,:) = [SQI SQI SQI 0]; sqiS_c(netw,:) = [SQI_C SQI_C SQI_C 0];
        fmiS(netw,:) = [FMI FMI FMI 0]; fmiS_c(netw,:) = [FMI_C FMI_C FMI_C 0];
        nmiS(netw,:) = [NMI NMI NMI 0]; nmiS_c(netw,:) = [NMI_C NMI_C NMI_C 0];
        simS(netw,:) = [Sim Sim Sim 0]; simS_c(netw,:) = [Sim_C Sim_C Sim_C 0];

        % --- F. RECONSTRUCTION (Unchanged Logic) ---
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
                allAcc_Ind(k) = 1; continue; 
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
        
        meanAcc = mean(allAcc_Ind);
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

    idx = floor(num_del/step) + 1;
    
    % Final stats (row = corruption level 0, 100, 200... 700)
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

% -----



% --- HELPER 1: Compute Spectral Features (Training) ---
function [specMatrix, V, D, binMatrix, degrees] = computeSpectralMatrix(ReadMatrix, K)
    % 1. Binarize
    binMatrix = double(ReadMatrix > 0);
    
    % 2. Jaccard Dist
    Dist = squareform(pdist(binMatrix', 'jaccard'));
    W = 1 - Dist;
    W(logical(eye(size(W)))) = 0;
    
    % 3. Norm Laplacian
    degrees = sum(W, 2);
    
    D_inv_sqrt = zeros(size(degrees));
    valid_idx = degrees > 0;
    D_inv_sqrt(valid_idx) = 1 ./ sqrt(degrees(valid_idx));
    
    D_mat = spdiags(D_inv_sqrt, 0, length(degrees), length(degrees));
    Norm_Affinity = D_mat * W * D_mat;
    
    % 4. Eigen Decomposition
    opts.issym = 1; opts.isreal = 1; opts.disp = 0;
    m = size(W,1);
    num_eigs = min(K, m);
    
    [V, D_diag] = eigs(Norm_Affinity, num_eigs, 'largestreal', opts);
    
    % Return eigenvalues as vector
    D = diag(D_diag);
    
    specMatrix = V';
    if num_eigs < K
        specMatrix = [specMatrix; zeros(K-num_eigs, m)];
    end
end




% --- HELPER 2: Project New Data into Existing Eigenspace (Nyström) ---
function specMatrix_Proj = projectSpectralFeatures(CorruptedReads, TrainBin, V, D, Deg_Train)
    % Nyström Extension for Spectral Clustering
    
    K = size(V, 2);
    [~, m_test] = size(CorruptedReads);
    
    % 1. Binarize Test Data
    TestBin = double(CorruptedReads > 0);
    
    % 2. Cross-Affinity Matrix (Test vs Train)
    % pdist2 computes dist between rows of X and rows of Y.
    % We want distance between TestIndivs (TestBin') and TrainIndivs (TrainBin')
    CrossDist = pdist2(TestBin', TrainBin', 'jaccard');
    W_cross = 1 - CrossDist; 
    
    % 3. Compute Degrees of Test Nodes (relative to Train)
    Deg_Test = sum(W_cross, 2);
    
    % 4. Normalize Association
    % Inverse sqrt degrees
    D_test_inv_sqrt = zeros(size(Deg_Test));
    valid_t = Deg_Test > 0;
    D_test_inv_sqrt(valid_t) = 1 ./ sqrt(Deg_Test(valid_t));
    
    D_train_inv_sqrt = zeros(size(Deg_Train));
    valid_tr = Deg_Train > 0;
    D_train_inv_sqrt(valid_tr) = 1 ./ sqrt(Deg_Train(valid_tr));
    
    % Vectorized Normalization
    W_norm = (W_cross .* D_test_inv_sqrt) .* (D_train_inv_sqrt');
    
    % 5. Projection
    D_inv = diag(1 ./ D);
    V_proj = W_norm * V * D_inv;
    
    % 6. Format for SOM (K x m_test)
    specMatrix_Proj = V_proj';
    
    % Handle potential NaNs
    specMatrix_Proj(isnan(specMatrix_Proj)) = 0;
    
    % Padding check
    if size(specMatrix_Proj, 1) < K
        specMatrix_Proj = [specMatrix_Proj; zeros(K - size(specMatrix_Proj,1), m_test)];
    end
end