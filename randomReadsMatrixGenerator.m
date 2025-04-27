function [reads,geno] = randomReadsMatrixGenerator(m, g, n,genoset,readsRange)
% Generate a random population from a genoset, save genotype of each indiv.
% m= population size, g= num genes, n= num alleles per gene
% genoset=[numGenotypes,2*g] matrix, each row is a genotype (MUST BE
% COHERENT WITH g,n
% readsRange= range of generated reads
% OUTPUT: reads=[g*n,m] matrix, geno(p)= index of genotype of indiv. p
reads = zeros(g*n, m);
numGenotypes= size(genoset,1);
geno = zeros(m, 1); % genotype of each indiv.
for k = 1:m % for each individual
    idx = randi([1,numGenotypes]); % Select a random genotype from the set
    geno(k,:)= idx; % store genotype for each individual in the testset
        for allele=genoset(idx,:) % for each allele of this genotype...
            reads(allele, k) = randi(readsRange); % ...assign random value between 20 and 200
        end
end

end