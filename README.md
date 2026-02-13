# ASE_SOM_Reconstruction
A two-phase procedure for recovering missing ASE data from RNA-seq experiments.

Reference paper: *Self-Organizing Maps for Allele Specific
Expression Data Reconstruction and
Identification of Anomalous Genomic Regions*
[Roberto Pagliarini, Francesco Nascimben, Alberto Policriti]
University of Udine, Italy

# Background on ASE
Allele Specific Expression data quantifies expression variation between the two haplotypes of
diploid individual distinguished by heterozygous sites. Current methodologies of genome-wide
sequencing produce large amounts of missing data that may affect statistical inference and bias
the outcome of experiments. Machine learning tools could be employed to explore the data and to
estimate missing signatures. 

# Method description
We present a two-phase procedure based on Self-Organizing Maps
(SOMs), an unsupervised clustering technique, to recover missing allele specific expression data
from RNA-seq experiments. Specifically, a SOM trained on a complete population *P* is used to
assign a so-called corrupted individual *p* to its most fitting cluster *c*; then, a completion rule based
on allele frequencies within the subpopulation of *Pc* ⊆ *P* defined by *c* is employed to reconstruct
 *p*. To evaluate our approach, we first apply it to purely artificial datasets, in order to have full
control over all experimental conditions. After that, we consider a real population of Vitis vinifera,
which we also extend by applying computational framework to generate synthetic individuals from
allele expression data. We then introduce two local feature relevance indices in order to assess
the influence of specific alleles on the topological placement of corrupted individuals in the SOM
structure. Our results, showing promising accuracy in the prediction of missing alleles, suggest
that the developed approach could be very useful for recovering incomplete samples in a dataset
instead of discarding them, mainly in situations where experiments are challenging.

# Artificial data
NN_SOM_Artificial.m trains a number of SOM networks to cluster corrupted individuals by their available ASE data and reconstruct them.
Populations are generated from a limited number of signatures, which are used as the ground-truth labeling G.

# Real data
We apply our SOM-based procedure to a real
dataset derived from 98 cultivars representative of the variability present in Vitis vinifera, by employing read
counts of genes in chromosome 1 of leaves. In this case, since each individual has a unique signature, we
require a different ground-truth labeling G to evaluate the SOM clustering: we use the labels produced by a
2-phase clustering procedure, based on spectral clustering, which we developed in a previous work [Pagliarini, Nascimben, Policriti. 2026].
We extend such dataset using a synthetic generator of ASE individuals.
We also repeat the experiments by training the SOM on a spectral representation of individuals, observing improvements in allele
prediction accuracy.

Reference paper for details on the spectral representation of ASE data and the 2-phase clustering procedure:
[Pagliarini R, Francesco N, Policriti A.] *A two-phase clustering procedure based on allele specific expression.*
BMC Bioinformatics (2026). doi:Toappear.

The grapevine dataset was obtained from:
[Magris, Gabriele and Jurman, Irena and Fornasiero, Alice and Paparelli, Eleonora and Schwope, Rachel
and Marroni, Fabio and Di Gaspero, Gabriele and Morgante, Michele] *The genomes of 204 vitis vinifera
accessions reveal the origin of european wine grapes.* Nat Commun 12 (2021) 7240. doi:10.1038/s41467-021-27487-y.


The software code related to the ASE synthetic generator, used to extended the original grapevine population, is available at the following URL:
https://github.com/RobertoPagliarini/ASESyntheticGenerator.git

# How to
Run the main programs to repeat the experiments described in the reference paper.
All additional function files (e.g. FMI) are required for the computation of quality indexes, etc.
Relevant output variables/vectors are clearly specified at the end of each main program.



