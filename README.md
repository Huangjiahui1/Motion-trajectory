# Animal Behavior and Genetic Analysis Code Repository 
This repository contains the supplementary code for our manuscript titled "Genetic and Evolutionary Insights into Panic-related Escape Behaviors in Ovine Animals." The code is organized into four main sections:

## 1. Running and Jumping Trajectory Analysis and Phenotype Calculation 
This folder contains the code used for analyzing running and jumping trajectories derived from high-speed videos of locomotion tests. The data were processed using DeepLabCut (DLC) for trajectory tracking, and subsequent analysis was performed to calculate running speed and jumping height.  
﻿ 
## 2. Random Forest Modeling 
This section includes the code used for random forest modeling to identify significant associations between morphological features and locomotion performance.  
﻿ 
## 3. GWAS and Haplotypes Analysis 
This folder contains the code used for Genome-Wide Association Study (GWAS) and haplotype analysis.  
### SNP similarity calculation.py
For the gene GRID2 identified in the GWAS analysis, we further investigated the association of its SNPs with jumping height. A total of 14,411 SNPs are located within GRID2 in the 227 individuals, of which 10,980 loci are identical homozygous SNPs across all the eight argali individuals. For each individual of Tibetan sheep and hybrid animals in the GWAS, we calculated similarity of the SNP genotypes in the 10,980 argali-associated homozygous SNPs against the argali genotypes. For example, scores of 1, 0.5 and 0 were given to the SNPs with the argali genotype, heterozygous genotype and non-argali genotype, respectively.
### haplotype analysis.py
We found a high linkage block covering the most significant one (Chr6:32,120,477) identified in the GWAS above and additional 122 SNPs. After dimensionality reduction via PCA and haplotype clustering with the k-means algorithm, we found 454 haplotypes in the block, which were distinguished into four distinct haplogroups (Fig. 4e,f). Haplogroup 4 (HAP4) contained the 16 haplotypes from the 8 argali individuals.
## 4. Mouse Behavior Experiments 
This section includes the code used for analyzing behavioral data from gene-edited mice experiments. 
