# Animal Behavior and Genetic Analysis Code Repository 
This repository contains the supplementary code for our manuscript titled "Genetic and Evolutionary Insights into Panic-related Escape Behaviors in Ovine Animals." The code is organized into four main sections:

## 1. Running and Jumping Trajectory Analysis and Phenotype Calculation 
This folder contains the code used for analyzing running and jumping trajectories derived from high-speed videos of locomotion tests. The data were processed using DeepLabCut (DLC) for trajectory tracking, and subsequent analysis was performed to calculate running speed and jumping height.  
```
python running_fix.py -h

usage: running_fix.py [-h] [--path PATH] [--scale SCALE]

Process some integers.

options:

-h, --help    show this help message and exit
--path    Path to the directory containing the pose data files
--scale    Scale of the video where 1 meter equals how many pixels
--out    Path of result output
```
### for example
```
python running_fix.py --path SampleData/Running_Fix --scale 150
```
## 2. Random Forest Modeling 
This section includes the code used for random forest modeling to identify significant associations between morphological features and locomotion performance.  
﻿ 
## 3. GWAS and Haplotypes Analysis 
This folder contains the code used for Genome-Wide Association Study (GWAS) manhattan map and haplotype analysis.  
## 4. Mouse Behavior Experiments 
This section includes the code used for analyzing behavioral data from gene-edited mice experiments. 
