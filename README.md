# Representation Space Learning Dynamics of Vision Transformers with SimCLR
## Abstract
Vision transformers with self-supervised training such as SimCLR have been successfully applied in a variety of computer vision research areas; thus, analyzing the dynamics of their training is crucial. While existing studies have approached this topic from different perspectives, the movement of data points in the representation space during training has not yet been fully examined. To address this research gap, we convert data point movements into edge flows on graphs, apply Hodge decomposition, quantitatively find characteristic flow patterns, and interpret the observed patterns through streamline visualizations. 
We will observe flows between clusters as they form in the representation space. These flows do not consistently move in the same direction but alternate among various directional patterns. This suggests that the clustering patterns in the representation space emerge from iterative flows in multiple directions rather than a steady motion toward a fixed separation direction. This study provides potential mechanistic insight into the formation of patterns within the representation space, thereby advancing the current understanding of vision transformer learning dynamics with SimCLR.
## Main Codes
### [main.ipynb](codes/main.ipynb)
Hodge decomposition, potential analysis codes.

### [visualization.ipynb](codes/visualization.ipynb)
Streamline visualizations.

### [SimCLR.py](codes/SimCLR.py)
SimCLR implementation for our paper.

### [SimSiam.py](codes/SimSiam.py)
SimSiam implementation for our paper.

## Supplementary Information
### Data Description
[**STL-10 dataset**](http://cs.stanford.edu/~acoates/stl10)

Adam Coates, Honglak Lee, Andrew Y. Ng An Analysis of Single Layer Networks in Unsupervised Feature Learning AISTATS, 2011. 

### Model Architecture and Learning Objectives
4-layer vision transformer with 256-dim embedding. 

Learning objectives are [**SimCLR**](https://arxiv.org/abs/2002.05709) and [**SimSiam**](https://arxiv.org/abs/2011.10566)

## Citation
Paper for this repository is currently under review. Meanwhile, please cite this as follows.

```bibtex
@online{oda2026patho,
  author = {Oda, H. and Komura, D. and Ishikawa, S.},
  title = {Representation Space Learning Dynamics of Vision Transformers with SimCLR},
  url = {https://github.com/TopologicalBird/ViT_dynamics_private},
  year = {2026}
}
``` 
