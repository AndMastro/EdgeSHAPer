<p align="center">
  <img src="docs/edgeshaper_logo.svg" alt="EdgeSHAPer logo" width=30%>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) [![DOI](https://zenodo.org/badge/429822987.svg)](https://zenodo.org/badge/latestdoi/429822987)

# EdgeSHAPer: Bond-Centric Shapley Value-Based Explanation Method for Graph Neural Networks

This is the official implementation for [**EdgeSHAPer: Bond-Centric Shapley Value-Based Explanation Method for Graph Neural Networks**](https://www.cell.com/iscience/fulltext/S2589-0042(22)01315-3).

The methodology relies on Shapley values approximations to determine edge importance for GNN prediction. It finds its application in the context of medicinal chemistry, but begin general-purpose it can be applied to many graph classification GNN-based tasks which require explanability.

EdgeSHAPer relies on PyTorch. It was tested with Python 3.8, PyTorch 1.10, CUDA 10.2 and PyTorch Geometric 2.0. We believe different vesions are fine as well, but we cannot fully guarantee it.

We suggest to create/use a Conda environment. If needed, we also provide a full conda config .yml file that can be used to create a brand new environment with the needed modules. For visualizations, [this](https://github.com/c-feldmann/rdkit_heatmaps) additional module is needed.

The file ```edgeshaper.py``` contains the method source code.

In the ```experiments``` folder it is possible to find and also reproduce the experiments (along with the visualizations) done in the paper.

If you want to use the EdgeSHAPer as a standalone tool to explain your own model, simply import the file ```edgeshaper.py```.

```python
from edgeshaper import edgeshaper

model = YOUR_GNN_MODEL
edge_index = YOUR_GRAPH_EDGE_INDEX
x = GRAPH_NODES_FEATURES
device = "cuda" or "cpu"
target_class = TARGET_CLASS #class label for which to perform explanations

edges_explanations = edgeshaper(model, x, edge_index, M = 100, target_class = TARGET_CLASS, device = "cuda")
```

The code above shows a basic usage of EdgeSHAPer to obtain explanations in terms of Shapley values for the edges of the graph under study. ```M``` is the number of Monte Carlo sampling steps to perform to obtain an approximation of the Shapley values. In the source file details on additional parameters can be found. 

The method works for graph classification tasks, but a node classification extension is planned in the future.

For any clarification on how to use the tool, feel free to drop an [email](mailto:mastropietro@diag.uniroma1.it).

## Citation

If you use our work, please cite our papers 😊

* Mastropietro, A., Pasculli, G., Feldmann, C., Rodríguez-Pérez, R., & Bajorath, J. (2022). EdgeSHAPer: Bond-centric Shapley value-based explanation method for graph neural networks. Iscience, 25 (10), 105043, doi: https://doi.org/10.1016/j.isci.2022.105043

* Mastropietro, A., Pasculli, G., & Bajorath, J. (2022). Protocol to explain graph neural network predictions using an edge-centric Shapley value-based approach. STAR Protocols, 3(4), 101887, doi: https://doi.org/10.1016/j.xpro.2022.101887

For the work/results on protein-ligand interaction, check out and cite the Nature Machine Intelligence paper 🧠

* Mastropietro, A., Pasculli, G. & Bajorath, J. [Learning characteristics of graph neural networks predicting protein–ligand affinities](https://rdcu.be/dqZlS
). Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00756-9 

Special thanks to [Simone Fiacco](https://www.linkedin.com/in/simone-fiacco-27bb5a25a/) for creating the EdgeSHAPer logo.
