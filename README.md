<p align="center">
  <img src="docs/edgeshaper_logo.svg" alt="EdgeSHAPer logo" width=30%>
</p>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) [![DOI](https://zenodo.org/badge/429822987.svg)](https://zenodo.org/badge/latestdoi/429822987)

# EdgeSHAPer: Bond-Centric Shapley Value-Based Explanation Method for Graph Neural Networks

This is the official implementation for [**EdgeSHAPer: Bond-Centric Shapley Value-Based Explanation Method for Graph Neural Networks**](https://www.cell.com/iscience/fulltext/S2589-0042(22)01315-3).

The methodology relies on Shapley values approximations to determine edge importance for GNN prediction. It finds its application in the context of medicinal chemistry, but being general-purpose it can be applied to many graph classification GNN-based tasks which require explainability.

## ⚙️ Prerequisites 
EdgeSHAPer relies on PyTorch. It was tested with Python 3.8, PyTorch 1.10, CUDA 10.2 and PyTorch Geometric 2.0. We believe different versions are fine as well, but we cannot fully guarantee it.

We suggest creating/using a Conda environment. If needed, we also provide a full conda config .yml file that can be used to create a brand new environment with the needed modules.

For visualizations, [this](https://github.com/c-feldmann/rdkit_heatmaps) additional module is needed.

The file ```edgeshaper.py``` contains the method source code.
For reproducibility of the experiments present in the [paper](https://www.cell.com/iscience/fulltext/S2589-0042(22)01315-3), see the section [Reproducing experiments from the paper](#-reproducing-experiments-from-the-paper).

## 🤖 EdgeSHAPer usage
If you want to use the EdgeSHAPer as a standalone tool to explain your own model, simply import the file ```edgeshaper.py``` and use the function ``edgeshaper()``:

```python
from edgeshaper import edgeshaper

# Define your model and data
model = YOUR_GNN_MODEL
edge_index = YOUR_GRAPH_EDGE_INDEX
x = GRAPH_NODE_FEATURES
device = "cuda"  #"cpu" or "cuda:x", if you have multiple GPUs 
target_class = TARGET_CLASS  # class label for which to perform explanations

# Run EdgeSHAPer to obtain edge explanations
edge_explanations = edgeshaper(
  model,
  x,
  edge_index,
  M=100,
  target_class=target_class,
  device=device
)
```

The code above shows a basic usage of EdgeSHAPer to obtain explanations in terms of Shapley values for the edges of the graph under study. ```M``` is the number of Monte Carlo sampling steps to perform to obtain an approximation of the Shapley values. The ```edgeshaper``` function returns a Python list containing Shapley values for edges, in the same order as in the provided ```edge_index```. Details on additional parameters can be found in the source file.


### 💻 EdgeSHAPer as a class
For more advanced usage, metrics, and visualizations, we suggest using the ``Edgeshaper`` class:

```python
from edgeshaper import Edgeshaper

TOLERANCE = False  # or a float value, e.g., 1e-3
SEED = 42  # seed used for random number generators

edgeshaper_explainer = Edgeshaper(model, x, edge_index, device=device)

edge_explanations = edgeshaper_explainer.explain(
  M=100,
  target_class=TARGET_CLASS,
  P=None,
  deviation=TOLERANCE,
  log_odds=False,
  seed=SEED
)
```

Among the parameters, ``deviation`` is an optional float argument for early stopping of the Monte Carlo sampling, upon reaching the defined deviation of approximation. If set to ``False``, all the ``M`` sampling steps will be performed. If ``log_odds`` is set to ``True``, Shapley values will be computed considering log odds instead of output probabilities (default). If ``target_class`` is not specified, a regression model will be assumed. ``P`` is the probability of an edge to exist in random graphs underlying the methodology for Shapley value approximation as described in the paper; we suggest leaving the default value, unless you know what works best for your application.

After having generated the explainations, the ``Edgeshaper`` class can be used to obtain the mimimal informative sets and the relatede Infidelity and Fidelity scores:

```python
# Compute the Pertinent Positive Set and infidelity score
pert_positive_set, infidelity_score = edgeshaper_explainer.compute_pertinent_positive_set()

# Compute the Minimal Top-K Set and fidelity score
minimal_top_k_set, fidelity_score = edgeshaper_explainer.compute_minimal_top_k_set()
```

Finally, when using our dataset or other molecular graphs, explantions can be mapped to molecular strucutres and visualized. It is also possible to specify if generating images for the minimal informative sets. 

```python
# Visualize explanations mapped to molecular structures
edgeshaper_explainer.visualize_molecule_explanations(
  smiles,
  save_path=SAVE_PATH,
  pertinent_positive=True,   # Generate mapping for the Pertinent Positive Set
  minimal_top_k=True         # Generate mapping for the Highlight Minimal Top-K Set
)
```

An example of feature importance mapping looks like:

<p align="center">
  <img src="experiments\results\explanations\single\P14416_P42336\Target 1 vs Random\CC(C)C(=O)NC1CCc2ccc(CCN3CCN(c4nsc5ccccc45)CC3)cc21\EdgeSHAPer_MC_100_train_FULL_heatmap.png" alt="EdgeSHAPer explanation example" width="30%">
</p>

Red edges indicate positive contribution to class prediction (active compound, in our case), and blue edges indicated negative contriution, opposing the prediction. 



## 🚀 Parallelization 🔜
🚧 **Work in progress:** A more efficient, parallel implementation to explain multiple samples simultaneously will be released soon!

## 📃 Reproducing experiments from the paper
In the ```experiments``` folder it is possible to find and also reproduce the experiments (along with the visualizations) presents in the [paper](https://www.cell.com/iscience/fulltext/S2589-0042(22)01315-3).

## 📧 Contacs and more

For any clarification on how to use the tool, feel free to drop an [email](mailto:mastropietro@diag.uniroma1.it).
<!-- For more information, see our [STAR Protocols publication](https://doi.org/10.1016/j.xpro.2022.101887) with detailed explanations on how to use our tool. -->

The method works for graph classification tasks, but a node classification extension is planned in the future.
## 📖 Citations

If you use our work, please cite our papers 😊

* Mastropietro, A., Pasculli, G., Feldmann, C., Rodríguez-Pérez, R., & Bajorath, J. (2022). EdgeSHAPer: Bond-centric Shapley value-based explanation method for graph neural networks. Iscience, 25 (10), 105043, doi: https://doi.org/10.1016/j.isci.2022.105043

* Mastropietro, A., Pasculli, G., & Bajorath, J. (2022). Protocol to explain graph neural network predictions using an edge-centric Shapley value-based approach. STAR Protocols, 3(4), 101887, doi: https://doi.org/10.1016/j.xpro.2022.101887

For the work/results on protein-ligand interaction, check out and cite the Nature Machine Intelligence paper 🧠

* Mastropietro, A., Pasculli, G. & Bajorath, J. [Learning characteristics of graph neural networks predicting protein–ligand affinities](https://rdcu.be/dqZlS
). Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00756-9 

Special thanks to [Simone Fiacco](https://www.linkedin.com/in/simone-fiacco-27bb5a25a/) for creating the EdgeSHAPer logo.
