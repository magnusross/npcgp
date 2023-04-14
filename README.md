# npcgp



Implementation of the GP with nonparametric covariances via multidimensional convolutions (NP-CGP), from our paper "Nonparametric Gaussian Process Covariances via Multidimensional Convolutions" in AISTATS 2023.

The code was jointly written by [magnusross](https://github.com/magnusross) and [tomcdonald](https://github.com/tomcdonald).


To install, clone the repo, and then run:

```
conda create --name npcgp_env python=3.9
conda activate npcgp_env
pip install -e .
```
This will install the CPU version of PyTorch, for the GPU version, follow the instructions [here](https://pytorch.org/get-started/locally/) to install the correct version for your GPU. 

You can run the model on the UCI data with `python uci.py`, and run `python uci.py -h` for help with possible arguments. 

![Diagram of the NP-CGP](model_diagram.png)

### Citation

```
@InProceedings{pmlr-v206-mcdonald23a,
  title = 	 {Nonparametric Gaussian Process Covariances via Multidimensional Convolutions},
  author =       {Mcdonald, Thomas M. and Ross, Magnus and Smith, Michael T. and \'Alvarez, Mauricio A.},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {8279--8293},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v206/mcdonald23a.html},
}
```
