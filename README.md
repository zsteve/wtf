# wtf: Wasserstein Tensor Factorisation

Non-negative matrix and tensor factorisations with a smoothed Wasserstein loss 

![pooh bear meme](images/wtf.jpg)

This repository contains a basic implementation of the method described in the article "Non-negative matrix and tensor factorisations with a smoothed Wasserstein loss".

## Requirements
 - PyTorch
 - [Tensorly](http://tensorly.org)
 - [PythonOT](https://pythonot.github.io/)
 - CUDA-compatible GPU (e.g. use [Colab](http://colab.research.google.com/)) for efficient autodiff

## Instructions
 - Clone this repo: `git clone https://github.com/zsteve/wtf.git`
 - Import the `wtf` module using
   ```python
   import sys
   sys.path.insert(0, "/content/wtf/src")
   import wtf
   ```
 - ???
 - Profit

## Example

A notebook for Figures 3, 4, 5 is located in the `examples/` directory.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zsteve/wtf/blob/main/examples/example.ipynb)

## Citing 
 - If you find this work relevant to your research project, please cite the [preprint](https://arxiv.org/abs/2104.01708)

```
Zhang, S. Non-negative matrix and tensor factorisations with a smoothed Wasserstein loss, arXiv preprint arXiv:2104.01708, 2021
```
