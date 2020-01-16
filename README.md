# IQT_tutorial
This code is a demo for [MedICSS project](https://medicss.cs.ucl.ac.uk/image-quality-transfer-in-mri-with-deep-neural-networks/). It is a patch-based 3D IQT implementation using two neural networks, i.e. [SR U-net](https://arxiv.org/pdf/1706.03142.pdf) and [Anisotropic U-net](https://arxiv.org/pdf/1909.06763.pdf). It is able to 4x-super resolve on cross-plane direction of MR images. We employ the [HCP dataset](http://www.humanconnectomeproject.org/) as training and testing data. The trained model and test sample data are available by request via emailing `harry.lin[AT]ucl.ac.uk`.

## Required configuration
* Download a [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version for your OS.
* Tensorflow 1.13 [requirements](https://www.tensorflow.org/install/gpu): CUDA 10.0, Nvidia GPU driver 418.x or higher, cuDNN (>=7.6). Select to download the suitable Nvidia GPU driver from [here](https://www.nvidia.com/download/index.aspx?lang=en-us).
* Download and install [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP3) to view nifty files.

## Installation

### Conda Environmental Set-up
On Miniconda: 
1. Create a virtual environment by `conda create -y -n iqt python=3.6.8`
2. Enter the environment by `source activate iqt`
3. Install required packages by:
```
pip install nibabel==2.1.0 # If any error, run `pip install nibabel==2.1.0 --user` instead.
conda install -y h5py=2.10.0 ipython=7.10.1 jupyter=1.0.0 scipy=1.3.2 scikit-image=0.16.2 scikit-learn=0.22 
pip install tensorflow==1.13.1 tensorflow-gpu==1.13.1 keras==2.3.1 # If any error, add `--user` to the end of this command line.
conda install -y cudatoolkit=10.0 cudnn pyyaml
```

### Python Package Dependency
------------------------------------------
Name                     |Version         
-------------------------|----------------
h5py                     | 2.10.0           
hdf5                     | 1.10.5           
Keras                    | 2.3.1          
matplotlib               | 3.1.1            
numpy                    | 1.17.3          
scikit-image             | 0.16.2           
scikit-learn             | 0.22             
scipy                    | 1.3.2            
tensorboard              | 1.13.1           
tensorflow               | 1.13.1           
tensorflow-gpu           | 1.13.1          
nibabel                  | 2.1.0          
------------------------------------------

### Get started (On Jupyter notebook)
1. In a linux terminal, run `git clone https://github.com/hongxiangharry/IQT_tutorial.git` to download source codes to your workspace.
2. Make sure you have run `source activate iqt`.
3. Under the workspace, launch Jupyter Notebook by `jupyter notebook`, then open `IQT.ipynb`.
4. Follow the further instruction in the notebook.


## Citation
Please cite the following paper if this software is useful for you.
```
@inproceedings{lin2019deep,
  title={Deep Learning for Low-Field to High-Field MR: Image Quality Transfer with Probabilistic Decimation Simulator},
  author={Lin, Hongxiang and Figini, Matteo and Tanno, Ryutaro and Blumberg, Stefano B and Kaden, Enrico and Ogbole, Godwin and Brown, Biobele J and D’Arco, Felice and Carmichael, David W and Lagunju, Ikeoluwa and others},
  booktitle={International Workshop on Machine Learning for Medical Image Reconstruction},
  pages={58--70},
  year={2019},
  organization={Springer}
}
```
