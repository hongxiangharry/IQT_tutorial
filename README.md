# IQT_tutorial
This is a repo for IQT tutorial.

## Environmental Set-up
On Miniconda: 
1. Create a virtual environment by `conda create -y -n iqt python=3.6.8`
2. Enter the environment by `source activate iqt`
3. Install required packages by:
```
pip install nibabel==2.1.0
conda install -y h5py=2.10.0 ipython=7.10.1 jupyter=1.0.0 scipy=1.3.2 scikit-image=0.16.2 scikit-learn=0.22 
pip install tensorflow==1.13.1 tensorflow-gpu==1.13.1 keras==2.3.1
conda install -y cudatoolkit=10.0.130 cudnn
```


## Python Package Dependency
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
