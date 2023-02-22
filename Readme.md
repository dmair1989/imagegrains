# ImageGrains  <img src="https://github.com/dmair1989/ImageGrains/blob/main/illustrations/logo_2.png?raw=true" width="100" title="logo" alt="logo" align="left">
   
    
    
       
A software library for segmenting and measuring of sedimentary particles in images. The segmentation is done with the *Cellpose* algorithm  (https://github.com/mouseland/cellpose), designed for cell and nucleus segmentation in biomedical images. Its segmentation capability is transferred to geoscientific applications throurgh re-training of the model with images of sediment particles. Pretrained models from our data (**link to repo**) are available or custom models can be trained (**link to preprint**).

## Citation  
   
If you use software and/or data from here in your research, please cite the following works:  
- **bibtex here**  

- Stringer, C.A., Pachitariu, M., (2021). Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106. https://doi.org/10.1038/s41592-020-01018-x.

If you use the human-in-the-loop approach for training custom models (see below), please also cite:  
- Pachitariu, M. & Stringer, C. (2022): Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641. https://doi.org/10.1038/s41592-022-01663-4.

If you use ImageGrains to calculate percentile uncertainties please also cite:
- Mair, D., Henrique, A., Prado, D., Garefalakis, P., Lechmann, A., Whittaker, A., and Schlunegger, F. (2022): Grain size of fluvial gravel bars from close-range UAV imagery-uncertainty in segmentation-based data, Earth Surf. Dyn., 10,953-973. https://doi.org/10.5194/esurf-10-953-2022.

## Installation 
    
### Local installation  

The easiest way to install the software is by using the conda package manager. If you do not have conda installed, please follow the instructions on the [conda website](https://docs.conda.io/en/latest/miniconda.html).  
   
To install the software, open an anaconda prompt / command prompt, then create a new environment with
```
conda create --name imagegrains python=3.8
```
and activate it with 
```
conda activate imagegrains
``` 

First, install cellpose, its GUI and dependancies (which includes [pytorch](https://pytorch.org/), [pyqtgrapgh](https://www.pyqtgraph.org/), [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/), [numpy](https://numpy.org/), [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html), [scipy](https://scipy.org/), [natsort](https://natsort.readthedocs.io/en/master/)) 
```
python -m pip install cellpose[gui]
```

Details and more installation options of cellpose (including GPU versions for Windows and Linux) are found [here](https://github.com/mouseland/cellpose#installation).

Then, install the additional dependancies:  

- jupyter (```pip install jupyterlab```)
- matplotlib (```pip install matplotlib```
- scikit-image (```python -m pip install -U scikit-image```)
- pandas (```pip install pandas```)  
- for tsne: scanpy (```pip install scanpy```)
   
## How does it work?
   
ImageGRains is organised in 3 main modules for *Segmentation*,*Grain size measurements* and *Grain size distribution (GSD) uncertainty estimation* (see below). Currently, the most convenient way to use its functionality is, by executing downloading the code and the running the jupyter notebooks in ```/notebooks``` in their order (A command-line executable version and complete online notebooks will follow soon). Of course, any functions can be combined in custom scripts or notebooks.
   
### Workflow  
<img src="https://github.com/dmair1989/ImageGrains/blob/main/illustrations/workflow.png?raw=true" width="900" title="wf" alt="wf" align="center">  
The main concept of ImageGrains is to first segment grains in images, then to measure and scale them with the respective image resolution before finally estimating the uncertainty on an image base. The whole workflow is designed to use individual images or a set of images in specific folder. During the processing steps, all intermediate outputs can be stored.

   
***TO DO: ***  
   
***-code:***   
***-- ship as package***  
***-- make it run in google colab***  
***-- create command-line executable***  
***-- do short manual***  
   
***-data:***   
***--change rescale range an evaluate effect***   
***--assess effect of image specific diam during eval()***   
***--improve set with 'human-in-the-loop' approach***

-------
Planned future modules:  
--> classifier model(s) for masks
--> shape analysis (angularity, )






