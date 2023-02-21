# ImageGrains  <img src="" width="250" title="logo" alt="logo" align="right" vspace = "50">
   
A software library for segmenting and measuring of sedimentary particles in images. The segmentation is done with the *Cellpose* algorithm  (https://github.com/mouseland/cellpose), designed for cell and nucleus segmentation in biomedical images. Its segmentation capability is transferred to geoscientific applications throurgh re-training of the algorithm with images of sediment particles (from our data (**link to repo**) or custom; **link to preprint**).

## Citation  
   
If you use this software in your research, please cite the following works:  
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

First install cellpose, its GUI and dependancies (which includes [pytorch](https://pytorch.org/), [pyqtgrapgh](https://www.pyqtgraph.org/), [PyQt5](https://www.riverbankcomputing.com/static/Docs/PyQt5/), [numpy](https://numpy.org/), [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html), [scipy](https://scipy.org/), [natsort](https://natsort.readthedocs.io/en/master/)) 
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






