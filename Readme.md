# ImageGrains  
   
A software library for segmenting and measuring of sedimentary particles in images. The segmentation is done with the *cellpose* algorithm  (https://github.com/mouseland/cellpose), designed cell and nucleus segmentation in biomedical images. Its segmentation capability is transferred to geoscientific applications throurgh re-training of the algorithm with images of sediment particles (from our data (**link to repo**) or custom; **link to preprint**).

## Citation  
   
If you use this software in your research, please cite the following works:  
- **bibtex here**  

- Stringer, C.A., Pachitariu, M., 2020. Cellpose: a generalist algorithm for cellular segmentation. bioRxiv. https://doi.org/10.1101/2020.01.28.925462 

If you use the human-in-the-loop approach for training custom models (see below), please also cite:  
- Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. Nature methods.

If you use ImageGrains to calculate percentile uncertainties please also cite:
- Mair, D., Henrique, A., Prado, D., Garefalakis, P., Lechmann, A., Whittaker, A., and Schlunegger, F.: Grain size of fluvial gravel bars from close-range UAV imagery-uncertainty in segmentation-based data, Earth Surf. Dyn. Discuss., 1–33, https://doi.org/10.5194/esurf-2022-19, 2022.

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

First install cellpose and its GUI 
```
python -m pip install cellpose[gui]
```

Then, install the additional dependancies:  

- jupyter
- matplotlib
- scikit-image
- pandas






