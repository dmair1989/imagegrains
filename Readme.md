# ImageGrains  <img src="https://github.com/dmair1989/ImageGrains/blob/main/illustrations/logo_2.png?raw=true" width="100" title="logo" alt="logo" align="left">

A software library for segmenting and measuring of sedimentary particles in images. The segmentation is done with the *Cellpose* algorithm  (<https://github.com/mouseland/cellpose>), designed for cell and nucleus segmentation in biomedical images. Its segmentation capability is transferred to geoscientific applications throurgh re-training of the model with images of sediment particles. Pretrained models from our data (**link to repo**) are available or custom models can be trained (**link to preprint**).

## Citation  

If you use software and/or data from here in your research, please cite the following works:  

- **bibtex here**  

- Stringer, C.A., Pachitariu, M., (2021). Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106. <https://doi.org/10.1038/s41592-020-01018-x>.

If you use the human-in-the-loop approach for training custom models (see below), please also cite:  

- Pachitariu, M. & Stringer, C. (2022): Cellpose 2.0: how to train your own model. Nat Methods 19, 1634–1641. <https://doi.org/10.1038/s41592-022-01663-4>.

If you use ImageGrains to calculate percentile uncertainties please also cite:

- Mair, D., Henrique, A., Prado, D., Garefalakis, P., Lechmann, A., Whittaker, A., and Schlunegger, F. (2022): Grain size of fluvial gravel bars from close-range UAV imagery-uncertainty in segmentation-based data, Earth Surf. Dyn., 10,953-973. <https://doi.org/10.5194/esurf-10-953-2022>.

## Installation

### Local installation  

The easiest way to install the software is by using the conda package manager. If you do not have conda installed, please follow the instructions on the [conda website](https://docs.conda.io/en/latest/miniconda.html).  

To install the software, open an anaconda prompt / command prompt, then create a new environment with:

```text
conda create --name imagegrains -c conda-forge python=3.8 imagecodecs
```

and activate it with:

```text
conda activate imagegrains
```

Then install the package using:

```text
pip install imagegrains
```

If you want access to the cellpose GUI for retraining use:

```text
pip install "imagegrains[gui]"
```

By default, 

```text
python -m pip install cellpose[gui]
```

By default, cellpose will run on the CPU. To use a GPU version, you will have to make sure you have a GPU compatible PyTorch version. For this:

1. Uninstall the PyTorch version that gets installed by default with Cellpose:

        pip uninstall torch

2. Make sure your have up-to-date drivers for your NVIDIA card installed.

3. Re-install a GPU version of PyTorch via conda using a command that you can find [here](https://pytorch.org/get-started/locally/) (this takes care of the cuda toolkit, cudnn etc. so **no need to install manually anything more than the driver**). The command will look like this:

        conda install pytorch torchvision cudatoolkit=11.3 -c pytorch


Details and more installation options of cellpose (including GPU versions for Windows and Linux) are found [here](https://github.com/mouseland/cellpose#installation).

## How does it work?

ImageGRains is organised in 3 main modules for *Segmentation*, *Grain size measurements* and *Grain size distribution (GSD) analysis* (see below). Currently, the most convenient way to use its functionality is, by downloading the code and running the jupyter notebooks in ```/notebooks``` in their order (A command-line executable version and complete online notebooks will follow soon). Of course, functions can be combined in custom scripts or notebooks.

### Workflow  

<img src="https://github.com/dmair1989/ImageGrains/blob/main/illustrations/workflow.png?raw=true" width="550" title="wf" alt="wf" align="center">  
The main concept of ImageGrains is to first segment grains in images, then to measure and scale them with the respective image resolution before finally estimating the uncertainty on an image base. The whole workflow is designed to use individual images or a set of images in specific folder. During the processing steps, all intermediate outputs can be stored.

### Segmentation of own images

If you want to segment own images with pre-trained models, simply use the corresponding jupyter notebook ```notebooks/2_image_segmentation.ipynb```. To do so locally, open the console and activate the environment (```conda activate imagegrains```) and start your jupyter instance (e.g., via```jupyter lab```). Then, open the notebook and follow the instructions. You can use any model provied in ```/models``` or train a custom model (see below).

### Grain size measurements

To measure grain sizes, use the jupyter notebook ```notebooks/3_grain_sizes.ipynb```. It will load the segmented images and calculate the grain size for each grain on an image-by-image basis. Several options for outline fitting are available. The grain size is then scaled with the image resolution and stored in an output file. It is also possible to export individual grain outlines for further analysis.

### Grain size distribution (GSD) and uncertainty

To analyze the GSD, use the jupyter notebook ```notebooks/4_gsd_analysis.ipynb```. It will load the grain size measurements and calculate the GSD. Several for the uncertainty estimation are available. The uncertainty by default is calculated for each perecentile as 95% confidence interval. The GSD is then stored in an output file.

### Training of custom models

If you want to train your own models, you can use the jupyter notebook ```notebooks/1_model_training.ipynb```, you can use the cellpose GUI (<https://www.cellpose.org/>; open it with ```python -m cellpose```) or train via console (<https://cellpose.readthedocs.io/en/latest/train.html>) with the full funcitionality of cellpose. To train custom models, you will first need manually annotated ground truth data ("labels"). This can be done either with the cellpose GUI or with any dedicated annotation tool. We used the labkit plugin for ImageJ (<https://imagej.net/Labkit>). Please note, that each grain has to have a unique class value.

***TO DO:***  

***-code:***
***-- make it run in google colab***  

***-data:***
***--improve set with 'human-in-the-loop' approach***

-------
Planned future modules:  
--> classifier model(s) for masks
--> shape analysis (angularity, )
