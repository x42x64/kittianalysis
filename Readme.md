# Analysis tools and helpers for the KITTI 3D Object Detection Dataset

This is a small collection of scripts (python and shell) to analyse the KITTI 3D Object Detection Dataset.
* kitti_reader.py: Python3 module to read the dataset into a Python dictionary and enrich the data with information from the KITTI raw dataset
* plot_helpers.py: Python3 module with helper functions to plot data specifically for this dataset
* get_kitti_data_corpus.sh: Bash script to download all required KITTI data for this analysis
* report.ipynb: Jupyter notebook with an initial analysis of the dataset

## Prerequisites & install
In order to install the prerequisites for this analysis, run
```
pip3 install -r requirements.txt
```

In order to run the jupyter notebook, additionally, jupyter needs to be installed:
```
pip3 install jupyter
```

To enable the Google Maps view, run following commands:
```
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension enable --py --sys-prefix gmaps
```

Then 2 things need to be obtained:
* The base URL for the KITTI dataset: replace <YOURBASEURL_HERE> by the proper URL in `get_kitti_data_corpus.sh`
* A Google Maps API key: replace <YOUR_GMAPS_API_KEY_HERE> by the proper API Key in `report.ipynb`
