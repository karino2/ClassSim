# ClassSim
Repository for ClassSim experiments.

Here is brief steps of reproduction.
For more detailed information, please check the comments of each notebook.



## Data Setup
Collect datasets for experiments.


### Step 1. Scrape image url list
If you use the url lists stored in `urls` folder, you can skip this step.
This is the same lists we used in the experiment in our paper.
If you want to add other classes or refresh the url list, you need this step.

For scraping url lists, we use selenium with headless chrome.
Use *Dockerfile.selenium* for building a docker image and run a docker container with "--cap-add=SYS_ADMIN" option for headless chrome.

The notebook for scraping is *datascraping.ipynb*.
The result of the url lists will be stored in `urls` folder.

Note that our result is already checked in to this repository.


### Step 2. Retrieve images based on url lists
Below here, you can use either *Dockerfile* (for GPU instance) or *Dockerfile.cpu* (for non-GPU instance) respectively.
For GPU instance, you need to use nvidia-docker instead of docker.
For CPU instance, due to we don't provide requirements, perhaps scripts do not work well.

The notebook for retrieving is *data_download.ipynb*.
This notebook also conducts filtering of corrupted images by checking whether the images can be loaded or not.
The images will be stored in `data` folder.



## Compute ClassSim
Once data setup is done, you can compute ClassSim by the following steps.


### Step 3. Train first level classifiers and multi-class classifier
The notebook for training first level one vs. rest classifiers is *train.ipynb*.
Trained classifiers will be stored in `trained_model` folder.

The notebook for training a multi-class classifier is *train_multiclass_classifier.ipynb*.
Trained classifiers will be stored in `trained_model` folder.


### Step 4. Compute similarities between classes (ClassSim)
The notebook is *classifier_similarity.ipynb*.
The result will be stored as "results/valid_sim_df.dat".

#### Step 4.5 (optional) Compute PD distances
If you want to compare our results to a previous study, you can reproduce the result by *GMM_similarity.ipynb*.
The result will be stored as "results/GMMDistances.dat"



## Two Level Model
This is the experiment of two level model that is a classification model based on two sets of one vs. rest classifiers.


### Step 5. Train second level classifiers
The notebook for training second level one vs. rest classifiers is *train_second.ipynb*.
Trained classifiers will be stored under "trained_model" folder.

This step requires "results/valid_sim_df.dat" created on Step 4.


### Step 6. Evaluate two level model
The notebook for evaluating the model is *evaluate_classifier.ipynb*.
The results will be stored as "results/1level_prediction.dat" and "results/2level_prediction.dat".

