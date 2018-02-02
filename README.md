# ClassSim
Repository for ClassSim experiments.

Here is brief steps of reproduction.
For more detailed information, please check the comments of each notebook.


# Data Setup

## Step 1. Scraping image url list


If you just use the same url list as ours under urls folder, you can skip this step.
If you want to add other classes or refresh url list, you need this step.

For scraping urlists, we use selenium with headless chrome.
Use Dockerfile.selenium for scraping image and run docker with "--cap-add=SYS_ADMIN" option for headless chrome.

The notebook is datascraping.ipynb.
The result url list will be stored under "urls" folder.

Note that our result is already check-ed in to this repository.


## Step 2. Retrieve images based on url list

Below here, you can use either Dockerfile.cpu (for none-GPU instance) or Dockerfile (for GPU instance) respectively. For GPU instance, you need to use nvidia-docker instead of docker.

The notebook is data_download.ipynb.

The images will be stored under "data" folder.
This notebook also conduct corrupt data filtering by opening images.



# Compute ClassSim

Once data setup is done, you can compute ClassSim by following steps.


## Step 3. Train first level classifier

The notebook is train.ipynb.
Trained classifiers will be stored under "trained_model" folder.


## Step 4. Compute classifier similarity

The notebook is classifier_similarity.ipynb.
The result will be stored at "results/valid_sim_df.dat".


### Step 4.5 (optional) Compute PD distance

If you want to compare our result to previous study, you can reproduce the result by GMM_similarity.ipynb.

The result will be stored at "results/GMMDistances.dat"



# Two Level Model



## Step 5. Train second level classifier

The notebook is train_second.ipynb.
Trained classifiers will be stored under "trained_model" folder.

This step requires "results/valid_sim_df.dat" created  on step 4.


## Step 6. Evaluate two level model

The notebook is evaluate_classifier.ipynb.

The results will be stored at "results/1level_prediction.dat" and "results/2level_prediction.dat".

