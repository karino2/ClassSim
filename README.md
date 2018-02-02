# ClassSim
Repository for ClassSim experiments.

Here is the breaf step of reproduction.


# Data Setup

## Step 1. Scraping image url list


If you just use the same url list as ours under urls folder, you can skip this step.
If you want to add other classes or refresh url list, you need this step.

For scraping urlists, we use selenium with headless chrome.
Use Dockerfile.selenium for scraping image and run docker with "--cap-add=SYS_ADMIN" option for headless chrome.

The notebook is datascraping.ipynb.


## Step 2. Retrieve data based on url list

Below here, you can use either Dockerfile.cpu (for none-GPU instance) or Dockerfile (for GPU instance) respectively. For GPU instance, you need to use nvidia-docker instead of docker.

The notebook is data_download.ipynb.


# Compute ClassSim

Once data setup is done, you can compute ClassSim by following step.


## Step 3. Train first level classifier

train.ipynb


## Step 4. Compute classifier similarity

classifier_similarity.ipynb


### Step 4.5 (optional) Compute PD distance

If you want to compare our result to previous study, you can reproduce the result by GMM_similarity.ipynb.


# Two Level Model



## Step 5. Train second level classifier

train_second.ipynb


## Step 6. Evaluate two level model

evaluate_classifier.ipynb


