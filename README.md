## Overview
This is the github repo for <br>
Lee, H., Achananuparp, P., Liu, Y., Lim, E. P., & Varshney, L. R. (2019). Estimating Glycemic Impact of Cooking Recipes via Online Crowdsourcing and Machine Learning. Short paper accepted at 9th International Digital Public Health Conference (DPH’ 19), ACM [arxiv](https://arxiv.org/pdf/1909.07881.pdf)

## Slides 
My slides and scripts presented in DPH'19 could be found at [here](https://drive.google.com/open?id=1bln5W9KmlxFwrpA3KRlpU30n4yGTg44U)

## Dataset
We crawled the 55k recipes from http://allrecipes.com and had 1000 recipes labelled by the AMT workers.
However, we were not allowed to release the 55k dataset (i.e. The dic_20190819.pickle I used in most of my notebooks)
As a result, we released the [dic_20191203.pickle](data/Downloads.md) file instead, which should be enough for reproducing more of our experiments except training the embeddings.
It contains 990 recipes instead of 55k recipes.

## Recipe54k-trained embeddings
This work involves a lot of embeddings trained on Recipe54k. As a result, I share the [trained embeddings](data/Downloads.md)

## Jupyter notebooks
### RQ1: Conduct data preprocessing and answer the Research Question 1
### RQ2: Train a lot of models to answer the Research Question 2 and save the results to csv/ and pickle/
We release the original code (RQ2_original) and a less messy version (RQ2_reproducible) to make it easier to reproduce the experiments.
* RQ2_original: How we prepare the results on paper, it is not well-structured and requires dic_20190819.pickle
* RQ2_reproducible: We selectively reproduced the best models in our study and re-organized the notebooks. 
  * Best non-nutritional model: NB-BoW + LR
  * Best overall model: Pre-trained GloVe + Nutritional information + LGBM
  * Second best overall model: NB-BoW + Nutritional information + LR
  * Nutritional only model: Nutritional information + LR
