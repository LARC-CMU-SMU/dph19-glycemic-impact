## Overview
This is the github repo for <br>
Lee, H., Achananuparp, P., Liu, Y., Lim, E. P., & Varshney, L. R. (2019). Estimating Glycemic Impact of Cooking Recipes via Online Crowdsourcing and Machine Learning. Short paper accepted at 9th International Digital Public Health Conference (DPH’ 19), ACM [arxiv](https://arxiv.org/pdf/1909.07881.pdf)

## Slides 
Our slides and scripts presented in DPH'19 could be found at [here](https://drive.google.com/open?id=1bln5W9KmlxFwrpA3KRlpU30n4yGTg44U)

## Project Structure
By default, the project assumes the following directory structure:

project <br>
└───data  <br>
|   │    dic_20191203.pickle: the textual description of recipe, AMT annotation, and nutritional properties <br>
 | │ Recipe54k-trained embeddings  <br>
 | │ combined.csv  <br>
└───RQ1 <br>
│   │   how we conduct data preprocessing and crowdsourcing to answer the Research Question 1  <br>
└───RQ2 <br>
│   │   Train a lot of models to answer the Research Question 2 and save the results to csv/ and pickle/ <br>
│   │   We release the original code (RQ2_original) and a less messy version (RQ2_reproducible) to make it easier to reproduce the experiments.  <br>
│   └───RQ2_original  <br>
│   │   │   How we prepare the results on paper, it is not well-structured and requires dic_20190819.pickle  <br>
│   └───RQ2_reproducible  <br>
│   │   │   We selectively reproduce the best models in our study and re-organize the notebooks. It requires dic_20191203.pickle  <br>
│   │   │   │   Best non-nutritional model: NB-BoW + LR  <br>
│   │   │   │   Best overall model: Pre-trained GloVe + Nutritional information + LGBM  <br>
│   │   │   │   Second best overall model: NB-BoW + Nutritional information + LR  <br>
│   │   │   │   Nutritional only model: Nutritional information + LR <br>
└───csv  <br>
└───pickle
└───reports
│   └───figures
All CSV data files should be put in the data folder. All notebooks should be put in the notebooks folder. Any generated reports and figures will be put in the reports folder.


## Dataset
We crawled the 55k recipes from http://allrecipes.com and had 1000 recipes labelled by the AMT workers.
However, we were not allowed to release the 55k dataset (i.e. The dic_20190819.pickle we used in most of the notebooks)
As a result, we released the [dic_20191203.pickle](data/Downloads.md) file instead, which should be enough for reproducing more of our experiments except training the embeddings.
It contains 990 recipes instead of 55k recipes.

## Recipe54k-trained embeddings
This work involves a lot of embeddings trained on Recipe54k. As a result, we share the [trained embeddings](data/Downloads.md)

## Jupyter notebooks
RQ1: Conduct data preprocessing and answer the Research Question 1 <br>
RQ2: Train a lot of models to answer the Research Question 2 and save the results to csv/ and pickle/ <br>
We release the original code (RQ2_original) and a less messy version (RQ2_reproducible) to make it easier to reproduce the experiments.
* RQ2_original: How we prepare the results on paper, it is not well-structured and requires dic_20190819.pickle
* RQ2_reproducible: We selectively reproduce the best models in our study and re-organize the notebooks. It requires dic_20191203.pickle
  * Best non-nutritional model: NB-BoW + LR
  * Best overall model: Pre-trained GloVe + Nutritional information + LGBM
  * Second best overall model: NB-BoW + Nutritional information + LR
  * Nutritional only model: Nutritional information + LR

## Python Version
We use python version 3.6.6 in this work
