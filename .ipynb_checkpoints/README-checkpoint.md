## Overview
This is the githib repo for <br>
[Lee, H., Achananuparp, P., Liu, Y., Lim, E. P., & Varshney, L. R. (2019). Estimating Glycemic Impact of Cooking Recipes via Online Crowdsourcing and Machine Learning. Short paper accepted at 9th International Digital Public Health Conference (DPH’ 19), ACM](https://arxiv.org/pdf/1909.07881.pdf)

## Slides 
I am going to release the slides of this work

## Dataset
We crawled the 55k recipes from http://allrecipes.com and had 1000 recipes labelled by the AMT workers.
However, we are not allowed to release the 55k dataset (i.e. The dic_20190819.pickle I used in most of my notebooks)
As a result, I want to release the [dic_20191203.pickle](data/Downloads.md) file instead, which should be enough for reproducing more of my experiments.
It contains 1000 recipes instead of 55k recipes.

## Recipe54k-trained models
This works involves a lot of embeddings trained on Recipe54k. As a result, I share the [trained embeddings](data/Downloads.md)

## Jupyter notebooks
### RQ1: Research Question 1
### RQ2: Research Question 2 ---> train models and save the results to csv/ and pickle/
