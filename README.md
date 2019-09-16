# Kleinlein, R. et al (2019) - Interspeech 2019

This repository contains instructions and data required to replicate the results of our paper 

- R. Kleinlein, C. Luna-Jiménez, J. M. Montero, Z. Callejas, F. Fernández-Martínez, "Predicting Group-Level Skin Attention to Short Movies from Audio-Based LSTM-Mixture of Experts Models". The 20th Annual Conference of the International Speech Communication Association - Interspeech 2019, Paper ID: 2799, Graz, Austria, Sep. 15-19 2019

which was selected for oral presentation at the conference within the special session [*Dynamics of Emotional Speech Exchanges in Multimodal Communication*](http://www.empathic-project.eu/index.php/ssinterspeech2019/).

## Sample script

We have developed a toy script that shows a case of use of our models applied to a randomly generated aural embedding sequence. In order to run it, type in your terminal:
```
	python eval_example.py [pretrained_model_path]
```

so, for instance, you could try the canonical model of the `neutral` MEX with
```
	python eval_example.py neutral/canonical.pth
```

## Pretrained models

The folders `2exps`, `genre`, `neutral-strict` and `neutral` contain pretrained models for each data partition considered in the paper. Please notice that the models uploaded are reduced versions, not trained over all the data available, so your results might differ a bit from our published results.

## Database

We provide the embeddings generated, as well as the Youtube URL addresses of each movie, with tags on attention, genre of the movies and the elements required to split the dataset by satisfaction measures.

You can download this database from [this MEGA folder](https://mega.nz/#F!fNIFwa7a!SIHjTzBjnmFGixJJXudfwg)