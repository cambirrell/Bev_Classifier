## Model Overview
Inspired by mixture of experts

99 catagories are divided into 16 subcatagories. These subcatagories are recofded in the catagories.json file.  One model determines which subcatagory and each subcatagroy has its own model.

The ipynb was to prototype the model. Later I put it on the bev_classifier_script.py so I could run it remotely on a A100 Cluster. That script saved the models which I classified with bev_eval.py
