# OpenAPI

## Setup Experiment Environment

The source code requires Python2.7 and Pip.

To setup the experiment environment, use the following commands. 
 environment for the experiments and install dependencies.


```
cd openapi
sh bin/setup_env.sh
```

The script `setup_env.sh` will create virtual

## Explainers

Black-box explainers (have no access to the model parameters) are implemented in separate files.


`lime_linear_regression.py`: Linear Regression LIME explainer

`lime_ridge_regression.py`: Ridge Regression LIME explainer

`naive_est.py`: The naive explainer

`openapi_est.py`: OpenAPI explainer

`zeroth.py`: ZOO explainer

All explainers implement the same interface `decision_f`, which returns the interpretation for the prediction on the  input instance `x_tensor`

```
"""
x_tensor: the input instance
model: the target PLM
clss: the predicted class
clss_num: the number of total classes
"""
explainer.decision_f(x_tensor, model, clss, clss_num)
```

White-box explainers (have access to the model parameters) and the original LIME are implemented as functions of the models.


The functions related to the explainers 

`integrated_gradien`: Intergrated Gradient. Interval value of the integrated line is set to 5 as being used in [2]

`gradient_input`: Gradient * Input

`saliency_map`: Saliency Maps

`lime_interpret`: LIME implemention in the original paper. We use the same parameter setting as in [1]


## Run the Experiment

Scripts to run the experiments are stored in the folder `bin`

### Train the Target Models

Train the PLNN and LMT on FMNIST and MNIST

```
sh bin/train_mdls.sh
```

### Train the Target Models

Run the experiments to compare the effectiveness of the results of different interpretation methods

```
sh bin/run_effective.sh
```


### Run the experiments

Run the experiments to compare the consistency of the results of different interpretation methods

```
sh bin/run_consistent.sh
```


Run the experiments to compare the samples quality of different interpretation methods

```
sh bin/run_consistent.sh
```


Run the experiments to compare the distance between the computed decision feature and the graound truth decision feature

```
sh bin/run_consistent.sh
```

[1] Exact and Consistent Interpretation for Piecewise Linear Neural Networks: A Closed Form Solution. L Chu, X Hu, J Hu, L Wang, J Pei

[2] Learning Important Features Through Propagating Activation Differences. A Shrikumar, P Greenside, A Kundaje
