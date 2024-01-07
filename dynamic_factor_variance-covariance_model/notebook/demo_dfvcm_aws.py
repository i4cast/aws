#!/usr/bin/env python
# coding: utf-8

# ## Train, tune, deploy and review ML algorithm/model DFVCM (dynamic factor variance-covariance model) from AWS Marketplace

# #### Overview of the algorithm
#   
# The dynamic factor variance-covariance model, [DFVCM](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa), is to make multi-step forecast of large variance-covariance matrix of large set of observed time-series, when the time-series are influenced by both (a) dynamic history of a set of unobserved factors commonly affecting all or many of the time-series and (b) dynamic histories of hidden components affecting idiosyncratic components of individual time-series.  
# 
# DFVCM applies [LMDFM](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa) algorithm to estimate and forecast volatility of common factors of all time-series. Then, DFVCM applies [YWpcAR](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa) algorithm to estimate and forecast volatility of idiosyncratic components of individual time-series.
#   
# [LMDFM](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa) applies dynamic principal components analysis (DPCA) with 1 or 2-dimensional discrete Fourier transforms (1/2D-DTFs). [YWpcAR](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa) applies principal components analysis on Yule-Walker equation of individual idiosyncratic component.
#   
# Therefore, the DFVCM algorithm can estimate influences of longer histories of unobserved common factors and hidden idiosyncratic components. The algorithm accommodates wider ranges of values of model learning parameters. The wider ranges can further enhance the power of machine learning.  
#   
# Current version of the [DFVCM](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa) algorithm estimates and/or forecasts (in multi-steps): (a) estimated matrix of factor loadings, (b) forecasted variance of common factors, (c) forecasted variance of idiosyncratic components, (d) forecasted variance of individual time-series, (e) forecasted variance of weighted aggregation of multiple time-series, and (f) forecasted variance-covariance matrix of multiple observed time-series. Other estimates and/or forecasts (such as forecasted auto-covariance matrixes) can be added in the future releases.

# #### Academic publications on multi-step forecasts and multivariate volatilities with dynamic factor models
#   
# L. Alessi, M. Barigozzi and M. Capasso  (2007).  "Dynamic factor GARCH: Multivariate volatility forecast for a large number of series".  LEM Working Paper Series, No. 2006/25, Laboratory of Economics and Management (LEM), Pisa.
#   
# C. Doz  and  P. Fuleky  (2020).  "Chapter 2,  Dynamic Factor Models" in Macroeconomic Forecasting in the Era of Big Data: Theory and Practice, Ed. P. Fuleky,  Advanced Studies in Theoretical and Applied Econometrics, Volume 52.  Springer.  
#   
# i4cast LLC  (2024).  "Introduction to Multi-step Forecast of Multivariate Volatility with Dynamic Factor Model".  https://github.com/i4cast/aws/blob/main/dynamic_factor_variance-covariance_model/publication/.

# #### This notebook
# 
# This sample notebook shows you how to train, tune, deploy and understand a custom ML algorithm/model: [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa), guided by common practices to [Use Algorithm and Model Package Resources](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-mkt-buy.html).
# 
# > **Note**: This is a reference notebook and it cannot run unless you make changes suggested in the notebook.

# #### Pre-requisites
# 
# 1. **Note**: This notebook contains elements which render correctly in Jupyter interface. Open this notebook from an Amazon SageMaker Notebook Instance or Amazon SageMaker Studio.
# 1. Ensure that IAM role used has **AmazonSageMakerFullAccess**
# 1. Some hands-on experience using [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
# 1. To use this algorithm successfully, ensure that:
#     1. Either your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used: 
#         1. **aws-marketplace:ViewSubscriptions**
#         1. **aws-marketplace:Unsubscribe**
#         1. **aws-marketplace:Subscribe**  
#     1. or your AWS account has a subscription to [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa)

# #### Contents
# 
# 1. [Subscribe to the algorithm](#1.-Subscribe-to-the-algorithm)
#     1. [Subscription](#1.1.-Subscription)
#     1. [Prepare relevant environment](#1.2.-Prepare-relevant-environment)
# 1. [Prepare dataset](#2.-Prepare-dataset)
#     1. [Dataset format expected by the algorithm](#2.1.-Dataset-format-expected-by-the-algorithm)
#     1. [Configure and visualize training dataset](#2.2.-Configure-and-visualize-training-dataset)
#     1. [Upload datasets to Amazon S3](#2.3.-Upload-datasets-to-Amazon-S3)
# 1. [Train a machine learning model](#3.-Train-a-machine-learning-model)
#     1. [Set hyperparameters](#3.1.-Set-hyperparameters)
#     1. [Train a model](#3.2.-Train-a-model)
# 1. [Tune your model (optional)](#4.-Tune-your-model-(optional))
#     1. [Tuning Guidelines](#4.1.-Tuning-guidelines)
#     1. [Define Tuning configuration](#4.2.-Define-tuning-configuration)
#     1. [Run a model tuning job](#4.3.-Run-a-model-tuning-job)
# 1. [Deploy model and verify results](#5.-Deploy-model-and-verify-results)
#     1. [Trained or tuned model](#5.1.-Trained-or-tuned-model)
#     1. [Deploy trained or tuned model](#5.2.-Deploy-trained-or-tuned-model)
#     1. [Create input payload](#5.3.-Create-input-payload)
#     1. [Perform real-time inference](#5.4.-Perform-real-time-inference)
# 1. [Perform Batch inference](#6.-Perform-batch-inference)
#     1. [Batch transform](#6.1.-Batch-transform)
#     1. [Delete the model](#6.2.-Delete-the-model)
# 1. [Model review by using Transformer (optional)](#7.-Model-review-by-using-Transformer-(optional))
#     1. [Available DFVCM model output data items](#7.1.-Available-DFVCM-model-output-data-items)
#     1. [Select DFVCM model output data item for review](#7.2.-Select-DFVCM-model-output-data-item-for-review)
#     1. [Model output review with Transformer](#7.3.-Model-output-review-with-Transformer)
# 1. [Clean-up](#8.-Clean-up)
#     1. [Delete endpoint and model](#8.1.-Delete-endpoint-and-model)
#     1. [Unsubscribe to the listing (optional)](#8.2.-Unsubscribe-to-the-listing-(optional))

# #### Usage instructions
# 
# You can run this notebook one cell at a time (By using Shift+Enter for running a cell).

# #### Sagemaker Notebook
# 
# For readers who like to review how to use Sagemaker Notebook in general, following Sagemaker documentation pages are best resources.  
#     [Get Started with Amazon SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html)  
#     [Step 1: Create an Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)  
#     [Step 2: Create a Jupyter Notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-prepare.html)  
#     [Step 3: Download, Explore, and Transform a Dataset](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-preprocess-data.html)  
#     [Step 4: Train a Model](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-train-model.html)  
#     [Step 5: Deploy the Model to Amazon EC2](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-model-deployment.html)  
#     [Step 6: Evaluate the Model](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-test-model.html)  
#     [Step 7: Clean Up](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html)

# ### 1. Subscribe to the algorithm

# #### 1.1. Subscription

# To subscribe to the algorithm:
# 
# 1. Open the algorithm listing page, [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview...applicationId=AWSMPContessa)
# 1. On the AWS Marketplace listing,  click on **Continue to subscribe** button.
# 1. On the **Subscribe to this software** page, review and click on **"Accept Offer"** if you agree with EULA, pricing, and support terms. 
# 1. Once you click on **Continue to configuration button** and then choose a **region**, you will see a **Product Arn**. This is the algorithm ARN that you need to specify while training a custom ML model. Copy the ARN corresponding to your region and specify the same in the following cell.
