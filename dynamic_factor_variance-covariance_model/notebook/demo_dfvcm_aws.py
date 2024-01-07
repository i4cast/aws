#!/usr/bin/env python
# coding: utf-8

# ## Train, tune, deploy and review ML algorithm/model DFVCM (dynamic factor variance-covariance model) from AWS Marketplace

# #### Overview of the algorithm
#   
# The dynamic factor variance-covariance model, [DFVCM](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa), is to make multi-step forecast of large variance-covariance matrix of large set of observed time-series, when the time-series are influenced by both (a) dynamic history of a set of unobserved factors commonly affecting all or many of the time-series and (b) dynamic histories of hidden components affecting idiosyncratic components of individual time-series.  
# 
# DFVCM applies [LMDFM](https://aws.amazon.com/marketplace/pp/prodview-da6ffrp4mlopg?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) algorithm to estimate and forecast volatility of common factors of all time-series. Then, DFVCM applies [YWpcAR](https://aws.amazon.com/marketplace/pp/prodview-prndys7tr7go6?sr=0-3&ref_=beagle&applicationId=AWSMPContessa) algorithm to estimate and forecast volatility of idiosyncratic components of individual time-series.
#   
# [LMDFM](https://aws.amazon.com/marketplace/pp/prodview-da6ffrp4mlopg?sr=0-1&ref_=beagle&applicationId=AWSMPContessa) applies dynamic principal components analysis (DPCA) with 1 or 2-dimensional discrete Fourier transforms (1/2D-DTFs). [YWpcAR](https://aws.amazon.com/marketplace/pp/prodview-prndys7tr7go6?sr=0-3&ref_=beagle&applicationId=AWSMPContessa) applies principal components analysis on Yule-Walker equation of individual idiosyncratic component.
#   
# Therefore, the DFVCM algorithm can estimate influences of longer histories of unobserved common factors and hidden idiosyncratic components. The algorithm accommodates wider ranges of values of model learning parameters. The wider ranges can further enhance the power of machine learning.  
#   
# Current version of the [DFVCM](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa) algorithm estimates and/or forecasts (in multi-steps): (a) estimated matrix of factor loadings, (b) forecasted variance of common factors, (c) forecasted variance of idiosyncratic components, (d) forecasted variance of individual time-series, (e) forecasted variance of weighted aggregation of multiple time-series, and (f) forecasted variance-covariance matrix of multiple observed time-series. Other estimates and/or forecasts (such as forecasted auto-covariance matrixes) can be added in the future releases.

# #### Academic publications on multi-step forecasts and multivariate volatilities with dynamic factor models
#   
# L. Alessi, M. Barigozzi and M. Capasso  (2007).  "Dynamic factor GARCH: Multivariate volatility forecast for a large number of series".  LEM Working Paper Series, No. 2006/25, Laboratory of Economics and Management (LEM), Pisa.
#   
# C. Doz  and  P. Fuleky  (2020).  "Chapter 2,  Dynamic Factor Models" in Macroeconomic Forecasting in the Era of Big Data: Theory and Practice, Ed. P. Fuleky,  Advanced Studies in Theoretical and Applied Econometrics, Volume 52.  Springer.  
#   
# i4cast LLC  (2024).  "Introduction to Multi-step Forecast of Multivariate Volatility with Dynamic Factor Model".  https://github.com/i4cast/aws/blob/main/dynamic_factor_variance-covariance_model/publication/.

# #### This notebook
# 
# This sample notebook shows you how to train, tune, deploy and understand a custom ML algorithm/model: [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa), guided by common practices to [Use Algorithm and Model Package Resources](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-mkt-buy.html).
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
#     1. or your AWS account has a subscription to [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa)

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
# 1. Open the algorithm listing page, [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa)
# 1. On the AWS Marketplace listing,  click on **Continue to subscribe** button.
# 1. On the **Subscribe to this software** page, review and click on **"Accept Offer"** if you agree with EULA, pricing, and support terms. 
# 1. Once you click on **Continue to configuration button** and then choose a **region**, you will see a **Product Arn**. This is the algorithm ARN that you need to specify while training a custom ML model. Copy the ARN corresponding to your region and specify the same in the following cell.

# In[ ]:


# specify your valid algorithm ARN
# my_algorithm_arn = 'arn:aws:sagemaker:{region}:123456789012:algorithm/{dfvcm_algorithm}'
# my_algorithm_arn = 'arn:aws:sagemaker:{}:{}:algorithm/{}'.format(
#     'your_region', 'your_aws_account_number', 'your_dfvcm_algorithm_label')
my_algorithm_arn = '{algorithm_arn_string}'
my_prefix = 'dfvcm'


# #### 1.2. Prepare relevant environment

# In[ ]:


# Python packages
import sagemaker
import os

# remind
print('Wait for Sagemaker values assigned to TWO important variables: my_bucket and my_role.\n')

# sagemaker session
my_session = sagemaker.session.Session()

# sagemaker attributes
my_bucket = my_session.default_bucket()
my_role = sagemaker.session.get_execution_role()

# review
print('my_session = {}'.format(my_session))
print('my_bucket = {}'.format(my_bucket))
print('my_role = {}'.format(my_role))


# To run this Sagemaker machine learning ('ml') notebook example, following S3 folders are expected to be in place:
# 
# 1. {my_bucket}/{my_prefix}/input/data/train/
# 1. {my_bucket}/{my_prefix}/input/data/inference/
# 1. {my_bucket}/{my_prefix}/model/
# 1. {my_bucket}/{my_prefix}/output/data/inference/

# In[ ]:


# aws s3 paths
my_input_data_train_path = 's3://{}/{}/input/data/train'.format(my_bucket, my_prefix)
my_input_data_infer_path = 's3://{}/{}/input/data/inference'.format(my_bucket, my_prefix)
my_model_path = 's3://{}/{}/model'.format(my_bucket, my_prefix)
my_output_data_infer_path = 's3://{}/{}/output/data/inference'.format(my_bucket, my_prefix)

# dfvcm Docker container training channel
training_input_channel = 'train'

# aws computing instance type: 'ml.m5.xlarge'
my_EC2 = 'ml.m5.xlarge'

# input CSV data file name
my_input_data_file = 'Weekly_VTS_6Yr.csv'

# information available model and endpoint
my_model_data = str()  # to be assigned / defined
my_model_name = str()  # to be assigned / defined 
my_endpoint_name = 'my-endpoint'


# If you are revisiting this demo notebook, and your model training job and/or your hyperparameter tuning job (to be defined later) were already run at least once, you can copy the resulted Sagemaker string values of your trained model data path and/or your tuned model data path to the variables, my_trained_model_data and/or my_tuned_model_data, in the cell below.

# In[ ]:


# trained model placeholder
# my_trained_model_data = str()
my_trained_model_data = str()
my_trained_model_name = 'my-trained-model'

# AVAILABLE trained model
# IF model is trained and not to be trained again, copy-paste or type the full model data path for my_trained_model_data
# my_trained_model_data = '{my_bucket}/{my_prefix}/model/{some_path}/model.tar.gz'
my_trained_model_data = 'string'

# review
print('Model data of trained model:')
print(my_trained_model_data)
print('Name of trained model:')
print(my_trained_model_name)

# ----------------------------------------------------------------------------------------------------

# tuned model placeholder
# my_tuned_model_data = str()
my_tuned_model_data = str()
my_tuned_model_name = 'my-tuned-model'

# AVAILABLE tuned model
# IF model is tuned and not to be tuned again, copy-paste or type the full model data path for my_tuned_model_data
# my_tuned_model_data = '{my_bucket}/{my_prefix}/model/{some_path}/model.tar.gz'
my_tuned_model_data = 'string'

# review
print('\nModel data of tuned model:')
print(my_tuned_model_data)
print('Name of tuned model:')
print(my_tuned_model_name)


# ### 2. Prepare dataset

# #### 2.1. Dataset format expected by the algorithm

# The DFVCM (dynamic factor variance-covariance model) algorithm takes, as input data, multiple time-series data contained in a CSV (comma separated value) data table, in a format of a CSV text-string or a CSV text-file.  
#   
# Each row of the data table is for values of an individual time-series (TS). Row header is the label or symbol of the time-series. Each column is for values of all time-series at a specific moment in time. Column header is the time-index or time-stamp of the moment. The first data column is for the earliest time and the last column for the most recent time. The input data is essentially in the form of "Row Time-Series of Column Vector". The current version of DFVCM requires equally spaced time-stamps.  
#   
# One of the simplest methods to generate such a CSV text-file is to save a Microsoft Excel spreadsheet as (into) a CSV file.  
#   
# You can also find more information about dataset format in **Usage Information** section of 
# [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa).

# #### 2.2. Configure and visualize training dataset

# A [sample data](https://github.com/i4cast/aws/blob/main/long_memory_vector_autoregressive_model/input/Weekly_VTS_6Yr.csv) provided with this product/example is six-year weekly (logarithmic) performances of mutual funds traded in the U.S. invested in equities, fixed income, and commodities. Each row is of an individual mutual fund. Each column is of a specific calendar week in history. The last week (the last column) was the week with a time-stamp as "2021-12-31". Following simple steps you can upload this sample data to your S3 location.

# #### 2.3. Upload datasets to Amazon S3

# To download the sample dataset from https://github.com/i4cast/aws/blob/main/dynamic_factor_variance-covariance_model/input/Weekly_VTS_6Yr.csv, and then upload the dataset to
# 
# 1. {my_bucket}/{my_prefix}/input/data/train/ for training
# 1. {my_bucket}/{my_prefix}/input/data/inference/ for inference
# 
# following simple steps can be used:

# 1. Open webpage https://github.com/i4cast/aws/blob/main/dynamic_factor_variance-covariance_model/input/Weekly_VTS_6Yr.csv
# 1. Click [Raw] option located at top right of the data table
# 1. In the Raw data window, right click [Save as]
# 1. Set local file folder and file name in the "Save As" window, then click [Save]
# 
# 1. Open AWS S3 Console
# 1. Go to S3 folder: {my_bucket}/{my_prefix}/input/data/train/
# 1. Upload the saved local data file to your AWS S3 folder
# 1. Go to S3 folder: {my_bucket}/{my_prefix}/input/data/inference/
# 1. Upload the saved local data file to your AWS S3 folder

# ### 3. Train a machine learning model

# #### 3.1. Set hyperparameters

# You can also find more information about dataset format in **Hyperparameters** section of [Dynamic Factor Variance-Covariance Model (DFVCM)](https://aws.amazon.com/marketplace/pp/prodview-yvaulquatt3v2?sr=0-6&ref_=beagle&applicationId=AWSMPContessa).

# In[ ]:


# define hyperparameters
# all individual elements must be individual strings
my_hyperparam = dict({
    'len_learn_window': '52',
    'var_order': '13',
    'num_factors': '5',
    'ar_order_idio': '13',
    'num_pcs': '2',
    'alt_ar_order': "dict: {}".format(dict()),
    'alt_num_pcs': "dict: {}".format(dict()),
    'max_forecast_step': '13',
    'target_type': 'Original',
    'num_forecasts': '13'
})
# Later in this example: Tuned hyperparameters = {
#     'len_learn_window': 105,
#     'var_order': 26,
#     'num_factors': 10,
#     'ar_order_idio': 13,
#     'num_pcs': 2,
#     ...}

# variance_type_list and variance_score_list
variance_type_list = list(['forecast', 'estimate', 'diff_FE', 'diff_FS'])
variance_score_list = list(['loglike', 'qstat'])

# 'MetricDefinitions': []
my_metrics = list([
    {
        'Name': '{}_{}'.format(variance_type_list[0], variance_score_list[0]),
        'Regex': '{}_{}=(.*?);'.format(variance_type_list[0], variance_score_list[0])
    },
    {
        'Name': '{}_{}'.format(variance_type_list[2], variance_score_list[0]),
        'Regex': '{}_{}=(.*?);'.format(variance_type_list[2], variance_score_list[0])
    },
    {
        'Name': '{}_{}'.format(variance_type_list[0], variance_score_list[1]),
        'Regex': '{}_{}=(.*?);'.format(variance_type_list[0], variance_score_list[1])
    },
    {
        'Name': '{}_{}'.format(variance_type_list[2], variance_score_list[1]),
        'Regex': '{}_{}=(.*?);'.format(variance_type_list[2], variance_score_list[1])
    }
])

# review
print('Hyperparameters: my_hyperparam =')
print(my_hyperparam)

# review
print('\nEvaluation metrics: my_metrics =')
print(my_metrics)


# #### 3.2. Train a model

# In[ ]:


# create an estimator object for running a training job
# Information on sagemaker.algorithm.AlgorithmEstimator():
# https://sagemaker.readthedocs.io/en/stable/api/training/algorithm.html
#
my_estimator = sagemaker.algorithm.AlgorithmEstimator(
    algorithm_arn=my_algorithm_arn,
    role=my_role,
    instance_count=1,
    instance_type=my_EC2,
    # volume_size=30,
    # volume_kms_key=None,
    # max_run=86400,
    input_mode='File',
    output_path=my_model_path,
    # output_kms_key=None,
    base_job_name='my-training-job',
    sagemaker_session=my_session,
    hyperparameters=my_hyperparam,
    # tags=None,
    # subnets=None,
    # security_group_ids=None,
    # model_uri=None,
    model_channel_name='model',
    metric_definitions=my_metrics # ,
    # encrypt_inter_container_traffic=False,
    # use_spot_instances=False,
    # max_wait=None,
    # **kwargs
)

# Information on sagemaker.inputs.TrainingInput():
# https://sagemaker.readthedocs.io/en/stable/api/utility/inputs.html
#
my_training_input = dict({
    training_input_channel:
        sagemaker.inputs.TrainingInput(
            s3_data=my_input_data_train_path,
            # distribution=None,
            compression=None,
            content_type='text/csv',
            # record_wrapping=None,
            s3_data_type='S3Prefix',
            # instance_groups=None,
            input_mode='File' # ,
            # attribute_names=None,
            # target_attribute_name=None,
            # shuffle_config=None
)})


# In the following cell, set the boolean indicator, run_training_job, to TRUE, in order to
# 1. run DFVCM model training job
# 1. save model artifacts of trained model

# In[ ]:


# run_training_job = True | False
run_training_job = False


# During waiting time after setting indicator run_training_job above to TRUE and running model training job in the cell below, you can re-set run_training_job indicator back to FALSE in order to avoid accidentally running model training job again.

# In[ ]:


# if TRUE then train the model and save the result
if run_training_job and (len(my_trained_model_data) < 0.5):
    
    # remind
    print('Train the model. Wait for training job completes with information:')
    print('Model data of trained model\n')
    
    # Information on sagemaker.algorithm.AlgorithmEstimator().fit()
    # https://sagemaker.readthedocs.io/en/stable/api/training/algorithm.html
    my_estimator.fit(
        inputs=my_training_input,
        wait=True,
        logs='All' # ,
        # job_name=None
    )
    
    # model data information
    my_trained_model_data = my_estimator.model_data
    
    # review
    print('\nModel data of trained model:')
    print(my_trained_model_data)


# For more information how to visualize metrics during the process, see [Easily monitor and visualize metrics while training models on Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/).
# 
# You can also open the training job from [Amazon SageMaker console](https://console.aws.amazon.com/sagemaker/home?#/jobs/) and monitor the metrics/logs in **Monitor** section.

# ### 4. Tune your model (optional)

# #### 4.1. Tuning guidelines

# Modeling and/or forecasting different sets of multiple time-series require different values of hyperparameters: len_learn_window, var_order, num_factors, ar_order_idio, and num_pcs.
# 
# Therefore, decisions on specific (integer) values of these hyperparameters need to be made before making meaningful training and inference. There are a variety of commonly practiced methods to estimate the appropriate hyperparameter values. When using AWS Sagemaker, it is natural to use Sagemaker's HyperparameterTuner class to search for appropriate hyperparameter values which result in better forecasts.
# 
# For information about Automatic model tuning, also see [Perform Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)

# #### 4.2. Define tuning configuration

# Possible ranges of appropriate hyperparameter values depend on specific dataset at hand. For the sample dataset used in this example, a set of reasonable ranges of hyperparameter values are as follows.

# In[ ]:


# Information on sagemaker.parameter.IntegerParameter():
# https://sagemaker.readthedocs.io/en/stable/api/training/parameter.html
int_hyperpar_range_example = dict({
    'len_learn_window':
        sagemaker.parameter.IntegerParameter(
        min_value=52, max_value=157, scaling_type='Auto'),
    'var_order':
        sagemaker.parameter.IntegerParameter(
        min_value=1, max_value=52, scaling_type='Auto'),
    'num_factors':
        sagemaker.parameter.IntegerParameter(
        min_value=1, max_value=30, scaling_type='Auto'),
    'ar_order_idio':
        sagemaker.parameter.IntegerParameter(
        min_value=1, max_value=52, scaling_type='Auto'),
    'num_pcs':
        sagemaker.parameter.IntegerParameter(
        min_value=1, max_value=20, scaling_type='Auto')
})


# Natural seasonality of time-series and some "rule of thumb for choices" may be utilized to focus on a few reasonable values within reasonable ranges. Following example can be used for a simpler model tuning.
# 
# For general information about AWS SageMaker Hyperparameter Tuning, referred to [How Hyperparameter Tuning Works](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html) and [Define Hyperparameter Ranges](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html).

# In[ ]:


# Information on sagemaker.parameter.CategoricalParameter():
# https://sagemaker.readthedocs.io/en/stable/api/training/parameter.html
my_hyperparam_range = dict({
    'len_learn_window':
        sagemaker.parameter.CategoricalParameter(['52', '105']),
    'var_order':
        sagemaker.parameter.CategoricalParameter(['13', '26']),
    'num_factors':
        sagemaker.parameter.IntegerParameter(
        min_value=2, max_value=10, scaling_type='Auto'),
    'ar_order_idio':
        sagemaker.parameter.CategoricalParameter(['13', '26']),
    'num_pcs':
        sagemaker.parameter.IntegerParameter(
        min_value=2, max_value=5, scaling_type='Auto')
})


# Different inference applications need to use different metrics to measure relevant goodness of fit. In this example, we try to forecast future performances of U.S. mutual funds. Proportionalities (a quantifiable version of similarity) between forecasted and realized absolute performances can serve as a useful measure of goodness of fit.
# 
# If we regard a set of forecasted or realized absolute performances as a multi-dimensional vector, projection of one vector (e.g. forecasted) onto the other (e.g. realized) is a measure of "proportionality (or similarity) between the two sets of absolute performances".
# 
# Therefore, we use the "projection coefficient" as the objective metric for tuning the hyperparameters.
# 
# For general information about AWS SageMaker Metrics, referred to [Define Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html)

# In[ ]:


# available choices for objective tuning metric
print('Available model evaulation metrics:')
print(my_metrics)

# name of objective tuning metric
# my_metrics = list([
#     ...,
#     {
#         'Name': '{}_{}'.format(variance_type_list[2] variance_score_list[0]),
#         'Regex': '{}_{}=(.*?);'.format(variance_type_list[2], variance_score_list[0])
#     },
#     ...
# ])
my_objective_metric = my_metrics[1]['Name']

# review
print('\nObjective tuning metric')
print(my_objective_metric)


# In general, minimizing error and/or maximizing similarity are desirable tuning directions. Therefore, we will maximize our objective metric, projection coefficient, in this hyperparameter tuning example.

# In[ ]:


# direction of hyperparameter optimization
my_objective_type = 'Maximize'


# #### 4.3. Run a model tuning job

# In[ ]:


# setting up hyperparameter tuning job
# Information on sagemaker.tuner.HyperparameterTuner():
# https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html
#
# Notes on an AWS Sagemaker requirement:
# when calling the CreateHyperParameterTuningJob operation,
# you can’t override the metric definitions for AWS Marketplace algorithms.
# try the request without specifying metric definitions.
#
my_tuner = sagemaker.tuner.HyperparameterTuner(
    estimator=my_estimator,
    objective_metric_name=my_objective_metric,
    hyperparameter_ranges=my_hyperparam_range,
    # metric_definitions=None,
    # strategy='Bayesian',
    objective_type=my_objective_type,
    max_jobs=1,
    max_parallel_jobs=1,
    # max_runtime_in_seconds=None,
    # tags=None,
    base_tuning_job_name='my-tuning-job',
    # warm_start_config=None,
    # strategy_config=None,
    # completion_criteria_config=None,
    early_stopping_type='Auto' # ,
    # estimator_name=None,
    # random_seed=None,
    # autotune=False,
    # hyperparameters_to_keep_static=None
)


# In the following cell, set the boolean indicator, run_tuning_job, to TRUE, in order to
# 1. run hyperparameter optimization job
# 1. save optimal model artifacts

# In[ ]:


# run_tuning_job = True | False
run_tuning_job = False


# During waiting time after setting indicator run_tuning_job above to TRUE and running hyperparameter tuning job in the cell below, you can re-set run_tuning_job indicator back to FALSE in order to avoid accidentally running hyperparameter tuning job again.

# In[ ]:


# if TRUE then optimize model and save the result
if run_tuning_job and (len(my_tuned_model_data) < 0.5):
    
    # remind
    print('Tune the model. Wait for tuning job completes with information:')
    print('Model data of tuned model\n')
    
    # tuning and waiting
    # Information on sagemaker.tuner.HyperparameterTuner().fit():
    # https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html
    my_tuner.fit(
        inputs=my_training_input)
    my_tuner.wait()
    
    # get tuned model and artfacts of the tuned model
    # Information on sagemaker.tuner.HyperparameterTuner().best_estimator():
    # https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html
    my_tuned_estimator = my_tuner.best_estimator()
    my_tuned_estimator.fit(
        inputs=my_training_input,
        wait=True,
        logs='All')
    
    # optimized hyperparameters
    my_tuned_hyperparam = my_tuned_estimator.hyperparameters()
    
    # optimal model artfacts
    my_tuned_model_data = my_tuned_estimator.model_data
    
    # review
    print('\nTuned hyperparameters:')
    print(my_tuned_hyperparam)
    
    # review
    print('\nModel data of tuned model:')
    print(my_tuned_model_data)


# As recommended by AWS Sagemaker Team, once you have completed a tuning job, (or even while the job is still running) you can [clone and use this notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb) to analyze the results to understand how each hyperparameter effects the quality of the model.

# ### 5. Deploy model and verify results

# #### 5.1. Trained or tuned model

# In[ ]:


# available trained model
if len(my_trained_model_data) > len('s3://.tar.gz'):
    my_model_data = my_trained_model_data
    my_model_name = my_trained_model_name

# available tuned model
if len(my_tuned_model_data) > len('s3://.tar.gz'):
    my_model_data = my_tuned_model_data
    my_model_name = my_tuned_model_name

# Information on sagemaker.model.ModelPackage():
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html
my_model = sagemaker.model.ModelPackage(
    role=my_role,
    model_data=my_model_data,
    algorithm_arn=my_algorithm_arn, # algorithm arn used to train the model
    # model_package_arn=None,
    # -----------------------
    # other **kwargs include:
    # image_uri,
    # predictor_cls=None,
    # env=None,
    name=my_model_name,
    # vpc_config=None,
    sagemaker_session=my_session # ,
    # enable_network_isolation=None,
    # model_kms_key=None,
    # image_config=None,
    # source_dir=None,
    # code_location=None,
    # entry_point=None,
    # container_log_level=20,
    # dependencies=None,
    # git_config=None,
    # resources=None
)

# review
print('Name of model:')
print(my_model_name)

# review
print('\nArtifacts of model:')
print(my_model_data)

# review
print('\nModel pacakge')
print(my_model)


# #### 5.2. Deploy trained or tuned model

# In[ ]:


# remind
print('Start endpoint for inference. Wait for endpoint becomes ready')

# Information on sagemaker.serializers.IdentitySerializer():
# https://sagemaker.readthedocs.io/en/stable/api/inference/serializers.html
my_serializer = sagemaker.serializers.IdentitySerializer()

# Information on sagemaker.deserializers.StreamDeserializer():
# https://sagemaker.readthedocs.io/en/stable/api/inference/deserializers.html
my_deserializer = sagemaker.deserializers.StreamDeserializer()

# Information on sagemaker.model.Model().deploy():
# https://sagemaker.readthedocs.io/en/stable/api/inference/model.html
my_endpoint = my_model.deploy(
    initial_instance_count=1,
    instance_type=my_EC2,
    serializer=my_serializer,
    deserializer=my_deserializer,
    # accelerator_type=None,
    endpoint_name=my_endpoint_name # ,
    # tags=None,
    # kms_key=None,
    # wait=True,
    # data_capture_config=None,
    # async_inference_config=None,
    # serverless_inference_config=None,
    # volume_size=None,
    # model_data_download_timeout=None,
    # container_startup_health_check_timeout=None,
    # inference_recommendation_id=None,
    # explainer_config=None,
    # accept_eula=None,
    # endpoint_logging=False
    # resources=None,
    # endpoint_type=<EndpointType.MODEL_BASED: 'ModelBased'>,
    # managed_instance_scaling=None,
    # **kwargs
)

# review
print('\nSagemaker endpoint, ' + my_endpoint_name + ', is now ready')


# In[ ]:


# Predictor
# Information on sagemaker.predictor.Predictor():
# https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
my_predictor = sagemaker.predictor.Predictor(
    endpoint_name=my_endpoint_name,
    sagemaker_session=my_session,
    serializer=my_serializer,
    deserializer=my_deserializer # ,
    # component_name=None,
    # **kwargs
)

# review
print(my_predictor)


# #### 5.3. Create input payload

# Input payload can be created by following functions of the class [S3 Utilities](https://sagemaker.readthedocs.io/en/stable/api/utility/s3.html)
# 
# 1. **sagemaker.s3.s3_path_join(*args)**: similarly to os.path.join()
# 1. **sagemaker.s3.S3Downloader.read_file(s3_uri, sagemaker_session=None)**: returns the contents of an s3 uri file body as a string

# In[ ]:


# data file for inference
my_infer_input_file = sagemaker.s3.s3_path_join(
    my_input_data_infer_path,
    my_input_data_file)

# CSV data: string
my_infer_input_str = sagemaker.s3.S3Downloader.read_file(
    my_infer_input_file, 
    sagemaker_session=my_session)

# CSV data: byte stream object
my_inference_input_obj = my_infer_input_str.encode()

# review
print('my_infer_input_file:')
print(my_infer_input_file + '\n')

# review
print('my_infer_input_str: ' + str(type(my_infer_input_str)))
print('my_inference_input_obj: ' + str(type(my_inference_input_obj)))


# #### 5.4. Perform real-time inference

# In[ ]:


# Information on sagemaker.predictor.Predictor().predict():
# https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
my_forecast = my_predictor.predict(
    data=my_inference_input_obj # ,
    # initial_args=None,
    # target_model=None,
    # target_variant=None,
    # inference_id=None,
    # custom_attributes=None,
    # component_name=None
)


# In[ ]:


# review
print('Output of real-time inference:')
print(my_forecast)

# review
# Information on botocore.response.StreamingBody()
# https://botocore.amazonaws.com/v1/documentation/api/latest/reference/response.html
print('\nReal-time forecasts of time-series')
print(my_forecast[0].read())


# Now that you have successfully performed a real-time inference, you do not need the endpoint any more. You can terminate it to avoid being charged.

# In[ ]:


# Information on sagemaker.predictor.Predictor().delete_endpoint():
# https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
my_predictor.delete_endpoint(
    delete_endpoint_config=True)


# ### 6. Perform batch inference

# #### 6.1. Batch transform

# In[ ]:


# default inference ENV variables
my_ENV = dict({
    'MODELOUTPUT': 'varcov'
})

# available output type
output_type_choice = dict({
    1: 'text/csv',
    2: 'application/json'
})

# output type
output_type = output_type_choice[
    1
]
# Notes: output data file of type 'text/csv'
# can be reviewed simply by a simple text edidtor

# Information sagemaker.transformer.Transformer():
# https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html
my_transformer = sagemaker.transformer.Transformer(
    model_name=my_model_name,
    instance_count=1,
    instance_type=my_EC2,
    # strategy=None,
    # assemble_with=None,
    output_path=my_output_data_infer_path,
    # output_kms_key=None,
    accept=output_type,
    # max_concurrent_transforms=None,
    # max_payload=None,
    # tags=None,
    env=my_ENV,
    # base_transform_job_name=None,
    sagemaker_session=my_session # ,
    # volume_kms_key=None
)


# Note: Batch-transform job input file is located in the S3 folder: {my_bucket}/{my_prefix}/input/data/inference/

# In[ ]:


# Information on sagemaker.inputs.TransformInput():
# https://sagemaker.readthedocs.io/en/stable/api/utility/inputs.html
my_transform_data_path = my_input_data_infer_path
my_transform_data_type = 'S3Prefix'
my_transform_content_type = 'text/csv'


# In[ ]:


# remind
print('Run batch transform. Wait for transform job completes with information:')
print('Batch transform output path')

# Information on sagemaker.transformer.Transformer().transform():
# https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html
my_transformer.transform(
    data=my_transform_data_path,
    data_type=my_transform_data_type,
    content_type=my_transform_content_type,
    compression_type=None,
    # split_type=None,
    # job_name=None,
    # input_filter=None,
    # output_filter=None, 
    # join_source=None,
    # experiment_config=None,
    # model_client_config=None,
    # batch_data_capture_config=None,
    wait=True,
    logs=True
)

# wait
my_transformer.wait()

# output is available on following path
my_transform_output_path = my_transformer.output_path
print('Batch transform output path:')
print(my_transform_output_path)


# Now you can display and review output generated by the batch transform job available in S3.

# In[ ]:


# transform output file name = {input_data_file}.csv.out
my_transform_output_file = my_input_data_file + '.out'

# data file for inference
my_inference_file = sagemaker.s3.s3_path_join(
    my_transform_output_path,
    my_transform_output_file)

# CSV data string
my_inference = sagemaker.s3.S3Downloader.read_file(
    my_inference_file, 
    sagemaker_session=my_session)

# review
print('Output of batch transform job:\n')
print(my_inference)


# You may change the transform output file name to keep the file from being overwritten.
# 
# Open AWS S3 Console, go to the batch transform output path shown above, re-name the file "{inference_input_data_file_name}.csv.out" to
# 1. "varcov.csv", if accept = output_type = 'text/csv', or
# 1. "varcov.json", if accept = output_type = 'application/json'

# #### 6.2. Delete the model

# Now that you have successfully performed a batch inference. IF you plan to review the trained or tuned model structure by using Transformer as demonstrated later, do NOT run the cell below. Otherwise, you can delete the model.

# In[ ]:


# need more batch transform?
more_batch_transform = True

# Information on sagemaker.session.Session().delete_model():
# https://sagemaker.readthedocs.io/en/stable/api/utility/session.html
if not more_batch_transform:
    my_session.delete_model(my_model_name)


# ### 7. Model review by using Transformer (optional)

# #### 7.1. Available DFVCM model output data items

# **"mean"**  
#   
# Data access method:  
#   
#     mean_vec = DFVCM_obj.get_mean()  
#   
# Data item(s) returned:  
#   
#     mean_vec : pandas.Series, index (ts_list)  
#         Sample mean vector of observed vector time-series  
#             of data points in last learning window  

# **"stdev"**  
#   
# Data access method:  
#   
#     stdev_vec = DFVCM_obj.get_stdev()  
#   
# Data item(s) returned:  
#   
#     stdev_vec : pandas.Series, index (ts_list)
#         Sample standard deviation vector of observed
#             vector time-series of data points in last
#             learning window

# **"varcov"**  
#   
# Data access method:  
#   
#     loadings_mat, dfs_variance, idio_variance = (  
#         DFVCM_obj.get_varcov())  
#   
# Data item(s) returned:  
#   
#     loadings_mat : pd.DataFrame,  
#             index (ts_list), columns (factor_list)  
#         Loadings matrix of common dynamic factors  
#             on observed vector time-series  
#     
#     dfs_variance : pd.DataFrame, index (factor_list),  
#             columns (0, 1, ..., max_forecast_step, 'asof')  
#         Out-of-sample multi-step forecasts of variance vectors  
#             of dynamic factor score time-series of observed  
#             vector time-series  
#         Notes:  
#             obj.loc[:, 0]: Estimated current, or nowcast of  
#                 variance vector  
#     
#     idio_variance : pd.DataFrame, index (ts_list),  
#             columns (0, 1, ..., max_forecast_step, 'asof')  
#         Forecasted variance vector of idiosyncratic component  
#             time-series of observed vector time-series  
#         Notes:  
#             obj.loc[:, 0]: Estimated current variance vector  

# **"aggVar"**  
#   
# Data access method:  
#   
#     agg_variance = DFVCM_obj.get_aggVar()  
#   
# Data item(s) returned:  
#   
#     agg_variance : pd.Series,  
#             index (0, 1, ..., max_forecast_step, 'asof')  
#         Forecasted variances of aggregate value of   
#             observed vector time-series  

# **"indivVar"**  
#   
# Data access method:  
#   
#     indiv_variance = DFVCM_obj.get_indivVar()  
#   
# Data item(s) returned:  
#   
#     indiv_variance : pd.DataFrame,  
#             index (ts_list),  
#             columns (0, 1, ..., max_forecast_step, 'asof')  
#         Out-of-sample multi-step forecasts of variance vectors  
#             of individual observed multiple time-series  
#         Notes:  
#             obj.loc[:, 0]: Estimated current, or nowcast of  
#                 variance vector  

# **"vcMatrix"**  
#   
# Data access method:  
#   
#     comm_varcov, idio_variance, vts_varcov = (  
#         DFVCM_obj.get_vcMatrix())  
#   
# Data item(s) returned:  
#   
#     comm_varcov : dict,  
#                 keys (0, 1, ..., max_forecast_step, 'asof')  
#             obj[key] : pd.DataFrame,  
#                 index (ts_list), columns (ts_list)  
#         Forecasted variance-covariance matrix of factor-based  
#             common component time-series of observed vector  
#             time-series  
#         Notes:  
#             obj[0]: Estimated current variance-covariance  
#     
#     idio_variance : pd.DataFrame, index (ts_list),  
#             columns (0, 1, ..., max_forecast_step, 'asof')  
#         Forecasted variance vector of idiosyncratic component  
#             time-series of observed vector time-series  
#         Notes:  
#             obj.loc[:, 0]: Estimated current variance vector  
#     
#     vts_varcov : dict,  
#                 keys (0, 1, ..., max_forecast_step, 'asof')  
#             obj[key] : pd.DataFrame,  
#                 index (ts_list), columns (ts_list)  
#         Forecasted variance-covariance matrix of observed  
#             vector time-series  
#         Notes:  
#             obj[0]: Estimated current variance-covariance  

# #### 7.2. Select DFVCM model output data item for review

# Trained or tuned DFVCM model output can be reviewed item by item using Transformer  
# with an environment variable, MODELOUTPUT  
#   
# Choices of values of MODELOUTPUT are:  

# In[ ]:


# available choices for MODELOUTPUT
model_output_choice = dict({
    1: 'mean',
    2: 'stdev',
    3: 'varcov',
    4: 'aggVar',
    5: 'indivVar',
    6: 'vcMatrix'
})

# available choices for output type
output_type_choice = dict({
    1: 'text/csv',
    2: 'application/json'
})


# You can make any valid pair of choices as exemplified as in following cell:

# In[ ]:


# choice for MODELOUTPUT (an integer between 1 and 6)
model_output = model_output_choice[
    6
]

# output type
output_type = output_type_choice[
    1
]
# Notes: output data file of type 'text/csv'
# can be reviewed simply by a simple text edidtor

# review
print('model_output = ' + model_output)
print('output_type = ' + output_type)


# #### 7.3. Model output review with Transformer

# In[ ]:


# ENV variables
my_ENV = dict({
    'MODELOUTPUT': model_output})

# sagemaker.transformer.Transformer()
# https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html
my_transformer = sagemaker.transformer.Transformer(
    model_name=my_model_name,
    instance_count=1,
    instance_type=my_EC2,
    # strategy=None,
    # assemble_with=None,
    output_path=my_output_data_infer_path,
    # output_kms_key=None,
    accept=output_type,
    # max_concurrent_transforms=None,
    # max_payload=None,
    # tags=None,
    env=my_ENV,
    # base_transform_job_name=None,
    sagemaker_session=my_session # ,
    # volume_kms_key=None
)


# In[ ]:


# sagemaker.inputs.TransformInput()
my_transform_data_path = my_input_data_infer_path
my_transform_data_type = 'S3Prefix'
my_transform_content_type = 'text/csv'

# remind
print('Run batch transform. Wait for transform job completes with information:')
print('Batch transform output path')

# Information on sagemaker.transformer.Transformer().transform():
# https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html
my_transformer.transform(
    data=my_transform_data_path,
    data_type=my_transform_data_type,
    content_type=my_transform_content_type,
    compression_type=None,
    # split_type=None,
    # job_name=None,
    # input_filter=None,
    # output_filter=None, 
    # join_source=None,
    # experiment_config=None,
    # model_client_config=None,
    # batch_data_capture_config=None,
    wait=True,
    logs=True
)

# wait
my_transformer.wait()

# output is available on following path
my_transform_output_path = my_transformer.output_path
print('Batch transform output path:')
print(my_transform_output_path)


# You can display and review output generated by the batch transform job available in S3.

# In[ ]:


# transform output file name = {input_data_file}.csv.out
my_transform_output_file = my_input_data_file + '.out'

# data file for inference
my_inference_file = sagemaker.s3.s3_path_join(
    my_transform_output_path,
    my_transform_output_file)

# CSV data string
my_inference = sagemaker.s3.S3Downloader.read_file(
    my_inference_file, 
    sagemaker_session=my_session)

# display
print('Selected output:\n')
print(my_inference)


# You may change the selected output file name to keep the file from being overwritten.
# 
# Open AWS S3 Console, go to the batch transform output path shown above, re-name the file "{inference_input_data_file_name}.csv.out" to
# 1. "{model_output}.csv", if accept = output_type = 'text/csv', or
# 1. "{model_output}.json", if accept = output_type = 'application/json'

# ### 8. Clean-up

# #### 8.1. Delete endpoint and model

# In[ ]:


# Information on sagemaker.predictor.Predictor().delete_endpoint():
# https://sagemaker.readthedocs.io/en/stable/api/inference/predictors.html
my_predictor.delete_endpoint(
    delete_endpoint_config=True)


# In[ ]:


# Information on sagemaker.session.Session().delete_model():
# https://sagemaker.readthedocs.io/en/stable/api/utility/session.html
my_session.delete_model(my_model_name)


# #### 8.2. Unsubscribe to the listing (optional)

# If you would like to unsubscribe to the algorithm, follow these steps. Before you cancel the subscription, ensure that you do not have any [deployable model](https://console.aws.amazon.com/sagemaker/home#/models) created from the model package or using the algorithm. Note - You can find this information by looking at the container name associated with the model. 
# 
# **Steps to unsubscribe to product from AWS Marketplace**:  
# 
# 1. Navigate to __Machine Learning__ tab on [__Your Software subscriptions page__](https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=mlmp_gitdemo_indust)
# 2. Locate the listing that you want to cancel the subscription for, and then choose __Cancel Subscription__  to cancel the subscription.
# 
# 
