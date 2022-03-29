#!/usr/bin/env python
# coding: utf-8
'''


	Rev
	Rev


i4c/aws/lmdfm/data
i4c/aws/lmdfm/notebook

1. left click link 4c/aws/lmdfm/data
2. left clike link data.csv
3. right click [raw] option located at top right of the data
4. in new data window in the browser, right click [Save as]
5. set local file folder and file anme in the dialog window
6. click [Save]


'''


'''
## Train, tune, deploy and understand ML algorithm/model LMDFM
 (long-memory dynamic factor model) from AWS Marketplace


<font color='red'>
For Seller to update: Add overview of the algorithm here
</font> overview

#### Overview of the algorithm

The long-memory dynamic factor model (LMDFM) algorithm is developed to analyze and
forecast large sets of time-series when the time-series are influenced by evolution histories of a
number of unobserved factors commonly affecting all or many of the time-series.

By applying objective data-driven constraints, the LMDFM algorithm can estimate the
influences of longer histories of common factors. The algorithm accommodates wider ranges of
values of model learning parameters. The wider ranges can further enhance the power of
machine learning.

Current version of the LMDFM algorithm estimates: (a) dynamic factor loadings matrixes, (b)
vector autoregressive (VAR) coefficients of the factors, (c) coefficients in the format of
structural VAR to estimate factor scores, (d) time-series of factor scores, (e) forecasts of
the observed time-series, and (f) impulse response of the time-series to several simultaneous
shocks. Other estimates will be added in the future releases.


<font color='red'>
For Seller to update: Add link to the research paper or a detailed description document of the algorithm here
</font>
An academic lecture notes on DFM model:


This sample notebook shows you how to train, tune, deploy and understand a custom ML algorithm/model using 
<font color='red'>
For Seller to update:[Title_of_your_Algorithm](Provide link to your marketplace listing of your product)
</font>
LMDFM (Long-Memory Dynamic Factor Model) from AWS Marketplace.

> **Note**: This is a reference notebook and it cannot run unless you make changes suggested in the notebook.

# #### Pre-requisites:
# 1. **Note**: This notebook contains elements which render correctly in Jupyter interface. Open this notebook from an Amazon SageMaker Notebook Instance or Amazon SageMaker Studio.
# 1. Ensure that IAM role used has **AmazonSageMakerFullAccess**
# 1. Some hands-on experience using [Amazon SageMaker](https://aws.amazon.com/sagemaker/).
# 1. To use this algorithm successfully, ensure that:
#     1. Either your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used: 
#         1. **aws-marketplace:ViewSubscriptions**
#         1. **aws-marketplace:Unsubscribe**
#         1. **aws-marketplace:Subscribe**  
#     2. or your AWS account has a subscription to 
#        <font color='red'> For Seller to update:[Title_of_your_algorithm](Provide link to your marketplace listing of your product)</font>
#        LMDFM (long-memory dynamic factor model). 


#### Sagemaker Notebook

For readers who like to review how to use Sagemaker Notebook,
 following Sagemaker documentation pages are best resources.  
  [Get Started with Amazon SageMaker Notebook Instances](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html).
  [Step 1: Create an Amazon SageMaker Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html).
  [Step 2: Create a Jupyter Notebook](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-prepare.html).
  [Step 3: Download, Explore, and Transform a Dataset](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-preprocess-data.html).
  [Step 4: Train a Model](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-train-model.html).
  [Step 5: Deploy the Model to Amazon EC2](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-model-deployment.html).
  [Step 6: Evaluate the Model](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-test-model.html).
  [Step 7: Clean Up](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-cleanup.html).

# #### Contents:

# 1. [Subscribe to the algorithm](#1.-Subscribe-to-the-algorithm)

# 1. [Prepare dataset](#2.-Prepare-dataset)
# 	1. [Dataset format expected by the algorithm](#A.-Dataset-format-expected-by-the-algorithm)
# 	1. [Configure and visualize train and test dataset](#B.-Configure-and-visualize-train-and-test-dataset)
# 	1. [Upload datasets to Amazon S3](#C.-Upload-datasets-to-Amazon-S3)
# 1. [Train a machine learning model](#3:-Train-a-machine-learning-model)
# 	1. [Set up environment](#3.1-Set-up-environment)
# 	1. [Train a model](#3.2-Train-a-model)
# 1. [Tune your model! (optional)](#5:-Tune-your-model!-(optional))
# 	1. [Tuning Guidelines](#A.-Tuning-Guidelines)
# 	1. [Define Tuning configuration](#B.-Define-Tuning-configuration)
# 	1. [Run a model tuning job](#C.-Run-a-model-tuning-job)
# 1. [Deploy model and verify results](#4:-Deploy-model-and-verify-results)
#     1. [Deploy trained model](#A.-Deploy-trained-model)
#     1. [Create input payload](#B.-Create-input-payload)
#     1. [Perform real-time inference](#C.-Perform-real-time-inference)
#     1. [Visualize output](#D.-Visualize-output)
#     1. [Calculate relevant metrics](#E.-Calculate-relevant-metrics)
#     1. [Delete the endpoint](#F.-Delete-the-endpoint)
# 1. [Perform Batch inference](#6.-Perform-Batch-inference)
# 1. [Clean-up](#7.-Clean-up)
# 	1. [Delete the model](#A.-Delete-the-model)
# 	1. [Unsubscribe to the listing (optional)](#B.-Unsubscribe-to-the-listing-(optional))


#### Usage instructions
You can run this notebook one cell at a time (By using Shift+Enter for running a cell).
'''


# In[ ]:
    

import sagemaker

my_image = '890771246647.dkr.ecr.us-east-1.amazonaws.com/lmdfm4aws:latest'
# my_role = 'arn:aws:iam::890771246647:role/AmazonSageMaker-ExecutionRole-20220129T120000'
my_role = sagemaker.get_execution_role()
my_EC2 = 'ml.m5.xlarge'
my_model_output_path = 's3://bucket4lmdfm/opt/ml/model'
my_dataset_input = 's3://bucket4lmdfm/opt/ml/input/data/train/Weekly_VTS_6Yr.csv'

my_estimator = sagemaker.estimator.Estimator(
    image_uri=my_image,
    role=my_role,
    instance_count=1,
    instance_type=my_EC2,
    output_path=my_model_output_path,
    sagemaker_session=sagemaker.Session())

my_estimator.fit(inputs=my_dataset_input)

my_predictor = my_estimator.deploy(
    initial_instance_count=1,
    instance_type=my_type)

my_transformer = my_estimator.transformer(
    instance_count=1,
    instance_type=my_EC2)

my_predictor.delete_endpoint()
my_predictor.delete_model()
my_transformer.delete_model()


# In[ ]:


'''
### 1. Subscribe to the algorithm
'''


# In[ ]:


'''
To subscribe to the algorithm:
1. Open the algorithm listing page 
   <font color='red'>
   For Seller to update:[Title_of_your_product](Provide link to your marketplace listing of your product).
   </font>
   LMDFM (long-memory dynamic Factor model)
1. On the AWS Marketplace listing,  click on **Continue to subscribe** button.
1. On the **Subscribe to this software** page, review and click on **"Accept Offer"** if you agree with EULA, pricing, and support terms. 
1. Once you click on **Continue to configuration button** and then choose a **region**, you will see a **Product Arn**. This is the algorithm ARN that you need to specify while training a custom ML model. Copy the ARN corresponding to your region and specify the same in the following cell.
'''


# In[ ]:


algo_arn = "<Customer to specify algorithm ARN corresponding to their AWS region>"


# In[ ]:


'''
### 2. Prepare dataset
'''


# In[ ]:


# Python packages
import sagemaker
import os
import json
import pandas as pd


# In[ ]:


'''
#### A. Dataset format expected by the algorithm
'''


# In[ ]:


'''
The LMDFM (long-memory dynamic factor model) algorithm takes, as input data,
multiple time-series data contained in a CSV (comma separated value) data table, in a format of a CSV text-string or a CSV text-file.

Each row of the data table is for values of an individual time-series (TS). Row header is the label or symbol of the time-series.
Each column is for values of all time-series at a specific monent in time. Column header is the time-index or time-stamp of the moment.
The first data column is for the earliest time and the last column for the most recent time.
Therefore, the first row of the CSV data table is "Label/Symbol/Description, earliest time-stamp, next time-stamp, ..., most recent time-stamp".
The first column of the CSV table is "Label/Symbol/Description, label of 1st TS, label of 2nd TS, ..., label of last TS".
The current version of LMDFM requires equally spaced time-stamps.

Since LMDFM forecasts future values of mulitple time-seres using "Vector Autoregressive (VAR)" model esitmated by "Dynamic Factor Model (DFM)",
the input data is essentially in the form of "Row Time-Series of Column Vector".

One of the simplest methods to generate such a CSV text-file is to save a Microsoft Excel spreadsheet as (into) a CSV file.
A sample data provided (web.link.com) is six-year weekly (logorithmic) perforances of mutual funds traded in the U.S. invested in
equiteis, fixed income, and commodities. Each row is of a individual mutual fund. Each column is of a specific calendar week in history.
The last week (the last column) was the week with a time-stamp as "2021-12-31".


You can also find more information about dataset format in **Usage Information** section of
<font color='red'>
For Seller to update:[Title_of_your_product](Provide link to your marketplace listing of your product).
</font>
LMDFM (Long-Memory Dynamic Factor Model), web.link.com
'''


# In[ ]:


'''
#### B. Configure and visualize train and test dataset
'''


# In[ ]:


# <font color='red'>
# For Seller to update: upload the sample training dataset into data/train directory and update the `training_dataset`
# parameter value in following cell.
# You are strongly recommended to either upload the dataset into data/train directory or download it from a reliable source at runtime.
# **If you intend to download it at run-time, add relevant code in following cell.**
# Do not hardcode your bucket name. 
# </font>


# In[ ]:


training_dataset = "data/train/<FileName.ext>"


# In[ ]:


# <font color='red'>
# For Seller to update/read: We recommend that you support a test channel and accept a test dataset to calculate your algorithm metrics on.
# Emit both - training as well as test metrics.
# </font>

# <font color='red'>
# For Seller to update: upload a test dataset into data/test directory. Alternately, you may want to download the test dataset on-the-fly.
# **If you intend to download it at run-time, add relevant code in following cell.**
# Update the test_dataset parameter value in following cell.
# </font>


# In[ ]:


test_dataset = "data/test/<FileName.ext>"


# In[ ]:


# <font color='red'>
# For Seller to update: Add code that displays a few rows from the training dataset.
# Also explain how the training dataset provided as part of the notebook was created.
# </font>


# In[ ]:


'''
#### C. Upload datasets to Amazon S3
'''


# In[ ]:


# <font color='red'>
# For Seller to read: Do not change bucket parameter value. Do not hardcode your S3 bucket name.
# </font>
sagemaker_session = sage.Session()
bucket = sagemaker_session.default_bucket()
bucket


# In[ ]:


# <font color='red'>
# For Seller to update: Update prefix with a unique S3 prefix for your algorithm.
# </font>
training_data = sagemaker_session.upload_data(
    test_dataset, bucket=bucket, key_prefix="<For Seller to update:S3 Prefix>"
)
test_data = sagemaker_session.upload_data(
    test_dataset, bucket=bucket, key_prefix="<For Seller to update:S3 Prefix>"
)


# In[ ]:


'''
## 3: Train a machine learning model
'''


# In[ ]:


# Now that dataset is available in an accessible Amazon S3 bucket, we are ready to train a machine learning model. 


# In[ ]:


'''
### 3.1 Set up environment
'''


# In[ ]:


import sagemaker

my_image = '890771246647.dkr.ecr.us-east-1.amazonaws.com/lmdfm4aws:latest'
# my_role = 'arn:aws:iam::890771246647:role/AmazonSageMaker-ExecutionRole-20220129T120000'
my_role = sagemaker.get_execution_role()
my_EC2 = 'ml.m5.xlarge'
my_model_output_path = 's3://bucket4lmdfm/opt/ml/model'
my_dataset_input = 's3://bucket4lmdfm/opt/ml/input/data/train/Weekly_VTS_6Yr.csv'


# <font color='red'>
# For Seller to update: Initialize required variables in following cell.
# </font>
role = get_execution_role()

# <font color='red'>
# For Seller to update: update algorithm sepcific unique prefix in following cell.
# </font>
output_location = "s3://{}/<For seller to Update:Update a unique prefix>/{}".format(
    bucket, "output"
)


# In[ ]:


'''
### 3.2 Train a model
'''


# In[ ]:


# <font color='red'>
# For Seller to update: Update following cell with appropriate
# hyperparameter values to be passed to the training job
# </font>

# You can also find more information about dataset format in
# **Hyperparameters** section of
# <font color='red'>
# For Seller to update:[Title_of_your_product](Provide link
# to your marketplace listing of your product).
# </font>

# Define hyperparameters
hyperparameters = {}


# In[ ]:


# <font color='red'>
# For Seller to update: Update appropriate values in estimator
# definition and ensure that fit call works as expected.
# </font>

# For information on creating an `Estimator` object, see [documentation]
# (https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html)

# Create an estimator object for running a training job
estimator = sage.algorithm.AlgorithmEstimator(
    algorithm_arn=algo_arn,
    base_job_name="<For Seller to update: Specify base job name>",
    role=role,
    train_instance_count=1,
    train_instance_type="<For Seller to update: Specify an instance-type recommended for training>",
    input_mode="File",
    output_path=output_location,
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
)
# Run the training job.
estimator.fit({"training": training_dataset, "test": test_dataset})

# See this [blog-post]
# (https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/)
# for more information how to visualize metrics during the process.
# You can also open the training job from
# [Amazon SageMaker console](https://console.aws.amazon.com/sagemaker/home?#/jobs/)
# and monitor the metrics/logs in **Monitor** section.


# In[ ]:


'''
### 5: Tune your model! (optional)
'''


# In[ ]:


# =============
# to be deleted
# =============
#
# <font color='red'>
# For Seller to update/read: It is important to provide hyperparameter
# tuning functionality as part of your algorithm. Users of algorithms
# range from new developers, to data scientists and ML practitioners.
# As an algorithm maker, you need to make your algorithm usable in
# production. To be able to do so, you need to give  tools such as
# capability to tune a custom ML model using Amazon SageMaker Automatic
# Model Tuning(HPO) SDK. Enabling your algorithm for automatic model
# tuning functionality is really easy. You need to mark appropriate
# hyperparameters as Tunable=True and emit multiple metrics that customers
# can choose to tune an ML model on.
#     
#     
# We recommend that you provide notes on how your customer can scale
# usage of your algorithm for really large datasets. 
#     
# **You are strongly recommended to provide this section with tuning
# guidelines and code for running an automatic tuning job**.
#     
# For information about Automatic model tuning, see
# [Perform Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)


# In[ ]:


'''
#### A. Tuning Guidelines
'''


# <font color='red'>
# For Seller to update: Provide guidelines on how customer can tune
# their ML model effectively using your algorithm in following cell.
# Provide details such as which parameter can be tuned for best results.
# </font>
'''
Modeling and/or forecasting different sets of mulitple time-series require different values of hyperparameters:
len_learn_window, var_order, and num_factors.

Therefore, decisions on specific (integer) values of these hyperparameters need to be made before making meaningful training and inference.
There are a variety of commonly practiced methods to estimate the appropriate hyperparameter values.
When using AWS Sagemaker, it is natural to use Sagemaker's HyperparameterTuner class to search for appropriate hyperparameter values
which result in better forecasts.

For information about Automatic model tuning, also see
[Perform Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
'''


# In[ ]:


'''
#### B. Define Tuning configuration
'''

# <font color='red'>
# For seller to update: Provide a recommended hyperparameter range
# configuration in the following cell. This configuration would be used
# for running an
# [HPO](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)
# job. For More information, see
# [Define Hyperparameters](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html)
# </font>
'''
Possible ranges of appropriate hyperparameter values depend on specific dataset at hand.
For the sample dataset used in this example, a set of reasonable ranges of hyperparameter
values is as follows.
'''


# In[ ]:


# ranges of hyperparameter values for LMDFM model tuning
hyperparameter_ranges = {}


# In[ ]:


# <font color='red'>
# For seller to update: As part of your algorithm, provide multiple
# objective metrics so that customer can choose a metric for tuning 
# a custom ML model. Update the following variable with a most
# suitable/popular metric that your algorithm emits. For more information, see
# [Define Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-metrics.html) </font>
'''
Different inference applicatoins need to use different metrics to measure relevant goodness of fit.
In this example, we try to forecast future performances of U.S. mutual funds.
Proportionalities (a quantifiable version of similarty) between foreacated and realized absolute performances can serve as
a useful measure of goodness of fit.

If we regard a set of forecasted or realized absolute performances as a multi-dimensional vector, projection of one vector
(e.g. forecasted) onto the other (e.g. realized) is a measure of "proportionality between the two sets of absolute performance".

Therefore, we use the "projection coefficient" as the objective metric for tuning the hyperparameters.
'''


# In[ ]:

# tuning metric
objective_metric_name = (
    "<For seller to update : Provide an appropriate objective metric emitted by the algorithm>"
)


# In[ ]:


# <font color='red'>
# For seller to update: Specify whether to maximize or minimize the
# objective metric, in following cell.
# </font>
'''
In general, minimizing error and/or maximiizing similarity are desirable tuning directions.
Therefore, we will maximize our objective metric, projection coefficient, in this hyperparameter
tuning example.
'''


# In[ ]:


# definition of hyperparameter optimization
tuning_direction =
    "<For seller to update: Provide tuning direction for objective metric specified>"


# In[ ]:


'''
#### C. Run a model tuning job
'''


# In[ ]:


# <font color='red'>
# For seller to update: Review/update the tuner configuration including
# but not limited to `base_tuning_job_name`, `max_jobs`, and
# `max_parallel_jobs`.
# </font>

# setting up hyperparameter tuning job
tuner = HyperparameterTuner(
    estimator=estimator,
    base_tuning_job_name="<For Seller to update: Specify base job name>",
    objective_metric_name=objective_metric_name,
    objective_type=tuning_direction,
    hyperparameter_ranges=hyperparameter_ranges,
    max_jobs=50,
    max_parallel_jobs=7,
)


# In[ ]:


# <font color='red'>
# For seller to update: Uncomment following lines, specify appropriate
# channels, and run the tuner to test it out.
# </font>
'''
Uncomment following codes, specify appropriate
# channels, and run the tuner to test it out.
'''

# Uncomment following two lines to run Hyperparameter optimization job.
tuner.fit({'training':  data})
tuner.wait()

# <font color='red'>
# For seller to update: Once you have tested the code written in the
# preceding cell, comment three lines in the preceding cell so that
# customers who choose to simply run entire notebook do not end up
# triggering a tuning job.
# </font>


# In[ ]:


'''
As recommended by AWS Sagemaker Team, once you have completed a tuning job, (or even while the job is still running) you can
[clone and use this notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb)
to analyze the results to understand how each hyperparameter effects the quality of the model.
'''


# In[ ]:


'''
### 4: Deploy model and verify results
'''


# In[ ]:


# Now you can deploy the model for performing real-time inference.

# <font color='red'>
# For seller to update: Update appropriate values in following cell.
# </font>

model_name = "For Seller to update:<specify-model_or_endpoint-name>"

content_type = "For Seller to update:<specify_content_type_accepted_by_trained_model>"

real_time_inference_instance_type = (
    "For Seller to update:<Update recommended_real-time_inference instance_type>"
)
batch_transform_inference_instance_type = (
    "For Seller to update:<Update recommended_batch_transform_job_inference instance_type>"
)


# In[ ]:


'''
#### A. Deploy trained model
'''


# In[ ]:


predictor = estimator.deploy(
    1, real_time_inference_instance_type, serializer="<For seller to update>"
)

# Once endpoint is created, you can perform real-time inference.


# In[ ]:

    
'''
#### B. Create input payload
'''


# In[ ]:


# <font color='red'>
# For Seller to update: Add code snippet that reads the input from
# 'data/inference/input/real-time/' directory 
# and converts it into format expected by the endpoint in the following cell
# </font>
'''
# <Add code snippet that shows the payload contents>
'''


# In[ ]:


# <font color='red'>
# For Seller to update: Ensure that the `file_name` variable points
# to the payload you created. 
# Ensure that the `output_file_name` variable points to a file-name
# in which output of real-time inference needs to be stored
# </font>
'''
# <Add code snippet>
'''


# In[ ]:


'''
#### C. Perform real-time inference
'''


# In[ ]:


# <font color='red'>
# For Seller to update: review/update `file_name`, `output_file name`,
# and custom attributes in the following AWS CLI example to perform a
# real-time inference using the payload file you created from 2.B
# </font>

get_ipython().system(
    'aws sagemaker-runtime invoke-endpoint ' +
    '--endpoint-name $predictor.endpoint ' +
    '--body fileb://$file_name ' +
    '--content-type $content_type ' +
    '--region $sagemaker_session.boto_region_name ' +
    '$output_file_name')


# In[ ]:


'''
#### D. Visualize output
'''


# In[ ]:


# <font color='red'>
# For Seller to update: Write code in the following cell to display
# the output generated by real-time inference. This output must match with
# output available in data/inference/output/real-time folder.
# </font>
'''
# <Add code snippet>
'''


# In[ ]:


'''
#### E. Calculate relevant metrics
'''


# In[ ]:


# <font color='red'>
# For seller to update: write code to calculate metrics such as accuracy
# or any other metrics relevant to the business problem, using the test
# dataset. **This is highly recommended if your algorithm does not support
# and calculate metrics on test channel**. For information on how to
# configure metrics for your algorithm, see
# [Step 4 of this blog post](https://aws.amazon.com/blogs/machine-learning/easily-monitor-and-visualize-metrics-while-training-models-on-amazon-sagemaker/).
# </font>
'''
# <Add code snippet>
'''


# In[ ]:


# If [Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html) supports the type of problem you are trying to solve using this algorithm, use the following examples to add Model Monitor support to your product:
# For sample code to enable and monitor the model, see following notebooks:
# 1. [Enable Amazon SageMaker Model Monitor](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_model_monitor/enable_model_monitor/SageMaker-Enable-Model-Monitor.ipynb)
# 2. [Amazon SageMaker Model Monitor - visualizing monitoring results](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/sagemaker_model_monitor/visualization/SageMaker-Model-Monitor-Visualize.ipynb)


# In[ ]:


'''
#### F. Delete the endpoint
'''


# In[ ]:


# Now that you have successfully performed a real-time inference,
# you do not need the endpoint any more. you can terminate the same
# to avoid being charged.

predictor = sage.RealTimePredictor(model_name, sagemaker_session, content_type)
predictor.delete_endpoint(delete_endpoint_config=True)


# In[ ]:


# Since this is an experiment, you do not need to run a hyperparameter
# tuning job. However, if you would like to see how to tune a model
# trained using a third-party algorithm with Amazon SageMaker's
# hyperparameter tuning functionality, you can run the optional tuning step.


# In[ ]:


'''
### 6. Perform Batch inference
'''


# In[ ]:


# In this section, you will perform batch inference using multiple
# input payloads together.


# In[ ]:


# upload the batch-transform job input files to S3
transform_input_folder = "data/inference/input/batch"
transform_input = sagemaker_session.upload_data(transform_input_folder, key_prefix=model_name)
print("Transform input uploaded to " + transform_input)


# In[ ]:


# Run the batch-transform job
transformer = model.transformer(1, batch_transform_inference_instance_type)
transformer.transform(transform_input, content_type=content_type)
transformer.wait()


# In[ ]:


# output is available on following path
transformer.output_path


# In[ ]:


# <font color='red'>
# For Seller to update: Add code that displays output generated by the
# batch transform job available in S3.  This output must match the output
# available in data/inference/output/batch folder.
# </font>


# In[ ]:


'''
### 7. Clean-up
'''


# In[ ]:


'''
#### A. Delete the model
'''


# In[ ]:


predictor.delete_model()


# In[ ]:


'''
#### B. Unsubscribe to the listing (optional)
'''


# In[ ]:


# If you would like to unsubscribe to the algorithm, follow these steps.
# Before you cancel the subscription, ensure that you do not have any
# [deployable model](https://console.aws.amazon.com/sagemaker/home#/models)
# created from the model package or using the algorithm. Note - You can
# find this information by looking at the container name associated with
# the model. 
# 
# **Steps to unsubscribe to product from AWS Marketplace**:
# 1. Navigate to __Machine Learning__ tab on
#    [__Your Software subscriptions page__](https://aws.amazon.com/marketplace/ai/library?productType=ml&ref_=mlmp_gitdemo_indust)
# 2. Locate the listing that you want to cancel the subscription for, and
#    then choose __Cancel Subscription__  to cancel the subscription.
# 
# 


# In[ ]:


'''
Create the notebook in Amazon SageMaker Studio
https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks.html

How Are Amazon SageMaker Studio Notebooks Different from Notebook Instances?
https://docs.aws.amazon.com/sagemaker/latest/dg/notebooks-comparison.html
'''


''' --------------------------
End of demo_lmdfm_aws_cloud.py
-------------------------- '''

