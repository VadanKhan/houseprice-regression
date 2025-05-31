# %% [markdown]
# # House Prices Prediction using TensorFlow Decision Forests

# %% [markdown]
# This notebook walks you through how to train a baseline Random Forest model using TensorFlow Decision Forests on the House Prices dataset made available for this competition.
#
# Roughly, the code will look as follows:
#
# ```
# import tensorflow_decision_forests as tfdf
# import pandas as pd
#
# dataset = pd.read_csv("project/dataset.csv")
# tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(dataset, label="my_label")
#
# model = tfdf.keras.RandomForestModel()
# model.fit(tf_dataset)
#
# print(model.summary())
# ```
#
# Decision Forests are a family of tree-based models including Random
# Forests and Gradient Boosted Trees. They are the best place to start
# when working with tabular data, and will often outperform (or provide a
# strong baseline) before you begin experimenting with neural networks.

# %% [markdown]
# ## Import the library

# %%
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ydf
print("ydf version:", ydf.__version__)
print(dir(ydf))

# %%

print("pandas version:", pd.__version__)
print("seaborn version:", sns.__version__)
print("matplotlib version:", matplotlib.__version__)


# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

# %% [markdown]
# ## Load the dataset
#

# %%
# Get the path to the current notebook (only works reliably in scripts or with Jupyter hacks)
current_path = Path().resolve()
ROOT_DIR = current_path.parents[1]  # Up 3 levels

# Example: Construct path to training data in ROOT_DIR/data/train.csv
train_file_path = ROOT_DIR / "data" / "train.csv"

print("Root directory:", ROOT_DIR)
print("Train file path:", train_file_path)


# %%
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

# %% [markdown]
# The data is composed of 81 columns and 1460 entries. We can see all 81
# dimensions of our dataset by printing out the first 3 entries using the
# following code:

# %%
dataset_df.head(3)

# %% [markdown]
# * There are 79 feature columns. Using these features your model has to predict the house sale price indicated by the label column named `SalePrice`.

# %% [markdown]
# We will drop the `Id` column as it is not necessary for model training.

# %%
dataset_df = dataset_df.drop('Id', axis=1)
dataset_df.head(3)

# %% [markdown]
# We can inspect the types of feature columns using the following code:

# %%
dataset_df.info()

# %% [markdown]
# ## House Price Distribution
#
# Now let us take a look at how the house prices are distributed.

# %%
print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.histplot(dataset_df['SalePrice'], color='g', bins=100, kde=True)

# %% [markdown]
# ## Numerical data distribution
#
# We will now take a look at how the numerical features are distributed.
# In order to do this, let us first list all the types of data from our
# dataset and select only the numerical ones.

# %%
list(set(dataset_df.dtypes.tolist()))

# %%
df_num = dataset_df.select_dtypes(include=['float64', 'int64'])
df_num.head()

# %% [markdown]
# Now let us plot the distribution for all the numerical features.

# %%
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

# %% [markdown]
# ## Prepare the dataset
#
# This dataset contains a mix of numeric, categorical and missing
# features. TF-DF supports all these feature types natively, and no
# preprocessing is required. This is one advantage of tree-based models,
# making them a great entry point to Tensorflow and ML.

# %% [markdown]
# Now let us split the dataset into training and testing datasets:

# %%


def split_dataset(dataset, test_ratio=0.30):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]


train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

# %% [markdown]
# There's one more step required before we can train the model. We need to convert the datatset from Pandas format (`pd.DataFrame`) into TensorFlow Datasets format (`tf.data.Dataset`).
#
# [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) is a high performance data loading library which is helpful when training neural networks with accelerators like GPUs and TPUs.

# %% [markdown]
# By default the Random Forest Model is configured to train classification
# tasks. Since this is a regression problem, we will specify the type of
# the task (`tfdf.keras.Task.REGRESSION`) as a parameter here.

# %% [markdown]
# **🔁 YDF equivalent**
#
# YDF (yggdrasil_decision_forests) works directly with Pandas DataFrames — no need to convert to tf.data.Dataset.
#
# You just pass the DataFrame directly to the model’s fit() method, and specify the label column as a parameter.
#
# So, instead of converting the data here, we just store the label name for later and proceed.

# %%
label = 'SalePrice'
# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
#     train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
# valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
#     valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

# %% [markdown]
# ## Select a Model
#
# There are several tree-based models for you to choose from.
#
# * RandomForestModel
# * GradientBoostedTreesModel
# * CartModel
# * DistributedGradientBoostedTreesModel
#
# To start, we'll work with a Random Forest. This is the most well-known of the Decision Forest training algorithms.
#
# A Random Forest is a collection of decision trees, each trained
# independently on a random subset of the training dataset (sampled with
# replacement). The algorithm is unique in that it is robust to
# overfitting, and easy to use.

# %% [markdown]
# We can list the all the available models in TensorFlow Decision Forests using the following code:

# %% [markdown]
# **YDF Doesn't really have a port**
#
# ✅ Why?
# YDF doesn’t use a unified registry like TF-DF does. Instead, you directly instantiate the model class you want from the public API:
#
# python
# Copy
# Edit
# ydf.RandomForestRegressor()
# ydf.GradientBoostedTreesRegressor()
# ydf.CartRegressor()
# (Or the Classifier versions for classification tasks.)
#
# 📘 Current available models in YDF:
# ydf.RandomForestRegressor()
#
# ydf.RandomForestClassifier()
#
# ydf.GradientBoostedTreesRegressor()
#
# ydf.GradientBoostedTreesClassifier()
#
# ydf.CartRegressor()
#
# ydf.CartClassifier()
#
# These are part of the documented public API, and are not dynamically
# discoverable via a method like get_all_models().

# %%
# tfdf.keras.get_all_models()

print("Available YDF models:")
print(" - ydf.RandomForestRegressor")
print(" - ydf.GradientBoostedTreesRegressor")
print(" - ydf.CartRegressor")
print(" - ...and their Classifier equivalents")

# %% [markdown]
# ## How can I configure them?
#
# TensorFlow Decision Forests provides good defaults for you (e.g. the top ranking hyperparameters on our benchmarks, slightly modified to run in reasonable time). If you would like to configure the learning algorithm, you will find many options you can explore to get the highest possible accuracy.
#
# You can select a template and/or set parameters as follows:
#
# ```rf = tfdf.keras.RandomForestModel(hyperparameter_template="benchmark_rank1", task=tfdf.keras.Task.REGRESSION)```
#
# Read more
# [here](https://www.tensorflow.org/decision_forests/api_docs/python/tfdf/keras/RandomForestModel).

# %% [markdown]
#
#

# %% [markdown]
# ## Create a Random Forest
#
# Today, we will use the defaults to create the Random Forest Model while
# specifiyng the task type as `tfdf.keras.Task.REGRESSION`.

# %% [markdown]
# **✅ YDF Equivalent**
#
# In YDF, the API is non-Keras and simpler. You instantiate a model from ydf.RandomForestRegressor() for regression problems.
#
# You do not call compile(), and you set the label during .fit().

# %%
# rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
# rf.compile(metrics=["mse"]) # Optional, you can use this to include a list of eval metrics
learner = ydf.RandomForestLearner(
    label=label,
    task=ydf.Task.REGRESSION,
    num_trees=300,  # Default is 300, same as TFDF
    max_depth=16,   # Optional: control tree depth
    min_examples=5  # Optional: minimum examples per node
)

# %% [markdown]
# ## Train the model
#
# We will train the model using a one-liner.
#
# Note: you may see a warning about Autograph. You can safely ignore this,
# it will be fixed in the next release.

# %% [markdown]
# **✅ YDF Equivalent**
#
# You train using the Pandas DataFrame directly, passing the label column name:

# %%
# rf.fit(x=train_ds)

model = learner.train(train_ds_pd)


# %% [markdown]
# ## Visualize the model (Not in YDF)
# One benefit of tree-based models is that you can easily visualize them.
# The default number of trees used in the Random Forests is 300. We can
# select a tree to display below.

# %%
# YDF doesn't have built-in tree plotting in 0.8.0, but we can:
# Only working option: Get model summary
print(model)

model.describe()

# model.print_tree()

# %% [markdown]
# ## Evaluate on the validation dataset
#
# Before training the dataset we have manually seperated 20% of the dataset for validation named as `valid_ds`.
#
# We can also use Out of bag (OOB) score to validate our RandomForestModel.
# To train a Random Forest Model, a set of random samples from training set are choosen by the algorithm and the rest of the samples are used to finetune the model.The subset of data that is not chosen is known as Out of bag data (OOB).
# OOB score is computed on the OOB data.
#
# Read more about OOB data [here](https://developers.google.com/machine-learning/decision-forests/out-of-bag).
#
# The training logs show the Root Mean Squared Error (RMSE) evaluated on the out-of-bag dataset according to the number of trees in the model. Let us plot this.
#
# Note: Smaller values are better for this hyperparameter.
#
# **✅ YDF Equivalent**
#
# Ignore all of the OOB stuff and just jump to validation dataset

# %% [markdown]
# Now, let us run an evaluation using the validation dataset.

# %%
predictions = model.predict(valid_ds_pd)
evaluation = model.evaluate(valid_ds_pd)
analysis = model.analyze(valid_ds_pd)

evaluation
# analysis
# for name, value in evaluation.items():
#   print(f"{name}: {value:.4f}")

# %% [markdown]
# ## Variable importances
#
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests.
# Let us list the available `Variable Importances` for Decision Trees:

# %%
# can simply get it from analysis in ydf

analysis

# %% [markdown]
# As an example, let us display the important features for the Variable Importance `NUM_AS_ROOT`.
#
# The larger the importance score for `NUM_AS_ROOT`, the more impact it has on the outcome of the model.
#
# By default, the list is sorted from the most important to the least.
# From the output you can infer that the feature at the top of the list is
# used as the root node in most number of trees in the random forest than
# any other feature.

# %%
inspector.variable_importances()["NUM_AS_ROOT"]

# %% [markdown]
# Plot the variable importances from the inspector using Matplotlib

# %%
plt.figure(figsize=(12, 4))

# Mean decrease in AUC of the class 1 vs the others.
variable_importance_metric = "NUM_AS_ROOT"
variable_importances = inspector.variable_importances()[variable_importance_metric]

# Extract the feature name and importance values.
#
# `variable_importances` is a list of <feature, importance> tuples.
feature_names = [vi[0].name for vi in variable_importances]
feature_importances = [vi[1] for vi in variable_importances]
# The feature are ordered in decreasing importance value.
feature_ranks = range(len(feature_names))

bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
plt.yticks(feature_ranks, feature_names)
plt.gca().invert_yaxis()

# TODO: Replace with "plt.bar_label()" when available.
# Label each bar with values
for importance, patch in zip(feature_importances, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

plt.xlabel(variable_importance_metric)
plt.title("NUM AS ROOT of the class 1 vs the others")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Submission
# Finally predict on the competition test data using the model.

# %%
test_file_path = "../input/house-prices-advanced-regression-techniques/test.csv"
test_data = pd.read_csv(test_file_path)
ids = test_data.pop('Id')

test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    test_data,
    task=tfdf.keras.Task.REGRESSION)

preds = rf.predict(test_ds)
output = pd.DataFrame({'Id': ids,
                       'SalePrice': preds.squeeze()})

output.head()


# %%
sample_submission_df = pd.read_csv(
    '../input/house-prices-advanced-regression-techniques/sample_submission.csv')
sample_submission_df['SalePrice'] = rf.predict(test_ds)
sample_submission_df.to_csv('/kaggle/working/submission.csv', index=False)
sample_submission_df.head()
