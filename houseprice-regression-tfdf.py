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
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this if the data visualisations doesn't work on your side
%matplotlib inline

# %%
print("TensorFlow v" + tf.__version__)
print("TensorFlow Decision Forests v" + tfdf.__version__)

# %% [markdown]
# ## Load the dataset
#

# %%
train_file_path = "../input/house-prices-advanced-regression-techniques/train.csv"
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
sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})

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

# %%
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    train_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    valid_ds_pd, label=label, task=tfdf.keras.Task.REGRESSION)

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

# %%
tfdf.keras.get_all_models()

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

# %%
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])  # Optional, you can use this to include a list of eval metrics

# %% [markdown]
# ## Train the model
#
# We will train the model using a one-liner.
#
# Note: you may see a warning about Autograph. You can safely ignore this,
# it will be fixed in the next release.

# %%
rf.fit(x=train_ds)

# %% [markdown]
# ## Visualize the model
# One benefit of tree-based models is that you can easily visualize them.
# The default number of trees used in the Random Forests is 300. We can
# select a tree to display below.

# %%
tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)

# %% [markdown]
# ## Evaluate the model on the Out of bag (OOB) data and the validation dataset
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

# %%
logs = rf.make_inspector().training_logs()
plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
plt.xlabel("Number of trees")
plt.ylabel("RMSE (out-of-bag)")
plt.show()

# %% [markdown]
# We can also see some general stats on the OOB dataset:

# %%
inspector = rf.make_inspector()
inspector.evaluation()

# %% [markdown]
# Now, let us run an evaluation using the validation dataset.

# %%
evaluation = rf.evaluate(x=valid_ds, return_dict=True)

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")

# %% [markdown]
# ## Variable importances
#
# Variable importances generally indicate how much a feature contributes to the model predictions or quality. There are several ways to identify important features using TensorFlow Decision Forests.
# Let us list the available `Variable Importances` for Decision Trees:

# %%
print(f"Available variable importances:")
for importance in inspector.variable_importances().keys():
    print("\t", importance)

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
