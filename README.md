# Unimelb COMP90051 Authorship Identification

## Acknowledgement
I would like to extend my sincere gratitude to the Unimelb COMP90051 2022S2 teaching team for providing me with the opportunity to work on this project, as well as for their guidance and feedback on my work.

## Project Introduction

This project is a typical machine learning project that focuses on the task of authorship attribution, which involves identifying the author of a given article.

- The label (not included in testing data) is **Author ID** of prolific authors range from 0 to 99, and those without prolific author are marked -1.

- The features contains:

  - **Coauthors**: The **Author ID** of non-prolific authors range from 100 to 21245.
  - **Year**: The year the paper was published, measured in years from the start of the training period.
  - **Venue**: The venue id mapped to a unique integer value {0,...,464} or empty if there is no specified venue.
  - **Title**: The sequence of words in paper title, after light preprocessing, where each word has been mapped to an index in {1,...,4999}.
  - **Abstract**: The sequence of words in paper abstract, processed and mapped same as **Title**.

This is a multi-label classification task because a single article may have one or multiple authors. 

For modeling, we will try an FNN based on feature engineering. The performance is evaluated by F1 score of classification on testing dataset through Kaggle: https://www.kaggle.com/competitions/comp90051-22-s2-authorship.

## File Description

Before running the project please prepare the following paths:

```
authorship_identification/
│ 
├── data/
│   ├── data_original/
│   ├── data_processed/
│   ├── data_feature/
│   └── data_result/
│
├── data_process/
│
├── feature_engineering/
│
└── method/
```

The folders are organized as follows:

- `data/`: This folder contains all the data used in the project.

  - `data_original/`: This subfolder contains the original training and testing datasets provided for the project. The data can be downloaded from [Kaggle](https://www.kaggle.com/competitions/comp90051-22-s2-authorship) or [link](https://pan.baidu.com/s/1KVCrEwxMFnnOGytlbCVthw?pwd=cmar) and should be placed in this folder before running the project.
  - `data_processed/`: This subfolder is where we will save the processed datasets after performing data preprocessing.
  - `data_feature/`: This subfolder is where we will save the feature-engineered datasets after performing feature engineering.
  - `data_result/`: This subfolder is where we will save the results of predictions after performing modeling.

- `data_process/`: This folder contains the code for data preprocessing.

- `feature_engineering/`: This folder contains the code for feature engineering.

- `method/`: This folder contains the code for the modeling approaches.

## Data Processing
To prepare our data for downstream feature engineering and modeling, we will perform the following data preprocessing steps:
1. Separate prolific authors from non-prolific authors in the training data by creating two columns: `authors_main` and `authors_coop` from the existing `authors` column. This is necessary since the testing data only contains a coauthors column with non-prolific authors.
2. Rename the list version of title and abstract to `list_title` and `list_abstract`, and merge them into a new column `list_combined`.
3. Create string formats for the title and abstract `str_title` and `str_abstract`, and merge them into a new column `str_combined`.
4. Fill any NA values in the venue column with `465`.

## Feature Engineering
First, we will implement some basic feature engineering methods without using machine learning techniques. The main goal is to establish writing history records, resulting in a total of 500 combined basic features.

### (1) Coauthors
For coauthors, we create a graph showing the co-occurrence among authors, where nodes represent authors and edge weights represent the frequency of co-occurrence between two authors.

For each paper, we create a temporary array of size 100, with each position representing a prolific author. For each coauthor in the paper's coauthors list, we locate them in the graph, find their prolific author neighbors, and add the corresponding edge weight to the appropriate position in the temporary array. We repeat this for all coauthors of the paper.

Next, we consider deeper collaborative relationships by searching the graph using depth-first search (DFS). Whenever we encounter a prolific author, we add the edge `weight * (1 / (depth * log depth))` to the corresponding position in the temporary array. This process is applied to all coauthors of the paper.

Finally, this temporary array of size 100 will serve as a feature in this section.

### (2) Venue
For the venue, we first establish a dictionary that records the frequency history of authors in relation to venues in the training dataset. For each paper, we locate its venue in the dictionary and transform the frequencies of the 100 prolific authors into an array of size 100, which will be used as a feature.

Additionally, we create another feature based on venue, considering the coauthor ties. For each coauthor of a given paper, we look at the author frequency of the venues where that coauthor has published and store this information in another array of size 100.

### (3) Text
For the text (which combines the title and abstract), we start by training a TFIDF vectorizer to transform each text into TFIDF vectors, identifying the top 20 unique words. We then record the frequencies of these unique words in a dictionary. For each paper's text, we identify its top 20 unique words, find them in the dictionary, and convert the frequencies of 100 prolific authors into an array of size 100 to use as a feature.

Similarly, we create another version of the text-based feature, considering the coauthor ties, and store it as an array of size 100.

## FNN Model

In this model, we construct a feed-forward neural network (FNN) based on basic features. The network consists of 3 hidden layers with 512, 256, and 128 nodes, respectively. The output activation function is sigmoid, which is suitable for multi-label classification. To prevent overfitting, we apply a dropout rate of 0.1 in the hidden layers and use L2 regularization on the parameters.

## Final Results

The final performance of models are evaluated by the F1 scores on test dataset on Kaggle:

| Dataset    | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| Training   | 0.9986    | 0.9911 | 0.9948   |
| Validation | 0.8824    | 0.7057 | 0.7725   |
| Testing    | 0.5500    | 0.5077 | 0.5279   |
