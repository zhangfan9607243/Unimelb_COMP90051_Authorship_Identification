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

For modeling, we will try two approaches:

1. A FNN based solely on basic feature engineering.
2. A empty BERT model based on text understanding.

The performance is evaluated by F1 score of classification on testing dataset through Kaggle: https://www.kaggle.com/competitions/comp90051-22-s2-authorship.

## File Description

Before running the project please prepare the following paths:

```
authorship_identification/
│ 
├── data/
│   ├── data_original/
│   ├── data_processed/
│   └── data_result/
│
├── data_process/
│
├── method1/
│
└── method2/
```

The folders are organized as follows:

- `data/`: This folder contains all the data used in the project.

  - `data_original/`: This subfolder contains the original training and testing datasets provided for the project. The data can be downloaded from [Kaggle](https://www.kaggle.com/competitions/comp90051-22-s2-authorship) or [link](https://pan.baidu.com/s/1KVCrEwxMFnnOGytlbCVthw?pwd=cmar) and should be placed in this folder before running the project.
  - `data_processed/`: This subfolder is where we will save the processed datasets after performing data preprocessing and feature engineering.
  - `data_result/`: This subfolder is where we will save the final results of predictions.

- `data_process/`: This folder contains the code for data preprocessing.

- `method1/`: This folder contains the code for the first modeling approach, which is based on basic feature engineering and a feed-forward neural network (FNN).

- `method2/`: This folder contains the code for the second modeling approach, which is based on text understanding using an empty BERT model.

## Data Processing
To prepare our data for downstream feature engineering and modeling, we will perform the following data preprocessing steps:
1. Separate prolific authors from non-prolific authors in the training data by creating two columns: `authors_main` and `authors_coop` from the existing `authors` column. This is necessary since the testing data only contains a coauthors column with non-prolific authors.
2. Rename the list version of title and abstract to `list_title` and `list_abstract`, and merge them into a new column `list_combined`.
3. Create string formats for the title and abstract `str_title` and `str_abstract`, and merge them into a new column `str_combined`.
4. Fill any NA values in the venue column with `465`.

## Method 1: Basic Features + FNN
### 1.1 Basic Feature Engineering
First, we will implement some basic feature engineering methods without using machine learning techniques. The main goal is to establish writing history records, resulting in a total of 500 combined basic features.

#### (1) Coauthors
For coauthors, we create a graph showing the co-occurrence among authors, where nodes represent authors and edge weights represent the frequency of co-occurrence between two authors.

For each paper, we create a temporary array of size 100, with each position representing a prolific author. For each coauthor in the paper's coauthors list, we locate them in the graph, find their prolific author neighbors, and add the corresponding edge weight to the appropriate position in the temporary array. We repeat this for all coauthors of the paper.

Next, we consider deeper collaborative relationships by searching the graph using depth-first search (DFS). Whenever we encounter a prolific author, we add the edge `weight * (1 / (depth * log depth))` to the corresponding position in the temporary array. This process is applied to all coauthors of the paper.

Finally, this temporary array of size 100 will serve as a feature in this section.

#### (2) Venue
For the venue, we first establish a dictionary that records the frequency history of authors in relation to venues in the training dataset. For each paper, we locate its venue in the dictionary and transform the frequencies of the 100 prolific authors into an array of size 100, which will be used as a feature.

Additionally, we create another feature based on venue, considering the coauthor ties. For each coauthor of a given paper, we look at the author frequency of the venues where that coauthor has published and store this information in another array of size 100.

#### (3) Text
For the text (which combines the title and abstract), we start by training a TFIDF vectorizer to transform each text into TFIDF vectors, identifying the top 20 unique words. We then record the frequencies of these unique words in a dictionary. For each paper's text, we identify its top 20 unique words, find them in the dictionary, and convert the frequencies of 100 prolific authors into an array of size 100 to use as a feature.

Similarly, we create another version of the text-based feature, considering the coauthor ties, and store it as an array of size 100.

### 1.2 FNN Model

In this model, we construct a feed-forward neural network (FNN) based on basic features. The network consists of 3 hidden layers with 512, 256, and 128 nodes, respectively. The output activation function is sigmoid, which is suitable for multi-label classification. To prevent overfitting, we apply a dropout rate of 0.1 in the hidden layers and use L2 regularization on the parameters.

## Method 2: Textual Feature + Empty BERT Model

### 2.1 Textual Feature Engineering
In this method, we focus on textual features and utilize an empty BERT model for text understanding. Since the textual features are represented as sequences of word indices, this kind of feature is suitable for transformer-based models like BERT.

First, we append authors, venue, and year into the vocabulary:

- The texts (title and abstract) are represented as sequences of word indices, where each word is mapped to an index in the range {1,...,4999}.
- The year is represented as an index in the range {0,...,19}. We add 5000 to the year indices to append them to the vocabulary without conflicts.
- The venue is represented as an index in the range {0,...,464}. So, we add an additional 5020 to the venue indices to append them to the vocabulary without conflicts.
- The authors are represented as indices in the range {0,...,21245}. We add 5486 to the author indices to append them to the vocabulary without conflicts.

Then, we create a combined text feature by concatenating all the integer-formatted features into a single sequence by the following order: year, venue, coauthors, title, abstract. This combined feature will be used as input for the BERT model, with max sequence length set to 128.

### 2.2 BERT Model

In this model, we utilize an empty BERT model for text understanding. Since the input features are represented as sequences of word indices, we can directly feed them into the BERT model. The pretrained BERT model is not used because the input features are not in natural language format.

The BERT model is downloaded from Hugging Face and the parameters are initialized randomly. The downstream classification head consists of a single linear layer with sigmoid activation, which is suitable for multi-label classification.

## Final Results

The final performance of models are evaluated by the F1 scores on test dataset on Kaggle:

| Method   | Dataset    | Precision | Recall | F1 Score |
|----------|------------|-----------|--------|----------|
| Method 1 | Training   | 0.316     | 0.290  | 0.302    |
| Method 1 | Validation | 0.306     | 0.281  | 0.293    |
| Method 1 | Testing    | 0.305     | 0.280  | 0.292    |
| Method 2 | Training   | 0.316     | 0.290  | 0.302    |
| Method 2 | Validation | 0.306     | 0.281  | 0.293    |
| Method 2 | Testing    | 0.305     | 0.280  | 0.292    |
