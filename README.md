# Unimelb COMP90051 Authorship Identification

## Project Introduction

This project is a typical machine learning project that focuses on the task of authorship attribution, which involves identifying the author of a given article.

The original data are `/data/data1/train.json` and `/data/data1/test.json`, with each row represents an academic article, which contains the following information:

* The label (not included in testing data) is **Author ID** of prolific authors range from 0 to 99, and those without prolific author are marked -1.
* The features contains:
  * **Coauthors**: The **Author ID** of non-prolific authors range from 100 to 21245.
  * **Year**: The year the paper was published, measured in years from the start of the training period.
  * **Venue**: The venue id mapped to a unique integer value {0,...,464} or empty if there is no specified venue.
  * **Title**: The sequence of words in paper title, after light preprocessing, where each word has been mapped to an index in {1,...,4999}.
  * **Abstract**: The sequence of words in paper abstract, proceessed and mapped same as **Title**.

This is a multi-label classification task because a single article may have one or multiple authors. 

In this project, we will experiment with various feature engineering techniques and multiple models. First, the basic feature engineering approach will not involve any machine learning methods; it will instead be constructed based on authors' writing history. Then, we will also create embeddings for text-based data such as titles and abstracts using doc2vec and word2vec models.

For modeling, we will try three approaches:

1. A FNN based solely on basic feature engineering.
2. A FNN combining basic feature engineering with doc2vec embeddings.
3. An RNN model incorporating basic feature engineering and word2vec embeddings.

Additionally, for all three methods, we will apply a rule-based adjustment: if none of the five types of basic feature engineering matches any of the prolific authors, we will assume that the article has no prolific author. Finally, we will compare the performance of these three models to evaluate their effectiveness.

The performance is evaluated by F1 score of classification on testing dataset through Kaggle: https://www.kaggle.com/competitions/comp90051-22-s2-authorship.

## File Description

Before running the project please prepare the following paths:

```
/
|---data
    |---data1
    |---data2
    |---data3
|---model
    |---model_doc2vec
    |---model_word2vec
```

The files in this project includs:
* `data/`: The original data and processed features.
  * `data1/`: The original datasets, which can be downloaded from the Kaggle link.
    * `train.json`: The original training dataset.
    * `test.json`: The original testing dataset.
    * `sample.csv`: The format of final prediction result on testing dataset to be uploaded to Kaggle for final evaluation.
  * `data2/`: The processed features from feature engineering that are ready for the following models.
    * `data_tran.json`: The simple preprocessed training data after running `/data_process/Data Loading & Preprocessing.ipynb`.
    * `data_test.json`: The simple preprocessed testing data after running `/data_process/Data Loading & Preprocessing.ipynb`.
    * `y_tran.npy`: The one-hot transformation of label after running `/data_process/Data Loading & Preprocessing.ipynb`.
    * `x_tran_coauthors.npy`: The basic feature engineering of training data for coauthors after running `/feature_engineering_basic/Basic Feature 1 - Coauthors.ipynb`.
    * `x_test_coauthors.npy`: The basic feature engineering of testing data for coauthors after running `/feature_engineering_basic/Basic Feature 1 - Coauthors.ipynb`.
    * `x_tran_venue_a.npy`: The basic feature engineering of training data for venue after running `/feature_engineering_basic/Basic Feature 2 - Venue a.ipynb`.
    * `x_test_venue_a.npy`: The basic feature engineering of testing data for venue after running `/feature_engineering_basic/Basic Feature 2 - Venue a.ipynb`.
    * `x_tran_venue_b.npy`: The basic feature engineering of training data for venue considering authors relationships, after running `/feature_engineering_basic/Basic Feature 2 - Venue b.ipynb`.
    * `x_test_venue_b.npy`: The basic feature engineering of testing data for venue considering authors relationships, after running `/feature_engineering_basic/Basic Feature 2 - Venue b.ipynb`.
    * `x_tran_text_a.npy`: The basic feature engineering of training data for combined texts of titles and articles after running `/feature_engineering_basic/Basic Feature 3 - Text a.ipynb`.
    * `x_test_text_a.npy`: The basic feature engineering of testing data for combined texts of titles and articles after running `/feature_engineering_basic/Basic Feature 3 - Text a.ipynb`.
    * `x_tran_text_b.npy`: The basic feature engineering of training data for combined texts of titles and articles considering authors relationships after running `/feature_engineering_basic/Basic Feature 3 - Text b.ipynb`.
    * `x_test_text_b.npy`: The basic feature engineering of testing data for combined texts of titles and articles considering authors relationships after running `/feature_engineering_basic/Basic Feature 3 - Text b.ipynb`.
    * `x_tran_abstract_doc2vec.npy`: The doc2vec embeddings of abstract on training data, after runing `/feature_engineering_text/Textual Feature 1 - Doc2Vec.ipynb`.
    * `x_test_abstract_doc2vec.npy`: The doc2vec embeddings of abstract on testing data, after runing `/feature_engineering_text/Textual Feature 1 - Doc2Vec.ipynb`.
    * `x_tran_title_doc2vec.npy`: The doc2vec embeddings of title on training data, after runing `/feature_engineering_text/Textual Feature 1 - Doc2Vec.ipynb`.
    * `x_test_title_doc2vec.npy`: The doc2vec embeddings of title on tesing data, after runing `/feature_engineering_text/Textual Feature 1 - Doc2Vec.ipynb`.
    * `x_tran_abstract_word_vectors.json`: The map of word embeddings trained by abstract on training data, after runing `/feature_engineering_text/Textual Feature 2 - Word2Vec.ipynb`.
    * `x_tran_title_word_vectors.json`: The map of word embeddings trained by title on training data, after runing `/feature_engineering_text/Textual Feature 2 - Word2Vec.ipynb`.
* `model/`: The embedding models for text features.
  * `model_doc2vec/`:
    * `model_doc2vec_title.bin`: The Doc2Vec model for title.
    * `model_doc2vec_abstract.bin`: The Doc2Vec model for abstract.
  * `model_word2vec/`:
    * `model_word2vec_title.bin`: The Word2Vec model for title.
    * `model_word2vec_abstract.bin`: The Word2Vec model for abstract.
* `data_process/Data Loading & Preprocessing.ipynb`: Load and preprocess data and make it ready for feature engineering.
* `feature_engineering_basic/`: The basic feature engineering based on writing history without machine learning techniques.
  * `Basic Feature 1 - Coauthors.ipynb`: The basic feature engineering for coauthors based on co-occurance history.
  * `Basic Feature 2 - Venue a.ipynb`: The basic feature engineering for venue based on writing history.
  * `Basic Feature 2 - Venue b.ipynb`: The basic feature engineering for venue based on writing history and coauthors ties.
  * `Basic Feature 3 - Text a.ipynb`: The basic feature engineering for text based on writing history.
  * `Basic Feature 3 - Text b.ipynb`: The basic feature engineering for text based on writing history and coauthors ties.
* `feature_engineering_text/`: The models for contextual embeddings of title & abstract.
  * `Textual Feature 1 - Doc2Vec.ipynb`: Train the doc2vec model for title & abstract and generate document embeddings for training & testing data.
  * `Textual Feature 2 - Word2Vec.ipynb`: Train the word2vec mode for title & abstract and save the word embedding maps as json files.
* `methods/`: The models for the main multi-label classification task.
  * `Method 1 - Basic Features + FNN.ipynb`: FNN based solely on basic feature engineering.
  * `Method 2 - Basic + Doc2Vec + FNN.ipynb`: FNN combining basic feature engineering with doc2vec embeddings.
  * `Method 3 - Basic + Word2Vec + LSTM.ipynb`: RNN model incorporating basic feature engineering and word2vec embeddings.

## Data Processing
In order to make our data ready for the downstream feature engineering and models, we make the following data preprocessing steps:
1. Seperate prolific authors & non-prolific authors in training data by creating two columns `author` and `coauthors` from existing column `authors`, since we only have a `coauthors` column in testing data with non-prolific authors only.
2. Create string format of title and abstract, namely `title_text` and `abstract_text`, and merge them into a new single column, namely `text`.
3. Fill NA in venue with 465.
4. Reduce the training data size, since there are too many papers in training data without non-prolific authors, which is about 18,000, and we keep only 1000 of them.

## Feature Engineering
### 1. Basic Feature Engineering
First we will try some basic feature engineering methods without machine learning techniques. The main idea is to establish writing history records.

#### (1) Coauthors
For coauthors, we built a graph showing the co-occurance among authors, with nodes representing authors, and edges weights representing co-occurance frequency between two authors. 

Then, for each paper, we create a temporary array with size 100, with each position represent a prolific author. For each coauthor in coauthors list of this paper, we locate this coauthor in the graph, and find its prolific authors neighbours, and add the edge weight on to the position of the temporary array. We do this for all the coauthors of this paper.

Then, we consider deeper collaborative relationships. We search the graph by DFS, and whenever we encounter a prolific author, we add the edge weight * (1 / (depth * log depth)) at the corresponding position in the temporary array. We do this for all the coauthors of this paper.

Finally, this temporary array with size 100 will be the feature in this part.

#### (2) Venue
For venue, we firstly establish a venue dictionary that record the venue - authors frequency history in training dataset. Then, similarly, and for each paper, we locate its venue in the dictionary, and transform its frequencies of 100 prolific authors into an array with size 100, and use it as a feature.

On the other hand, we also create another feature based on venue, considering the coauthors ties. For each coauthor of a given paper, we also need to consider the author frequency of the venues where this co-author has published. Also, we store it as an array with size 100.

#### (3) Text

### 2. Textual Feature Engineering
#### (1) Doc2Vec

#### (2) Word2Vec

## Models




## Final Results
