# Unimelb COMP90051 Authorship Identification

## Acknowledgement
I would like to extend my sincere gratitude to the Unimelb COMP90051 2022S2 teaching team for providing me with the opportunity to work on this project, as well as for their guidance and feedback on my work.

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
3. An LSTM model incorporating basic feature engineering and word2vec embeddings.

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
  * `Method 3 - Basic + Word2Vec + LSTM.ipynb`: LSTM model incorporating basic feature engineering and word2vec embeddings.

## Data Processing
To prepare our data for downstream feature engineering and modeling, we will perform the following data preprocessing steps:
1. Separate prolific authors from non-prolific authors in the training data by creating two columns: author and coauthors from the existing authors column. This is necessary since the testing data only contains a coauthors column with non-prolific authors.
2. Create string formats for the title and abstract, named title_text and abstract_text, and merge them into a new column called text.
3. Fill any NA values in the venue column with 465.
4. Reduce the training data size, as there are approximately 18,000 papers in the training set without non-prolific authors. We will retain only 1,000 of these papers.

## Feature Engineering
### 1. Basic Feature Engineering
First, we will implement some basic feature engineering methods without using machine learning techniques. The main goal is to establish writing history records, resulting in a total of 500 combined basic features.

#### (1) Coauthors
For coauthors, we create a graph showing the co-occurrence among authors, where nodes represent authors and edge weights represent the frequency of co-occurrence between two authors.

For each paper, we create a temporary array of size 100, with each position representing a prolific author. For each coauthor in the paper's coauthors list, we locate them in the graph, find their prolific author neighbors, and add the corresponding edge weight to the appropriate position in the temporary array. We repeat this for all coauthors of the paper.

Next, we consider deeper collaborative relationships by searching the graph using depth-first search (DFS). Whenever we encounter a prolific author, we add the edge weight * (1 / (depth * log depth)) to the corresponding position in the temporary array. This process is applied to all coauthors of the paper.

Finally, this temporary array of size 100 will serve as a feature in this section.

#### (2) Venue
For the venue, we first establish a dictionary that records the frequency history of authors in relation to venues in the training dataset. For each paper, we locate its venue in the dictionary and transform the frequencies of the 100 prolific authors into an array of size 100, which will be used as a feature.

Additionally, we create another feature based on venue, considering the coauthor ties. For each coauthor of a given paper, we look at the author frequency of the venues where that coauthor has published and store this information in another array of size 100.

#### (3) Text
For the text (which combines the title and abstract), we start by training a TFIDF vectorizer to transform each text into TFIDF vectors, identifying the top 20 unique words. We then record the frequencies of these unique words in a dictionary. For each paper's text, we identify its top 20 unique words, find them in the dictionary, and convert the frequencies of 100 prolific authors into an array of size 100 to use as a feature.

Similarly, we create another version of the text-based feature, considering the coauthor ties, and store it as an array of size 100.

### 2. Textual Feature Engineering
#### (1) Doc2Vec
The Doc2Vec model generates embeddings for entire documents. In this case, we train two separate Doc2Vec models using the titles and abstracts in the training data, respectively. We then use these models to produce embeddings for both titles and abstracts in the training and testing datasets.

#### (2) Word2Vec
The Word2Vec model is utilized to create word embeddings. Similarly, we train two Word2Vec models using the titles and abstracts in the training data, and we store the resulting word embeddings in JSON files. The text matrix for titles and abstracts, which consists of a sequence of embeddings for the individual words, will be generated within the dataset and data loader during model training and prediction.

## Models
### 1. FNN + Basic Features
In this model, we construct a feed-forward neural network (FNN) based solely on basic features. The network consists of 3 hidden layers with 512, 256, and 128 nodes, respectively. The output activation function is sigmoid, which is suitable for multi-label classification. To prevent overfitting, we apply a dropout rate of 0.1 in the hidden layers and use L2 regularization on the parameters.

### 2. FNN + Basic Features & Doc2Vec Features
In this model, we also construct a feed-forward neural network (FNN). The input features consist of the concatenated basic features and the Doc2Vec features for the title and abstract. We maintain the same model settings as Model 1, but this time we use a new FNN architecture with four hidden layers containing 1024, 512, 256, and 128 nodes, respectively.

### 3. LSTM + Basic Features & Word2Vec Features
In this model, we first construct two LSTM models for the title and abstract text matrices, which consist of sequences of word embeddings for their constituent words. We then concatenate the outputs of the two LSTM models with the basic features and feed them into an FNN classifier. The other model settings are similar to those in Model 1 and Model 2.

## Final Results
The final performance of models are evaluated by the F1 scores on test dataset on Kaggle:
* Method 1: 0.5279 (Final Score with Rank 19/187)
* Method 2: 0.4810
* Method 3: 0.4580

Finally, Method 1 achieved the best prediction performance, indicating that the basic features derived from writing history contain sufficient information for predicting the authors of a given paper.
