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

## Data Processing




## Feature Engineering




## Models




## Final Results
