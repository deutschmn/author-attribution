# Author Attribution – DerStandard Forum Writing Style

a.k.a. *authorship attribution* or *author identification*.

## Problem Setting (Patrick)
- *consider formal notation* (maybe see other author identification papers?)

## Related Work (Patrick)
### Papers
- https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1174/reports/2760185.pdf
- https://www.aclweb.org/anthology/C18-1029.pdf
  
    - > We find that the most effective features for datasets can be predicted by applying topic modeling and feature analysis. Content-based features tend to be suitable for datasets with high topical diversity such as the one constructed from on-line news. Datasets with less topical variance, e.g. legal judgments and movie reviews, benefit more from style- based features.

### Useful links
- https://towardsdatascience.com/a-machine-learning-approach-to-author-identification-of-horror-novels-from-text-snippets-3f1ef5dba634
- https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
- https://www.kaggle.com/christopher22/stylometry-identify-authors-by-sentence-structure

## Setup (Patrick)
- two different "branches" of the network
- Keras
- etc.

## Features (Lukas)

### Post Specific 

* CreatedAt time
* Number of Pos/Neg votes 
* Exist Parent Post
  * is always just responding or does create own posts
* Parent Post User
  * always responds to same user
  * parent votes pos/neg (maybe root post)
* Named Entities from Articles/Posts (Lukas)

### Stylometric

<img src="assets/image-20200606082356512.png" alt="image-20200606082356512" style="zoom:67%;" />

* Same features for title
* Total length of post/title

## Evaluation (Lukas)

### Baseline

### Results

### Interpretation

## Discussion

>  *The discussion section should provide the key insights and the context (i.e., not only list what worked (or not), but also provide possible reasons why it worked (or not), or when it should work)*

- feature selection
  - date stats (a bit surprisingly) not very helpful
  - content very helpful (embeddings, article entities)
  - style also quite helpful
    - even though we only used rather simple features
    - → more advanced are future work

- data preparation
  - one-hot-encoding of categories essential
  - normalisation of features very important (timestamp for post date was very hard to train)
  - embeddings for post content work well
    - any issues?
- Network architecture

  - RNN for post embeddings
    - best type: GRU; LSTM harder to train
  - dense layer after concatenation is important
    - but more than one don't really make a difference
- Network hyper params

  - early stopping very useful
  - dropout absolutely essential (overfitting otherwise)
  - extensive search necessary

## Conclusion

* good results for given task
* hard to find a good baseline to compare to
* semantics that are considered in the post contribute greatly to results (style and text itself wouldn't yield such high rates)