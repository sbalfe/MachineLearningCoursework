# Practical 4 - NLP

## Supervised approach

- Approach **topic analysis** via *supervised machine learning*  
  - As for usual with **supervised ML tasks** there are *several key components* to consider:
    1. *Data* **labelled** with the **classes of interest**.
    2. *Algorithm* to apply to **this** *multi class classification* task.
    3. *Evaluation strategy* that helps you **check** that your **approach** *that works well*.
- Since, **within** the *supervised* ML *setting* 
  - You are **applying** *classification* to **distinguish** between *various topics*
    - Call this **==topic classification==**

---

### Data

- For **supervised ML** scenarios $\to$ *high quality data* that is labelled with **classes of interest** are of *utmost importance*
  - For this *practical* $\to$ we use the **famous 20 newsgroup dataset**
    - That is **well suited** for the *topic classification task* and is **easily accessible** via `sklearn`.

---

- The **20 newsground dataset** has **18,000 newsgroup** *posts* on **20 topics**
  - The *dataset* $\to$ has **been widely** used in the **NLP community** for various task involving **text classification** / **topic classification** 
    - For the *ease of comparison* $\to$ the **dataset** is *already split* into **training / test** subsets.

---

```python
from sklearn.datasets import fetch_20newsgroups
import numpy as np
```

- Define `load_dataset` 
  - This **initialises** a **subset** of data as the *train / test* chunk of the **20 newsgroups** dataset
  - Enables us to **select particular categories** listed under `cats` 
- We **shuffle** this dataset and remove extraneous (irrelevant) information such as **headers / footers / quotes**

```python
def load_dataset(a_set, cats):
    dataset = fetch_20newsgroups(subset= a_set, 
                                 categories= cats ,
                                 remove=('headers', 'footers', 'quotes'),
                                 shuffle= True
                                )
    return dataset
```

- This code **enables** you to *specify* a list of **categories** of *interest as input* 

  - The **list** of *10 topics* below is used as **an example** (*i.e.* you can **experiment** with a **different selection later**)
- Finally we **initialise** the `newsgroup_train` and `newsgroup_test` subsets
  - If we use `"all"`  instead of `"train" / "test"` 
    - You obtain **access** to the *full 20 newsgroup dataset* 
  - Use `None` instead of **categories** will *help will help you access* **all topics**

![image-20220209043504404](D:\University\Notes\DiscreteMaths\Resources\image-20220209043504404.png)

```python
categories = ["comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball"]
categories += ["rec.sport.hockey", "sci.crypt", "sci.med", "sci.space", "talk.politics.mideast"]

newsgroups_train = load_dataset("train", categories)
newsgroups_test = load_dataset("test", categories)
```

- Obtain **5913** *training posts*
- Obtain **3937** *training posts* 

---

- The code below shows how to **check** what **data** got **uploaded** and **how many posts** are **included** in **each** **subset**. 

  - In this code, you **should** first **check** what **categories** are **uploaded** using `target_names` field – this list **should** **coincide** with the **one** that you **defined** in `categories` earlier.

  -  Then, you **check** the **number** of **posts** (`filenames` field) and the **number** of **labels** assigned to them (`target` field) and **confirm** that the **two** **numbers** are the **same**.

  -  The `filenames` field **stores** **file** **locations** for the **posts** on your **computer**: 

    - for example, you can **access** the **very** **first** one via `filenames[0]`. 

  - The `data` field **stores** **file** contents for the **posts** in the **dataset**:

    - for example, you can **access** the very **first** one via `data[0]`. 

- As a final **sanity** check, you can also print out **category** **labels** for the first 10 posts from the dataset using `target[:10]`:

```python
def check_data(dataset):
    print(list(dataset.target_names))
    print(dataset.filenames.shape)
    print(len(dataset.target))
    if dataset.filenames.shape[0]==dataset.target.shape[0]:
        print("Equal sizes for data and targets")
    print(dataset.filenames[0])
    print(dataset.data[0])
    print(dataset.target[:10])
```

```python
check_data(newsgroups_train)
print("\n***\n")
check_data(newsgroups_test)
```

- First line should **confirm** the **categories** have *been loaded correctly*
- Number of **posts in training data** to $5913$ , test $3937$
- `dataset.filenames` returns a **list** 
- `dataset.target` returns **array** 

---

- `sklearn` allows you **to not only access** the **dataset**
  - Also *represents* some **object** with **relevant attributes** that may *be directly accessed* via `dataset.attribute` 
    - `target_names` returns the **list** of the **names** for the **target** **classes** (categories);
    - `filenames` is the list of **paths** where the **files** are stored on your **computer**;
    - `target` **returns** an **array** with the **target** **labels** (note that the **category** **names** are cast to the **numerical** format);
    - `data` **returns** the **list** of the **contents** of the **posts**.

- The list of **targets** represents *categories numerically*
  - This is as basic **machine learning classifiers** implemented in `sklearn` prefer to work in **numerical format** for the **labels**
- Numbers are **assigned** to *categories* in an *alphabetical order*
  - For example $\to$ `comp.windows.x` corresponds to *numerical label* $0$ , `misc.foresale` $\to$ $1$.
- Output such as $[4,3,9,7,4,3,0,5,7,8]$ tells that **the posts** on *different topics* are **shuffled**
  - The *first one* is on `rec.sport.baseball` 
  - The *second* is on `rec.motorcycles` 

### Feature Selection

- Next, let's create **word** **vectors** based on the **content** of the **posts**.

-  First of all, note that, **compared** to the **previous** **applications**, we’ve made the **detection** task **more** **complex**. 

- We are considering **10** **topics** and a **vast** **range** of words (all but [stopwords](https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words), i.e., very frequent words that are often uninformative for prediction tasks) occurring in newsgroups posts. 

- Even **after** **stopwords** **removal**, many of the **remaining** **words** will **occur** not in a **single** **topic** but rather **across** **lots** of posts on various topics.

  - **Consider** the **word** “*post*” itself as one **example** of such **frequent** and **widely** **spread** **word**:

  -  it might mean a **new** ***post*** that **someone** has **got** and, as **such**, might be **more** **relevant** to the **texts** on **politics**; at the same time, you **will** also **see** it **frequently** used in **contexts** like *“I have posted the logos of the NL East teams to …”.*
  -  That is, **despite** the word “***post***” not being a stopword, it is **quite** **similar** to **stopwords** in **nature** – it might be used **frequently** across **many** texts on **multiple** topics, and **thus** **lose** its **value** for the **task**. 
  - How can you **make** sure that **words** that **occur** **frequently** in the **data** are **given** **less** **weight** than **more** **meaningful** **words** that **occur** **frequently** in a **restricted** set of **texts** (e.g. restricted by a topic)?

---

- We apply a **technique** to *allow* us to **downweigh terms** that occur **more frequently** across many documents and *upvalue* terms that **occur frequently** only *in some documents*, but **not across** the **whole collection**

  - This *technique* is *called* $TF-IDF$ for **==Term Frequency - Inverse Document Frequency==**

    1. Ensure **contribution** of some word is *not affected* by the **document length** 

       - 100 $\to$ car said **2 times**.
       - 200 $\to$ car said **4 times**.
       - The second one is **not more prevalent**
         - Calculate its $TF$ $\to$ $TF(\text{“car”})=4/200=2/100=0.02$, thus **they are the same**
           - This is **==term frequency==**.

    2. You *would also* **like** to *ensure* the **word contribution** is *measured against* **specificity**

       -  See `post` in nearly **every text** $\to$ its *contribution* must be *lower* than `car` 

         - This is **what** the **==inverse document frequency (idf)==** allows us to **take into account**.

           - If some word **post** is used in **80 posts** out **100**.
           - If Some word **car** is used in **25 posts** out **100**.

           - $IDF(\text{“post”})=100/80=1.25 < IDF(\text{“car”})=100/25=4$ 
           - Thus **car has more weight**

    3. Finally placing **these two information together** $\to$ $TF-IDF=tf*idf$ 

       - This **gives** *higher weights* to *words*  that are **used frequently** within some documents but **not across a wide variety of documents**.
         - This is **very useful for our task**.


- Apply in `sklearn` 

- You use ***vectorizers*** – **functions** that are **capable** of **counting** **word** **occurrences** in texts and then **presenting** **each** **text** as a **vector** of **such** **word** counts:

  - for instance, [`CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) simply **counts** word **occurrences**.
  -  [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), as its name **suggests**, performs word **counting** and **TF**-**IDF** **weighing** in one go.
- In the code below, you first **initialize** the **vectorizer** to **apply** to all **words** but **stopwords**. 
- The **vectorizer** **estimates** **word** counts and **learns** the **tf-idf** weights on the **training** **data** (therefore, you **use** **method** `.fit_transform` and **apply** it to the `train_set`) and then **applies** the **weights** to the **words** in the **test** **data** (this is **done** **using** **method** `.transform` **applied** to the `test_set`).
-  Using the **vectorizer**, you **convert** **training** and **test** **texts** into **vectors** and **store** the **resulting** vectors as `vectors_train` and `vectors_test`.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

vectorizer = TfidfVectorizer(stop_words = text.ENGLISH_STOP_WORDS)
def text2vec(vectorizer, train_set, test_set):
    vectors_train = vectorizer.fit_transform(train_set.data)
    vectors_test = vectorizer.transform(test_set)
    return vectors_train, vectors_test

vectors_train, vectors_test = text2vec(vectorizer, newsgroups_train, newsgroups_test)
```

- Lets check how the **data looks now**
  - You can **run  some checks** on the *vectors*
    - e.g. $\to$ check the **dimensionality** of the *vector* structures using `.shape` 
      - See how the **training** text is **represented**, then *check* which **word corresponds** to a **particular** *id*
        - For instance $\to$ $33404$. 

```python
print(vectors_train.shape)
print(#apply to the test set)
print(#check how the first training text is represented)
print(vectorizer.get_feature_names()[#use feature id here])
```

- The first two lines should tell you that `vectors_train` is a **matrix** of 5,913 rows and 52,746 columns (or a similar number)
- `vectors_test` is a matrix of 3,937 rows and 52,746 columns: 

  - you can **imagine** **two** **large** **tables** here, with **each** of the **rows** **representing** a **text** (remember, there are **5**,**913** training posts and **3**,**937** **test** posts) and **each** **column** **representing** a **word**.
  -  It is no coincidence that both matrices contain the same number of columns: the `TfidfTransformer` identified 52,746 non-stopwords in the training data, and it is this set of words that are used to classify texts into topics here. 
- The method `fit_transform` then **calculates** *tf-idf* scores **based** on the **training** **texts** (with the `fit` part of the method) and transforms the raw counts in the training data to these scores. 
- Finally, it **applies** the **same** **transformations** to the **occurrences** of the **same** **52**,**746** words in the test data (with the `transform` method).
-  It is **important** that the **tf-idf** scores are **learned** on the **training** **set** only: 
  - this is why we only use `transform` **method** on the **test** **data** and do not apply `fit_transform` as **this** will **rewrite** our **tf-idf** scores **based** on the **test data** and we will **end** up with **two separate sets** of **tf-idf scores** – one **fit** to the **training** **data** and **another** to the **test** **data**.
  -  **Remember** that in a **real**-**life** **application** you **would** only **have** **access** to the **training** **data** and **your** **test** set **might** **come**, for example, from the **future** **posts** on your news platform.

---

- Glimpse into the **first text** $\to$ shows a **list of references** and **scores**
  - Such as $(0, 15218)$ with some **rounded up score** of $0.32$ 
    - $0$ $\to$ refers to **first text**
    - $15218$ $\to$ index of the **15,219th** word in the **total set** of **52,746** words used for **classification**.
      - Use the `vectorizer.get_feature_names()[index]` to find out the **word** 
        - This is **sorted alphabetically**.

### Algorithm

- Train the **multinomial naive bayes classifier**
  - Then *classify* the **posts** from the **test set** into **topics**
    - In the **following code** $\to$ method `.fit` trains the **classifier** on the *training set features*
      - These are *stored in* `vectors_train` 
- The **gold standard** training set labels are also applied
  - These are `target` values of the `newsgroups_train`
- Then the **classifier** is applied to the **test set feature vectors**.

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=0.1)
clf.fit(vectors_train, newsgroups_train.target)
predictions = clf.predict(vectors_test)
```

- *Training* and *Testing* routine should **look pretty familiar** by now
  -  There is only **one parameter** `alpha` 
    - This **code specifies** for the **naive bayes**.

- This is **smoothing parameter** $\to$  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html and https://en.wikipedia.org/wiki/Additive_smoothing

### Evaluation

- Finally, let's **evaluate** the **results**, **extract** the most **informative** **terms** per topic, and **print** out and **visualise** the confusion matrix. 
- In the following code, you rely on `sklearn`’s [`metrics`](https://scikit-learn.org/stable/modules/model_evaluation.html) functionality that allows you to quickly evaluate your output.
-  To identify the most **informative** **features** in each **category**, you first **iterate** **through** the **categories** using `enumerate(categories)` – this **allows** you to **iterate** through the **tuples** of *(category id, category name)*. 
- **Within** this loop, `classifier.coef_[i]` **returns** a list of **probabilities** for the **features** in the **i-th category**, and `np.argsort` sorts this list in the increasing order (from the smallest to the largest) and returns the list of identifiers for the features. 
- As a **result**, you can **extract** $n$ most **informative** **features** using `[-n:]`. 
- You can **access** the **word** **features** via their **unique** **identifiers** using `vectorizer.get_feature_names()` and **print** out the **name** of the **category** and the **corresponding** most **informative** words. 
- In the **end**, you **print** out the **full** `classification_report` as **well** as the **top** 10 **informative** **features** per **category**.

```python
from sklearn import metrics

def show_top(classifier, categories, vectorizer, n):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top = np.argsort(classifier.coef_[i])[-n:]
        print(f'{category}: {" ".join(feature_names[top])}')
        
full_report = metrics.classification_report(newsgroups_test.target, predictions, target_names=newsgroups_test.target_names)
print(full_report)
show_top(clf, categories, vectorizer, 10)
```

---

![image-20220209223256307](D:\University\Notes\DiscreteMaths\Resources\image-20220209223256307.png)

![image-20220209223243578](D:\University\Notes\DiscreteMaths\Resources\image-20220209223243578.png)

Lowest precision is for `rec.sport.hockey`, Lowest recall is for `rec.autos`. The low precision value for `rec.sport.hockey` suggests that the words used to generate a class of `rec.sport.hockey` lack the specificity required to be able to incorrectly label a sample that is negative as actually being positive, however it does manage to return many more results. In fact the `rec.sport.hockey` has the highest recall which would suggest that its threshold for a positive prediction is too high and must be lowered into not letting as many classes through. For `rec.autos` low recall suggest it does not cover the threshold to classify a specific class as many times as the others and does not have any significantly high precision and therefore the features used in the class may just be not specific enough to provide an overall accurate prediction of classes. Overall the f1-scores of each sample is very similar and accuracy wise is good, `comp.windows.x` seems to be the most successful regarding the f1-score which suggests it features are extremely good for classification correctly having a solid high balance of precision and recall, this could simply come down to it being a much more explicit topic than the others which have words which may be relevant to said topic but not particularly exclusive to that topic as a whole thus affecting the overall score in the long run, which explains the `rec.sport.hockey` which has overlapping words applicable to various topics

---

- Finally, the code below shows how to **explore** the **confusions** that the **classifier** makes. 
- In this code, you rely on `sklearn`’s [`plot_confusion_matrix`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html) **functionality** and `matplotlib`’s plotting functionality. 
- The `plot_confusion_matrix`’s functionality **allows** you to **plot** the predictions that the **classifier** makes on `vectors_test` against the **actual** **labels** from `newsgroups_test.target` using a **heatmap**.

  - Additionally, you can set some **further** parameters: 

    - for instance,

      1. represent the number of correct and incorrect predictions using integer values format (i.e., `values_format=”0.0f”`) a#

      2. highlight the decisions on the heatmap with a particular color scheme. 
- In this code, you use **blue** **color** scheme, with the **darker** **color** **representing** **higher** numbers.
-  Finally, you **print** out the confusion **matrix** and **visualize** correct **predictions** and **confusions** with a **heatmap**. 
  - For **reference**, you can also **print** out the **categories**’ **ids** **corresponding** to the **categories**’ **names**.

```python
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

classifier = clf.fit(vectors_train, newsgroups_train.target)

disp = plot_confusion_matrix(classifier, vectors_test, 
                             newsgroups_test.target,
                             values_format="0.0f",
                             cmap=plt.cm.Blues)
    
print(disp.confusion_matrix)

plt.show()
for i, category in enumerate(newsgroups_train.target_names):
    print(i, category)
```

![image-20220209230157798](D:\University\Notes\DiscreteMaths\Resources\image-20220209230157798.png)



![image-20220209230142816](D:\University\Notes\DiscreteMaths\Resources\image-20220209230142816.png)

`rec.sport.hockey` is the most confused one as it clearly has many words which make it classify as the incorrect class, which does intuitively make sense considering `rec.sport.baseball` which has the most confused classes with is very similar topic and thus expected to have very similar words for its topic and corresponds exactly the low precision and high recall value that `rec.sport.hockey` obtained in the results above. Similarly further evidence is with `rec.motorcylces` classes confusing itself most commonly with `rec.autos` which is the same idea of very related topics which bound to have many similar terms to identify the topics, amongst also being related to `misc.forsale` as its about selling used cars in some regards as used cars are a big part of the automobiles market and thus that explains those seriously noticeable misclassifications. 

## Unsupervised approach

- Let’s now apply the **unsupervised** **approach** to our **data** from the 20 **Newsgroups** dataset.
-  In the **previous** section, you have **already** defined a **set** of **posts** on the **selected** 10 **categories** to work with. 
- You are **going** to use the **same** **set**, only this **time** you will **approach** it as if you **don’t** **know** what the **actual** topic labels are.
-  **Why** is this a **good** idea? First of all, since you **know** what the **labels** in this **data** **actually** are, you can **evaluate** your **algorithm** at the **end**.
-  **Secondly**, you will be **able** to see **what** the **algorithm** **identifies** in the **data** by itself, i.e., **regardless** of any **assigned** **labels**.
- After all, it is **always** **possible** that **someone** who **posted** to one **topic** actually **talked** **more** about **another** **topic**. **This** is **exactly** what **you** are **going** to **find** out.

### Data Preparation

- First, let’s prepare the data for ***clustering***.

-  Recall that you have already **extracted** the **data** from the **20** **Newsgroups** dataset: 

  - There are **5**,**913** posts in the `newsgroups_train` and **3**,**937** in the `newsgroups_test`. 

- Since **clustering** is an **unsupervised** technique, you **don’t** have to **separate** the **data** into **two** sets, so let’s **combine** them **together** in one set, `all_news_data`

  -  which should then contain 5,913+3,937=9,850 posts all together. 

- You are **going** to **cluster** posts **based** on their **content** (which you can **extract** **using** the `dataset.data` field);

-  finally, let’s **extract** the **correct** **labels** from the **data**

  >  (recall from the earlier code that they are stored in the `dataset.target` field) and set them **aside** – you can use **them** **later** to **check** how the **topics** **discovered** in this **unsupervised** way **correspond** to the labels originally assigned to the posts.
  >
  
- The code below walks you through these steps.

  -  Recall that it is a **good** idea to **shuffle** the data **randomly**, so let’s **import** `random` **functionality** and set the **seed** to a particular **value** (e.g., 42) to make sure **future** runs of your **code** return **same** **results**.

  -  Next, the code **suggests** that you **combine** the **data** from `newsgroups_train` and `newsgroups_test` into a **single** list, `all_news`, **mapping** the **content** of each **post** (accessible via `.data`) to its **label** (`.target`) and **using** `zip` function.

  -  After that, you **shuffle** the **tuples** and store the **contents** and **labels** separately: 

    - you will use the **contents** of the **posts** in `all_news_data` for **clustering** and the **actual** labels from `all_news_labels` to **evaluate** the results. 

  - Finally, you **should** **check** how many **posts** you h**a**ve (length of `all_news_data` should equal 9,850) and **how** many **unique** **labels** you have using `np.unique` (the answer should be 10)
    - Take a **look** into the **labels** to make sure you have a **random** **shuffle** of **posts** on **different** **topics**.

```python
import random
random.seed(42)

all_news = list(zip(newsgroups_train.data, newsgroups_train.target))
all_news += list(zip(newsgroups_test.data, newsgroups_test.target))
random.shuffle(all_news)

all_news_data = [text for (text, label) in all_news]
all_news_labels = [label for (text,label) in all_news]

print("Data:")
print(str(len(all_news_data)) + " posts in "+ str(np.unique(all_news_labels).shape[0]) + " categories\n")

print("Labels: ")
print(all_news_labels[:10])
num_clusters = np.unique(all_news_labels).shape[0]
print("Assumed number of clusters: " + str(num_clusters))
```

### Feature Selection

- Now the data is initialized, let’s **extract** the **features**. 
  - As before, you will use **words** as **features** and **represent** each **post** as an **array**, or **vector**, where **each** **dimension** will keep the **count** or *tf*-*idf* score **assigned** to the **corresponding** word:
    - For a **particular** **post** such an **array** may look like $[word \ 0=0, word \ 1=5, word \ 2=0, …, word \ 52745=3]$. To begin with, this **looks** exactly like the **pre**-**processing** and **feature** extraction **steps** that you **did** **earlier** for the supe**r**vised **approach**

---

- This time there are **two issues** that *need to be addressed*

  1. **Remember** that to assign **data** points to **clusters** you will need to **calculate** **distances** from each data point to **each** **cluster’s** **centroid**. 
     - This means **calculating** **differences** **between** the **coordinates** for **9**,**850** data **points** and **10** **centroids** in **52**,**746** **dimensions**, and then **comparing** the **results** to **detect** the **closest** centroid. 
     - Moreover, **remember** that **clustering** uses an **iterative** **algorithm**, and you **will** have to **perform** these **calculations** **repeatedly** for, e.g., **100** **iterations**. This **amounts** to a **lot** of **calculations**, which **will** **likely** make your **algorithm** very slow.

  2. In **addition**, a **typical** **post** in this **data** is **relatively** **short** – it might **contain** a **couple** of **hundreds** of **words**, and **assuming** that not **all** of these **words** are **unique** (some may be **stopwords** and **some** may be **repeated** several times), the **actual** word **observations** for **each** post will **fill** in a **very** **small** **fraction** of **52**,**746** dimensions, **filling** most of **them** with **zeros**. 
     - That is, it would be **impossible** to **see** any **post** that will **contain** a **substantial** amount of the **vocabulary** in it and, **realistically**, **every** **post** will have a **very** **small** **number** of **dimensions** filled with **actual** **occurrence** numbers, while the **rest** will **contain** **zeros**. 
     - What a **waste** – not **only** will you **end** up with a **huge** **data** structure of **9**,**850** posts by **52**,**746** word **dimensions** that **will** **slow** your **algorithm** down, but you **will** also be **using** **most** of this **structure** for **storing** zeros. 
       - This will make the **algorithm** very **inefficient**.

---

- What **you can be done** to *address* these **problems**

  - You *come across* solutions to these problems before , while some **others** are *new for you here*:
    1. First of all, you can **ignore** **stopwords**.
    2. **Next**, you can **take** into **account** only the **words** that are **contained** in a **certain** **number** of **documents**:
       - it would **make** **sense** to **ignore** **rare** **words** that **occur** in **less** than **some** **minimal** **number** of **documents** (e.g., 2) or that **occur** across too **many** **documents** (e.g., above 50% of the dataset). 
       - You can **perform** all **word** **filtering** **steps** in **one** go using `TfidfVectorizer`.
    3. Finally, you can **further** **compress** the **input** **data** using **dimensionality** **reduction** techniques. 
       - One of such widely used **techniques** is [`Singular Value Decomposition`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD) (SVD), which **tries** to **capture** the **information** from the **original** **data** **matrix** with a **more** **compact** **matrix**. 
         - **SVD** is an **alternative** for **PCA** (which we discussed in lectures), that is **widely** applied to **NLP** tasks, **thus** you will apply this **technique** here.


---

- In the following code, you use [`TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to **convert** **text** content to **vectors** **ignoring** all words that **occur** in **less** that 2 **documents** (with `min_df=2`) or in more than **50**% of the **documents** (with `max_df=0.5`).
  - In **addition**, you **remove** **stopwords** and **apply** **inverse** **document** **frequency** weights (`use_idf=True`). 
  - **Within** the `transform` function, you **first** **transform** the **original** **data** using a **vectorizer** and **print** out the dimensionality of this transformed data. 
  - **Next**, you **reduce** the **number** of **original** **dimensions** to a **much** **smaller** number using [`TruncatedSVD`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD). `TruncatedSVD` is particularly **suitable** for **sparse** data like the one you are working with here (e.g., see more examples of its application to text data: [here](https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html#sphx-glr-auto-examples-text-plot-document-clustering-py)).
  - **Then**, you add `TruncatedSVD` to a pipeline ([`make_pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) from `sklearn`) **together** with a [`Normalizer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html#sklearn.preprocessing.Normalizer), **which** helps **adjust** **different** ranges of values to the **same** **range**, thus **helping** **clustering** **algorithm’s** efficiency.
    - As the **output** of the `transform` function, **you** **return** both the **data** with the **reduced** **dimensionality** and the `svd` **mapping** **between** the **original** and the reduced data. **Finally**, you **apply** the transformations to `all_news_data` to compress the **original** **data** **matrix** to a **smaller** number of features (e.g., 300) and **print** out the **dimensionality** of the **new** data **structure**.

---

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# calculate the tfid of each words using stop words of the english.

#ignore words that appear in less than 2 documents or in more than 50% of the documents.

# use idf weightings.
vectorizer = TfidfVectorizer(min_df=2, max_df=0.5,
                             stop_words='english',
                             use_idf=True)

# takes in data and vectorized used to transform data into the word vectors
def transform(data, vectorizer, dimensions):
    trans_data = vectorizer.fit_transform(data)
    print("Transformed data contains: " + str(trans_data.shape[0]) +
          " with " + str(trans_data.shape[1]) + " features =>")

    # dimensionality reduction
    svd = TruncatedSVD(dimensions)
    
    # create preprocessing pipeline
    pipe = make_pipeline(svd, Normalizer(copy=False))
    reduced_data = pipe.fit_transform(trans_data)
    return reduced_data, svd
```

```python
reduced_data, svd = transform(all_news_data, vectorizer, 300)
print("Reduced data contains: " + str(reduced_data.shape[0]) +
        " with " + str(reduced_data.shape[1]) + " features")
```

### Algorithm

- The data is **ready** $\to$ apply **clustering algorithms**
  - Specifically the code below use `Kmeans` clustering **algorithms** from `sklearn` 
    -  You apply the **algorithm** with `n_clusters` defining the **number of clusters** to *form* and *centroids* to **estimate** (*10 here for examp*le)
  - `k-means++` defines an **efficient** way to *initialize* the **centroids**

- Parameter `max_iter` $\to$ defines the **number** of *times you iterate* through the **dataset**
- `random_state` $\to$ set to some **particular value** ensures that you **get same results** every **time you run the algorithm**
- Finally **run the algorithm** on the `reduced_data` with the **number** of *clusters* equivalent to $10$ 
  - Recall this value is stored in `num_clusters` 
    - Based on the **number of categories** in the **input data**.

```python
from sklearn.cluster import KMeans

# take the transformed data and select the cluster using the efficient algorithm
def cluster(data, num_clusters):
    km = KMeans(n_clusters=num_clusters, init='k-means++', 
                max_iter=100, random_state=0)
    
    # fit our data which returns th
    km.fit(data)
    return km

km = cluster(#pass in appropriate arguments)
```

