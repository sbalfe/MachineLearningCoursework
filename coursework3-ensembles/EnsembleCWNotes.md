# Ensemble learning

## Learning objectives

In this practical you will learn about:
- simple **voting** **classifiers** using **hard** and **soft** voting strategies
- **bagging** and **pasting** ensembles
- **boosting** and **early** **stopping**
- popular **ensemble**-**based** **learning** algorithms including `RandomForests` and `AdaBoost`

> In addition, you will get more practice with `sklearn's` machine learning routines, and you will learn how to implement ensembles both from scratch and using `sklearn's` implementation.

---

## Simple voting Classifiers - hard / soft voting strategies

- Combine **multiple classifiers together** the resulting vote is **better** than **any single predictor**

```python
import scipy, math
import scipy.special

def probability(p, n, x):
    binom = scipy.special.comb(n, x, exact=True) * math.pow(p, x) * math.pow((1-p), n-x)
    return binom

prob_maj = 0.0
for x in range(501, 1001):
    prob_maj += probability(0.51, 1000, x)
    
print(prob_maj)

# output approx = 73%, 51% chance but there are 1000 thus increases the percentage overall.
```

- Works only they are **independent** and **diverse**.
  - Often trained on the **same data** therefore this is hard.

---

- Apply voting strategy to some data
  - Working with **synthetic** moons dataset
- Includes **two interleaving half circles** and provides a suitable **toy example** for *testing* out the **classification strategies**. 

---

- Generate data points and visualise this

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```

![image-20211128142509386](D:\University\Notes\DiscreteMaths\Resources\image-20211128142509386.png)

- Split dataset into training / test sets and train some classifiers on the **training data**

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

- Code below *applies* many **classifiers** to *this data*:

```python
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)
```

![image-20211128142812164](D:\University\Notes\DiscreteMaths\Resources\image-20211128142812164.png)

- Look into **how correlated** the **predictions** of the *classifiers are*
  1. Obtain **predictions** on the **test data**
  2. Store as **pandas** *dataframe*
  3. Apply `.corr()` function.

```python
import pandas as pd

def get_predictions(clf):
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


preds = {'lr': get_predictions(log_clf), 
        'rf': get_predictions(rnd_clf), 
        'svc': get_predictions(svm_clf)}
df = pd.DataFrame(data=preds)
df[:100]
```

![image-20211128142918576](D:\University\Notes\DiscreteMaths\Resources\image-20211128142918576.png)

```python
df.corr()
```

![image-20211128143621403](D:\University\Notes\DiscreteMaths\Resources\image-20211128143621403.png)

- Will the **correlation** in the **individual classifiers predictions** be *sufficient for an ensemble*
  - Check this by **combining** votes with a **hard voting strategy**
    - The *ensemble classifier* will **simply choose the majority** *class* predicted by the **three classifiers**.
- Code below prints **individual classifiers** *accuracy scores* along with the **ensemble accuracy score**

```python
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

![image-20211128144050824](D:\University\Notes\DiscreteMaths\Resources\image-20211128144050824.png)

- If one has `predict_proba()` method then use the **soft voting strategy** 
  - Estimated **highest class** *probability* **averaged** over the **individual classifiers**
    - In comparison to **hard voting strategy** , soft obtains **more weight** on the **higher confidence votes**
- `SVC` does **not estimate** *class probabilities* therefore you must set `probability` hyperparameter to `True` 

```python
log_clf = LogisticRegression(random_state=42)
rnd_clf = RandomForestClassifier(random_state=42)
svm_clf = SVC(probability=True, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='soft')
voting_clf.fit(X_train, y_train)
```

![image-20211128144554489](D:\University\Notes\DiscreteMaths\Resources\image-20211128144554489.png)

- Then estimate the accuracy of the voting classifier in this mode.

```python
from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
```

![image-20211128144623389](D:\University\Notes\DiscreteMaths\Resources\image-20211128144623389.png)

## Bagging ensembles - Bagging and Pasting

https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#use bootstrap=False for pasting
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
#n_jobs = use all of the available CPU cores
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

- Calculate the accuracy score for the bagging classifier that combines 500 decision tree estimators with a prediction of a single decision tree trained on the same data.

```python
print(accuracy_score(y_test, y_pred)) # 0.904
```

```python
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_tree)) # 0.856
```

- Obtain more **insight results** and plot its **decision boundary** for the **two classifiers**

```python
from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_decision_boundary(tree_clf, X, y)
plt.title("Decision Tree", fontsize=14)
plt.subplot(122)
plot_decision_boundary(bag_clf, X, y)
plt.title("Decision Trees with Bagging", fontsize=14)
plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20211128145215945.png" alt="image-20211128145215945" style="zoom:80%;" />

- Even though the **ensemble** makes a **comparable** number of errors on the **training set** as a **single decision tree**
  - Its **decision boundary** is *less irregular*
    - This suggests the **ensembles predictions** will *likely generalise better* than the **single** `Decision Tree` predictions when applied **to the new data**

### Out of bag evaluation

- Instances are **sampled multiple times** for *some predictor*
  - Some *not at all*.
- Default `BaggingClassifier` samples $m$ training instances with **replacement** where $m$ is the size **of the training set**.
  - As $m$ **increases** $\to$ the **ratio** of **training instances** that are *sampled* for every **predictor** reaches $1-e^{(-1)}$ 
    - Around $63\%$ 
  - Remaining $37\%$ of the **training instances** that are **never used** by the **predictors** are called **==oob==** instances.

![image-20211128145618729](D:\University\Notes\DiscreteMaths\Resources\image-20211128145618729.png)

- A predictor never see these instances **during training**
  - They can be used for **evaluation** without the need for **additional validation set** or **cross validation experiments**
    - You can evaluate the ensemble itself by **averaging each predictors** *performance* on the *oob instances*.
- Code below shows show this
  - Setting `oob_score = True` 

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_ # 0.898666666
```

- Therefore this **classifier** is *likely* to **achieve** around **90%** accuracy on the **test data**
  - Check the results:

```python
from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred) # 0.912
```

- Very close
  - Obtain insights by `oob_decision_function_` 
    -  These are the **class probabilities** assigned by the **classifier** to **negative / positive class** for *each instance*

```python
bag_clf.oob_decision_function_
```

![image-20211128150215968](D:\University\Notes\DiscreteMaths\Resources\image-20211128150215968.png)

## Random Forests

- `RandomForestClassifier` is an **ensemble** of **decision trees** trained via the **bagging method**
  - The **number of training instances** (`max_samples`) often just set to the **size** of the **training set**
    - So in fact the `RandomForestClassifer` is **roughly equivalent** to the `BaggingClassifier` that takes **decision trees** as the **base estimators** and the following parameters.

```python 
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
    n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)
```

- The **key difference** rather than using `BaggingClassifier` and passing it a `DecisionTreeClassifier` 
  - Can rely on the `sklearns` `RandomForestClassifier` 
    - Which is more **optimised** for **decision trees**

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, 
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_test)
```

- Estimate the difference between the **two classifiers predictions**

```python
np.sum(y_pred == y_pred_rf) / len(y_pred)  # see to what extent predictions are identical , 0.976 
```

- Two classifiers are **nearly identical**
  - Random forest introduces some **extra randomness** when **growing tree**
- Rather than searching for the **best feature** when **splitting a node**
  - Searched for the **best feature** among a **random subset of features**
    - As a result the **model is more diverse** and **yields better results overall**

- Visualise the **decision boundary** for a **random set**  of $15$ decision trees to obtain a better idea where the **diversity comes from**

```python
plt.figure(figsize=(6, 4))

for i in range(15):
    tree_clf = DecisionTreeClassifier(max_leaf_nodes=16, random_state=42 + i)
    indices_with_replacement = np.random.randint(0, len(X_train), len(X_train))
    tree_clf.fit(X[indices_with_replacement], y[indices_with_replacement])
    plot_decision_boundary(tree_clf, X, y, axes=[-1.5, 2.5, -1, 1.5], 
                           alpha=0.02, contour=False)

plt.show()
```

![image-20211128153826446](D:\University\Notes\DiscreteMaths\Resources\image-20211128153826446.png)

### Feature importance

- They can measure the **relative importance** of **each feature** by looking at **how much the tree nodes** that use that *features* reduces the **impurity** of the **nodes on average**
  - As in **across all trees in the forest**
    - Results are **scaled** so the **sum** of **all feature importance's** *equals* $1$ 

- using iris dataset

```python
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42)
rnd_clf.fit(iris["data"], iris["target"])
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
    print(name, score)
```

![image-20211128154021512](D:\University\Notes\DiscreteMaths\Resources\image-20211128154021512.png)

```python
rnd_clf.feature_importances_
```

![image-20211128165858430](D:\University\Notes\DiscreteMaths\Resources\image-20211128165858430.png)

```python
from sklearn import datasets
digits = datasets.load_digits()

rnd_clf = RandomForestClassifier(random_state=42)
rnd_clf.fit(digits["data"], digits["target"])
```

![image-20211128165913789](D:\University\Notes\DiscreteMaths\Resources\image-20211128165913789.png)

```python
def plot_digit(data):
    image = data.reshape(8, 8)
    plt.imshow(image, cmap = matplotlib.cm.hot,
               interpolation="nearest")
    plt.axis("off")
    
plot_digit(rnd_clf.feature_importances_)

cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'])

plt.show()
```

![image-20211128170400553](D:\University\Notes\DiscreteMaths\Resources\image-20211128170400553.png)

## AdaBoost

- Step by Step:
  - Start with the first classifier (i.e., this can be a Decision Trees classifier)
  - train it and use it to make predictions on the training set
  - increase the relative weight of misclassified training instances
  - train a second classifier with these new updated weights and make new predictions
  - update weights using the new predictions
  - continue until stopping criteria are satisfied.

---
For instance, suppose each instance's original weight $w^{(i)}$ is set to $\frac{1}{m}$, where $m$ is the number of instances. The first classifier is applied, and its error rate $r_1$ is computed on the training set using the equation below:

$$
\begin{equation}
r_j = \frac{\sum_{\hat{y}^{(i)}_j \neq y^{(i)}} w^{(i)}}{\sum_{i=1}^m w^{(i)}}
\end{equation}
$$


where $\hat{y}^{(i)}_j$ is the prediction of the $j$-th classifier on the $i$-th instance. The predictor's weight $\alpha_j$ is then estimated using:

$$
\begin{equation}
\alpha_j = \eta log \frac{1-r_j}{r_j}
\end{equation}
$$
where $\eta$ is the learning rate, a hyperparameter that defaults to $1$. The more accurate the predictor is, the higher its weight will be; if a predictor is guessing randomly, its weight will be close to $0$; and if it performs worse than random guessing it will get a high negative weight.

Next, the weights are updated and the misclassified instances are boosted as follows:

$$
\begin{equation}
  w^{(i)}=\begin{cases}
    w^{(i)}, & \text{if $\hat y_j^{(i)} = y_j^{(i)}$}\\
    w^{(i)} exp(\alpha_j), & \text{if $\hat y_j^{(i)} \neq y_j^{(i)}$}
  \end{cases}
\end{equation}
$$
Then all the instances weights are normalised (i.e., divided by $\sum_{i=1}^m w^{(i)}$). The new predictor is trained on the updated training instances, applied to the training set, its weight is computed, weights are updated again, and so on. The algorithm stops when either the predefined number of predictors is reached or a perfect predictor is found.

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
```

![image-20211128171529594](D:\University\Notes\DiscreteMaths\Resources\image-20211128171529594.png)

- Plot the **decision boundary**

```python
plot_decision_boundary(ada_clf, X, y)
```

![image-20211128171555097](D:\University\Notes\DiscreteMaths\Resources\image-20211128171555097.png)

- To obtain more insight into how the decision boundaries change from one step to another
  - Use a different learning rate.
    - Plot the decision boundaries for **5 consecutive predictors**
      - For this case $\to$ using `SVC` as the estimator
- Note how the **first classifiers** often get **many instances wrong**
  - While the **following predictors** are *gradually getting better*.
    - The **plot** on the **right presents** the *very* same **5 consecutive classifiers**
      - But assigns **half the learning rate**
        - The **misclassified** instance weights are **only boosted half as much every iteration**

```python
m = len(X_train)

plt.figure(figsize=(11, 4))
for subplot, learning_rate in ((121, 1), (122, 0.5)):
    sample_weights = np.ones(m)
    plt.subplot(subplot)
    for i in range(5):
        svm_clf = SVC(kernel="rbf", C=0.05, gamma="auto", random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = svm_clf.predict(X_train)
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
    if subplot == 121:
        plt.text(1.70,  -0.90, "1", fontsize=14)
        plt.text(-0.40, -0.35, "2", fontsize=14)
        plt.text(-0.55, -0.05, "3", fontsize=14)
        plt.text(-0.70,  0.20, "4", fontsize=14)
        plt.text(-0.85,  0.45, "5", fontsize=14)

plt.show()
```

![image-20211128172427976](D:\University\Notes\DiscreteMaths\Resources\image-20211128172427976.png)

- Additionally, here is the full list of parameters and attributes of the algorithm

```python
list(m for m in dir(ada_clf) if not m.startswith("_") and m.endswith("_"))
```

![image-20211128172548774](D:\University\Notes\DiscreteMaths\Resources\image-20211128172548774.png)

- Can check classification errors for **each estimator** in the **boosted ensemble as follows**

```python
ada_clf.estimator_errors_
```

![image-20211128172841876](D:\University\Notes\DiscreteMaths\Resources\image-20211128172841876.png)

# Answers to questions at the bottom

1. Yes. Hard voting obtains its results from various predictors based on the highest majority one. Mode is the same concept in this case as being the most common result from the predictors.

2. Hard / Soft. There is a choice of whether to use the soft or hard, whilst hard is often used as its a yes or no output, there is the probability output method which can be used to calculate the average outputs of predicotrs. predicted probability varies on implementations , ranging from oob scores, mean of scores for examp.e
3. It just shows the areas where the most prevalence is being placed which is clearly just going to be the features that are in common with most images.
4. No because adaboost passes the weights down the chain of predictors so you would have to wait on outputs , so a sort of pipeline could be setup instead
5. Learning rate controls how much the weights are adjusted on each iteration, which in turn affects the optimal value, but too high can make it go way beyond the value and reach a value far off, so it must be adjusted to be optimised to reach the proper minimum. 

