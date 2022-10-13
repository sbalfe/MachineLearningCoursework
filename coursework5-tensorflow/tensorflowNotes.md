# Tensorflow Coursework notes

## Learning objectives

![image-20220301163358596](D:\University\Notes\DiscreteMaths\Resources\image-20220301163358596.png)

## Minimal Tensorflow Example

- Take input vector > multiply by weight matrix > add weight vector

```python

# tf.Variable defines model parameters which are trained.

# init this as 3 x 3 matrix, every entry being one tf.ones
weight_matrix = tf.Variable(tf.ones(shape=(3,3)))

# init weight as 3 x 1 vector variable every entry being 0
weight_vector = tf.Variable(tf.zeros(shape=(3,)))

def affine_transformation(input_vector):
    
    # matvec in linear algbera library takes the vector and matrix , multiplies and adds weight vector.
    return tf.linalg.matvec(weight_matrix, input_vector) + weight_vector

result = affine_transformation([2.,3.,7.])
print(result)
```

 ![image-20220301171917073](D:\University\Notes\DiscreteMaths\Resources\image-20220301171917073.png)

- Reset function `tf.keras.backend.clear_session` 
- Must reset from time to tiume as there are many small networks in one notebook that we **dont want to interfere** 
- This is a **preemptive measure** to occasionally reset the **computation graph**

## Training parameters

- Show how to **optimise parameters** in the **model**.

```python
tf.keras.backend.clear_session

weight_matrix = tf.Variable(tf.ones(shape=(3,3)))
weight_vector = tf.Variable(tf.zeros(shape=(3,)))

# network taking the input vector
def network(input_vector):
    
    # sum elements using reduce_sum which just sums the matrix 
    return tf.math.reduce_sum(affine_transformation(input_vector))

# square error loss function, given specific input and output

def loss_fn(predicted, gold):
    return tf.square(predicted - gold)

input = [2.,3.,7.]
gold_output = 20

# calculate loss of applying the network to the input
def loss():
    return loss_fn(network(input), gold_output)

# define optimiser using SGD with learnign rate 0.001
opt = tf.keras.optimizers.SGD(learning_rate=1e-3)

# use this optimiser to train the network for 10 epochs, over this single training point
# optimises the output towards target value 20.
for epoch in range(10):
    opt.minimize(loss, var_list=[weight_matrix, weight_vector])
    print(network(input))
```

![image-20220301172654186](D:\University\Notes\DiscreteMaths\Resources\image-20220301172654186.png)

- Optional $\to$ try changing the **learning rate** and **numper of epochs**
  - What results are **you getting**.

## Network Layers

- Most case , do **not actually need** to create the **trainable variables manually**
  - Instead $\to$ feedforward layer available as **pre defined module**

```python
# define network as sequence of operations using tf.keras.sequential
model = tf.keras.Sequential([
    
    # first layer is dense feedforward , acts like affine transformation define earlier
    tf.keras.layers.Dense(3, input_shape=(3,)),
    
    # second step sums the elements of the vector, this is not a standard operation therefore
    # use tf.keras.layers.
    tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=1))
])
```

- Model expects data as **==minibatch==** 
  - The *input tensor* should have some **extra index** which ranges over **datapoints**
    - For our case $\to$ since we have a **single datapoint** 
      - Rather than passing **3 dimensional input vector** $\to$ pass an $N x 3$ matrix where $N$ is **number of datapoints**
        - i.e. we apply model to **single datapoint**

```python
model.predict(tf.constant([[2.,3.,7.]]))
```

![image-20220301174600807](D:\University\Notes\DiscreteMaths\Resources\image-20220301174600807.png)

- Model defined in terms of layers
  - Replace manually created variables of the **previous section**

```python
tf.keras.backend.clear_session

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(3,)),
    tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis=1))
])

def loss_fn(predicted, gold):
    return tf.square(predicted - gold)

input = tf.constant([[2.,3.,7.]])
gold_output = 20

def loss():
    return loss_fn(model(input), gold_output)

opt = tf.keras.optimizers.SGD(learning_rate=1e-3)

for epoch in range(10):
    opt.minimize(loss, var_list=model.trainable_variables)
    print(model(input))
    weights, biases = model.layers[0].get_weights()
    print(weights)
    print(biases)
```

![image-20220301174752826](D:\University\Notes\DiscreteMaths\Resources\image-20220301174752826.png)

- For stadnard **optimisers** and **loss functions**
  - The *tensorflow* API makes it **even easier for us**

```python
tf.keras.backend.clear_session

model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(3,)),
    tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x))
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
             loss='mean_squared_error')

input = tf.constant([[2.,3.,7.]])
gold_output = tf.constant([[20.]])

for epoch in range(10):
    model.train_on_batch(input, gold_output)
    print(model(input).numpy())
```

![image-20220301175019666](D:\University\Notes\DiscreteMaths\Resources\image-20220301175019666.png)

## Activation functions

- Model **non linear patterns** in *data*.
  - After applying **an affine transformation** 
    - We apply a **non linear activation function** to *each element*
- There are various **activation functions**

```python
hidden = tf.keras.layers.Dense(100, activation='sigmoid')
```

```python
hidden = tf.keras.layers.Dense(100, activation='tanh') # used more in modern networks, has flexibility as it transforms input to -1 and 1, thus outputs negative values
```

```python
hidden = tf.keras.layers.Dense(100, activation='relu') # linear function above 0, ranges from 0 to infinity.
```

- Partial linear property of **ReLU** helps it **converge** faster *on some tasks*
  - Although **in practice** $\to$ `tanh` may be **some more robust option**

- Softmax function important

![image-20220301175426669](D:\University\Notes\DiscreteMaths\Resources\image-20220301175426669.png)

- The value of the **denominator** depends on **all other values**
- Softmax used in **output layer** to *determine the class probability distribution*.

```python
model = tf.keras.Sequential([
    # takes 20 dimensional input, maps to 50 dim hiddne,  maps to over 10 output classses
    tf.keras.layers.Dense(50, input_shape=(20,), activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## Operations and useful functions

- various math functions in `tf.math` 

```python
tf.abs # absolute value
tf.negative # computes the negative value
tf.sign # returns 1, 0 or -1 depending on the sign of the input
tf.math.reciprocal # reciprocal 1/x
tf.square # return input squared
tf.round # return rounded value
tf.sqrt # square root
tf.math.rsqrt # reciprocal of square root
tf.pow # power
tf.exp # exponential
```

- Operations applied to **scalar vlaues**
  - Also to **vectors / matrices / higher order tensors**
- In latter case $\to$ they are applied element wise.

```python
print(tf.negative([3.2,-2.7]))
print(tf.square([1.5,-2.1]))
```

![image-20220301175733378](D:\University\Notes\DiscreteMaths\Resources\image-20220301175733378.png)

- Some useful operations are performed over a **whole vector / matrix / tensor** to return a **single value**

## Adaptive learning rates

- Used SGD to train our model, uses **fixed learning rates** to update the **parameters**
  - Several opimsiation algorithms are based on SGD but adapatively adjust the learning rate 
    - Often for **each paramter seperately**
- Different **adaptive learning rates** strategies are **often implemented** in *tensorflow* as functions

```python
tf.keras.optimizers.SGD
tf.keras.optimizers.Adadelta
tf.keras.optimizers.Adam
tf.keras.optimizers.RMSprop
```

https://ruder.io/optimizing-gradient-descent/

## Training an XOR function

- Takes 2 binary > output 1 if values are different, zero otherwise.
- Dataset consists of **possible states** XOR can take.

```python
xor_input = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
xor_output = tf.constant([0.0, 1.0, 1.0, 0.0])
```

- Construct a **linear network** and **optmize** on the **dataset** , printing the predictions at **each epoch**

```python
tf.keras.backend.clear_session

linear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,))
])

linear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                     loss='mean_squared_error')

for epoch in range(100):
    linear_model.train_on_batch(xor_input, xor_output)
    if (epoch + 1) % 10 == 0:
        print('after {} epochs:'.format(epoch+1), 
              linear_model(xor_input).numpy().reshape((4,)))
```

![image-20220301180247643](D:\University\Notes\DiscreteMaths\Resources\image-20220301180247643.png)

- It does terrible, predictions should be [0,1,1,0]
  - This hovers around 0.5 for each input case
    - Improve architecture adding some **non linear layers** into the **models**

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(2,), activation='relu'), # note that these settings can be changed
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1),
                        loss='mean_squared_error')

for epoch in range(100):
    nonlinear_model.train_on_batch(xor_input, xor_output)
    if (epoch + 1) % 10 == 0:
        print('after {} epochs:'.format(epoch+1), nonlinear_model(xor_input).numpy().reshape((4,)))
```

![image-20220301180345812](D:\University\Notes\DiscreteMaths\Resources\image-20220301180345812.png)

- Result as expected
- Random init.
- Restore a particular model $\to$ https://www.tensorflow.org/tutorials/keras/save_and_load
- Had to **increase learning rate** for the **network**
  - Still **smaller learning rate** but *converges slowly*
    - Learning rate is **hyperparameter** that vary quite a lot **on the network architecture** and **dataset**.

## XOR classification

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    
    # use softmax function
    tf.keras.layers.Dense(2, activation='softmax')
])

# SGD constant learning rate, using new loss cross entropy as correct output has probability 1 for the correct class and probability 0 for the other ones, assign higher proporiton to more likely values
nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1),
                        loss='sparse_categorical_crossentropy')

for epoch in range(50):
    nonlinear_model.train_on_batch(xor_input, xor_output)
    if (epoch + 1) % 10 == 0:
        print('after {} epochs:'.format(epoch+1), nonlinear_model(xor_input).numpy(), sep='\n')
```

![image-20220301181110230](D:\University\Notes\DiscreteMaths\Resources\image-20220301181110230.png)

- Convert **these probabilities** into *class probabilities* into **class predictions** and also *report some of the more familiar* **evalution metrics** such as *accuracy*.

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])  

for epoch in range(50):
    nonlinear_model.train_on_batch(xor_input, xor_output)
    predictions = nonlinear_model.predict(xor_input)
    result = tf.argmax(predictions, axis=1)
    
    if (epoch + 1) % 10 == 0:
        print('\nAfter {} epochs:'.format(epoch+1), " ".join([str(x) for x in result.numpy()]))
        test_loss, test_acc = nonlinear_model.evaluate(xor_input, xor_output, verbose=2)
        print('\nAccuracy:', test_acc)
```

![image-20220301181428991](D:\University\Notes\DiscreteMaths\Resources\image-20220301181428991.png)

- Should see in **this printout** that it starts with **incorrect predictions**, later returns to **correct sequence** [0,1,1,0]
  - Finally here is **how you print** out **confusion matrix**
    - Since we *are looking* into a **simple case** and the *predictions* from the **above** are *quite accurate*
      - There is **not to much to be learned** from the **confusion matrix** at *point* (*but* not the functionality comes in handy later) 

```python
conf_mx = tf.math.confusion_matrix(xor_output, result.numpy()).numpy()
print(conf_mx)
```

![image-20220301181854208](D:\University\Notes\DiscreteMaths\Resources\image-20220301181854208.png)

## Minibatching

- For XOR data there are **4 datapoints**
  - For **realistic datasets**
    - This is *inefficient* to train **on the whole dataset on once** as it **requires** a *lot of computation* in *order* to **make a single update step**
- Instead $\to$ train on a  **batch of data** at a *time*
  - Example $\to$ how we take batches of **2 datapoints** for the **XOR data**.

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                        loss='sparse_categorical_crossentropy')

BATCH_SIZE = 2

for epoch in range(50):
    for i in range(0,len(xor_input),BATCH_SIZE):
        input_batch = xor_input[i:i+BATCH_SIZE]
        output_batch = xor_output[i:i+BATCH_SIZE]
        nonlinear_model.train_on_batch(input_batch, output_batch)
    if (epoch + 1) % 10 == 0:
        print('after {} epochs:'.format(epoch+1), nonlinear_model(xor_input).numpy(), sep='\n')
```

![image-20220301182159656](D:\University\Notes\DiscreteMaths\Resources\image-20220301182159656.png)

- This is built into **tensorflow**
  - Following code trains the model with the given batch size and number of epochs

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                        loss='sparse_categorical_crossentropy')

nonlinear_model.fit(xor_input, xor_output, batch_size=2, epochs=50)

print('final loss:', nonlinear_model.evaluate(xor_input, xor_output))
print('final predictions:', nonlinear_model.predict(xor_input), sep='\n')
```

## Tensor Board

- Print statements are fine for simple models but more complicated models required better analysis.
- Visualise for debugging / inspecting / reporting the network results.

![image-20220301183241961](D:\University\Notes\DiscreteMaths\Resources\image-20220301183241961.png)

```python
# Load the TensorBoard notebook extension
%load_ext tensorboard
```

- Likely introduce **changes** into the **network** and *rerunning code*
  - Important to be able to **distinguish** between these different runs to **track changes**
    - Each time you **run a new model** $\to$ stored in *log files* then added to *tensor board*
      - Add **timestamps** to each model

```python
import datetime
```

- Clean other logs using `rm -rf ./logs/` in terminal.
- Once completed $\to$ train network and store details in the **log files**

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                        loss='sparse_categorical_crossentropy')

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

nonlinear_model.fit(xor_input, xor_output, 
                    batch_size=2, epochs=50, 
                    validation_data=(xor_input, xor_output),
                    callbacks=[tensorboard_callback])

print('final loss:', nonlinear_model.evaluate(xor_input, xor_output))
print('final predictions:', nonlinear_model.predict(xor_input), sep='\n')
```

![image-20220301183548479](D:\University\Notes\DiscreteMaths\Resources\image-20220301183548479.png)

```python
%tensorboard --logdir logs/fit
```

![image-20220301183610312](D:\University\Notes\DiscreteMaths\Resources\image-20220301183610312.png)

- Explore results under `scalars` tab.
- network architecture under `graphs` tab.

![image-20220301183800026](D:\University\Notes\DiscreteMaths\Resources\image-20220301183800026.png)

- Additional plugins available

## Keeping track of history

- Other methods to get more information and description of the model
  - Which are useful when introducig more complexity to the model and would like to keep track of changes
    - Summarise them in this section

```python
tf.keras.backend.clear_session

nonlinear_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
```

```python
# how we return the information on networks layers and types
nonlinear_model.layers
```

![image-20220301183959208](D:\University\Notes\DiscreteMaths\Resources\image-20220301183959208.png)

```python
# obtain concise summary of the network layters (first dim in output shape column specified as None)
# This is to denote that the dimension is variable as it depends on the batch size.
nonlinear_model.summary()
```

![image-20220301184100232](D:\University\Notes\DiscreteMaths\Resources\image-20220301184100232.png)

```python
# plot the model summary like so

tf.keras.utils.plot_model(nonlinear_model, show_shapes = True)
```

```python
# below are number of ways to extract (then store) information on individual layers, alongside weights and biases in the network
```

```python
hidden1 = nonlinear_model.layers[1]
hidden1.name

nonlinear_model.get_layer(hidden1.name) is hidden1

weights, biases = hidden1.get_weights()
weights.shape
biases.shape
```

---

- Now train model and **track changes** in the *loss accuracy* on training and validation data
  - Code shows how to **apply** to **training and validation data** but the subsets are not actually *different for this small example*
    - However later $\to$ obtain chance to apply to different subsets of data.

```python
nonlinear_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                        loss='sparse_categorical_crossentropy',
                       metrics = ['accuracy'])
```

```pyth
history = nonlinear_model.fit(xor_input, xor_output, batch_size=2, epochs=50,
                    validation_data=(xor_input, xor_output))


```

![image-20220301184610560](D:\University\Notes\DiscreteMaths\Resources\image-20220301184610560.png)

```python
history.params
```

![image-20220301184539974](D:\University\Notes\DiscreteMaths\Resources\image-20220301184539974.png)

```python
print(history.epoch)
```

![image-20220301184627367](D:\University\Notes\DiscreteMaths\Resources\image-20220301184627367.png)

```python
history.history.keys()
```

![image-20220301184640994](D:\University\Notes\DiscreteMaths\Resources\image-20220301184640994.png)

```python

# plot changes in results across all epochs

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```

![image-20220301184725125](D:\University\Notes\DiscreteMaths\Resources\image-20220301184725125.png)

## Case of regression

```python
np.random.seed(42)
tf.random.set_seed(42)

# practical 1 use custom version of california dataset
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
```

```python
from sklearn.model_selection import train_test_split

# split data into traiing , validation and testing.

# access data from the dataset with housing.data, labels with housing.target
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, 
                                                              housing.target, 
                                                              random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, 
                                                      y_train_full, 
                                                      random_state=42)
```

```python
from sklearn.preprocessing import StandardScaler

# scale data using standardisation.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
```

```python
# implement regression model using tensorlfow, similar for classification with minor change

reg_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    # output single predicted value thus the dimensionality of output layer changes from classification.
    tf.keras.layers.Dense(1)
])

#loss function now mean squared error
reg_model.compile(loss="mean_squared_error", 
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3))
history = reg_model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = reg_model.evaluate(X_test, y_test)
```

```python
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```

![image-20220301185040040](D:\University\Notes\DiscreteMaths\Resources\image-20220301185040040.png)

```python
# explore model prediction on some selected datapoints, compare them to the true values for these datapoints

X_new = X_test[:3]
y_pred = reg_model.predict(X_new)

y_pred
'''
array([[0.38856643],
       [1.6792021 ],
       [3.1022794 ]], dtype=float32)
'''

y_test[:3]
'''
	array([0.477  , 0.458  , 5.00001])
'''
```

