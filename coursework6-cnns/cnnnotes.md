# Convolutional Neural Networks

## Learning objectives

![image-20220408130257330](D:\University\Notes\DiscreteMaths\Resources\image-20220408130257330.png)

## Convolutional Layers

- Using the **sample images** from the *sklearn dataset*. First **load the images** and *inspect the pixel values*

```python
from sklearn.datasets import load_sample_image

# Load sample images
china_img = load_sample_image("china.jpg")
flower_img = load_sample_image("flower.jpg")
```

```python
plt.figure()
plt.imshow(china_img)
plt.colorbar()
plt.grid(False)
plt.show()
```

![image-20220408130432668](D:\University\Notes\DiscreteMaths\Resources\image-20220408130432668.png)

- Values are in range $0$ to $255$.
- Scale in range $0$ to $1$ before feeding them **to our models**.
- Dividing them by $255$ 

```python
china = china_img / 255.0
flower = flower_img / 255.0
images = np.array([china, flower])
print(images.shape)
```

- Images are **3D tensors** of $[height, width, channels]$ 
- Store this information in relevant variables then **create two filters** (*convolutional kernels*).

---

- First filter **filter** $\to$ **black square** with a *vertical line* in the *middle* $\to$ $7 \times 7$ full of $0s$ apart from the **central column** of $1s$.

- Neurons using **these weights** will *ignore everything* in their *receptive field* except for the **central vertical line**.

---

- The **second filter** is a *black square* with a **horizontal line** in the *middle instead*
- Neurons using these weights will *ignore everything* in their **receptive field** except for the **central horizontal line**

  ```python
  batch_size, height, width, channels = images.shape
  
  # Create 2 filters
  filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
  filters[:, 3, :, 0] = 1  # vertical line in the middle
  filters[3, :, :, 1] = 1  # horizontal line in the middle
  ```

- Apply the filters

```python

# images as the input
# filters created above are aplied as filters to the input
# stride defines the sliding window for each dimension of the input
# padding defines the padding strategy, using "SAME". This uses zero padding if necessary.
# Zeros are added as evenly as possible around the inputs.
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")

plt.imshow(outputs[0, :, :, 1], cmap="gray") # this will plot 1st image's 2nd feature map
plt.axis("off")
plt.show()
```

![image-20220408131809667](D:\University\Notes\DiscreteMaths\Resources\image-20220408131809667.png)

- Inputs to `tf.nn.conv2d` are **hyperparameters** 
- We can explore the number of **filters, heights , widths, strides** and the **padding type**.
- In practice $\to$ cross validation experiments are **quite expensive** on CNNs, may rely on the common CNN architectures in Keras/tf.

---

- Inspect result of applying **different feature maps** to each of the images
- The *output above* shows the *neurons* with the **horizontal line filter** applied
- This makes the **horizontal white lines enhanced**  , blurring the rest of the image.

---

- For each of the **images** and **feature maps**

```python
for image_index in (0, 1):
    for feature_map_index in (0, 1):
        plt.subplot(2, 2, image_index * 2 + feature_map_index + 1)
        plot_image(outputs[image_index, :, :, feature_map_index])

plt.show()
```

![image-20220408133502267](D:\University\Notes\DiscreteMaths\Resources\image-20220408133502267.png)

- Crop them by **extracting sub images**
- Save new images with the **feature maps applied to them** as follows:

```python
def crop(images, rh1, rh2, rw1, rw2):
    return images[rw1:rw2, rh1:rh2]

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join("./images/", fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

- Crop the input image and inspect the results

```python
plot_image(crop(images[0, :, :, 0], 150, 220, 130, 250))
#save_fig("china_original", tight_layout=False)
plt.show()

for feature_map_index, filename in enumerate(["china_vertical", "china_horizontal"]):
    plot_image(crop(outputs[0, :, :, feature_map_index], 150, 220, 130, 250))
    #save_fig(filename, tight_layout=False)
    plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20220408133631852.png" alt="image-20220408133631852" style="zoom:67%;" />

```python
plot_image(crop(images[1, :, :, 0], 200, 270, 250, 370))
#save_fig("flower_original", tight_layout=False)
plt.show()

for feature_map_index, filename in enumerate(["flower_vertical", "flower_horizontal"]):
    plot_image(crop(outputs[1, :, :, feature_map_index], 200, 270, 250, 370))
    #save_fig(filename, tight_layout=False)
    plt.show()
```

- We finally visualise the feature maps (filters).

```python
plot_image(filters[:, :, 0, 0]) #horizontal
plt.show()
plot_image(filters[:, :, 0, 1]) #vertical
plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20220408134004445.png" alt="image-20220408134004445" style="zoom:67%;" />

- Finally visualise the feature maps

```python
plot_image(filters[:, :, 0, 0]) #horizontal
plt.show()
plot_image(filters[:, :, 0, 1]) #vertical
plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20220408134044648.png" alt="image-20220408134044648" style="zoom:50%;" />

- To apply a 2D conv layer using keras `keras.layers.Conv2D()`.
  1. We apply 32 *filters* here 
     - Let the network figure out the filters, rather than initialising them ourselves.
  2. Using *kernel size* of $3$ 
     - Defines equal height and width for the filters
  3. Using *stride* of $1$ 
  4. Using `"SAME"` padding scheme as above
  5. Using **ReLU activation function**

```python
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="SAME", activation="relu")
```

```python
tf.keras.backend.set_floatx('float64')
outputs = conv(images)
```

```python
for image_index in (0, 1):
    plt.subplot(1, 2, image_index + 1)
    plot_image(crop(outputs[image_index, :, :, 0], 150, 220, 130, 250))

plt.show()

plot_image(crop(outputs[0, :, :, 0], 150, 220, 130, 250))
plt.show()

plot_image(crop(outputs[1, :, :, 0], 200, 270, 250, 370))
plt.show()
```

![image-20220408135025011](D:\University\Notes\DiscreteMaths\Resources\image-20220408135025011.png)

## Pooling Layers

### Max pooling

- Easy within keras. `keras.layers.MaxPool2D` 
- Below we are applying **max pooling** (*subsampling* taking the **maximum value** with the **pooling kernel** of $2 \times 2$) 

```python
max_pool = keras.layers.MaxPool2D(pool_size=2)
```

- See what effect has **on our images**.
- Below we *crop the images* for a **better visibility** and then apply the **max pooling layer**.

```python
tf.keras.backend.set_floatx('float32')
cropped_images = np.array([crop(image, 150, 220, 130, 250) for image in images], dtype=np.float32)
output = max_pool(cropped_images)
```

- Output becomes a **bit smaller**
  - Pooling kernel $2\times 2$ and stride $2$ $\to$ obtain an output $4$ times smaller dropping the $75\%$ of the **input values**.
- The *goal of this modification* is to reduce the **computational load**, the **memory usage**, the **number of parameters** and make the **network tolerate** some **image shift**.

```python
fig = plt.figure(figsize=(12, 8))
gs = mpl.gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[2, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("Input", fontsize=14)
ax1.imshow(cropped_images[0])  # plot the 1st image
ax1.axis("off")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("Output", fontsize=14)
ax2.imshow(output[0])  # plot the output for the 1st image
ax2.axis("off")
#save_fig("china_max_pooling")
plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20220408140756044.png" alt="image-20220408140756044" style="zoom:80%;" />

### Depth-wise pooling

- Pooling often applies to **every input channel independently**. This makes the **output depth** the *same* as the **input depth**.
- Can apply **depth wise pooling** $\to$ preserving the **images input dimensions** (*height / width*) reducing depth (*number of channels*)
- May implement this **using** a *custom* `Lambda` layer with `tf.nn.max_pool`:

```python
depth_pool = keras.layers.Lambda(lambda X: tf.nn.max_pool(
    X, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3), padding="VALID")) # for pool size = 3
depth_output = depth_pool(cropped_images)
depth_output.shape
```

```python
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Input", fontsize=14)
plot_color_image(cropped_images[0])  # plot the 1st image
plt.subplot(1, 2, 2)
plt.title("Output", fontsize=14)
plot_image(depth_output[0, ..., 0])  # plot the output for the 1st image
plt.axis("off")
plt.show()
```

<img src="D:\University\Notes\DiscreteMaths\Resources\image-20220408140742038.png" alt="image-20220408140742038" style="zoom:80%;" />

## Fashion MNIST with a CNN

- Working with an image datasets
- First load it and split **into training, validation and test sets**.

```python
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
```

- Lets *run some sanity checks* $\to$ check the dimensionality of the **training set input** (should tell you the *number of images* and their height / width)

```python
X_train.shape
```

- Clearly the **number of labels** should be *equal* to *number of input images*

```python
len(y_train)
```

- We have *looked the images before* $\to$ lets inspect the **first image** from this *dataset*, too.

```python
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

- Once again, the *pixels have values* in the range of $0$ to $255$ 
- Scale them to the **range** of $[0,1]$ as we did before.

```python
train_images = X_train / 255.0
valid_images = X_valid / 255.0
test_images = X_test / 255.0
```

- Let us view what the first 10 labels are in the **training subset**

```python
y_train[:10]
```

- Class labels are **represented numerically**. This is what we want for the **classifier**.
- However , this makes it **harder** to *interpret* the **results** us.
- Class names are **not included** in the *dataset* therefore store them here and **see** how they **correspond** to the **image**.

```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
    
plt.show()
```

- Now apply the **standardisation** to the inputs

```python
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
```

```python
from functools import partial

# create partial function, allows us to change parameters in the conv layers as we go


DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

#our model below applies a sequence of conv layers with varied number of filters (thus making the net deeper with feature maps)

# Followed by max pooling layers 

# Finally applies a softmax activation function at last layer to predict across 10 classes
model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
])
```

```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10] # pretend we have new images
y_pred = model.predict(X_new)
```

```python
plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_test[i]])
plt.show()

result = tf.argmax(y_pred, axis=1)

for i in range(10):
    print(class_names[result.numpy()[i]], "?=", class_names[y_test[i]])
```

