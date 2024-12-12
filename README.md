## 1. Introduction
- **Pattern Name**: Transfer Learning  
- **Type**: Patterns That Modify Model Training
- **Introduction**: Transfer learning is a machine learning design pattern aimed at enhancing model performance by leveraging pre-existing knowledge from related tasks. This paradigm reduces the need for extensive data collection and computational resources by utilizing pre-trained models as a foundation for new tasks. It is particularly valuable in scenarios where labeled data is scarce, or the cost of training models from scratch is prohibitive. Transfer learning works by adapting the learned features of a pre-trained model to solve a different but related problem, thereby accelerating development and improving accuracy. For example, a convolutional neural network pre-trained on the ImageNet dataset can be fine-tuned to classify medical images, even when the medical dataset is small. Similarly, in natural language processing (NLP), models like BERT or GPT are fine-tuned for downstream tasks like sentiment analysis or question answering, significantly cutting training time and boosting performance. The approach is not only efficient but also helps mitigate overfitting, especially in cases with limited training data, as the pre-trained models already encompass robust, generalizable features derived from large-scale datasets. Transfer learning has revolutionized fields like computer vision, NLP, and audio processing, making it a cornerstone of modern AI solutions.
---

## 2. YouTube Resource
- [Transfer Learning - DeepLearningAI](https://www.youtube.com/watch?v=yofjFQddwHE&t=24s&ab_channel=DeepLearningAI)
- [Transfer Learning - simple version](https://www.youtube.com/watch?v=DyPW-994t7w)
---

## 3. Rationale
-The primary rationale behind transfer learning is efficiency. Training models from scratch often requires large datasets and significant computational resources. Transfer learning addresses these issues by reusing a model trained on a similar problem. This enables rapid prototyping, reduces training time, and allows practitioners to achieve high performance with less data. For example, a model trained on ImageNet can serve as a strong foundation for classifying specific types of medical images. Moreover, it helps mitigate overfitting in scenarios with limited training data, as pre-trained models already contain generalizable features.
- Many machine learning tasks lack sufficient labeled data to train complex models from scratch.  
- In Conclusion Transfer Learning reuses knowledge from existing models trained on large datasets, enabling quicker adaptation to new tasks and improving efficiency.

---

## 4. UML Diagram
![Image Description](https://github.com/Paschal78/Transfer_Learning_Design_Pattern/blob/patch-1/LAST%20UML-10.png?raw=true))

---

## 5. Simple Code Example
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# Load pre-trained model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Create a new model with additional layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Freeze base model layers
base_model.trainable = False

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```
## 6. Common usage in the software industry 
## Transfer learning is widely used in:

### a. **Image Classification**
- Pre-trained models, such as those trained on ImageNet, are commonly used to extract features. These features serve as input to train a new classifier for a specific dataset or task.

### b. **Object Detection**
- Models like Faster R-CNN are used as feature extractors. Pre-trained layers identify features, and additional layers handle object detection for new datasets.

### c. **Natural Language Processing (NLP)** 
- Pre-trained models like BERT are fine-tuned for tasks such as sentiment analysis or text translation, leveraging their existing language understanding.

### d. **Speech Recognition**
 - Pre-trained models like DeepSpeech extract key audio features, which are then used to build a specialized speech recognition system.

### e. **Anomaly Detection**
- Pre-trained models learn normal patterns and detect irregularities by analyzing deviations in new data.

### f. **Recommendation Systems**
-Pre-trained models trained on user behavior initialize features in recommendation systems, enhancing predictions using learned patterns.

### g. **Medical Imaging**
- Transfer learning is commonly used for detecting diseases in medical images (e.g., X-rays or MRIs), where datasets are often small. Pre-trained CNNs fine-tune on these specialized tasks, achieving state-of-the-art accuracy.

### h. **Time-Series Analysis**
- Pre-trained models, particularly in fields like finance and energy, detect patterns and predict trends or anomalies in time-series data, reducing computational costs while improving accuracy.

### i. **Robotics**
- Transfer learning allows robotic systems to adapt pre-trained motor control or image recognition models for specific tasks, such as object manipulation or navigation in different environments.

## 7. A complex code problem
### Requirements
```
pip install tensorflow
```

### Complex code
The full code is available in the attached file: transferLearning.py. You can find it in the repository for this project.

### Output
```
C:\Users\mirac\OneDrive\Desktop\Seneca Learnings\Term 5\Design Patterns\Assignment 6\transferLearning>python transferLearning.py
2024-12-09 20:25:34.266189: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2024-12-09 20:25:35.590326: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
Found 1257 images belonging to 3 classes.
Found 3 images belonging to 3 classes.
Classes in dataset: {'birds': 0, 'cats': 1, 'dogs': 2}
Number of training samples: 1257
Number of validation samples: 3
2024-12-09 20:25:39.069405: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\mirac\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
Epoch 1/5
629/629 ━━━━━━━━━━━━━━━━━━━━ 0s 271ms/step - accuracy: 0.3951 - loss: 7.7640C:\Users\mirac\AppData\Local\Programs\Python\Python312\Lib\site-packages\keras\src\trainers\data_adapters\py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.
  self._warn_if_super_not_called()
629/629 ━━━━━━━━━━━━━━━━━━━━ 178s 275ms/step - accuracy: 0.3951 - loss: 7.7572 - val_accuracy: 0.3333 - val_loss: 1.1041
Epoch 2/5
629/629 ━━━━━━━━━━━━━━━━━━━━ 170s 270ms/step - accuracy: 0.3951 - loss: 1.1056 - val_accuracy: 0.3333 - val_loss: 1.1164
Epoch 3/5
629/629 ━━━━━━━━━━━━━━━━━━━━ 169s 269ms/step - accuracy: 0.4067 - loss: 1.1079 - val_accuracy: 0.3333 - val_loss: 1.1266
Epoch 4/5
629/629 ━━━━━━━━━━━━━━━━━━━━ 170s 270ms/step - accuracy: 0.4246 - loss: 1.0821 - val_accuracy: 0.3333 - val_loss: 1.1329
Epoch 5/5
629/629 ━━━━━━━━━━━━━━━━━━━━ 171s 272ms/step - accuracy: 0.4034 - loss: 1.1126 - val_accuracy: 0.3333 - val_loss: 1.1370
3/3 ━━━━━━━━━━━━━━━━━━━━ 0s 69ms/step - accuracy: 0.5417 - loss: 1.1095
Validation Accuracy: 33.33%
```

## 8. References
1. Crawford, C. (2018, February 16). *Cat dataset*. Kaggle. [https://www.kaggle.com/datasets/crawford/cat-dataset](https://www.kaggle.com/datasets/crawford/cat-dataset)

2. *Labeled image datasets for Computer Vision.* (n.d.). Labeled image datasets for computer vision. [https://images.cv/download/dog/19921](https://images.cv/download/dog/19921)

3. Prashanth, C. M. (2023, June 13). *Birds Image Dataset*. Kaggle. [https://www.kaggle.com/datasets/klu2000030172/birds-image-dataset](https://www.kaggle.com/datasets/klu2000030172/birds-image-dataset)

4. Tensorflow. *TensorFlow.* (n.d.). [https://www.tensorflow.org/](https://www.tensorflow.org/)

5. YouTube. (n.d.). *YouTube.* [https://www.youtube.com/watch?v=yofjFQddwHE&t=24s&ab_channel=DeepLearningAI](https://www.youtube.com/watch?v=yofjFQddwHE&t=24s&ab_channel=DeepLearningAI)
