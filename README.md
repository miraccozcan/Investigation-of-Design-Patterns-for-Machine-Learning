# Lesson Plan: Transfer Learning

## 1. Introduction
- **Pattern Name**: Transfer Learning  
- **Type**: Patterns That Modify Model Training (Machine Learning)  
- **Short Introduction**: Transfer Learning uses a pre-trained model to solve new, related tasks. It helps reduce the need for large datasets and extensive computational resources.

---

## 2. YouTube Resource
- [Transfer Learning - DeepLearningAI](https://www.youtube.com/watch?v=yofjFQddwHE&t=24s&ab_channel=DeepLearningAI)

---

## 3. Rationale
- Many machine learning tasks lack sufficient labeled data to train complex models from scratch.  
- Transfer Learning reuses knowledge from existing models trained on large datasets, enabling quicker adaptation to new tasks and improving efficiency.

---

## 4. Diagram
### Workflow Components:
1. **Pre-Trained Model**: A model trained on a large dataset (e.g., ImageNet for vision tasks).  
2. **Feature Extractor**: Extracts relevant features for the new task.  
3. **Fine-Tuning**: Adjusts the model’s layers to optimize for the specific dataset.

### Diagram Description:
The workflow shows how the pre-trained model's knowledge is transferred and fine-tuned for new applications.

---

## 5. Simple Code Example (Python using TensorFlow)
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
