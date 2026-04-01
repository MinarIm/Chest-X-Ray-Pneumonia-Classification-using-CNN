<<<<<<< HEAD
# Chest X-Ray Pneumonia Classification Using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Normal** or **Pneumonia**. The model is built using TensorFlow/Keras and trained on a medical imaging dataset with data augmentation techniques to improve generalization.

## Dataset
- **Structure**: Organized into three sets (train, validation, test)
- **Classes**: 
  - NORMAL: Healthy X-ray images
  - PNEUMONIA: X-ray images showing pneumonia
- **Image Resolution**: 150×150 pixels (RGB)
- **Directory Layout**:
  ```
  dataset/
  ├── train/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  ├── val/
  │   ├── NORMAL/
  │   └── PNEUMONIA/
  └── test/
      ├── NORMAL/
      └── PNEUMONIA/
  ```

## Model Architecture

### CNN Layers
```
Input (150, 150, 3)
  ↓
Conv2D (32 filters, 3×3, ReLU) → MaxPooling2D (2×2)
  ↓
Conv2D (64 filters, 3×3, ReLU) → MaxPooling2D (2×2)
  ↓
Conv2D (128 filters, 3×3, ReLU) → MaxPooling2D (2×2)
  ↓
Flatten
  ↓
Dense (128 units, ReLU) → Dropout (0.5)
  ↓
Dense (1 unit, Sigmoid) [Binary Classification Output]
```

### Key Specifications
- **Total Convolutional Layers**: 3
- **Filters**: 32 → 64 → 128
- **Activation Function**: ReLU (hidden layers), Sigmoid (output)
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Regularization**: Dropout (0.5)
- **Training Epochs**: 5
- **Batch Size**: 16

## Data Augmentation
To improve model robustness and reduce overfitting:
- **Rescaling**: 1/255 normalization
- **Zoom Range**: 0.2
- **Shear Range**: 0.2
- **Horizontal Flip**: Enabled

## Results
The model produces:
- **Training Accuracy Plot**: Shows accuracy improvement over epochs
- **Validation Accuracy Plot**: Monitors generalization performance
- **Test Accuracy**: Final evaluation on unseen test data
- **Saved Model**: `pneumonia_cnn_model.h5` in H5 format

## Requirements
```
tensorflow>=2.0
keras
numpy
matplotlib
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Model
```bash
python main.py
```

### Script Flow
1. Loads and preprocesses training, validation, and test data
2. Builds and compiles the CNN model
3. Trains the model on training data with validation monitoring
4. Displays accuracy and loss graphs
5. Evaluates performance on test set
6. Saves the trained model as `pneumonia_cnn_model.h5`

### Loading a Saved Model
```python
from tensorflow.keras.models import load_model
model = load_model('pneumonia_cnn_model.h5')

# Make predictions
predictions = model.predict(test_data)
```

## Model Performance
- Binary classification task (Normal vs Pneumonia)
- Evaluated on test set accuracy metric
- Output: Probability score (0 = Normal, 1 = Pneumonia)

## Files
- `main.py`: Main training and evaluation script
- `pneumonia_cnn_model.h5`: Trained model weights
- `README.md`: Project documentation
- `dataset/`: Training, validation, and test data folders

## Author
Minar Imtiaz

## License
This project is open source and available under the MIT License.
