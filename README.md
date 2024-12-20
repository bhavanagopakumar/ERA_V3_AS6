# ERA-V3 Session 6 Assignment

A PyTorch CNN implementation for MNIST digit classification that achieves exceptional accuracy with minimal parameters.

## Key Achievements 🏆

### 1. High Accuracy
- **Peak Performance**: 99.39% (9939/10000)
- **Consistent Results**: Maintained 99%+ accuracy across multiple epochs
- **Stable Training**: No signs of overfitting

### 2. Efficient Architecture (6.6K Parameters)
- **Lightweight Design**: Only 6,618 total parameters
- **Smart Feature Extraction**: 
  - Strategic use of 1x1 convolutions
  - Optimal channel reduction
  - Well under 20K parameter limit

### 3. Quick Convergence
- Reached 98%+ accuracy in first epoch
- Achieved 99%+ by sixth epoch
- Best accuracy - 99.39%

## Model Architecture

### Key Features
1. **Optimized Layer Structure**:
   - Input → 1 channel
   - Max channels: 16
   - Output → 10 classes
   - Two 1x1 convolution layers for channel reduction

2. **Regularization**:
   - Batch Normalization after convolutions
   - Dropout (7%) for preventing overfitting
   - MaxPooling for spatial reduction

3. **Training Configuration**:
   - Optimizer: SGD with momentum (0.9)
   - Learning Rate: 0.01
   - Batch Size: 64

## Training Results 
```
Epoch 1: 98.51% (9851/10000)
Epoch 2: 98.77% (9877/10000)
Epoch 3: 98.96% (9896/10000)
Epoch 4: 98.98% (9898/10000)
Epoch 5: 98.78% (9978/10000)
Epoch 6: 99.17% (9917/10000)
Epoch 7: 98.92% (9892/10000)
Epoch 8: 99.19% (9919/10000)
Epoch 9: 99.30% (9930/10000)
Epoch 10: 99.17% (9917/10000)
Epoch 11: 99.26% (9926/10000)
Epoch 12: 99.33% (9933/10000)
Epoch 13: 99.09% (9909/10000)
Epoch 14: 99.30% (9930/10000)
Epoch 15: 99.26% (9926/10000)
Epoch 16: 99.35% (9935/10000)
Epoch 17: 99.38% (9938/10000) 
Epoch 18: 99.33% (9933/10000)
Epoch 19: 99.39% (9939/10000) ← Best
```

## Note
```
* Detailed test results are in the ERA_v3_S6.ipynb file.
* Model architecture is in the model.py file.
```
