# ERA-V3 Session 6 Assignment

A PyTorch CNN implementation for MNIST digit classification that achieves exceptional accuracy with minimal parameters.

## Key Achievements 🏆

### 1. High Accuracy (99.4%+)
- **Peak Performance**: 99.49% (9949/10000)
- **Consistent Results**: Maintained 99%+ accuracy across multiple epochs
- **Stable Training**: No signs of overfitting

### 2. Efficient Architecture (13.5K Parameters)
- **Lightweight Design**: Only 13,514 total parameters
- **Smart Feature Extraction**: 
  - Strategic use of 1x1 convolutions
  - Optimal channel reduction
  - Well under 20K parameter limit

### 3. Quick Convergence
- Reached 98%+ accuracy in first epoch
- Achieved 99%+ by fifth epoch
- Best accuracy (99.49%) in less than 17 epochs

## Model Architecture

### Key Features
1. **Optimized Layer Structure**:
   - Input → 1 channel
   - Max channels: 32
   - Output → 10 classes
   - Two 1x1 convolution layers for channel reduction

2. **Regularization**:
   - Batch Normalization after convolutions
   - Dropout (8%) for preventing overfitting
   - MaxPooling for spatial reduction

3. **Training Configuration**:
   - Optimizer: SGD with momentum (0.9)
   - Learning Rate: 0.01
   - Batch Size: 64

## Training Results 
```
Epoch 1: 98.56% (9856/10000)
Epoch 2: 98.70% (9870/10000)
Epoch 3: 98.90% (9890/10000)
Epoch 4: 98.83% (9883/10000)
Epoch 5: 99.13% (9913/10000)
Epoch 6: 99.17% (9917/10000)
Epoch 7: 99.26% (9926/10000)
Epoch 8: 99.35% (9935/10000)
Epoch 9: 99.27% (9927/10000)
Epoch 10: 99.31% (9931/10000)
Epoch 11: 99.40% (9940/10000)
Epoch 12: 99.30% (9930/10000)
Epoch 13: 99.24% (9924/10000)
Epoch 14: 99.45% (9945/10000)
Epoch 15: 99.40% (9940/10000)
Epoch 16: 99.33% (9933/10000)
Epoch 17: 99.49% (9949/10000) ← Best
Epoch 18: 99.38% (9938/10000)
Epoch 19: 99.32% (9932/10000)
```
