# UR5e Joint State Predictor

Train a neural network to predict UR5e robot joint positions using MuJoCo simulation data.

## What it does
- Collects robot movement data from MuJoCo simulation
- Trains a neural network to predict next joint positions
- Tests prediction accuracy on new data

## Project Structure
```
UR5e-JointState-Predictor/
├── Models/ur5e_urdf/          # Robot model files
├── Src/
│   ├── DataCollector.py       # Collect training data
│   ├── JointPredictorTrainer.py    # Train the model
│   └── JointPredictorEvaluator.py  # Test the model
├── Utils/
│   ├── DataVisualizer.py      # Plot results
│   └── UR5eSimTest.py         # Test robot viewer
└── requirements.txt
```

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Test robot simulation**
```bash
python Utils/UR5eSimTest.py
```

3. **Collect data**
```bash
python Src/DataCollector.py
```

4. **Train model**
```bash
python Src/JointPredictorTrainer.py
```

5. **Test predictions**
```bash
python Src/JointPredictorEvaluator.py
```

## Model Details
- **Input**: Current joint positions (6) + velocities (6) + control torques (6) = 18 features
- **Output**: Next joint positions (6)
- **Architecture**: 18 → 128 → 64 → 6 (fully connected)
- **Training data**: ~50k samples from random robot movements

## Results
- Prediction accuracy: ~0.6° average error per joint
- Training time: ~5 minutes
- Inference speed: Real-time capable

## Files Generated
- `joint_trajectory_data.npz` - Training dataset
- `best_joint_predictor.pth` - Trained model
- `training_results.png` - Training curves
- `prediction_accuracy_analysis.png` - Error analysis

## Requirements
- Python 3.8+
- PyTorch
- MuJoCo
- NumPy, Matplotlib

---
**Author**: Priyam  
**Status**: In Progress - Needs Improveent