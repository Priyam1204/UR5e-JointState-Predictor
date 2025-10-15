import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path

class JointPredictor(nn.Module):
    """Neural network to predict next joint positions"""
    def __init__(self, InputDim=18, HiddenDim=128, OutputDim=6):
        super().__init__()
        # Input: 6 positions + 6 velocities + 6 controls = 18 features
        # Output: 6 next joint positions
        
        self.Network = nn.Sequential(
            nn.Linear(InputDim, HiddenDim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HiddenDim, HiddenDim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HiddenDim, HiddenDim//2),
            nn.ReLU(),
            nn.Linear(HiddenDim//2, OutputDim)
        )
        
        # Initialize weights
        for Module in self.modules():
            if isinstance(Module, nn.Linear):
                nn.init.xavier_uniform_(Module.weight)
                nn.init.zeros_(Module.bias)
    
    def forward(self, Input):
        return self.Network(Input)

def LoadAndPrepareData():
    """Load and prepare training data"""
    print("📊 Loading trajectory data...")
    
    try:
        Data = np.load('joint_trajectory_data.npz')
        Positions = Data['positions']
        Velocities = Data['velocities']
        Controls = Data['controls']
        NextPositions = Data['next_positions']
        
        print(f"✅ Loaded {len(Positions)} samples")
        print(f"   Positions: {Positions.shape}")
        print(f"   Velocities: {Velocities.shape}")
        print(f"   Controls: {Controls.shape}")
        print(f"   Next positions: {NextPositions.shape}")
        
    except FileNotFoundError:
        print("❌ Error: joint_trajectory_data.npz not found!")
        print("Please run: python DataCollector.py first")
        return None, None, None, None
    
    # Create input features (current pos + vel + controls)
    InputFeatures = np.concatenate([Positions, Velocities, Controls], axis=1)
    TargetOutput = NextPositions
    
    print(f"📋 Input features shape: {InputFeatures.shape}")
    print(f"📋 Target shape: {TargetOutput.shape}")
    
    # Data normalization
    InputMean = np.mean(InputFeatures, axis=0)
    InputStd = np.std(InputFeatures, axis=0) + 1e-8  # Avoid division by zero
    OutputMean = np.mean(TargetOutput, axis=0)
    OutputStd = np.std(TargetOutput, axis=0) + 1e-8
    
    InputNormalized = (InputFeatures - InputMean) / InputStd
    OutputNormalized = (TargetOutput - OutputMean) / OutputStd
    
    # Save normalization parameters
    np.savez('normalization_params.npz',
             X_mean=InputMean, X_std=InputStd,
             y_mean=OutputMean, y_std=OutputStd)
    
    print("✅ Data normalized and parameters saved")
    
    return InputNormalized, OutputNormalized, (InputMean, InputStd, OutputMean, OutputStd)

def TrainModel(InputFeatures, TargetOutput, NormalizationParams):
    """Train the joint predictor neural network"""
    
    # Convert to PyTorch tensors
    InputTensor = torch.FloatTensor(InputFeatures)
    OutputTensor = torch.FloatTensor(TargetOutput)
    
    # Train/validation split (80/20)
    SplitIndex = int(0.8 * len(InputFeatures))
    TrainInput, ValidationInput = InputTensor[:SplitIndex], InputTensor[SplitIndex:]
    TrainOutput, ValidationOutput = OutputTensor[:SplitIndex], OutputTensor[SplitIndex:]
    
    print(f"🔄 Training set: {len(TrainInput)} samples")
    print(f"🔄 Validation set: {len(ValidationInput)} samples")
    
    # Data loaders
    TrainDataset = TensorDataset(TrainInput, TrainOutput)
    ValidationDataset = TensorDataset(ValidationInput, ValidationOutput)
    TrainLoader = DataLoader(TrainDataset, batch_size=128, shuffle=True)
    ValidationLoader = DataLoader(ValidationDataset, batch_size=128, shuffle=False)
    
    # Model, loss, optimizer
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {Device}")
    
    Model = JointPredictor(InputDim=18, HiddenDim=128, OutputDim=6).to(Device)
    Criterion = nn.MSELoss()
    Optimizer = optim.Adam(Model.parameters(), lr=0.001, weight_decay=1e-4)
    Scheduler = optim.lr_scheduler.ReduceLROnPlateau(Optimizer, patience=10, factor=0.5)
    
    # Training loop
    TrainLosses = []
    ValidationLosses = []
    BestValidationLoss = float('inf')
    PatienceCounter = 0
    
    print("🚀 Starting training...")
    
    for Epoch in range(100):
        # Training phase
        Model.train()
        TrainLoss = 0.0
        for BatchInput, BatchOutput in TrainLoader:
            BatchInput, BatchOutput = BatchInput.to(Device), BatchOutput.to(Device)
            
            Optimizer.zero_grad()
            Predictions = Model(BatchInput)
            Loss = Criterion(Predictions, BatchOutput)
            Loss.backward()
            Optimizer.step()
            
            TrainLoss += Loss.item()
        
        AverageTrainLoss = TrainLoss / len(TrainLoader)
        TrainLosses.append(AverageTrainLoss)
        
        # Validation phase
        Model.eval()
        ValidationLoss = 0.0
        with torch.no_grad():
            for BatchInput, BatchOutput in ValidationLoader:
                BatchInput, BatchOutput = BatchInput.to(Device), BatchOutput.to(Device)
                Predictions = Model(BatchInput)
                Loss = Criterion(Predictions, BatchOutput)
                ValidationLoss += Loss.item()
        
        AverageValidationLoss = ValidationLoss / len(ValidationLoader)
        ValidationLosses.append(AverageValidationLoss)
        
        # Learning rate scheduling
        Scheduler.step(AverageValidationLoss)
        
        # Early stopping
        if AverageValidationLoss < BestValidationLoss:
            BestValidationLoss = AverageValidationLoss
            PatienceCounter = 0
            # Save best model
            torch.save(Model.state_dict(), 'best_joint_predictor.pth')
        else:
            PatienceCounter += 1
        
        # Print progress
        if Epoch % 10 == 0 or PatienceCounter == 0:
            print(f'Epoch {Epoch:3d} | Train Loss: {AverageTrainLoss:.6f} | Val Loss: {AverageValidationLoss:.6f} | LR: {Optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping
        if PatienceCounter >= 20:
            print(f"⏹️  Early stopping at epoch {Epoch}")
            break
    
    # Load best model
    Model.load_state_dict(torch.load('best_joint_predictor.pth'))
    
    print(f"✅ Training completed!")
    print(f"   Best validation loss: {BestValidationLoss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(TrainLosses, label='Training Loss', alpha=0.8)
    plt.plot(ValidationLosses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test predictions on validation set
    Model.eval()
    with torch.no_grad():
        ValidationPredictions = Model(ValidationInput.to(Device)).cpu().numpy()
        ValidationTargets = ValidationOutput.numpy()
    
    # Denormalize for visualization
    InputMean, InputStd, OutputMean, OutputStd = NormalizationParams
    ValidationPredictionsDenormalized = ValidationPredictions * OutputStd + OutputMean
    ValidationTargetsDenormalized = ValidationTargets * OutputStd + OutputMean
    
    # Plot prediction accuracy
    plt.subplot(1, 2, 2)
    JointNames = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
    Errors = np.abs(ValidationPredictionsDenormalized - ValidationTargetsDenormalized)
    MeanErrors = np.mean(Errors, axis=0)
    
    plt.bar(range(6), MeanErrors)
    plt.xlabel('Joint')
    plt.ylabel('Mean Absolute Error (rad)')
    plt.title('Prediction Accuracy by Joint')
    plt.xticks(range(6), [Name.split()[0] for Name in JointNames], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Mean prediction errors (rad):")
    for i, Name in enumerate(JointNames):
        print(f"   {Name}: {MeanErrors[i]:.4f}")
    
    return Model

def Main():
    print("🤖 UR5e Joint Position Predictor Training")
    print("=" * 50)
    
    # Load and prepare data
    InputFeatures, TargetOutput, NormalizationParams = LoadAndPrepareData()
    if InputFeatures is None:
        return
    
    # Train model
    Model = TrainModel(InputFeatures, TargetOutput, NormalizationParams)
    
    print("\n✅ Training pipeline complete!")
    print("Files saved:")
    print("  📁 best_joint_predictor.pth - Trained model")
    print("  📁 normalization_params.npz - Data normalization parameters") 
    print("  📁 training_results.png - Training curves and accuracy")
    print("\nNext step: python test_predictor.py")

if __name__ == "__main__":
    Main()