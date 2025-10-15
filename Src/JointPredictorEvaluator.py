import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import torch
from JointPredictorTrainer import JointPredictor
import matplotlib.pyplot as plt

MODEL_PATH = "Models/ur5e_urdf/ur5e.mjcf.xml"

class JointPredictorTester:
    def __init__(self):
        # Load our trained neural network model
        print("Loading trained model from disk...")
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.NeuralNetwork = JointPredictor(input_dim=18, hidden_dim=128, output_dim=6).to(self.Device)
        self.NeuralNetwork.load_state_dict(torch.load('best_joint_predictor.pth'))
        self.NeuralNetwork.eval()
        
        # Load the normalization parameters that we saved during training
        NormalizationData = np.load('normalization_params.npz')
        self.InputMean = NormalizationData['X_mean']
        self.InputStandardDeviation = NormalizationData['X_std'] 
        self.OutputMean = NormalizationData['y_mean']
        self.OutputStandardDeviation = NormalizationData['y_std']
        
        print("Model and normalization parameters successfully loaded")
    
    def PredictNextJointPositions(self, CurrentPositions, CurrentVelocities, ControlCommands):
        """Use our trained neural network to predict what the next joint positions will be"""
        
        # Combine all input features into a single array
        CombinedInputFeatures = np.concatenate([CurrentPositions, CurrentVelocities, ControlCommands])
        
        # Normalize the input using the same statistics from training
        NormalizedInput = (CombinedInputFeatures - self.InputMean) / self.InputStandardDeviation
        
        # Make a prediction using our neural network
        with torch.no_grad():
            InputTensor = torch.FloatTensor(NormalizedInput).unsqueeze(0).to(self.Device)
            NormalizedPrediction = self.NeuralNetwork(InputTensor).cpu().numpy()[0]
        
        # Convert the normalized prediction back to real joint angles
        ActualPrediction = NormalizedPrediction * self.OutputStandardDeviation + self.OutputMean
        
        return ActualPrediction

def TestPredictionsWithVisualization():
    """Watch our neural network predictions in action alongside the real robot simulation"""
    
    print("Visual Prediction Test - Comparing Neural Network vs Real Physics")
    print("Green robot = Real physics simulation")
    print("Red dots = Neural network predictions")
    
    PredictionTester = JointPredictorTester()
    
    # Load the robot model for simulation
    RobotModel = mj.MjModel.from_xml_path(MODEL_PATH)
    SimulationData = mj.MjData(RobotModel)
    
    # Set the robot to a comfortable starting position
    JointNames = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 
                  'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    ComfortableStartingPose = np.deg2rad([0, -90, 90, -90, -90, 0])  # Nice upright pose
    
    for JointIndex, JointName in enumerate(JointNames):
        JointId = mj.mj_name2id(RobotModel, mj.mjtObj.mjOBJ_JOINT, JointName)
        SimulationData.qpos[JointId] = ComfortableStartingPose[JointIndex]
    
    SimulationData.qvel[:] = 0.0  # Start with no movement
    mj.mj_forward(RobotModel, SimulationData)
    
    # Keep track of how well our predictions match reality
    ActualJointPositions = []
    PredictedJointPositions = []
    PredictionErrors = []
    
    with viewer.launch_passive(RobotModel, SimulationData) as RobotViewer:
        SimulationStep = 0
        while RobotViewer.is_running() and SimulationStep < 2000:
            
            # Get the current state of all joints
            CurrentJointPositions = SimulationData.qpos[:6].copy()
            CurrentJointVelocities = SimulationData.qvel[:6].copy()
            
            # Create some smooth, random movement commands
            if SimulationStep == 0:
                ControlCommands = np.random.uniform(-0.1, 0.1, 6)
                ControlCommands[3:] *= 0.5  # Be gentler with the wrist joints
            else:
                # Make the control changes smooth and gradual
                NewControlCommands = np.random.uniform(-0.1, 0.1, 6) 
                NewControlCommands[3:] *= 0.5
                ControlCommands = 0.9 * ControlCommands + 0.1 * NewControlCommands
            
            SimulationData.ctrl[:] = ControlCommands
            
            # Ask our neural network: what will happen next?
            PredictedNextPositions = PredictionTester.PredictNextJointPositions(
                CurrentJointPositions, CurrentJointVelocities, ControlCommands)
            
            # Now let's see what actually happens in the real simulation
            mj.mj_step(RobotModel, SimulationData)
            ActualNextPositions = SimulationData.qpos[:6].copy()
            
            # Calculate how wrong our prediction was
            PredictionError = np.abs(PredictedNextPositions - ActualNextPositions)
            PredictionErrors.append(PredictionError)
            
            # Save this data for later analysis
            ActualJointPositions.append(ActualNextPositions)
            PredictedJointPositions.append(PredictedNextPositions)
            
            # Give us updates on how well we're doing every 200 steps
            if SimulationStep % 200 == 0:
                AverageError = np.mean(PredictionError)
                WorstError = np.max(PredictionError)
                print(f"Step {SimulationStep:4d} | Average Error: {AverageError:.4f} rad | Worst Error: {WorstError:.4f} rad")
                
                # Show how each individual joint is doing
                JointAbbreviations = ['SP', 'SL', 'EL', 'W1', 'W2', 'W3']
                ErrorBreakdown = " | ".join([f"{name}: {error:.3f}" for name, error in zip(JointAbbreviations, PredictionError)])
                print(f"           | {ErrorBreakdown}")
            
            SimulationStep += 1
            RobotViewer.sync()
    
    # Convert our collected data into arrays for analysis
    ActualJointPositions = np.array(ActualJointPositions)
    PredictedJointPositions = np.array(PredictedJointPositions)
    PredictionErrors = np.array(PredictionErrors)
    
    return ActualJointPositions, PredictedJointPositions, PredictionErrors

def AnalyzePredictionAccuracy(ActualPositions, PredictedPositions, Errors):
    """Create detailed charts and statistics about how accurate our predictions were"""
    
    JointNames = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
    
    print("\nDetailed Analysis of Neural Network Performance")
    print("=" * 50)
    
    # Calculate various error statistics
    AverageErrors = np.mean(Errors, axis=0)
    ErrorVariability = np.std(Errors, axis=0)
    WorstErrors = np.max(Errors, axis=0)
    
    print("Prediction accuracy for each joint (in radians):")
    print(f"{'Joint':<15} {'Average':<8} {'Variability':<8} {'Worst':<8} {'Average(deg)':<10}")
    print("-" * 55)
    for JointIndex, JointName in enumerate(JointNames):
        print(f"{JointName:<15} {AverageErrors[JointIndex]:<8.4f} {ErrorVariability[JointIndex]:<8.4f} {WorstErrors[JointIndex]:<8.4f} {np.rad2deg(AverageErrors[JointIndex]):<10.2f}")
    
    OverallPerformance = np.mean(AverageErrors)
    print(f"\nOverall neural network performance: {OverallPerformance:.4f} rad ({np.rad2deg(OverallPerformance):.2f} degrees)")
    
    # Create detailed visualizations
    MainFigure, ChartGrid = plt.subplots(2, 3, figsize=(18, 12))
    MainFigure.suptitle('Neural Network vs Reality: Joint Position Predictions', fontsize=16)
    
    for JointIndex in range(6):
        ChartRow = JointIndex // 3
        ChartColumn = JointIndex % 3
        CurrentChart = ChartGrid[ChartRow, ChartColumn]
        
        # Plot what actually happened vs what we predicted
        TimeSteps = np.arange(len(ActualPositions))
        CurrentChart.plot(TimeSteps, ActualPositions[:, JointIndex], 'g-', label='Reality', alpha=0.8, linewidth=1.5)
        CurrentChart.plot(TimeSteps, PredictedPositions[:, JointIndex], 'r--', label='Neural Network Prediction', alpha=0.7, linewidth=1)
        
        CurrentChart.set_title(f'{JointNames[JointIndex]}')
        CurrentChart.set_xlabel('Time Step')
        CurrentChart.set_ylabel('Joint Angle (radians)')
        CurrentChart.legend()
        CurrentChart.grid(True, alpha=0.3)
        
        # Add a summary box with key statistics
        SummaryText = f'Average Error: {AverageErrors[JointIndex]:.4f} rad\nWorst Error: {WorstErrors[JointIndex]:.4f} rad'
        CurrentChart.text(0.02, 0.98, SummaryText, 
                         transform=CurrentChart.transAxes, verticalalignment='top', 
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('prediction_accuracy_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Create error distribution charts
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([Errors[:, JointIndex] for JointIndex in range(6)], 
                labels=[JointName.split()[0] for JointName in JointNames])
    plt.title('Prediction Error Distribution by Joint')
    plt.ylabel('Prediction Error (radians)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    OverallErrors = np.mean(Errors, axis=1)
    plt.hist(OverallErrors, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(np.mean(OverallErrors), color='red', linestyle='--', 
                label=f'Average: {np.mean(OverallErrors):.4f}')
    plt.title('Overall Prediction Quality Distribution')
    plt.xlabel('Average Prediction Error (radians)')
    plt.ylabel('Number of Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def RunQuickBenchmarkTest():
    """Test our neural network quickly using saved data without the visual simulation"""
    
    print("Quick Performance Test Using Previously Saved Data")
    
    PredictionTester = JointPredictorTester()
    
    # Load some test data we saved earlier
    SavedData = np.load('joint_trajectory_data.npz')
    TestPositions = SavedData['positions'][-1000:]  # Use the most recent 1000 samples
    TestVelocities = SavedData['velocities'][-1000:]
    TestControls = SavedData['controls'][-1000:]
    KnownCorrectAnswers = SavedData['next_positions'][-1000:]
    
    # Test our neural network on all these samples
    AllPredictions = []
    for SampleIndex in range(len(TestPositions)):
        Prediction = PredictionTester.PredictNextJointPositions(
            TestPositions[SampleIndex], TestVelocities[SampleIndex], TestControls[SampleIndex])
        AllPredictions.append(Prediction)
    
    AllPredictions = np.array(AllPredictions)
    AllErrors = np.abs(AllPredictions - KnownCorrectAnswers)
    
    # Report the results
    AverageError = np.mean(AllErrors)
    print(f"Tested neural network on {len(TestPositions)} different scenarios")
    print(f"   Average prediction error: {AverageError:.4f} radians ({np.rad2deg(AverageError):.2f} degrees)")
    print(f"   Worst prediction error: {np.max(AllErrors):.4f} radians ({np.rad2deg(np.max(AllErrors)):.2f} degrees)")

def RunMainProgram():
    print("UR5e Robot Joint Predictor - Neural Network Testing Suite")
    print("=" * 60)
    
    UserChoice = input("\nHow would you like to test the neural network?\n" +
                      "1. Visual test (watch predictions in the robot simulator)\n" +
                      "2. Quick benchmark (fast test with numbers only)\n" +
                      "3. Both tests\n" +
                      "Enter your choice (1/2/3): ")
    
    if UserChoice in ['1', '3']:
        print("\nStarting visual test where you can observe the AI predictions in real-time...")
        ActualData, PredictedData, ErrorData = TestPredictionsWithVisualization()
        AnalyzePredictionAccuracy(ActualData, PredictedData, ErrorData)
    
    if UserChoice in ['2', '3']:
        print("\nRunning quick benchmark test...")
        RunQuickBenchmarkTest()
    
    print("\nTesting complete! Generated files:")
    print("prediction_accuracy_analysis.png - Detailed charts comparing predictions vs reality")
    print("error_distribution.png - Statistical analysis of prediction quality")

if __name__ == "__main__":
    RunMainProgram()