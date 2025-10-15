import mujoco as mj
import mujoco.viewer as viewer
import numpy as np
import time
import sys

from Visualizer import VisualizeData    

MODEL_PATH = "Models/ur5e_urdf/ur5e.mjcf.xml"

# UR5e Real Joint Limits 
UR5E_JOINT_LIMITS = {
    'position': np.array([
        [-2*np.pi, 2*np.pi],     # Shoulder Pan: ±360°
        [-2*np.pi, 2*np.pi],     # Shoulder Lift: ±360°
        [-np.pi, np.pi],         # Elbow: ±180°
        [-2*np.pi, 2*np.pi],     # Wrist 1: ±360°
        [-2*np.pi, 2*np.pi],     # Wrist 2: ±360°
        [-2*np.pi, 2*np.pi]      # Wrist 3: ±360°
    ]),
    'velocity': np.array([3.15, 3.15, 3.15, 6.28, 6.28, 6.28]),  # rad/s
    'torque': np.array([150, 150, 150, 28, 28, 28])  # Nm
}

def EnforceSafetyLimits(Data, NumJoints):
    """Enforce industrial safety limits on robot state"""
    # Clamp joint positions
    for i in range(NumJoints):
        Data.qpos[i] = np.clip(Data.qpos[i], 
                              UR5E_JOINT_LIMITS['position'][i][0], 
                              UR5E_JOINT_LIMITS['position'][i][1])
    
    # Clamp joint velocities
    for i in range(NumJoints):
        Data.qvel[i] = np.clip(Data.qvel[i], 
                              -UR5E_JOINT_LIMITS['velocity'][i], 
                              UR5E_JOINT_LIMITS['velocity'][i])

def GenerateSafeControl(NumJoints, Step, PreviousControl=None):
    """Generate control inputs within safe torque limits"""
    # Use industrial-grade torque limits (reduced for safety margin)
    MaxTorque = UR5E_JOINT_LIMITS['torque'] * 0.1  # 10% of max torque
    
    Control = np.random.uniform(-MaxTorque, MaxTorque)
    
    # Additional safety for wrist joints
    Control[3:] *= 0.3  # Further reduce wrist torques
    
    # Smooth control for industrial applications
    if Step > 0 and PreviousControl is not None:
        Control = 0.8 * Control + 0.2 * PreviousControl  # More conservative smoothing
    
    return Control

def CheckSafetyViolation(Data, NumJoints):
    """Check if robot state violates safety constraints"""
    for i in range(NumJoints):
        # Check position limits
        if (Data.qpos[i] < UR5E_JOINT_LIMITS['position'][i][0] or 
            Data.qpos[i] > UR5E_JOINT_LIMITS['position'][i][1]):
            return True
        
        # Check velocity limits
        if abs(Data.qvel[i]) > UR5E_JOINT_LIMITS['velocity'][i]:
            return True
    
    return False

def CollectJointTrajectories(NumEpisodes=50, StepsPerEpisode=1000, Visual=False):
    Model = mj.MjModel.from_xml_path(MODEL_PATH)
    Data = mj.MjData(Model)
    
    NumJoints = Model.nu
    print(f"Collecting data from {NumJoints} joints with industrial safety limits")
    
    AllPositions = []
    AllVelocities = []
    AllControls = []
    AllNextPositions = []
    
    SafetyViolations = 0
    
    def CollectEpisodeData(Episode):
        nonlocal SafetyViolations
        print(f"Episode {Episode+1}/{NumEpisodes}")
        
        # Initialize within safe working envelope (60% of limits)
        for i in range(NumJoints):
            SafeRange = UR5E_JOINT_LIMITS['position'][i]
            Data.qpos[i] = np.random.uniform(SafeRange[0] * 0.6, SafeRange[1] * 0.6)
        
        Data.qvel[:] = 0.0
        mj.mj_forward(Model, Data)
        
        EpisodePositions = []
        EpisodeVelocities = []
        EpisodeControls = []
        EpisodeNextPositions = []
        
        for Step in range(StepsPerEpisode):
            PreviousControl = Data.ctrl[:NumJoints] if Step > 0 else None
            Control = GenerateSafeControl(NumJoints, Step, PreviousControl)
            
            Data.ctrl[:NumJoints] = Control
            
            CurrentPosition = Data.qpos[:NumJoints].copy()
            CurrentVelocity = Data.qvel[:NumJoints].copy()
            CurrentControl = Data.ctrl[:NumJoints].copy()
            
            mj.mj_step(Model, Data)
            
            # INDUSTRIAL SAFETY: Enforce limits after each step
            EnforceSafetyLimits(Data, NumJoints)
            
            # Check for safety violations
            if CheckSafetyViolation(Data, NumJoints):
                SafetyViolations += 1
                # In real industrial setting, this would trigger emergency stop
                print(f"Warning: Safety violation detected at Episode {Episode+1}, Step {Step+1}")
            
            NextPosition = Data.qpos[:NumJoints].copy()
            
            EpisodePositions.append(CurrentPosition)
            EpisodeVelocities.append(CurrentVelocity)
            EpisodeControls.append(CurrentControl)
            EpisodeNextPositions.append(NextPosition)
        
        return EpisodePositions, EpisodeVelocities, EpisodeControls, EpisodeNextPositions
    
    if Visual:
        print("Visual data collection mode with safety limits")
        
        with viewer.launch_passive(Model, Data) as Viewer:
            mj.mj_forward(Model, Data)
            Viewer.sync()
            time.sleep(1.0)
            
            Episode = 0
            while Viewer.is_running() and Episode < NumEpisodes:
                # Initialize within safe working envelope
                for i in range(NumJoints):
                    SafeRange = UR5E_JOINT_LIMITS['position'][i]
                    Data.qpos[i] = np.random.uniform(SafeRange[0] * 0.6, SafeRange[1] * 0.6)
                
                Data.qvel[:] = 0.0
                mj.mj_forward(Model, Data)
                Viewer.sync()
                time.sleep(0.5)
                
                EpisodePositions = []
                EpisodeVelocities = []
                EpisodeControls = []
                EpisodeNextPositions = []
                
                for Step in range(StepsPerEpisode):
                    if not Viewer.is_running():
                        break
                    
                    PreviousControl = Data.ctrl[:NumJoints] if Step > 0 else None
                    Control = GenerateSafeControl(NumJoints, Step, PreviousControl)
                    
                    Data.ctrl[:NumJoints] = Control
                    
                    CurrentPosition = Data.qpos[:NumJoints].copy()
                    CurrentVelocity = Data.qvel[:NumJoints].copy()
                    CurrentControl = Data.ctrl[:NumJoints].copy()
                    
                    mj.mj_step(Model, Data)
                    
                    # INDUSTRIAL SAFETY: Enforce limits
                    EnforceSafetyLimits(Data, NumJoints)
                    
                    NextPosition = Data.qpos[:NumJoints].copy()
                    
                    EpisodePositions.append(CurrentPosition)
                    EpisodeVelocities.append(CurrentVelocity)
                    EpisodeControls.append(CurrentControl)
                    EpisodeNextPositions.append(NextPosition)
                    
                    Viewer.sync()
                
                if not Viewer.is_running():
                    break
                
                AllPositions.extend(EpisodePositions)
                AllVelocities.extend(EpisodeVelocities)
                AllControls.extend(EpisodeControls)
                AllNextPositions.extend(EpisodeNextPositions)
                
                Episode += 1
                if Episode < NumEpisodes:
                    time.sleep(1.0)
    else:
        print("Headless data collection with safety limits")
        
        for Episode in range(NumEpisodes):
            EpPos, EpVel, EpCtrl, EpNext = CollectEpisodeData(Episode)
            
            AllPositions.extend(EpPos)
            AllVelocities.extend(EpVel)
            AllControls.extend(EpCtrl)
            AllNextPositions.extend(EpNext)
    
    Positions = np.array(AllPositions)
    Velocities = np.array(AllVelocities)
    Controls = np.array(AllControls)
    NextPositions = np.array(AllNextPositions)
    
    print(f"Collected {len(Positions)} samples")
    print(f"Safety violations detected: {SafetyViolations}")
    
    # Industrial logging
    with open('data_collection_log.txt', 'w') as f:
        f.write(f"Data Collection Report\n")
        f.write(f"Total samples: {len(Positions)}\n")
        f.write(f"Safety violations: {SafetyViolations}\n")
        f.write(f"Joint limits enforced: UR5e Industrial Standard\n")
    
    np.savez('joint_trajectory_data.npz', 
             positions=Positions, 
             velocities=Velocities,
             controls=Controls,
             next_positions=NextPositions)
    
    return Positions, Velocities, Controls, NextPositions



if __name__ == "__main__":
    VisualMode = len(sys.argv) > 1 and sys.argv[1] == "--visual"
    
    print("UR5e Joint Trajectory Data Collection")
    print(f"Mode: {'Visual' if VisualMode else 'Headless'}")
    
    NumEpisodes = 10 if VisualMode else 50
    
    Positions, Velocities, Controls, NextPositions = CollectJointTrajectories(
        NumEpisodes=NumEpisodes, 
        StepsPerEpisode=1000,
        Visual=VisualMode
    )
    
    VisualizeData(Positions, Velocities, Controls, NextPositions)
    
    print("Data collection complete")
    print("Next: python train_joint_predictor.py")