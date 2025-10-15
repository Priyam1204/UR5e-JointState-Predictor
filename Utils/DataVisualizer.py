import matplotlib.pyplot as plt

def VisualizeData(Positions, Velocities, Controls, NextPositions):
    JointNames = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
    
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(Positions[:2000, i], label=f'{JointNames[i]} Position', alpha=0.8)
        plt.plot(NextPositions[:2000, i], label=f'{JointNames[i]} Next Position', alpha=0.6)
        plt.title(f'{JointNames[i]}')
        plt.ylabel('Angle (rad)')
        plt.xlabel('Time step')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('joint_trajectories.png', dpi=150)
    plt.show()
    
    plt.figure(figsize=(15, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(Controls[:2000, i], alpha=0.8)
        plt.title(f'{JointNames[i]} Control Input')
        plt.ylabel('Torque (Nm)')
        plt.xlabel('Time step')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('control_inputs.png', dpi=150)
    plt.show()
    
    print("\nData Statistics:")
    for i, Name in enumerate(JointNames):
        print(f"{Name}: [{Positions[:, i].min():.3f}, {Positions[:, i].max():.3f}] rad")