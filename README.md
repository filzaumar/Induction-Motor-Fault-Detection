# Three-Phase Squirrel-Cage Induction Motor Fault Detection

- Used a MATLAB starting example: https://www.mathworks.com/help/sps/ug/three-phase-asynchronous-machine-starting.html
- Induced **electrical fault**: Voltage Unbalance
- Induced **mechanical faults**: Torque Overload and Torque Braking
- Created a dataset focused on **transient states**
- Split data into **train/test** and classified into **4 classes**:
  - Healthy
  - Voltage Unbalance
  - Torque Overload
  - Torque Braking
- Achieved **~99% accuracy**, even with a **50/50 train-test split**
- Added a **rotational damper** to make the waveforms clearer for presentation
