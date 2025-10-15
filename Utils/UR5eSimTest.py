import mujoco as mj
import mujoco.viewer as viewer
import numpy as np

MODEL = "Models/ur5e_urdf/ur5e.mjcf.xml"

m = mj.MjModel.from_xml_path(MODEL)
d = mj.MjData(m)

# Remove this pose setting - it's now in the MJCF
# if m.nq >= 6:
#     d.qpos[:6] = np.deg2rad([0, -90, 90, -90, -90, 0])
#     mj.mj_forward(m, d)

# Just run forward kinematics once to initialize
mj.mj_forward(m, d)

with viewer.launch_passive(m, d) as v:
    while v.is_running():
        mj.mj_step(m, d)
        v.sync()
