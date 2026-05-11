from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

from isaaclab.sim import SimulationContext

import isaaclab.sim as sim_utils
from Grip.tasks.manager_based.grip.grip_env_cfg import GripSceneCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg



def main():
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    scene_cfg = GripSceneCfg(num_envs=1, env_spacing=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    sim_dt = sim.get_physics_dt()
    
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()
