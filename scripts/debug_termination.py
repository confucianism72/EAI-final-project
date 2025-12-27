import torch
import gymnasium as gym
import mani_skill.envs
from scripts import track1_env
from mani_skill.utils.structs.pose import Pose

def test_termination():
    num_envs = 1
    # Create environment directly via gym.make to avoid Hydra cfg dependencies
    env = gym.make(
        "Track1-v0", 
        num_envs=num_envs, 
        task="lift", 
        obs_mode="state", 
        reward_mode="dense",
        control_mode="pd_joint_target_delta_pos"
    )
    
    # Reset
    obs, info = env.reset()
    unwrapped = env.unwrapped
    
    # Move cube to fallen position
    print(f"\nInitial cube pos: {unwrapped.red_cube.pose.p}")
    
    # Manually move cube to trigger fail_fallen
    # Fallen threshold in Track1Env is -0.05
    out_pos = torch.tensor([[0.5, 0.25, -0.1]], device=unwrapped.device)
    unwrapped.red_cube.set_pose(Pose.create_from_pq(p=out_pos))
    
    print("\nChecking evaluation results directly...")
    eval_res = unwrapped.evaluate()
    print(f"Eval Success: {eval_res.get('success')}")
    print(f"Eval Fail: {eval_res.get('fail')}")
    
    # Step environment
    # Note: BaseEnv.step should now handle termination automatically based on info['fail']
    print("\nStepping environment...")
    action = torch.zeros((num_envs, 6), device=unwrapped.device)
    next_obs, reward, terminated, truncated, next_info = env.step(action)
    
    print(f"Terminated from env.step: {terminated}")
    
    if terminated.any():
        print("\nSUCCESS: Environment correctly terminated on fail using standard ManiSkill logic!")
    else:
        # Check if the next_info has fail
        print(f"Next Info Fail: {next_info.get('fail')}")
        print("\nFAILURE: Environment DID NOT terminate on fail!")

    env.close()

if __name__ == "__main__":
    test_termination()
