import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

@torch.jit.script
def optimized_gae(
    rewards: torch.Tensor,
    vals: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float
):
    """
    Standard GAE calculation with post-step dones.
    GAE = r_t + gamma * V_{t+1} * (1-d_t) - V_t + gamma * lambda * (1-d_t) * GAE_{t+1}
    """
    num_steps: int = rewards.shape[0]
    next_value = next_value.reshape(-1)
    
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros_like(rewards[0])
    
    # nextvalues starts at V(s_{T+1})
    nextvalues = next_value
    # next_non_terminal is (1 - d_T)
    next_non_terminal = 1.0 - next_done.float()
    
    # Loop backwards from T-1 to 0
    for t in range(num_steps - 1, -1, -1):
        # Advantages are calculated based on the return/value at step t
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        # However, our loop usually uses 'nextvalues' which is V(s_{t+1})
        # And 'next_non_terminal' which is (1 - d_t)
        
        # In our storage, dones[t] is the done signal after step t.
        # So it applies to the transition from t to t+1.
        non_terminal = 1.0 - dones[t].float()
        
        delta = rewards[t] + gamma * nextvalues * non_terminal - vals[t]
        lastgaelam = delta + gamma * gae_lambda * non_terminal * lastgaelam
        advantages[t] = lastgaelam
        
        # Set next values for step t-1
        nextvalues = vals[t]
        # Although not strictly used in standard PPO loop for delta calculation inside the loop,
        # we can keep track of it if needed.
        # non_terminal = 1.0 - dones[t].float()
        
    return advantages, advantages + vals


def make_ppo_update_fn(agent, optimizer, cfg):
    """Factory function to create PPO update TensorDictModule.
    
    Args:
        agent: The training agent (with get_action_and_value method)
        optimizer: The optimizer for agent parameters
        cfg: Config with ppo.clip_coef, ppo.ent_coef, ppo.vf_coef, ppo.max_grad_norm
    
    Returns:
        TensorDictModule wrapping the update function
    """
    import tensordict
    
    clip_coef = cfg.ppo.clip_coef
    ent_coef = cfg.ppo.ent_coef
    vf_coef = cfg.ppo.vf_coef
    max_grad_norm = cfg.ppo.max_grad_norm
    
    def update(obs, actions, logprobs, advantages, returns, vals):
        optimizer.zero_grad(set_to_none=True)
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
        logratio = newlogprob - logprobs
        ratio = logratio.exp()
        
        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss (clipped)
        newvalue = newvalue.view(-1)
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(newvalue - vals, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        
        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
        
        loss.backward()
        gn = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        
        return approx_kl, v_loss, pg_loss, entropy_loss, old_approx_kl, clipfrac, gn
    
    return tensordict.nn.TensorDictModule(
        update,
        in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
        out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
    )
