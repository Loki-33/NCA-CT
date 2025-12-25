import torch.nn as nn 
import torch 
from nca import NCA 
import numpy as np 
import torch.nn.functional as F 
import matplotlib.pyplot as plt 

def compute_lyapunov_exponent(nca, grid, steps=50, epsilon=1e-5, device='cpu'):
    nca.eval()

    grid_ref = grid.clone().detach()

    perturbation = torch.randn_like(grid) * epsilon
    perturbation[:, 0] = 0 # no perturb visible channel only hidden 
    grid_pert = grid_ref + perturbation
    lyapunov_sum = 0.0 
    divergence_history = []

    with torch.no_grad():
        for t in range(steps):
            grid_ref = nca(grid_ref)
            grid_pert = nca(grid_pert)

            diff = grid_pert - grid_ref
            distance = torch.norm(diff) / diff.numel()**0.5

            divergence_history.append(distance.item())

            lyapunov_sum += torch.log(distance/epsilon) # dist/eps -> growth rate 

            # RENORMALIZATION TRICK 
            grid_pert = grid_ref + (diff/distance) * epsilon 

    lyapunov_exponent = (lyapunov_sum/steps).item()
    return lyapunov_exponent, divergence_history 


def compute_spatial_entropy(grid, bins=50):
    visible = grid[:, 0]
    dx = visible[:, :, 1:] - visible[:, :, :-1]
    dy = visible[:, 1:, :] - visible[:, :-1, :]

    gradients = torch.cat([dx.flatten(), dy.flatten()])
    hist = torch.histc(gradients, bins=bins, min=-1, max=1)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return (-probs * torch.log2(probs)).sum().item()


def compute_temporal_activity(trajectory):
    if len(trajectory) < 2:
        return 0.0 

    differences = []
    for i in range(len(trajectory)-1):
        diff = torch.mean((trajectory[i+1] - trajectory[i])**2).item()
        differences.append(diff)

    temporal_activity = np.std(differences)
    return temporal_activity


def apply_damage(grid, damage_type='random', damage_rate=0.3):
    damaged = grid.clone()
    batch, channels, h, w = grid.shape 
    if damage_type=='random':
        mask = torch.randn(batch, 1, h, w, device=grid.device) > damage_rate 
        damaged = damaged * mask 
    
    elif damage_type == 'block':
        block_size = int(h*0.3)
        x = torch.randint(0, h-block_size, (1,)).item()
        y = torch.randint(0, w-block_size, (1,)).item()
        damaged[:, :, x:x+block_size, y:y+block_size] = 0
    
    elif damage_type=='noise':
        noise = torch.randn_like(grid)*0.5 
        damaged = damaged + noise 
        damaged = torch.clamp(damaged, -2.0, 2.0)
    
    elif damage_type=='channel':
        channel_dmg = torch.randperm(channels)[:channels//2]
        for c in channel_dmg:
            damaged[:, c, :, :] = 0 
    return damaged 

def compute_recovery_metrics(
    nca, healthy_grid, damage_type='random',
    damage_rate=0.3, recovery_steps=150
):
    nca.eval()

    damaged = apply_damage(healthy_grid, damage_type, damage_rate)
    current = damaged.clone()

    metrics = {
        'mse_to_healthy': [],
        'recovery_score': [],
        'activity': [],
        'timesteps': list(range(recovery_steps))
    }

    with torch.no_grad():
        for _ in range(recovery_steps):
            prev = current.clone()
            current = nca(current)

            visible_curr = nca.get_visible_channel(current)
            visible_healthy = nca.get_visible_channel(healthy_grid)

            mse = F.mse_loss(visible_curr, visible_healthy).item()
            recovery_score = 1.0 - mse

            activity = torch.norm(current - prev).item()

            metrics['mse_to_healthy'].append(mse)
            metrics['recovery_score'].append(recovery_score)
            metrics['activity'].append(activity)

    return metrics, current



def analysis(nca, grid):
    print("======== ANALYSIS ========")

    results = {
        'lyapunov': {},
        'entropy': {},
        'recovery': {}
    }

    print("Computing Lyapunov exponent...")
    lyap, hist = compute_lyapunov_exponent(nca, grid, steps=100, epsilon=1e-4)
    results['lyapunov']['exponent'] = lyap
    results['lyapunov']['history'] = hist

    regime = (
        "ORDERED" if lyap < -0.01 else
        "CHAOTIC" if lyap > 0.01 else
        "EDGE OF CHAOS"
    )
    print(f"λ = {lyap:.5f} → {regime}")

    print("Computing entropy...")
    spatial = compute_spatial_entropy(grid)
    results['entropy']['spatial'] = spatial
    print(f"Spatial entropy: {spatial:.4f}")

    trajectory = [grid.clone()]
    with torch.no_grad():
        for _ in range(50):
            grid = nca(grid)
            trajectory.append(grid.clone())

    temporal = compute_temporal_activity(trajectory)
    results['entropy']['temporal_activity'] = temporal
    print(f"Temporal activity: {temporal:.6f}")

    print("Testing recovery...")
    for dmg in ['random', 'block', 'noise']:
        print(f"  Damage type: {dmg}")
        metrics, _ = compute_recovery_metrics(nca, trajectory[-1], dmg)
        results['recovery'][dmg] = metrics

        print(
            f"    Final MSE: {metrics['mse_to_healthy'][-1]:.4f}, "
            f"Recovery score: {metrics['recovery_score'][-1]:.4f}"
        )

    return results
    

def plot_metrics(results, save_path='chaos_analysis1_pool.png'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Neural CA Chaos Analysis", fontsize=16)

    # Lyapunov history
    ax = axes[0, 0]
    ax.plot(results['lyapunov']['history'])
    ax.set_yscale('log')
    ax.set_title(f"Lyapunov divergence (λ={results['lyapunov']['exponent']:.4f})")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Distance")
    ax.grid(True)

    # Entropy
    ax = axes[0, 1]
    ax.bar(
        ['Spatial', 'Temporal Activity'],
        [
            results['entropy']['spatial'],
            results['entropy']['temporal_activity']
        ]
    )
    ax.set_title("Complexity Measures")

    # Recovery plots
    for i, dmg in enumerate(['random', 'block', 'noise']):
        ax = axes[1, i]
        m = results['recovery'][dmg]

        ax.plot(m['timesteps'], m['mse_to_healthy'], label='MSE')
        ax.plot(m['timesteps'], m['recovery_score'], label='Recovery score')
        ax.plot(m['timesteps'], m['activity'], label='Activity')

        ax.set_title(f"Recovery: {dmg}")
        ax.set_xlabel("Timestep")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    
def measure_visual_divergence(nca, grid, steps=100, epsilon=1e-4):
    """Like Lyapunov but measure VISIBLE divergence only"""
    nca.eval()
    
    grid_ref = grid.clone().detach()
    grid_pert = grid + torch.randn_like(grid) * epsilon
    
    visual_divergences = []
    
    with torch.no_grad():
        for t in range(steps):
            grid_ref = nca(grid_ref)
            grid_pert = nca(grid_pert)
            
            # Compare ONLY visible channels
            vis_ref = nca.get_visible_channel(grid_ref)
            vis_pert = nca.get_visible_channel(grid_pert)
            
            visual_diff = torch.mean((vis_ref - vis_pert) ** 2).item()
            visual_divergences.append(visual_diff)
    
    plt.plot(visual_divergences)
    plt.yscale('log')
    plt.title("Visual MSE divergence over time")
    plt.xlabel("Timestep")
    plt.ylabel("Visible channel MSE")
    plt.savefig('divergence_pool', dpi=150)

def analyze_channel_sensitivity(nca, grid, epsilon=1e-4, steps=50):
    """Measure how much each channel contributes to visible divergence"""
    
    results = {}
    
    for ch in range(grid.shape[1]):  # for each channel
        nca.eval()
        
        # Perturb ONLY this channel
        grid_ref = grid.clone()
        grid_pert = grid.clone()
        
        perturbation = torch.randn_like(grid[:, ch:ch+1, :, :]) * epsilon
        grid_pert[:, ch:ch+1, :, :] += perturbation
        
        visible_divergences = []
        
        with torch.no_grad():
            for t in range(steps):
                grid_ref = nca(grid_ref)
                grid_pert = nca(grid_pert)
                
                # Measure VISIBLE divergence
                vis_ref = nca.get_visible_channel(grid_ref)
                vis_pert = nca.get_visible_channel(grid_pert)
                
                mse = F.mse_loss(vis_ref, vis_pert).item()
                visible_divergences.append(mse)
        
        results[f'channel_{ch}'] = visible_divergences

    fig, ax = plt.subplots(figsize=(12, 6))
    for ch, divergence in results.items():
        ax.plot(divergence, label=ch, alpha=0.7)

    ax.set_yscale('log')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Visible MSE')
    ax.set_title('How much does perturbing each channel affect visible output?')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('channel_sens_pool', dpi=150)

def compare_perturbation_vs_damage(nca, grid, steps=150):
    nca.eval()
    
    # Original
    original = grid.clone()
    
    # Tiny perturbation (Lyapunov-style)
    perturbed = grid.clone() + torch.randn_like(grid) * 1e-4
    
    # Large damage (recovery-style)
    damaged = apply_damage(grid.clone(), damage_type='random', damage_rate=0.3)
    
    # Track all three
    trajectories = {
        'original': [original.clone()],
        'perturbed': [perturbed.clone()],
        'damaged': [damaged.clone()]
    }
    
    metrics = {
        'pert_mse': [],  # perturbed vs original
        'dmg_mse': []    # damaged vs original
    }
    
    with torch.no_grad():
        for t in range(steps):
            # Evolve all three
            original = nca(original)
            perturbed = nca(perturbed)
            damaged = nca(damaged)
            
            trajectories['original'].append(original.clone())
            trajectories['perturbed'].append(perturbed.clone())
            trajectories['damaged'].append(damaged.clone())
            
            # Measure divergence
            vis_orig = nca.get_visible_channel(original)
            vis_pert = nca.get_visible_channel(perturbed)
            vis_dmg = nca.get_visible_channel(damaged)
            
            pert_mse = F.mse_loss(vis_pert, vis_orig).item()
            dmg_mse = F.mse_loss(vis_dmg, vis_orig).item()
            
            metrics['pert_mse'].append(pert_mse)
            metrics['dmg_mse'].append(dmg_mse)
    

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MSE comparison
    axes[0, 0].plot(metrics['pert_mse'], label='Tiny perturbation', linewidth=2)
    axes[0, 0].plot(metrics['dmg_mse'], label='Large damage', linewidth=2)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('MSE to original')
    axes[0, 0].set_title('Divergence: Perturbation vs Damage')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

# Zoom on early behavior
    axes[0, 1].plot(metrics['pert_mse'][:50], label='Tiny perturbation', linewidth=2)
    axes[0, 1].plot(metrics['dmg_mse'][:50], label='Large damage', linewidth=2)
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('MSE to original')
    axes[0, 1].set_title('Early dynamics (first 50 steps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

# Show final frames
    axes[1, 0].imshow(nca.get_visible_channel(trajectories['perturbed'][-1])[0, 0].cpu(), cmap='gray')
    axes[1, 0].set_title(f'Perturbed (final MSE: {metrics["pert_mse"][-1]:.4f})')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(nca.get_visible_channel(trajectories['damaged'][-1])[0, 0].cpu(), cmap='gray')
    axes[1, 1].set_title(f'Damaged (final MSE: {metrics["dmg_mse"][-1]:.4f})')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('dmg_vs_perturbation_pool', dpi=150)

if __name__ == '__main__':
    nca_model = torch.load('nca_model.pth',map_location=torch.device('cpu'), weights_only=False)
    nca_model.eval()

    grid = nca_model.initialize_grid(batch_size=1)
    with torch.no_grad():
        for _ in range(100):
            grid = nca_model(grid)

    results = analysis(nca_model, grid)

    plot_metrics(results)
    grid = nca_model.initialize_grid(batch_size=1)
    for _ in range(30):  # early growth phase
        grid = nca_model(grid)
    lyap_growth, _ = compute_lyapunov_exponent(nca_model, grid, steps=50)

    # After long convergence
    for _ in range(200):  # way past convergence
        grid = nca_model(grid)
    lyap_converged, _ = compute_lyapunov_exponent(nca_model, grid, steps=50)

    print(f"λ during growth: {lyap_growth}")
    print(f"λ after convergence: {lyap_converged}")

    nca_model.fire_rate = 1.0 
    grid = nca_model.initialize_grid(batch_size=1)
    for _ in range(100):
        grid = nca_model(grid)

    lyap_deterministic, _ = compute_lyapunov_exponent(nca_model, grid, steps=100)

    print(f"λ with fire_rate=0.5: 2.5")
    print(f"λ with fire_rate=1.0: {lyap_deterministic}")
    nca_model.fire_rate=0.5 

    #analyze_channel_sensitivity(nca_model, grid, epsilon=1e-4, steps=50)

    compare_perturbation_vs_damage(nca_model, grid, steps=150)
