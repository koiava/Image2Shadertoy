import torch
import os
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torch.nn.functional as F
import time
from torch.amp import autocast  # Updated import
import argparse

# Parameters
DEFAULT_NUM_GAUSSIANS_INITIAL = 110
DEFAULT_ITERATIONS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_COLORS = False
MIN_SIGMA = 1e-2
BATCH_SIZE = 128  # For intra-tile batching (used only in tiled mode)
PRUNE_INTERVAL = 100
PRUNE_THRESHOLD = 1e-5
TILE_SIZE = 64  # Used only in tiled mode
TILE_BATCH_SIZE = 4  # Used only in tiled mode
GAUSSIAN_THRESHOLD = 500  # Switch to tiled rendering above this count

def gaussian_2d(x_grid, y_grid, params, min_dim):
    """Compute 2D Gaussian with RGB contribution and alpha using batched operations."""
    amp_r, amp_g, amp_b = params[:, 0].view(-1, 1, 1), params[:, 1].view(-1, 1, 1), params[:, 2].view(-1, 1, 1)
    mu_x, mu_y = params[:, 3].view(-1, 1, 1), params[:, 4].view(-1, 1, 1)
    sigma = params[:, 5].view(-1, 1, 1)
    anisotropy = params[:, 6].view(-1, 1, 1)
    theta = params[:, 7].view(-1, 1, 1)

    exp_aniso = torch.exp(0.5 * anisotropy)
    sigma_1 = sigma * exp_aniso
    sigma_2 = sigma / exp_aniso

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    x_diff = x_grid - mu_x
    y_diff = y_grid - mu_y
    x_rot = cos_theta * x_diff + sin_theta * y_diff
    y_rot = -sin_theta * x_diff + cos_theta * y_diff

    alpha = torch.exp(-0.5 * (x_rot**2 / sigma_1**2 + y_rot**2 / sigma_2**2))
    rgb = torch.stack([amp_r, amp_g, amp_b], dim=-1)

    return rgb, alpha

def compute_gaussian_contribution(params, width, height, device, min_dim, x_grid, y_grid):
    """Compute the maximum alpha contribution of each Gaussian."""
    num_gaussians = params.shape[0]
    x = x_grid.expand(num_gaussians, -1, -1)
    y = y_grid.expand(num_gaussians, -1, -1)
    # Use autocast only if CUDA is available
    with autocast('cuda', enabled=device.type == 'cuda'):
        _, alpha = gaussian_2d(x, y, params, min_dim)
        max_alpha = torch.amax(alpha, dim=(1, 2))
    return max_alpha

def prune_gaussians(params, width, height, device, min_dim, x_grid, y_grid, threshold=PRUNE_THRESHOLD):
    """Remove Gaussians with negligible contribution."""
    max_alpha = compute_gaussian_contribution(params, width, height, device, min_dim, x_grid, y_grid)
    mask = max_alpha > threshold
    pruned_params = params[mask].clone().detach().requires_grad_(True)
    print(f"Pruned {params.shape[0] - pruned_params.shape[0]} Gaussians with max alpha < {threshold}")
    return pruned_params

def get_gaussian_tile_associations(params, width, height, tile_size, device):
    """Associate Gaussians with tiles (used in tiled mode)."""
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size
    tile_gaussians = {(tx, ty): [] for tx in range(num_tiles_x) for ty in range(num_tiles_y)}

    tile_x_starts = torch.arange(0, width, tile_size, device=device)
    tile_y_starts = torch.arange(0, height, tile_size, device=device)
    tile_x_ends = torch.clamp(tile_x_starts + tile_size, max=width)
    tile_y_ends = torch.clamp(tile_y_starts + tile_size, max=height)

    mu_x = params[:, 3]
    mu_y = params[:, 4]
    sigma = params[:, 5]
    influence_radius = 3 * sigma

    left = torch.clamp((mu_x - influence_radius) / tile_size, 0, num_tiles_x - 1).int()
    right = torch.clamp((mu_x + influence_radius) / tile_size, 0, num_tiles_x - 1).int()
    top = torch.clamp((mu_y - influence_radius) / tile_size, 0, num_tiles_y - 1).int()
    bottom = torch.clamp((mu_y + influence_radius) / tile_size, 0, num_tiles_y - 1).int()

    for i in range(params.shape[0]):
        for tx in range(left[i].item(), right[i].item() + 1):
            for ty in range(top[i].item(), bottom[i].item() + 1):
                tile_gaussians[(tx, ty)].append(i)

    return tile_gaussians, num_tiles_x, num_tiles_y

def render_tile_batch(tile_batch, params, tile_gaussians, width, height, device, min_dim, x_grid, y_grid, tile_size, batch_size):
    """Render a batch of tiles (used in tiled mode)."""
    batch_images = []
    for (tile_x, tile_y) in tile_batch:
        gaussian_indices = tile_gaussians[(tile_x, tile_y)]
        if not gaussian_indices:
            tile_height = min(tile_size, height - tile_y * tile_size)
            tile_width = min(tile_size, width - tile_x * tile_size)
            batch_images.append(torch.zeros((tile_height, tile_width, 3), dtype=torch.float32, device=device))
            continue

        tile_params = params[gaussian_indices]
        num_gaussians = len(gaussian_indices)

        y_start = tile_y * tile_size
        y_end = min(y_start + tile_size, height)
        x_start = tile_x * tile_size
        x_end = min(x_start + tile_size, width)

        tile_x_grid = x_grid[y_start:y_end, x_start:x_end]
        tile_y_grid = y_grid[y_start:y_end, x_start:x_end]
        tile_height = y_end - y_start
        tile_width = x_end - x_start

        image = torch.zeros((tile_height, tile_width, 3), dtype=torch.float32, device=device)

        x = tile_x_grid.expand(num_gaussians, -1, -1)
        y = tile_y_grid.expand(num_gaussians, -1, -1)
        with autocast('cuda', enabled=device.type == 'cuda'):
            rgb, alpha = gaussian_2d(x, y, tile_params, min_dim)
            rgb = rgb.expand(-1, tile_height, tile_width, -1)
            alpha = alpha.unsqueeze(-1)
            weighted_rgb = rgb * alpha
            cum_alpha = torch.cumprod(1 - alpha, dim=0)
            cum_alpha = torch.cat([torch.ones_like(cum_alpha[:1]), cum_alpha[:-1]], dim=0)
            image = torch.sum(weighted_rgb * cum_alpha, dim=0)

        batch_images.append(image)

    return batch_images

def render_gaussians(gaussians, width, height, device, min_dim, x_grid, y_grid, tile_size=TILE_SIZE, batch_size=BATCH_SIZE, tile_batch_size=TILE_BATCH_SIZE):
    """Render Gaussians with hybrid approach: vectorized for small counts, tiled for large counts."""
    start_time = time.time()
    if not isinstance(gaussians, torch.Tensor) or gaussians.dim() != 2 or gaussians.shape[1] != 8:
        raise ValueError("Expected gaussians to be a tensor of shape [N, 8]")

    params = gaussians
    num_gaussians = params.shape[0]
    image = torch.zeros((height, width, 3), dtype=torch.float32, device=device)

    if num_gaussians < GAUSSIAN_THRESHOLD:
        # Vectorized rendering for small Gaussian counts
        x = x_grid.expand(num_gaussians, -1, -1)
        y = y_grid.expand(num_gaussians, -1, -1)
        with autocast('cuda', enabled=device.type == 'cuda'):
            rgb, alpha = gaussian_2d(x, y, params, min_dim)
            rgb = rgb.expand(-1, height, width, -1)
            alpha = alpha.unsqueeze(-1)
            weighted_rgb = rgb * alpha
            cum_alpha = torch.cumprod(1 - alpha, dim=0)
            cum_alpha = torch.cat([torch.ones_like(cum_alpha[:1]), cum_alpha[:-1]], dim=0)
            image = torch.sum(weighted_rgb * cum_alpha, dim=0)
    else:
        # Tiled rendering for large Gaussian counts
        tile_gaussians, num_tiles_x, num_tiles_y = get_gaussian_tile_associations(params, width, height, tile_size, device)
        tile_list = [(tx, ty) for ty in range(num_tiles_y) for tx in range(num_tiles_x)]
        for batch_start in range(0, len(tile_list), tile_batch_size):
            batch_end = min(batch_start + tile_batch_size, len(tile_list))
            tile_batch = tile_list[batch_start:batch_end]
            batch_images = render_tile_batch(tile_batch, params, tile_gaussians, width, height, device, min_dim, x_grid, y_grid, tile_size, batch_size)
            for (tile_x, tile_y), tile_image in zip(tile_batch, batch_images):
                y_start = tile_y * tile_size
                y_end = min(y_start + tile_size, height)
                x_start = tile_x * tile_size
                x_end = min(x_start + tile_size, width)
                image[y_start:y_end, x_start:x_end] = tile_image

    end_time = time.time()
    if DEBUG_COLORS:
        print(f"Render time: {end_time - start_time:.4f}s, Mode: {'Vectorized' if num_gaussians < GAUSSIAN_THRESHOLD else 'Tiled'}")
    return torch.clamp(image, 0, 255)

def save_gaussian_image(gaussians, width, height, output_path, device, min_dim, x_grid, y_grid):
    """Save the rendered Gaussian splatting image."""
    rendered_image = render_gaussians(gaussians, width, height, device, min_dim, x_grid, y_grid)
    rendered_image = rendered_image.permute(2, 0, 1) / 255
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_image(rendered_image, output_path)
    print(f"Image saved to {output_path}")

def initialize_gaussians(num_gaussians, width, height, sigma_range, low_res_tensor, device, mu_x=None, mu_y=None):
    """Initialize Gaussians with colors from low-res image."""
    if mu_x is None or mu_y is None:
        mu_x = torch.rand(num_gaussians, device=device) * width
        mu_y = torch.rand(num_gaussians, device=device) * height
    else:
        assert mu_x.shape == (num_gaussians,) and mu_y.shape == (num_gaussians,), "mu_x and mu_y must match num_gaussians"

    params = torch.zeros((num_gaussians, 8), dtype=torch.float32, device=device)

    coords = torch.stack([mu_x / (width - 1), mu_y / (height - 1)], dim=-1) * 2 - 1
    coords = coords.view(1, num_gaussians, 1, 2)
    colors = F.grid_sample(low_res_tensor.unsqueeze(0), coords, align_corners=True)
    colors = colors.squeeze().permute(1, 0)

    if DEBUG_COLORS:
        for i in range(min(3, num_gaussians)):
            print(f"Gaussian {i}: mu_x={mu_x[i].item():.2f}, mu_y={mu_y[i].item():.2f}, "
                  f"color=[{colors[i, 0].item():.2f}, {colors[i, 1].item():.2f}, {colors[i, 2].item():.2f}]")

    params[:, 0:3] = colors
    params[:, 3] = mu_x
    params[:, 4] = mu_y
    sigma_min, sigma_max = sigma_range
    params[:, 5] = torch.rand(num_gaussians, device=device) * (sigma_max - sigma_min) + sigma_min
    params[:, 6] = torch.zeros(num_gaussians, device=device)
    params[:, 7] = torch.rand(num_gaussians, device=device) * torch.pi
    return params.requires_grad_(True)

def generate_shadertoy_code(params, width, height):
    """Generate optimized Shadertoy GLSL code."""
    params_np = params.detach().cpu().numpy()
    num_gaussians = params_np.shape[0]
    aspect_ratio = width / height

    packed_data = []
    for i in range(num_gaussians):
        amp_r, amp_g, amp_b, mu_x, mu_y, sigma, anisotropy, theta = params_np[i]
        exp_aniso = np.exp(0.5 * anisotropy)
        sigma_1 = sigma * exp_aniso
        sigma_2 = sigma / exp_aniso

        color_R = max(0, min(255, int(round(amp_r))))
        color_G = max(0, min(255, int(round(amp_g))))
        color_B = max(0, min(255, int(round(amp_b))))
        mu_x_quant = max(0, min(65535, int(round((mu_x / width) * 65535))))
        mu_y_quant = max(0, min(65535, int(round((mu_y / height) * 65535))))
        sigma_1_norm = min(sigma_1 / height, 1.0)
        sigma_2_norm = min(sigma_2 / height, 1.0)
        sigma_1_quant = max(0, min(65535, int(round(sigma_1_norm * 65535))))
        sigma_2_quant = max(0, min(65535, int(round(sigma_2_norm * 65535))))
        theta_quant = max(0, min(255, int(round((theta / np.pi) * 255))))

        u0 = (color_R << 0) | (color_G << 8) | (color_B << 16) | (theta_quant << 24)
        u1 = (mu_x_quant << 0) | (mu_y_quant << 16)
        u2 = (sigma_1_quant << 0) | (sigma_2_quant << 16)

        packed_data.append(f"    uvec3({u0}u, {u1}u, {u2}u)")

    packed_array = ",\n".join(packed_data)

    shader_code = f"""
#define NUM_GAUSSIANS {num_gaussians}
const uvec3 gaussian_data[NUM_GAUSSIANS] = uvec3[](
{packed_array}
);

struct Gaussian {{
    vec3 rgb;
    vec2 pos;
    vec2 scale;
    float cos_theta;
    float sin_theta;
}};

Gaussian unpackGaussian(uvec3 data) {{
    uint u0 = data.x;
    uint u1 = data.y;
    uint u2 = data.z;

    uint R = (u0 >> 0u) & 255u;
    uint G = (u0 >> 8u) & 255u;
    uint B = (u0 >> 16u) & 255u;
    uint theta_quant = (u0 >> 24u) & 255u;

    uint mu_x_quant = u1 & 65535u;
    uint mu_y_quant = (u1 >> 16u) & 65535u;

    uint sigma_1_quant = u2 & 65535u;
    uint sigma_2_quant = (u2 >> 16u) & 65535u;

    vec3 rgb = vec3(float(R), float(G), float(B)) / 255.0;
    vec2 pos = vec2(float(mu_x_quant), float(mu_y_quant)) / 65535.0;
    vec2 scale = vec2(float(sigma_1_quant), float(sigma_2_quant)) / 65535.0;
    float theta = float(theta_quant) / 255.0 * 3.1415926535;
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    Gaussian g;
    g.rgb = rgb;
    g.pos = pos;
    g.scale = scale;
    g.cos_theta = cos_theta;
    g.sin_theta = sin_theta;
    return g;
}}

precision highp float;
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{{
    float aspect = {aspect_ratio:.6f};
    vec2 uv = fragCoord / iResolution.xy;
    uv.y = 1.0 - uv.y;
    float canvasAspect = iResolution.x / iResolution.y;
    if (canvasAspect > aspect) {{
        float scale = canvasAspect / aspect;
        uv.x = (uv.x - 0.5) * scale + 0.5;
    }} else {{
        float scale = aspect / canvasAspect;
        uv.y = (uv.y - 0.5) * scale + 0.5;
    }}
    vec3 color = vec3(0.0);
    for (int i = NUM_GAUSSIANS - 1; i >= 0; --i)
    {{
        Gaussian g = unpackGaussian(gaussian_data[i]);
        vec2 d = uv - g.pos;
        vec2 d_scaled = vec2(d.x * aspect, d.y);
        vec2 d_rot = vec2(
            g.cos_theta * d_scaled.x + g.sin_theta * d_scaled.y,
            -g.sin_theta * d_scaled.x + g.cos_theta * d_scaled.y
        );
        float alpha = exp(-0.5 * (
            (d_rot.x * d_rot.x) / (g.scale.x * g.scale.x) +
            (d_rot.y * d_rot.y) / (g.scale.y * g.scale.y)
        ));
        color = mix(color, g.rgb, alpha);
    }}
    fragColor = vec4(color, 1.0);
}}
"""
    return shader_code

def main():
    """Perform Gaussian splatting with error-based initialization."""
    parser = argparse.ArgumentParser(description="Gaussian splatting script with command-line arguments.")
    parser.add_argument('-i', '--input', required=True, help="Path to input image (e.g., neo.png)")
    parser.add_argument('-o', '--output', default='result.png', help="Path to output image (default: result.png)")
    parser.add_argument('-osh', '--output-shadertoy', default='result.shadertoy', help="Path to output Shadertoy file (default: result.shadertoy)")
    parser.add_argument('-n', '--num-gaussians', type=int, default=DEFAULT_NUM_GAUSSIANS_INITIAL, help=f"Number of initial Gaussians (default: {DEFAULT_NUM_GAUSSIANS_INITIAL})")
    parser.add_argument('-it', '--iterations', type=int, default=DEFAULT_ITERATIONS, help=f"Number of initial iterations (default: {DEFAULT_ITERATIONS})")
    args = parser.parse_args()

    input_image_path = args.input
    output_image_path = args.output
    output_shadertoy_path = args.output_shadertoy
    num_gaussians_initial = args.num_gaussians
    initial_iterations = args.iterations

    if num_gaussians_initial <= 0:
        print(f"Error: Number of initial Gaussians must be positive, got {num_gaussians_initial}")
        return

    if initial_iterations <= 0:
        print(f"Error: Number of initial iterations must be positive, got {initial_iterations}")
        return

    torch.autograd.set_detect_anomaly(True)

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, running on CPU")

    print(f"Initial number of Gaussians: {num_gaussians_initial}")
    print(f"Initial number of iterations: {initial_iterations}")

    if not os.path.exists(input_image_path):
        print(f"Error: Input image file {input_image_path} not found.")
        return

    img = Image.open(input_image_path).convert('RGB')
    to_tensor = ToTensor()
    target_image = to_tensor(img).to(DEVICE, dtype=torch.float32) * 255
    target_image = target_image.permute(1, 2, 0)
    height, width = target_image.shape[:2]
    min_dim = min(width, height)

    x_base = torch.arange(0, width, dtype=torch.float32, device=DEVICE)
    y_base = torch.arange(0, height, dtype=torch.float32, device=DEVICE)
    x_grid, y_grid = torch.meshgrid(x_base, y_base, indexing='xy')

    low_res_size = (width // 8, height // 8)
    low_res_img = img.resize(low_res_size, Image.Resampling.BILINEAR)
    low_res_tensor = to_tensor(low_res_img).to(DEVICE) * 255

    stages = [
        {"new_gaussians": num_gaussians_initial*1, "iterations": initial_iterations, "sigma_range": (min_dim / 32, min_dim / 4)},
        {"new_gaussians": num_gaussians_initial*2, "iterations": initial_iterations+1000, "sigma_range": (min_dim / 64, min_dim / 16)},
        {"new_gaussians": num_gaussians_initial*3, "iterations": initial_iterations+2000, "sigma_range": (min_dim / 128, min_dim / 32)},
        {"new_gaussians": num_gaussians_initial*4, "iterations": initial_iterations+3000, "sigma_range": (4, min_dim / 64)}
    ]

    total_iterations = sum(stage["iterations"] for stage in stages)
    params = None
    current_iteration = 0
    total_gaussians = 0

    for stage_idx, stage in enumerate(stages):
        num_new_gaussians = stage["new_gaussians"]
        total_gaussians += num_new_gaussians
        print(f"Starting stage {stage_idx + 1}: {total_gaussians} total Gaussians, {stage['iterations']} iterations")

        if stage_idx == 0:
            params = initialize_gaussians(num_new_gaussians, width, height, stage["sigma_range"], low_res_tensor, DEVICE)
        else:
            print(f"Adding {num_new_gaussians} new Gaussians based on error")
            with torch.no_grad():
                rendered = render_gaussians(params, width, height, DEVICE, min_dim, x_grid, y_grid)
                error = (rendered - target_image) ** 2
                error_per_pixel = error.sum(dim=2)
                error_flat = error_per_pixel.view(-1)
                prob = error_flat / error_flat.sum()
                indices = torch.multinomial(prob, num_new_gaussians, replacement=True)
                y_coords = torch.div(indices, width, rounding_mode='trunc').float()
                x_coords = (indices % width).float()
                mu_x = x_coords + 0.5
                mu_y = y_coords + 0.5
            new_params = initialize_gaussians(
                num_new_gaussians, width, height, stage["sigma_range"], low_res_tensor, DEVICE, mu_x=mu_x, mu_y=mu_y
            )
            params = torch.cat([params, new_params], dim=0).clone().detach().requires_grad_(True)

        lr = 0.1 if stage_idx == 0 else 0.01
        optimizer = torch.optim.AdamW([params], lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-6)  # Removed verbose=True

        for iter_idx in range(stage["iterations"]):
            start_iter = time.time()
            optimizer.zero_grad()

            with autocast('cuda', enabled=DEVICE.type == 'cuda'):
                rendered = render_gaussians(params, width, height, DEVICE, min_dim, x_grid, y_grid)
                mse_loss = torch.mean((rendered - target_image) ** 2)

                sigma = params[:, 5]
                anisotropy = params[:, 6]
                exp_aniso = torch.exp(0.5 * anisotropy)
                sigma_2 = sigma / exp_aniso

                sigma_reg = torch.mean(sigma**2) * 1e-5
                sigma_upper_penalty = torch.mean(torch.clamp(sigma - min_dim / 2, min=0)**2) * 1e-3
                anisotropy_reg = torch.mean(anisotropy**2) * 1e-1
                min_sigma2 = 0.5
                sigma2_penalty = torch.mean(torch.clamp(min_sigma2 - sigma_2, min=0)**2) * 1e-1

                _, alpha = gaussian_2d(x_grid.expand(params.shape[0], -1, -1), y_grid.expand(params.shape[0], -1, -1), params, min_dim)
                sparsity_reg = torch.mean(torch.sum(alpha, dim=(1, 2))) * 1e-4

                loss = mse_loss + sigma_reg + sigma_upper_penalty + anisotropy_reg + sigma2_penalty + sparsity_reg

            loss.backward()
            optimizer.step()

            if (current_iteration + 1) % 10 == 0:
                with torch.no_grad():
                    params[:, 0:3].clamp_(0, 255)
                    params[:, 3].clamp_(0, width)
                    params[:, 4].clamp_(0, height)
                    params[:, 5].clamp_(MIN_SIGMA, min_dim)
                    params[:, 6].clamp_(0, 2)
                    params[:, 7].clamp_(0, torch.pi)

            if (current_iteration + 1) % PRUNE_INTERVAL == 0:
                with torch.no_grad():
                    params = prune_gaussians(params, width, height, DEVICE, min_dim, x_grid, y_grid, threshold=PRUNE_THRESHOLD)
                    if params.shape[0] == 0:
                        print("All Gaussians pruned. Stopping training.")
                        break
                    params = params.clone().detach().requires_grad_(True)
                    optimizer = torch.optim.AdamW([params], lr=optimizer.param_groups[0]['lr'], weight_decay=1e-4)

            if (current_iteration + 1) % 100 == 0:
                with torch.no_grad():
                    exp_aniso = torch.exp(0.5 * params[:, 6])
                    sigma_2 = params[:, 5] / exp_aniso
                    mask = sigma_2 >= min_sigma2
                    if not mask.all():
                        print(f"Pruning {(~mask).sum().item()} thin Gaussians at iteration {current_iteration + 1}")
                        params = params[mask].clone().detach().requires_grad_(True)
                        optimizer = torch.optim.AdamW([params], lr=optimizer.param_groups[0]['lr'], weight_decay=1e-4)

            scheduler.step(mse_loss)

            if (current_iteration + 1) % 100 == 0:
                end_iter = time.time()
                print(f"Iteration {current_iteration + 1}/{total_iterations}, Stage {stage_idx + 1}, "
                      f"Loss: {loss.item():.4f}, MSE: {mse_loss.item():.4f}, Iter time: {end_iter - start_iter:.4f}s, "
                      f"Num Gaussians: {params.shape[0]}")

            current_iteration += 1

    os.makedirs(os.path.dirname(output_shadertoy_path) or '.', exist_ok=True)
    shader_code = generate_shadertoy_code(params, width, height)
    with open(output_shadertoy_path, 'w') as f:
        f.write(shader_code)
    print(f"Shadertoy code saved to {output_shadertoy_path}")

    save_gaussian_image(params, width, height, output_path=output_image_path, device=DEVICE, min_dim=min_dim, x_grid=x_grid, y_grid=y_grid)

if __name__ == "__main__":
    main()