import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Set publication style
plt.rcParams['font.size'] = 8

fig = plt.figure(figsize=(14, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1.8, 1.4, 1.2])

# -------------------------------------------------------------------------
# 1. Base ERA5 Spatial Grid Plane (Z = 0)
# -------------------------------------------------------------------------
grid_x = np.linspace(100.0, 100.3, 4)
grid_y = np.linspace(30.0, 30.3, 4)

# Draw ERA5 0.1 deg Grid Lines on Base Plane
for x in grid_x:
    ax.plot([x, x], [30.0, 30.3], [0, 0], color='#A0A0A0', linestyle='--', linewidth=0.8)
for y in grid_y:
    ax.plot([100.0, 100.3], [y, y], [0, 0], color='#A0A0A0', linestyle='--', linewidth=0.8)

# ERA5 Grid Points (Red Squares)
X_mesh, Y_mesh = np.meshgrid(grid_x[:-1] + 0.05, grid_y[:-1] + 0.05)
ax.scatter(X_mesh.ravel(), Y_mesh.ravel(), np.zeros_like(X_mesh.ravel()), 
           color='#D9534F', marker='s', s=35, label='ERA5 Grid Points (0.1°)', zorder=5)

# Target ERA5 Grid Center
target_lon, target_lat = 100.15, 30.15
ax.scatter([target_lon], [target_lat], [0], color='#D9534F', marker='s', s=80, edgecolors='black', lw=1.2, zorder=6)

# -------------------------------------------------------------------------
# 2. 3D Spatiotemporal Bounding Volume
# -------------------------------------------------------------------------
cell_x, cell_y = 100.1, 30.1
dx, dy, dz = 0.1, 0.1, 15.0

def draw_cuboid(ax, corner, size, color='#E0E0E0', edgecolor='#333333', alpha=0.3):
    x, y, z = corner
    dx, dy, dz = size
    vertices = np.array([
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]], # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]], # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]], # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]], # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]], # Right
        [vertices[0], vertices[3], vertices[7], vertices[4]]  # Left
    ]
    poly = Poly3DCollection(faces, facecolors=color, linewidths=0.8, edgecolors=edgecolor, alpha=alpha)
    ax.add_collection3d(poly)

# Render Target Matching Cylinder / Box
draw_cuboid(ax, (cell_x, cell_y, 0), (dx, dy, dz), color='#BBDEFB', edgecolor='#1976D2', alpha=0.25)

# -------------------------------------------------------------------------
# 3. ACDL Lidar Orbit Tracks & Profile Curtains
# -------------------------------------------------------------------------
np.random.seed(42)
track_lon = np.linspace(100.02, 100.28, 14)
track_lat = np.linspace(30.02, 30.28, 14)

r_spatial = 0.05 * np.sqrt(2)
dist = np.sqrt((track_lon - target_lon)**2 + (track_lat - target_lat)**2)
inside_mask = dist <= r_spatial

# Ground Track Line
ax.plot(track_lon, track_lat, np.zeros_like(track_lon), color='#0275D8', linestyle='-', linewidth=1.5, zorder=4)

# Vertical Extinction Profiles
for i in range(len(track_lon)):
    x_i, y_i = track_lon[i], track_lat[i]
    is_in = inside_mask[i]
    
    p_color = '#5CB85C' if is_in else '#0275D8'
    p_alpha = 0.9 if is_in else 0.3
    p_lw = 1.8 if is_in else 0.8
    
    # 3D Profile Curtain Vertical Line
    ax.plot([x_i, x_i], [y_i, y_i], [0, 15], color=p_color, alpha=p_alpha, linewidth=p_lw)
    ax.scatter([x_i], [y_i], [0], color=p_color, s=25 if is_in else 15, alpha=p_alpha, zorder=5)

# -------------------------------------------------------------------------
# 4. Unobstructed Floating Callout Banners
# -------------------------------------------------------------------------
def add_floating_label(ax, text, target_pos, height_offset=3.5, bg_color='#FFFFFF', border_color='#777777'):
    tx, ty, tz = target_pos
    lz = tz + height_offset
    ax.plot([tx, tx], [ty, ty], [tz, lz], color=border_color, linestyle=':', linewidth=0.8, zorder=10)
    ax.text(tx, ty, lz, text, ha='center', va='bottom', fontsize=7.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, edgecolor=border_color, lw=0.8, alpha=0.95),
            zorder=11)

add_floating_label(ax, 'ERA5 Grid Node\n(0.1° × 0.1°)', (100.05, 30.05, 0), height_offset=5.0, bg_color='#FFEBEE', border_color='#D9534F')
add_floating_label(ax, 'Spatiotemporal Matching Box\nRadius r ≤ 0.05√2°\nWindow |dt| ≤ 30 min', (100.15, 30.15, 15), height_offset=3.0, bg_color='#E3F2FD', border_color='#1976D2')
add_floating_label(ax, 'Matched ACDL Profiles\n(Averaged across 1291 Levels)', (100.20, 30.20, 8), height_offset=6.0, bg_color='#E8F5E9', border_color='#4CAF50')
add_floating_label(ax, 'ACDL Orbit Track', (100.26, 30.26, 0), height_offset=4.0, bg_color='#E0F7FA', border_color='#0275D8')

# -------------------------------------------------------------------------
# 5. Axes Formatting
# -------------------------------------------------------------------------
ax.set_xlabel('Longitude (°E)', labelpad=8, fontsize=8.5, fontweight='bold')
ax.set_ylabel('Latitude (°N)', labelpad=8, fontsize=8.5, fontweight='bold')
ax.set_zlabel('Altitude z (km) / Vertical Layers', labelpad=8, fontsize=8.5, fontweight='bold')

ax.set_xlim(100.0, 100.3)
ax.set_ylim(30.0, 30.3)
ax.set_zlim(0, 22)

ax.view_init(elev=22, azim=-55)

plt.title('3D Spatiotemporal Matching Principle (ACDL Lidar vs. ERA5)', fontsize=11, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('ACDL_ERA5_3D_Matching.png', dpi=300, bbox_inches='tight')
plt.show()