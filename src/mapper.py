import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
sys.path.insert(0, '.')
from src.ekf_tracker import EKFTracker

class ArenaMapper:
    """
    Builds a 2D top-down map of detected Mars terrain features.
    
    As drone flies over arena, each YOLOv8 detection gets assigned
    a stable world-frame position using EKF tracking.
    
    Usage:
        mapper = ArenaMapper(arena_size=5.0)
        mapper.add_detection(drone_x, drone_y, bbox, confidence)
        mapper.show()
    """

    def __init__(self, arena_size=5.0, cell_size=0.1):
        self.arena_size = arena_size  # metres
        self.cell_size  = cell_size   # metres per grid cell
        self.detections = []          # list of (x, y, conf) tuples
        self.drone_path = []          # drone trajectory

        # EKF for x and y axes independently
        self.ekf_x = EKFTracker(dt=0.1)
        self.ekf_y = EKFTracker(dt=0.1)

        # Grid map: counts detections per cell
        grid_dim = int(arena_size / cell_size)
        self.grid = np.zeros((grid_dim, grid_dim))

    def update_drone_position(self, imu_ax, imu_ay, flow_vx, flow_vy):
        """
        Update drone position estimate using EKF.
        Returns (x, y) estimated position.
        """
        x = self.ekf_x.step(imu_ax, flow_vx)
        y = self.ekf_y.step(imu_ay, flow_vy)
        self.drone_path.append((x, y))
        return x, y

    def add_detection(self, drone_x, drone_y, confidence=1.0):
        """
        Register a feature detection at current drone position.
        drone_x, drone_y: EKF-estimated position in metres
        confidence: YOLOv8 detection confidence score
        """
        # Only record high-confidence detections
        if confidence < 0.3:
            return

        self.detections.append((drone_x, drone_y, confidence))

        # Update grid map
        gx = int((drone_x + self.arena_size/2) / self.cell_size)
        gy = int((drone_y + self.arena_size/2) / self.cell_size)

        grid_dim = self.grid.shape[0]
        if 0 <= gx < grid_dim and 0 <= gy < grid_dim:
            self.grid[gy, gx] += confidence

    def show(self, title="Mars Arena Survey Map"):
        """Render the 2D map with drone path and detections."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── Left: Heatmap ─────────────────────────────────────
        im = axes[0].imshow(
            self.grid,
            cmap='hot',
            origin='lower',
            extent=[-self.arena_size/2, self.arena_size/2,
                    -self.arena_size/2, self.arena_size/2]
        )
        plt.colorbar(im, ax=axes[0], label='Detection Density')
        axes[0].set_title('Feature Detection Heatmap')
        axes[0].set_xlabel('X position (m)')
        axes[0].set_ylabel('Y position (m)')

        # ── Right: Scatter map with drone path ────────────────
        if self.drone_path:
            px, py = zip(*self.drone_path)
            axes[1].plot(px, py, 'b-', alpha=0.4,
                        linewidth=1, label='Drone Path')
            axes[1].plot(px[0], py[0], 'go', markersize=10,
                        label='Start')
            axes[1].plot(px[-1], py[-1], 'rs', markersize=10,
                        label='End')

        if self.detections:
            dx = [d[0] for d in self.detections]
            dy = [d[1] for d in self.detections]
            dc = [d[2] for d in self.detections]
            scatter = axes[1].scatter(dx, dy, c=dc, cmap='YlOrRd',
                                      s=50, zorder=5,
                                      label='Detections',
                                      vmin=0.3, vmax=1.0)
            plt.colorbar(scatter, ax=axes[1], label='Confidence')

        axes[1].set_xlim(-self.arena_size/2, self.arena_size/2)
        axes[1].set_ylim(-self.arena_size/2, self.arena_size/2)
        axes[1].set_title('Drone Path + Feature Locations')
        axes[1].set_xlabel('X position (m)')
        axes[1].set_ylabel('Y position (m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save to results
        plt.savefig('results/plots/arena_map.png', dpi=150,
                    bbox_inches='tight')
        print("Map saved to results/plots/arena_map.png")
        plt.show()

    def stats(self):
        """Print mapping statistics."""
        print(f"\n── Arena Mapping Stats ──────────────────")
        print(f"Total detections : {len(self.detections)}")
        print(f"Drone path points: {len(self.drone_path)}")
        if self.detections:
            confs = [d[2] for d in self.detections]
            print(f"Avg confidence   : {np.mean(confs):.3f}")
            print(f"Coverage area    : {np.sum(self.grid > 0) * self.cell_size**2:.2f} m²")


# ── Simulation test when run directly ────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    mapper = ArenaMapper(arena_size=5.0)

    # Simulate lawnmower survey pattern with true positions
    rows = 7
    for i, row_y in enumerate(np.linspace(-2, 2, rows)):
        if i % 2 == 0:
            xs = np.linspace(-2, 2, 80)
        else:
            xs = np.linspace(2, -2, 80)

        for x in xs:
            # True position + small noise (simulating VIO output)
            est_x = x + np.random.normal(0, 0.05)
            est_y = row_y + np.random.normal(0, 0.05)

            mapper.drone_path.append((est_x, est_y))

            # 25% detection rate along path
            if np.random.random() < 0.25:
                conf = np.random.uniform(0.4, 0.95)
                mapper.add_detection(est_x, est_y, conf)

    mapper.stats()
    mapper.show()