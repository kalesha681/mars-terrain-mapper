import numpy as np

class EKFTracker:
    """
    Extended Kalman Filter for fusing IMU + Optical Flow
    State vector: [position, velocity]
    
    Used in Mars Terrain Mapper to track detected feature 
    positions as drone moves across the arena.
    """

    def __init__(self, dt=0.01, imu_noise=0.5, flow_noise=0.3):
        self.dt = dt

        # ── State transition matrix ───────────────────────────
        self.F = np.array([[1, dt],
                           [0,  1]])

        # ── Control input matrix (IMU acceleration) ───────────
        self.B = np.array([[0.5 * dt**2],
                           [dt          ]])

        # ── Observation matrix (optical flow → velocity) ──────
        self.H = np.array([[0, 1]])

        # ── Process noise ─────────────────────────────────────
        self.Q = np.array([[1e-5, 0   ],
                           [0,    1e-3]])

        # ── Measurement noise ─────────────────────────────────
        self.R = np.array([[flow_noise**2]])

        # ── Initial state and covariance ──────────────────────
        self.x = np.zeros((2, 1))   # [position, velocity]
        self.P = np.eye(2)          # uncertainty

    def reset(self, initial_pos=0.0, initial_vel=0.0):
        """Reset filter to new initial conditions"""
        self.x = np.array([[initial_pos],
                           [initial_vel]])
        self.P = np.eye(2)

    def predict(self, imu_acc):
        """
        PREDICT step — propagate state using IMU
        imu_acc: scalar acceleration from IMU (m/s²)
        """
        u = np.array([[imu_acc]])
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0]  # return predicted position

    def update(self, flow_vel):
        """
        UPDATE step — correct with optical flow measurement
        flow_vel: scalar velocity from optical flow (m/s)
        """
        z = np.array([[flow_vel]])
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]  # return corrected position

    def step(self, imu_acc, flow_vel):
        """
        Run one full predict+update cycle
        Returns corrected position estimate
        """
        self.predict(imu_acc)
        return self.update(flow_vel)

    @property
    def position(self):
        return self.x[0, 0]

    @property
    def velocity(self):
        return self.x[1, 0]


# ── Quick self-test when run directly ────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.01
    t  = np.arange(0, 10, dt)
    N  = len(t)

    np.random.seed(42)
    true_pos  = np.sin(t)
    true_vel  = np.cos(t)
    true_acc  = -np.sin(t)
    imu_meas  = true_acc + 0.05 * t + np.random.normal(0, 0.5, N)
    flow_meas = true_vel + np.random.normal(0, 0.3, N)

    ekf = EKFTracker(dt=dt)
    est_pos = np.array([ekf.step(imu_meas[k], flow_meas[k]) for k in range(N)])

    rmse = np.sqrt(np.mean((est_pos - true_pos)**2))
    print(f"EKFTracker self-test RMSE: {rmse:.4f} m")

    plt.figure(figsize=(10, 4))
    plt.plot(t, true_pos, label='Ground Truth', color='green', linewidth=2)
    plt.plot(t, est_pos,  label=f'EKF (RMSE={rmse:.4f}m)', 
             color='blue', linewidth=1.5)
    plt.legend()
    plt.title("EKFTracker Class — Self Test")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()