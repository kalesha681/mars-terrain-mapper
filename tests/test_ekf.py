"""
Unit tests for EKFTracker
Run with: python tests/test_ekf.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.ekf_tracker import EKFTracker



def test_ekf_initializes():
    ekf = EKFTracker()
    assert ekf.position == 0.0
    assert ekf.velocity == 0.0
    print("[PASS] test_ekf_initializes")

def test_ekf_predict_moves():
    ekf = EKFTracker(dt=0.1)
    # Apply constant acceleration
    for _ in range(10):
        ekf.predict(imu_acc=1.0)
    assert ekf.velocity > 0
    print("[PASS] test_ekf_predict_moves")

def test_ekf_update_corrects():
    ekf = EKFTracker()
    # Push state away from zero
    for _ in range(5):
        ekf.predict(imu_acc=2.0)
    vel_before = ekf.velocity
    # Correct with zero velocity measurement
    ekf.update(flow_vel=0.0)
    vel_after = ekf.velocity
    assert abs(vel_after) < abs(vel_before)
    print("[PASS] test_ekf_update_corrects")

def test_ekf_rmse_below_threshold():
    np.random.seed(42)
    dt = 0.01
    t = np.arange(0, 10, dt)
    N = len(t)
    true_pos = np.sin(t)
    true_vel = np.cos(t)
    true_acc = -np.sin(t)
    imu  = true_acc + np.random.normal(0, 0.5, N)
    flow = true_vel + np.random.normal(0, 0.3, N)

    ekf = EKFTracker(dt=dt)
    est = np.array([ekf.step(imu[k], flow[k]) for k in range(N)])
    rmse = np.sqrt(np.mean((est - true_pos)**2))

    assert rmse < 0.5, f"RMSE too high: {rmse:.4f}"
    print(f"[PASS] test_ekf_rmse_below_threshold (RMSE={rmse:.4f})")

def test_ekf_reset():
    ekf = EKFTracker()
    for _ in range(20):
        ekf.step(1.0, 0.5)
    ekf.reset(initial_pos=0.0, initial_vel=0.0)
    assert ekf.position == 0.0
    assert ekf.velocity == 0.0
    print("[PASS] test_ekf_reset")

if __name__ == "__main__":
    print("\n-- EKFTracker Unit Tests --------------------")
    test_ekf_initializes()
    test_ekf_predict_moves()
    test_ekf_update_corrects()
    test_ekf_rmse_below_threshold()
    test_ekf_reset()
    print("\n[PASS] All tests passed!")