#!/usr/bin/env python3
"""
Analyze and visualize drone telemetry (attitude and position) from a JSON log.
Used by hackathon organizers to judge the quality and safety of landings.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze drone telemetry log.")
    parser.add_argument("log_path", help="Path to the trajectory JSON file.")
    parser.add_argument("--save-plot", help="Path to save the resulting plot (e.g. 'logs/plot.png').")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"Error: Log file {log_path} not found.")
        return

    with open(log_path, "r") as f:
        data = json.load(f)

    if not data:
        print("Error: Log file is empty.")
        return

    # Extract data
    t = [step["t"] for step in data]
    drone_pos = np.array([step["drone_pos"] for step in data])
    drone_rpy = np.array([step["drone_rpy"] for step in data]) # Roll, Pitch, Yaw in radians
    
    # Convert to degrees for easier reading
    drone_rpy_deg = np.degrees(drone_rpy)

    # Calculate statistics
    max_tilt_deg = np.max(np.abs(drone_rpy_deg[:, :2])) # Max of Roll or Pitch
    max_descent_vel = np.max(-np.diff(drone_pos[:, 2]) / np.diff(t)) if len(t) > 1 else 0

    print("--- Telemetry Analysis ---")
    print(f"Total Flight Time: {t[-1]:.2f} s")
    print(f"Max Tilt (Roll/Pitch): {max_tilt_deg:.2f}°")
    print(f"Max Descent Velocity: {max_descent_vel:.2f} m/s")
    
    # Safety thresholds (example)
    TILT_THRESHOLD = 45.0
    if max_tilt_deg > TILT_THRESHOLD:
        print(f"WARNING: Drone exceeded safety tilt limit ({TILT_THRESHOLD}°)! Maneuver may be unsafe.")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Attitude
    ax1.plot(t, drone_rpy_deg[:, 0], label="Roll", color="red")
    ax1.plot(t, drone_rpy_deg[:, 1], label="Pitch", color="green")
    ax1.plot(t, drone_rpy_deg[:, 2], label="Yaw", color="blue")
    ax1.axhline(y=TILT_THRESHOLD, color="orange", linestyle="--", alpha=0.5, label="Limit")
    ax1.axhline(y=-TILT_THRESHOLD, color="orange", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Attitude [deg]")
    ax1.set_title("Drone Attitude over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Altitude
    ax2.plot(t, drone_pos[:, 2], label="Altitude (Z)", color="black")
    ax2.set_ylabel("Altitude [m]")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Drone Altitude over Time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    if args.save_plot:
        plot_path = Path(args.save_plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    else:
        print("Showing plot... (close window to finish)")
        plt.show()

if __name__ == "__main__":
    main()
