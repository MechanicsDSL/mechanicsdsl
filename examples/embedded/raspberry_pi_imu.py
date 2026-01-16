"""
Raspberry Pi IMU Integration Example
=====================================

Demonstrates reading real sensor data from an MPU6050 IMU and using
MechanicsDSL for state estimation and physics simulation.

Hardware Requirements:
- Raspberry Pi (3/4/5 recommended)
- MPU6050 IMU sensor connected via I2C
  - VCC -> 3.3V (Pin 1)
  - GND -> GND (Pin 6)
  - SDA -> GPIO 2 (Pin 3)
  - SCL -> GPIO 3 (Pin 5)

Setup:
1. Enable I2C: sudo raspi-config -> Interface Options -> I2C
2. Install dependencies: pip install smbus2 numpy scipy matplotlib
3. Install MechanicsDSL: pip install mechanicsdsl-core

Author: MechanicsDSL Team
License: MIT
"""

import time
import numpy as np
from collections import deque

# =============================================================================
# MPU6050 IMU Driver
# =============================================================================

class MPU6050:
    """
    Driver for MPU6050 6-axis IMU.
    
    Provides accelerometer and gyroscope data for physics applications.
    """
    
    # MPU6050 Register addresses
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    
    # Scale factors
    ACCEL_SCALE = 16384.0  # LSB/g for ±2g range
    GYRO_SCALE = 131.0     # LSB/°/s for ±250°/s range
    
    def __init__(self, address=0x68, bus_number=1):
        """Initialize IMU.
        
        Args:
            address: I2C address (0x68 default, 0x69 if AD0 high)
            bus_number: I2C bus (1 for Pi 3/4/5)
        """
        self.address = address
        self._mock_mode = False
        
        try:
            import smbus2
            self.bus = smbus2.SMBus(bus_number)
            # Wake up MPU6050
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0)
            print(f"MPU6050 initialized at 0x{address:02x}")
        except Exception as e:
            print(f"Could not initialize MPU6050: {e}")
            print("Running in mock mode")
            self._mock_mode = True
            self.bus = None
    
    def _read_word(self, reg):
        """Read 16-bit signed value from register."""
        if self._mock_mode:
            return 0
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        value = (high << 8) + low
        if value >= 0x8000:
            value = -((65535 - value) + 1)
        return value
    
    def get_accel(self) -> tuple:
        """Get accelerometer data (m/s²)."""
        if self._mock_mode:
            # Return simulated gravity + noise
            return (
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                9.81 + np.random.normal(0, 0.1)
            )
        
        ax = self._read_word(self.ACCEL_XOUT_H) / self.ACCEL_SCALE * 9.81
        ay = self._read_word(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE * 9.81
        az = self._read_word(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE * 9.81
        return (ax, ay, az)
    
    def get_gyro(self) -> tuple:
        """Get gyroscope data (rad/s)."""
        if self._mock_mode:
            return (
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01),
                np.random.normal(0, 0.01)
            )
        
        gx = self._read_word(self.GYRO_XOUT_H) / self.GYRO_SCALE * np.pi / 180
        gy = self._read_word(self.GYRO_XOUT_H + 2) / self.GYRO_SCALE * np.pi / 180
        gz = self._read_word(self.GYRO_XOUT_H + 4) / self.GYRO_SCALE * np.pi / 180
        return (gx, gy, gz)
    
    def get_tilt_angle(self) -> float:
        """Estimate tilt angle from accelerometer (rad)."""
        ax, ay, az = self.get_accel()
        return np.arctan2(ay, np.sqrt(ax**2 + az**2))


# =============================================================================
# Complementary Filter for Sensor Fusion
# =============================================================================

class ComplementaryFilter:
    """
    Complementary filter for combining accelerometer and gyroscope data.
    
    Provides smooth angle estimates that combine:
    - Low-frequency: accelerometer (drift-free but noisy)
    - High-frequency: gyroscope (smooth but drifts)
    """
    
    def __init__(self, alpha: float = 0.98):
        """
        Args:
            alpha: Filter coefficient (0.95-0.99 typical)
                   Higher = more gyro trust
        """
        self.alpha = alpha
        self.angle = 0.0
        self.last_time = None
    
    def update(self, accel_angle: float, gyro_rate: float) -> float:
        """Update filter with new sensor readings.
        
        Args:
            accel_angle: Angle from accelerometer (rad)
            gyro_rate: Angular velocity from gyro (rad/s)
            
        Returns:
            Filtered angle estimate (rad)
        """
        now = time.time()
        
        if self.last_time is None:
            self.angle = accel_angle
            self.last_time = now
            return self.angle
        
        dt = now - self.last_time
        self.last_time = now
        
        # Complementary filter equation
        self.angle = self.alpha * (self.angle + gyro_rate * dt) + \
                     (1 - self.alpha) * accel_angle
        
        return self.angle


# =============================================================================
# Real-time Pendulum State Estimation
# =============================================================================

def run_realtime_estimation():
    """Run real-time pendulum state estimation using IMU data."""
    from mechanics_dsl import PhysicsCompiler
    
    print("=" * 60)
    print("MechanicsDSL - Real-time IMU State Estimation")
    print("=" * 60)
    
    # Initialize hardware
    imu = MPU6050()
    filter = ComplementaryFilter(alpha=0.98)
    
    # Data buffers
    times = deque(maxlen=200)
    angles = deque(maxlen=200)
    rates = deque(maxlen=200)
    
    print("Collecting data... Press Ctrl+C to stop")
    print("-" * 60)
    print(f"{'Time':>8} {'Angle (deg)':>12} {'Rate (deg/s)':>14}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        while True:
            # Get sensor data
            accel_angle = imu.get_tilt_angle()
            gx, gy, gz = imu.get_gyro()
            
            # Apply complementary filter
            filtered_angle = filter.update(accel_angle, gx)
            
            # Store data
            t = time.time() - start_time
            times.append(t)
            angles.append(filtered_angle)
            rates.append(gx)
            
            # Print status
            print(f"{t:8.2f} {np.degrees(filtered_angle):12.3f} {np.degrees(gx):14.3f}")
            
            time.sleep(0.05)  # 20 Hz sampling
            
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("Data collection complete")
    
    # Save data
    data = np.column_stack([
        list(times),
        list(angles),
        list(rates)
    ])
    np.savetxt('imu_data.csv', data, delimiter=',',
               header='time,angle,angular_rate', comments='')
    print(f"Saved {len(times)} samples to imu_data.csv")
    
    # Plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(list(times), np.degrees(list(angles)), 'b-')
        ax1.set_ylabel('Angle (deg)')
        ax1.set_title('IMU Tilt Angle')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(list(times), np.degrees(list(rates)), 'r-')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Rate (deg/s)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('imu_data.png', dpi=150)
        print("Plot saved to imu_data.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_realtime_estimation()
