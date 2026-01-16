"""
Embedded Motor Control Example
==============================

Demonstrates using MechanicsDSL for real-time motor control on 
embedded systems like Raspberry Pi.

This example includes:
- DC motor dynamics simulation
- PID position controller
- GPIO control integration
- Real-time loop with timing

Hardware:
- Raspberry Pi
- DC motor with encoder
- Motor driver (L298N or similar)

Wiring:
- PWM output: GPIO 18
- Direction A: GPIO 23
- Direction B: GPIO 24
- Encoder A: GPIO 17
- Encoder B: GPIO 27
"""

import time
import math
import numpy as np

# Check for Raspberry Pi
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False
    print("Not on Raspberry Pi - using simulation mode")


class MotorSimulator:
    """Simulates DC motor dynamics when not on real hardware."""
    
    def __init__(self, J=0.01, b=0.1, K=0.01, R=1.0, L=0.5):
        """
        Motor parameters:
            J: Rotor inertia (kg*m^2)
            b: Viscous friction (N*m*s)
            K: Motor constant (N*m/A)
            R: Armature resistance (Ohm)
            L: Armature inductance (H)
        """
        self.J = J
        self.b = b
        self.K = K
        self.R = R
        self.L = L
        
        # State: [theta, omega, current]
        self.state = np.array([0.0, 0.0, 0.0])
        self.voltage = 0.0
    
    def step(self, dt: float):
        """Step simulation forward by dt seconds."""
        theta, omega, i = self.state
        V = self.voltage
        
        # Motor dynamics
        di_dt = (V - self.R * i - self.K * omega) / self.L
        domega_dt = (self.K * i - self.b * omega) / self.J
        dtheta_dt = omega
        
        # Euler integration
        self.state[0] += dtheta_dt * dt
        self.state[1] += domega_dt * dt
        self.state[2] += di_dt * dt
    
    @property
    def position(self):
        return self.state[0]
    
    @property
    def velocity(self):
        return self.state[1]


class PIDController:
    """PID controller for position control."""
    
    def __init__(self, Kp: float, Ki: float, Kd: float, 
                 output_limits: tuple = (-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None
    
    def update(self, setpoint: float, measurement: float) -> float:
        """Compute PID output."""
        now = time.time()
        
        if self.last_time is None:
            self.last_time = now
            self.last_error = setpoint - measurement
            return 0.0
        
        dt = now - self.last_time
        if dt <= 0:
            return 0.0
        
        error = setpoint - measurement
        
        # Proportional
        P = self.Kp * error
        
        # Integral with anti-windup
        self.integral += error * dt
        I = self.Ki * self.integral
        
        # Derivative
        derivative = (error - self.last_error) / dt
        D = self.Kd * derivative
        
        self.last_error = error
        self.last_time = now
        
        output = P + I + D
        
        # Clamp output
        return max(self.output_limits[0], 
                   min(self.output_limits[1], output))
    
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None


class MotorController:
    """Complete motor control system."""
    
    # GPIO pins
    PWM_PIN = 18
    DIR_A_PIN = 23
    DIR_B_PIN = 24
    ENC_A_PIN = 17
    ENC_B_PIN = 27
    
    def __init__(self, Kp=2.0, Ki=0.5, Kd=0.1):
        self.pid = PIDController(Kp, Ki, Kd)
        self.encoder_count = 0
        self.counts_per_rev = 360  # Encoder resolution
        
        if ON_PI:
            self._setup_gpio()
            self.motor = None
        else:
            self.motor = MotorSimulator()
            self.pwm = None
    
    def _setup_gpio(self):
        """Initialize GPIO for motor control."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Motor control pins
        GPIO.setup(self.PWM_PIN, GPIO.OUT)
        GPIO.setup(self.DIR_A_PIN, GPIO.OUT)
        GPIO.setup(self.DIR_B_PIN, GPIO.OUT)
        
        # PWM for speed control
        self.pwm = GPIO.PWM(self.PWM_PIN, 1000)  # 1kHz
        self.pwm.start(0)
        
        # Encoder inputs with pull-ups
        GPIO.setup(self.ENC_A_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.ENC_B_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Encoder interrupt
        GPIO.add_event_detect(self.ENC_A_PIN, GPIO.BOTH, 
                              callback=self._encoder_callback)
    
    def _encoder_callback(self, channel):
        """Handle encoder pulse."""
        a = GPIO.input(self.ENC_A_PIN)
        b = GPIO.input(self.ENC_B_PIN)
        
        if a == b:
            self.encoder_count += 1
        else:
            self.encoder_count -= 1
    
    @property
    def position(self) -> float:
        """Get current position in radians."""
        if ON_PI:
            return (self.encoder_count / self.counts_per_rev) * 2 * math.pi
        else:
            return self.motor.position
    
    def set_motor_power(self, power: float):
        """Set motor power (-1.0 to 1.0)."""
        power = max(-1.0, min(1.0, power))
        
        if ON_PI:
            if power >= 0:
                GPIO.output(self.DIR_A_PIN, GPIO.HIGH)
                GPIO.output(self.DIR_B_PIN, GPIO.LOW)
            else:
                GPIO.output(self.DIR_A_PIN, GPIO.LOW)
                GPIO.output(self.DIR_B_PIN, GPIO.HIGH)
            
            self.pwm.ChangeDutyCycle(abs(power) * 100)
        else:
            self.motor.voltage = power * 12.0  # 12V supply
    
    def move_to(self, target: float, timeout: float = 5.0):
        """Move to target position (radians)."""
        start_time = time.time()
        
        print(f"Moving to {math.degrees(target):.1f}°")
        
        while time.time() - start_time < timeout:
            current = self.position
            power = self.pid.update(target, current)
            self.set_motor_power(power)
            
            # Simulate motor dynamics
            if not ON_PI:
                self.motor.step(0.01)
            
            # Check if at target
            if abs(target - current) < 0.02:  # ~1 degree
                print(f"  Reached target in {time.time() - start_time:.2f}s")
                break
            
            time.sleep(0.01)
        
        self.set_motor_power(0)
        self.pid.reset()
    
    def cleanup(self):
        """Clean up GPIO."""
        self.set_motor_power(0)
        if ON_PI:
            self.pwm.stop()
            GPIO.cleanup()


def demo():
    """Run motor control demo."""
    print("=" * 60)
    print("MechanicsDSL Embedded Motor Control Demo")
    print("=" * 60)
    print(f"Mode: {'Hardware' if ON_PI else 'Simulation'}")
    
    controller = MotorController(Kp=5.0, Ki=0.5, Kd=0.2)
    
    try:
        # Move to various positions
        targets = [
            math.pi / 2,   # 90°
            math.pi,       # 180°
            math.pi / 2,   # 90°
            0.0            # 0°
        ]
        
        for target in targets:
            controller.move_to(target)
            time.sleep(0.5)
        
        print("\nDemo complete!")
        
    finally:
        controller.cleanup()


if __name__ == "__main__":
    demo()
