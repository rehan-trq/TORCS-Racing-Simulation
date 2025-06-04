import sys
import socket
import time
import os
from datetime import datetime
from pynput import keyboard
import carState
import carControl
import msgParser

def main():
    print("\n=== MANUAL TORCS CONTROL WITH AUTO GEAR SHIFTING ===")
    print("Controls:")
    print("  W        - Accelerate")
    print("  S        - Brake")
    print("  A / D    - Steer Left/Right")
    print("  R        - Reverse Gear (-1)")
    print("  N        - Neutral Gear (0)")
    print("  1-6      - Manual Gear Selection")
    print("  G        - Toggle Auto Gear Shifting")
    print("  X        - Emergency Stop (zero all controls)")
    print("  ESC      - Quit")
    print("====================================================\n")
    
    # Set up telemetry logging
    log_filename = setup_telemetry_logging()
    print(f"Telemetry logging to: {log_filename}")
    
    # Telemetry buffer to reduce I/O overhead
    telemetry_buffer = []
    last_write_time = time.time()
    
    # Create socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as msg:
        print(f"Socket error: {msg}")
        return
    
    sock.settimeout(1.0)
    
    # TORCS connection parameters
    host = 'localhost'
    port = 3001
    bot_id = 'SCR'
    
    # Create objects
    parser = msgParser.MsgParser()
    state = carState.CarState()
    control = carControl.CarControl()
    
    # Initialize controls
    control.setAccel(0.0)
    control.setBrake(0.0)
    control.setSteer(0.0)
    control.setGear(1)
    control.setMeta(0)
    
    # Initialize rangefinder angles
    angles = [0 for _ in range(19)]
    for i in range(5):
        angles[i] = -90 + i * 15
        angles[18 - i] = 90 - i * 15
    for i in range(5, 9):
        angles[i] = -20 + (i-5) * 5
        angles[18 - i] = 20 - (i-5) * 5
    
    # Connect to TORCS
    print(f"Connecting to TORCS at {host}:{port}...")
    connected = False
    attempts = 0
    max_attempts = 20
    initial_timeout = 2.0
    
    sock.settimeout(initial_timeout)
    while not connected and attempts < max_attempts:
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts}...")
        try:
            init_string = bot_id + parser.stringify({'init': angles})
            sock.sendto(init_string.encode(), (host, port))
            data, addr = sock.recvfrom(1000)
            data_str = data.decode()
            if "***identified***" in data_str:
                print("Successfully connected to TORCS!")
                connected = True
                sock.settimeout(0.02)  # 20ms for game loop
                break
        except socket.timeout:
            print("Timeout waiting for server response. Retrying...")
        except Exception as e:
            print(f"Connection failed: {e}. Retrying...")
        time.sleep(1)
    
    if not connected:
        print(f"Failed to connect after {max_attempts} attempts. Ensure TORCS is running and configured.")
        sock.close()
        return
    
    print("\n=== CONNECTION ESTABLISHED ===")
    print("IMPORTANT: Use W,A,S,D for control, not arrow keys")
    print("Auto Gear Shifting is ON by default")
    
    # Control variables
    accel = 0.0
    brake = 0.0
    steer = 0.0
    gear = 1
    meta = 0
    
    # Auto gear shifting settings
    auto_gear_shift = True
    in_reverse = False
    
    # Gear shift thresholds in km/h
    gear_thresholds = {1: 40, 2: 80, 3: 140, 4: 200, 5: 260}
    downshift_thresholds = {2: 35, 3: 70, 4: 130, 5: 190, 6: 250}
    
    # Keyboard state
    key_states = {
        'w': False, 's': False, 'a': False, 'd': False, 'g': False,
        'r': False, 'n': False, '1': False, '2': False, '3': False,
        '4': False, '5': False, '6': False, 'x': False, 'esc': False
    }
    
    def on_press(key):
        try:
            k = key.char.lower() if hasattr(key, 'char') else str(key).lower()
            if k in key_states:
                key_states[k] = True
                print(f"Key pressed: {k}")
        except Exception as e:
            print(f"Error processing key press: {e}")
    
    def on_release(key):
        try:
            k = key.char.lower() if hasattr(key, 'char') else str(key).lower()
            if k in key_states:
                key_states[k] = False
                print(f"Key released: {k}")
            if k == 'esc':
                return False  # Stop listener
        except Exception as e:
            print(f"Error processing key release: {e}")
    
    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    def reset_controls():
        nonlocal accel, brake, steer
        accel = 0.0
        brake = 0.0
        steer = 0.0
        print("CONTROLS RESET TO ZERO")
    
    # Main control loop
    running = True
    while running:
        try:
            # Process keyboard inputs
            if key_states['w']:
                accel = 1.0
                brake = 0.0
                if in_reverse:
                    in_reverse = False
                    gear = 1
                    print("Switching to forward gear")
            elif key_states['s']:
                accel = 0.0
                brake = 1.0
            else:
                accel = 0.0
                brake = 0.0
            
            if key_states['a']:
                steer = 0.5
            elif key_states['d']:
                steer = -0.5
            else:
                steer = 0.0
            
            if key_states['g'] and not key_states.get('g_last', False):
                auto_gear_shift = not auto_gear_shift
                print(f"Auto Gear Shifting: {'ON' if auto_gear_shift else 'OFF'}")
                key_states['g_last'] = True
            elif not key_states['g']:
                key_states['g_last'] = False
            
            if key_states['r']:
                gear = -1
                in_reverse = True
                auto_gear_shift = False
                print("REVERSE GEAR SELECTED - Auto shift disabled")
            elif key_states['n']:
                gear = 0
                in_reverse = False
                auto_gear_shift = False
                print("NEUTRAL GEAR SELECTED - Auto shift disabled")
            elif any(key_states[k] for k in ['1', '2', '3', '4', '5', '6']):
                for k in ['1', '2', '3', '4', '5', '6']:
                    if key_states[k]:
                        gear = int(k)
                        in_reverse = False
                        auto_gear_shift = False
                        print(f"GEAR {gear} SELECTED - Auto shift disabled")
                        break
            
            if key_states['x']:
                reset_controls()
                auto_gear_shift = True
                print("AUTO GEAR SHIFT RE-ENABLED")
            
            if key_states['esc']:
                meta = 1
                running = False
                print("Quitting...")
            
            # Apply controls
            control.setAccel(accel)
            control.setBrake(brake)
            control.setSteer(steer)
            control.setGear(gear)
            control.setMeta(meta)
            
            # Receive message from server
            try:
                data, addr = sock.recvfrom(1000)
                msg = data.decode()
            except socket.timeout:
                continue
            except socket.error as e:
                print(f"Socket error: {e}")
                continue
            
            if "***shutdown***" in msg:
                print("Server shutting down.")
                running = False
                break
            if "***restart***" in msg:
                print("Server restarting.")
                reset_controls()
                gear = 1
                auto_gear_shift = True
                in_reverse = False
                log_filename = setup_telemetry_logging()
                print(f"New telemetry file: {log_filename}")
                telemetry_buffer.clear()
                last_write_time = time.time()
                continue
            
            # Parse message and update state
            try:
                state.setFromMsg(msg)
            except Exception as e:
                print(f"Error parsing message: {e}")
                continue
            
            # Validate sensor data
            if not validate_state(state):
                print("Incomplete sensor data, skipping telemetry log")
                continue
            
            # Auto gear shifting
            speed = state.getSpeedX() or 0
            speed_kmh = abs(speed * 3.6)
            if auto_gear_shift and not in_reverse and brake == 0 and accel > 0:
                if gear < 6 and speed_kmh > gear_thresholds.get(gear, 0):
                    gear += 1
                    print(f"AUTO UPSHIFT: Speed {speed_kmh:.1f} km/h -> Gear {gear}")
                elif gear > 1 and speed_kmh < downshift_thresholds.get(gear, 0):
                    gear -= 1
                    print(f"AUTO DOWNSHIFT: Speed {speed_kmh:.1f} km/h -> Gear {gear}")
            
            # Log telemetry to buffer
            telemetry_row = get_telemetry_row(state, control, auto_gear_shift)
            telemetry_buffer.append(telemetry_row)
            
            # Write buffer to file every second
            current_time = time.time()
            if current_time - last_write_time >= 1.0:
                if telemetry_buffer:
                    write_telemetry_buffer(log_filename, telemetry_buffer)
                    telemetry_buffer.clear()
                last_write_time = current_time
            
            # Display status
            car_gear = state.getGear() or 0
            rpm = state.getRpm() or 0
            gear_mode = "AUTO" if auto_gear_shift else "MANUAL"
            print(f"CAR: Speed={speed_kmh:.1f} km/h, RPM={rpm:.0f}, Gear={car_gear} | "
                  f"Controls: Accel={accel:.2f}, Brake={brake:.2f}, Steer={steer:.2f}, Gear={gear} | Mode: {gear_mode}")
            
            # Send controls
            try:
                response = control.toMsg()
                sock.sendto(response.encode(), (host, port))
            except socket.error as e:
                print(f"Error sending controls: {e}")
                continue
        
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue
    
    # Write remaining buffer
    if telemetry_buffer:
        write_telemetry_buffer(log_filename, telemetry_buffer)
    
    # Cleanup
    listener.stop()
    sock.close()
    print(f"Client closed. Telemetry saved to {log_filename}")

def setup_telemetry_logging():
    """Setup telemetry logging to file"""
    if not os.path.exists('logs'):
        os.makedirs('logs')
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_filename = f"logs/telemetry_manual_{timestamp}.csv"
    with open(log_filename, 'w') as f:
        headers = [
            "Time", "Speed", "Gear", "RPM", "TrackPos", "Angle", "Damage", "Fuel",
            "DistRaced", "RacePos", "SpeedY", "SpeedZ", "Z"
        ] + [f"Track{i}" for i in range(19)] + [f"Opponent{i}" for i in range(36)] + \
          [f"WheelSpinVel{i}" for i in range(4)] + ["Accel", "Brake", "Steer", "Gear", "ControlMode"]
        f.write(",".join(headers) + "\n")
    return log_filename

def validate_state(state):
    """Validate that essential sensor data is present"""
    essential_sensors = [
        state.getCurLapTime(), state.getSpeedX(), state.getGear(), state.getRpm(),
        state.getTrackPos(), state.getAngle(), state.getTrack(), state.getOpponents()
    ]
    return all(s is not None for s in essential_sensors) and \
           len(state.getTrack() or []) == 19 and len(state.getOpponents() or []) == 36

def get_telemetry_row(state, control, auto_gear_shift):
    """Generate a telemetry row from state and control"""
    lap_time = state.getCurLapTime() or 0
    speed_x = state.getSpeedX() or 0
    gear = state.getGear() or 0
    rpm = state.getRpm() or 0
    track_pos = state.getTrackPos() or 0
    angle = state.getAngle() or 0
    damage = state.getDamage() or 0
    fuel = state.getFuel() or 0
    dist_raced = state.getDistRaced() or 0
    race_pos = state.getRacePos() or 0
    speed_y = state.getSpeedY() or 0
    speed_z = state.getSpeedZ() or 0
    z = state.getZ() or 0
    track = (state.getTrack() or []) + [0] * (19 - len(state.getTrack() or []))
    opponents = (state.getOpponents() or []) + [0] * (36 - len(state.getOpponents() or []))
    wheel_spin_vel = (state.getWheelSpinVel() or []) + [0] * (4 - len(state.getWheelSpinVel() or []))
    accel = control.getAccel()
    brake = control.getBrake()
    steer = control.getSteer()
    control_gear = control.getGear()
    control_mode = "Auto" if auto_gear_shift and control_gear >= 1 else "Manual"
    
    return [
        f"{lap_time:.3f}", f"{speed_x:.3f}", f"{gear}", f"{rpm:.1f}",
        f"{track_pos:.3f}", f"{angle:.3f}", f"{damage:.1f}", f"{fuel:.1f}",
        f"{dist_raced:.3f}", f"{race_pos}", f"{speed_y:.3f}", f"{speed_z:.3f}", f"{z:.3f}"
    ] + [f"{v:.3f}" for v in track] + [f"{v:.3f}" for v in opponents] + \
      [f"{v:.3f}" for v in wheel_spin_vel] + \
      [f"{accel:.3f}", f"{brake:.3f}", f"{steer:.3f}", f"{control_gear}", control_mode]

def write_telemetry_buffer(filename, buffer):
    """Write telemetry buffer to file"""
    try:
        with open(filename, 'a') as f:
            for row in buffer:
                f.write(",".join(map(str, row)) + "\n")
    except Exception as e:
        print(f"Error writing telemetry: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Program crashed: {e}")