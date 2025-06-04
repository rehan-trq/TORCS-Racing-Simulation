import socket
import sys
import getopt
import os
import time
import math
import random
import subprocess
import psutil
import xml.etree.ElementTree as ET

# Configuration constants
PI_VALUE = math.pi
MAX_BUFFER = 2**17
VERSION_TAG = "2025-05-11-v2"

# Command-line help text
USAGE_GUIDE = """Usage: {} [options]
Options:
  -H, --host <host>      Hostname of TORCS server [localhost]
  -p, --port <port>      Port for TORCS server [3001]
  -i, --id <id>          Client identifier [SCR]
  -m, --maxsteps <#>     Maximum steps for simulation [100000]
  -e, --episodes <#>     Maximum training episodes [1]
  -t, --track <name>     Track name for reference [unknown]
  -s, --stage <#>        Stage: 0=warmup, 1=qualifying, 2=race, 3=unknown [3]
  -d, --debug            Enable detailed telemetry output
  -h, --help             Display this usage guide
  -v, --version          Show version information
"""

def restrict(value, min_val, max_val):
    """Limit a value to a specified range."""
    return max(min_val, min(value, max_val))

def ascii_graph(value, min_val, max_val, width, symbol='X'):
    """Create an ASCII representation of a value as a bar graph."""
    if width <= 0:
        return ""
    value = restrict(value, min_val, max_val)
    range_span = max_val - min_val
    if range_span <= 0:
        return "invalid range"
    units_per_char = range_span / width
    if units_per_char <= 0:
        return "zero division"
    
    graph = ""
    if min_val < 0:
        neg_chars = int(max(0, -min_val) / units_per_char) if value >= 0 else int(-value / units_per_char)
        graph += "-" * neg_chars
    if value < 0:
        graph += symbol * int(abs(value) / units_per_char)
    else:
        graph += symbol * int(value / units_per_char)
    if max_val > 0:
        pos_chars = int(max(0, max_val - value) / units_per_char) if value <= 0 else int((max_val - value) / units_per_char)
        graph += "_" * pos_chars
    return f"[{graph}]"

class RaceStatus:
    """Tracks the current state received from the TORCS server."""
    def __init__(self):
        self.state_data = {}
        self.raw_message = ""

    def parse_message(self, message):
        """Convert raw server message into a structured dictionary."""
        self.raw_message = message.strip()[:-1]
        segments = self.raw_message.lstrip('(').rstrip(')').split(')(')
        for segment in segments:
            parts = segment.split()
            self.state_data[parts[0]] = self._parse_values(parts[1:])

    def _parse_values(self, values):
        """Transform string values into appropriate numeric or list types."""
        if not values:
            return values
        if len(values) == 1:
            try:
                return float(values[0])
            except ValueError:
                return values[0]
        return [self._parse_values([v]) for v in values]

    def __str__(self):
        return self.format_output()

    def format_output(self, debug=False):
        """Generate a formatted display of the server state."""
        if not debug:
            return str(self.state_data)
        output = []
        keys = ['angle', 'trackPos', 'speedX', 'speedY', 'speedZ', 'rpm', 'gear', 'fuel', 'track', 'opponents']
        for key in keys:
            value = self.state_data.get(key)
            if isinstance(value, list):
                if key == 'track':
                    formatted = ' '.join([f"{x:.1f}" for x in value[:9]]) + f"_{value[9]:.1f}_" + ' '.join([f"{x:.1f}" for x in value[10:]])
                elif key == 'opponents':
                    formatted = ''.join(['.' if v > 90 else chr(int(v/2)+97-19) if v > 39 else '?' for v in value])
                    formatted = f" -> {formatted[:18]} {formatted[18:]} <-"
                else:
                    formatted = ', '.join(map(str, value))
            else:
                if key == 'gear':
                    gear = 'R' if value == -1 else 'N' if value == 0 else str(int(value))
                    formatted = f"[{gear}]"
                elif key == 'fuel':
                    formatted = f"{value:6.0f} {ascii_graph(value, 0, 100, 50, 'F')}"
                elif key == 'speedX':
                    formatted = f"{value:6.1f} {ascii_graph(value, -30, 300, 50, 'X' if value >= 0 else 'R')}"
                else:
                    formatted = str(value)
            output.append(f"{key}: {formatted}")
        return "\n".join(output)

class ControlCommand:
    """Represents the control inputs to be sent to the TORCS server."""
    def __init__(self):
        self.command_str = ""
        self.controls = {
            'accel': 0.2,
            'brake': 0,
            'clutch': 0,
            'gear': 1,
            'steer': 0,
            'focus': [-90, -45, 0, 45, 90],
            'meta': 0
        }

    def enforce_limits(self):
        """Ensure control values stay within valid ranges."""
        self.controls['steer'] = restrict(self.controls['steer'], -1, 1)
        self.controls['brake'] = restrict(self.controls['brake'], 0, 1)
        self.controls['accel'] = restrict(self.controls['accel'], 0, 1)
        self.controls['clutch'] = restrict(self.controls['clutch'], 0, 1)
        if self.controls['gear'] not in [-1, 0, 1, 2, 3, 4, 5, 6]:
            self.controls['gear'] = 0
        if self.controls['meta'] not in [0, 1]:
            self.controls['meta'] = 0
        if not isinstance(self.controls['focus'], list) or min(self.controls['focus']) < -180 or max(self.controls['focus']) > 180:
            self.controls['focus'] = 0

    def __str__(self):
        self.enforce_limits()
        command = ""
        for key, value in self.controls.items():
            command += f"({key} "
            if not isinstance(value, list):
                command += f"{value:.3f}" if isinstance(value, float) else str(value)
            else:
                command += ' '.join(map(str, value))
            command += ")"
        return command

    def detailed_output(self):
        """Provide a detailed view of control inputs for monitoring."""
        output = ""
        controls = self.controls.copy()
        for key in ['accel', 'brake', 'clutch', 'steer']:
            value = controls[key]
            if key in ['accel', 'brake', 'clutch']:
                graph = ascii_graph(value, 0, 1, 50, key[0].upper())
                output += f"{key}: {value:6.3f} {graph}\n"
            elif key == 'steer':
                graph = ascii_graph(-value, -1, 1, 50, 'S')
                output += f"{key}: {value:6.3f} {graph}\n"
        return output

class TorcsClient:
    """Manages communication with the TORCS server for the RL agent."""
    def __init__(self, host=None, port=None, sid=None, episodes=None, track=None, stage=None, debug=None,
                 vision=False, process_id=None, race_config_path=None, race_speed=1.0,
                 rendering=True, damage=False, lap_limiter=2, recdata=False,
                 noisy=False, rec_index=0, rec_episode_limit=1, rec_timestep_limit=3600,
                 rank=0):
        self.vision_enabled = vision
        self.server_host = 'localhost'
        self.server_port = 3001
        self.client_id = 'SCR'
        self.max_episodes = 1
        self.track_name = 'unknown'
        self.race_stage = 3
        self.debug_mode = False
        self.max_steps = 1000000000
        self.process_id = process_id
        self.race_config = race_config_path
        self.race_speed = race_speed
        self.render_enabled = rendering
        self.damage_enabled = damage
        self.lap_limit = lap_limiter
        self.record_data = recdata
        self.noise_enabled = noisy
        self.record_index = rec_index
        self.episode_limit = rec_episode_limit
        self.timestep_limit = rec_timestep_limit
        self.rank = rank

        self.parse_options()
        if host: self.server_host = host
        if port: self.server_port = port
        if sid: self.client_id = sid
        if episodes: self.max_episodes = episodes
        if track: self.track_name = track
        if stage: self.race_stage = stage
        if debug: self.debug_mode = debug

        self.S = RaceStatus()
        self.R = ControlCommand()
        self.initialize_connection()

    def parse_options(self):
        """Process command-line arguments."""
        try:
            opts, args = getopt.getopt(sys.argv[1:], 'H:p:i:m:e:t:s:dhv',
                                       ['host=', 'port=', 'id=', 'maxsteps=',
                                        'episodes=', 'track=', 'stage=',
                                        'debug', 'help', 'version'])
        except getopt.GetoptError as err:
            print(f"Option error: {err}\n{USAGE_GUIDE.format(sys.argv[0])}")
            sys.exit(-1)
        for opt, val in opts:
            if opt in ('-h', '--help'):
                print(USAGE_GUIDE.format(sys.argv[0]))
                sys.exit(0)
            if opt in ('-d', '--debug'):
                self.debug_mode = True
            if opt in ('-H', '--host'):
                self.server_host = val
            if opt in ('-p', '--port'):
                self.server_port = int(val)
            if opt in ('-i', '--id'):
                self.client_id = val
            if opt in ('-t', '--track'):
                self.track_name = val
            if opt in ('-s', '--stage'):
                self.race_stage = int(val)
            if opt in ('-e', '--episodes'):
                self.max_episodes = int(val)
            if opt in ('-m', '--maxsteps'):
                self.max_steps = int(val)
            if opt in ('-v', '--version'):
                print(f"{sys.argv[0]} {VERSION_TAG}")
                sys.exit(0)
        if args:
            print(f"Unexpected arguments: {', '.join(args)}\n{USAGE_GUIDE.format(sys.argv[0])}")
            sys.exit(-1)

    def initialize_connection(self):
        """Establish a UDP connection to the TORCS server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except socket.error as err:
            print(f"Failed to create socket: {err}")
            sys.exit(-1)
        self.socket.settimeout(1)

        attempts_left = 5
        sensor_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
        init_message = f"{self.client_id}(init {sensor_angles})"

        while True:
            try:
                self.socket.sendto(init_message.encode(), (self.server_host, self.server_port))
            except socket.error as err:
                print(f"Error sending init message: {err}")
                sys.exit(-1)
            try:
                data, addr = self.socket.recvfrom(MAX_BUFFER)
                data = data.decode('utf-8')
            except socket.error:
                print(f"Waiting for server on port {self.server_port}... Attempts left: {attempts_left}")
                attempts_left -= 1
                if attempts_left < 0:
                    if self.process_id:
                        try:
                            proc = psutil.Process(self.process_id)
                            for child in proc.children():
                                child.terminate()
                            proc.terminate()
                        except Exception:
                            self.process_id = None
                    args = ["torcs", "-nofuel", "-nolaptime", "-a", str(self.race_speed)]
                    if not self.damage_enabled:
                        args.append("-nodamage")
                    if self.noise_enabled:
                        args.append("-noisy")
                    if self.vision_enabled:
                        args.append("-vision")
                    if not self.render_enabled:
                        args.append("-T")
                    if self.race_config:
                        args.extend(["-raceconfig", self.race_config])
                    if self.record_data:
                        args.extend([f"-rechum {self.record_index}",
                                     f"-recepisodelim {self.episode_limit}",
                                     f"-rectimesteplim {self.timestep_limit}"])
                    args.append("&")
                    self.process_id = subprocess.Popen(args, shell=False).pid
                    attempts_left = 5
                continue
            if "***identified***" in data:
                print(f"Connected to server on port {self.server_port}")
                break

    def fetch_server_data(self):
        """Retrieve and process data from the TORCS server."""
        if not self.socket:
            return
        while True:
            try:
                data, addr = self.socket.recvfrom(MAX_BUFFER)
                data = data.decode('utf-8')
            except socket.error:
                print(".", end=' ', flush=True)
                continue
            if "***identified***" in data:
                print(f"Client reconnected on port {self.server_port}")
                continue
            if "***shutdown***" in data:
                print(f"Server shutdown on port {self.server_port}. Position: {self.S.state_data.get('racePos')}")
                self.terminate()
                return
            if "***restart***" in data:
                print(f"Server restarted race on port {self.server_port}")
                self.terminate()
                return
            if not data:
                continue
            self.S.parse_message(data)
            if self.debug_mode:
                print(f"\033[2J\033[H{self.S.format_output(debug=True)}")
            print(f"Received: speedX={self.S.state_data.get('speedX', 0):.2f}, gear={self.S.state_data.get('gear', 0)}")
            break

    def send_controls(self):
        """Transmit control commands to the TORCS server."""
        if not self.socket:
            return
        try:
            message = str(self.R)
            self.socket.sendto(message.encode(), (self.server_host, self.server_port))
            print(f"Sent: {message}")
        except socket.error as err:
            print(f"Error sending controls: {err}")
            sys.exit(-1)
        if self.debug_mode:
            print(self.R.detailed_output())

    def terminate(self):
        """Close the connection and clean up."""
        if self.socket:
            print(f"Terminating race on port {self.server_port} after {self.max_steps} steps")
            self.socket.close()
            self.socket = None

    def get_servers_input(self):
        """Wrapper for fetching server data."""
        self.fetch_server_data()

    def respond_to_server(self):
        """Wrapper for sending controls."""
        self.send_controls()

    def shutdown(self):
        """Alias for terminate."""
        self.terminate()

def control_example(client):
    """Sample control logic for testing (not used in DDPG)."""
    state, cmd = client.S.state_data, client.R.controls
    target_speed = 300
    cmd['steer'] = state['angle'] * 10 / PI_VALUE - state['trackPos'] * 0.10
    if state['speedX'] < target_speed - (cmd['steer'] * 50):
        cmd['accel'] += 0.01
    else:
        cmd['accel'] -= 0.01
    if state['speedX'] < 10:
        cmd['accel'] += 1 / (state['speedX'] + 0.1)
    if ((state['wheelSpinVel'][2] + state['wheelSpinVel'][3]) -
        (state['wheelSpinVel'][0] + state['wheelSpinVel'][1]) > 5):
        cmd['accel'] -= 0.2
    if state['rpm'] > 6500:
        cmd['gear'] = min(cmd['gear'] + 1, 6)
    elif state['rpm'] < 3000 and cmd['gear'] > 1:
        cmd['gear'] = max(cmd['gear'] - 1, 1)

if __name__ == "__main__":
    client = TorcsClient(port=3001)
    for step in range(client.max_steps, 0, -1):
        client.get_servers_input()
        control_example(client)
        client.respond_to_server()
    client.shutdown()