import aliengo_py
import numpy as np
import threading
import time
from typing import Optional
import h5py
from scipy.spatial.transform import Rotation as R
from plot_client import LivePlotClient

Robot = aliengo_py.Robot
AliengoCommand = aliengo_py.AliengoCommand
AliengoState = aliengo_py.AliengoState

ORBIT_JOINT_ORDER = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

SDK_JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


# fmt: off
default_joint_pos = np.array(
    [
        0.1, -0.1, 0.1, -0.1,
        0.8, 0.8, 0.8, 0.8,
        -1.5, -1.5, -1.5, -1.5,
    ],
)
# fmt on


def normalize(v: np.ndarray):
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, alpha: float):
    return a * (1 - alpha) + b * alpha

def wrap_to_pi(a: np.ndarray):
    # wrap to -pi to pi
    a = np.mod(a + np.pi, 2 * np.pi) - np.pi
    return a

def orbit_to_sdk(joints: np.ndarray):
    return np.flip(joints.reshape(3, 2, 2), axis=2).transpose(1, 2, 0).reshape(-1)

def sdk_to_orbit(joints: np.ndarray):
    return np.flip(joints.reshape(2, 2, 3), axis=1).transpose(2, 0, 1).reshape(-1)

print(sdk_to_orbit(np.array(SDK_JOINT_ORDER)) == np.array(ORBIT_JOINT_ORDER))
print(orbit_to_sdk(np.array(ORBIT_JOINT_ORDER)) == np.array(SDK_JOINT_ORDER))
print(orbit_to_sdk(default_joint_pos).reshape(4, 3))


def _moving_average(data_buf: np.ndarray) -> np.ndarray:
    return np.mean(data_buf, axis=0)


class Robot_py:
    def __init__(
        self,
        control_freq: int = 500,
        log_file: Optional[h5py.File] = None,
        window_size: int = 4,
        update_rate_hz: float = 200.0,
        debug: bool = True
    ) -> None:
        self._robot = aliengo_py.Robot(control_freq)
        self._running = False
        self._debug = debug
        self.log_file = log_file

        self.window_size = window_size
        self.update_period = 1.0 / update_rate_hz

        self._smooth_state = AliengoState()
        self._init_empty_state(self._smooth_state)

        self._field_buffers = {
            "jpos": np.zeros((window_size, 12)),
            "jvel": np.zeros((window_size, 12)),
            "jvel_diff": np.zeros((window_size, 12)),
            "jtorque": np.zeros((window_size, 12)),
            "quat": np.zeros((window_size, 4)),
            "rpy": np.zeros((window_size, 3)),
            "gyro": np.zeros((window_size, 3)),
            "projected_gravity": np.zeros((window_size, 3)),
        }
        self._buf_index = 0

        self.step_count = 0
        if self.log_file is not None:
            default_len = 20000
            self.log_file.attrs["cursor"] = 0
            # TODO
            self.log_file.create_dataset("jpos", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("jvel", (default_len, 12), maxshape=(None, 12))
            self.log_file.create_dataset("rpy", (default_len, 3), maxshape=(None, 3))

        self._thread = threading.Thread(target=self._update_loop, daemon=True)

    def _init_empty_state(self, state: AliengoState):
        # breakpoint()
        state.jpos = [0.0] * 12
        state.jvel = [0.0] * 12
        state.jvel_diff = [0.0] * 12
        state.jtorque = [0.0] * 12
        state.quat = [0.0] * 4
        state.rpy = [0.0] * 3
        state.gyro = [0.0] * 3
        state.projected_gravity = [0.0] * 3
        state.lxy = [0.0, 0.0]
        state.rxy = [0.0, 0.0]
        state.timestamp = 0.0

    def start_control(self) -> None:
        self._robot.start_control()
        self._thread.start()
        time.sleep(0.5)
        while not self._running:
            print("wait init...")
            time.sleep(0.5)
        print("init finished!")

    def stop_control(self) -> None:
        self._running = False
        self._thread.join()

    def set_command(self, command: AliengoCommand) -> None:
        command.jpos_des = (orbit_to_sdk(np.array(command.jpos_des))).tolist()
        # command.jvel_des = (orbit_to_sdk(np.array(command.jvel_des))).tolist()
        # command.tau_ff = (orbit_to_sdk(np.array(command.tau_ff))).tolist()
        # print(command.jpos_des)
        if self._debug:
            return
        self._robot.set_command(command)

    def get_state(self) -> AliengoState:
        return self._smooth_state

    def _update_loop(self):
        self._running = True
        while self._running:
            t0 = time.time()
            raw_state = self._robot.get_state()

            self._push_raw_data(raw_state)
            self._smooth_state = self._compute_smoothed_state(raw_state)
            self._maybe_log(raw_state)

            elapsed = time.time() - t0
            sleep_time = self.update_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _push_raw_data(self, state: AliengoState):
        idx = self._buf_index
        bf = self._field_buffers

        bf["jpos"][idx] = state.jpos
        bf["jvel"][idx] = state.jvel
        bf["jvel_diff"][idx] = state.jvel_diff
        bf["jtorque"][idx] = state.jtorque
        bf["quat"][idx] = state.quat
        bf["rpy"][idx] = state.rpy
        bf["gyro"][idx] = state.gyro
        bf["projected_gravity"][idx] = state.projected_gravity

        self._buf_index = (idx + 1) % self.window_size

    def _compute_smoothed_state(self, raw_state: AliengoState) -> AliengoState:
        bf = self._field_buffers

        jpos_smoothed = sdk_to_orbit(_moving_average(bf["jpos"]))
        jvel_smoothed = sdk_to_orbit(_moving_average(bf["jvel"]))
        jvel_diff_smoothed = sdk_to_orbit(_moving_average(bf["jvel_diff"]))
        jtorque_smoothed = sdk_to_orbit(_moving_average(bf["jtorque"]))
        quat_smoothed = _moving_average(bf["quat"])
        rpy_smoothed = _moving_average(bf["rpy"])
        gyro_smoothed = _moving_average(bf["gyro"])
        proj_g_smoothed = _moving_average(bf["projected_gravity"])

        smooth_state = AliengoState()
        self._init_empty_state(smooth_state)

        smooth_state.jpos = jpos_smoothed.tolist()
        smooth_state.jvel = jvel_smoothed.tolist()
        smooth_state.jvel_diff = jvel_diff_smoothed.tolist()
        smooth_state.jtorque = jtorque_smoothed.tolist()
        smooth_state.quat = quat_smoothed.tolist()
        smooth_state.rpy = rpy_smoothed.tolist()
        smooth_state.gyro = gyro_smoothed.tolist()
        smooth_state.projected_gravity = proj_g_smoothed.tolist()

        smooth_state.lxy = raw_state.lxy
        smooth_state.rxy = raw_state.rxy
        smooth_state.timestamp = raw_state.timestamp

        return smooth_state

    def _maybe_log(self, raw_state: AliengoState):
        if self.log_file is None:
            return

        idx = self.step_count
        
        self.log_file["jpos"][idx] = np.array(raw_state.jpos)
        self.log_file["jvel"][idx] = np.array(raw_state.jvel)
        self.log_file["rpy"][idx] = np.array(raw_state.rpy)

        if idx == self.log_file["jpos"].len() - 1:
            new_len = idx + 1 + 3000
            print(f"Extend log size to {new_len}.")
            for key, value in self.log_file.items():
                value.resize((new_len, value.shape[1]))

        self.log_file.attrs["cursor"] = idx
        self.step_count += 1

class CommandManager:

    command_dim: int

    def __init__(self, robot: Robot_py) -> None:
        self._robot = robot
        self.command = np.zeros(self.command_dim)

    def update(self):
        pass


class JoyStickFlat(CommandManager):

    command_dim = 4

    max_speed_x = 1.0
    max_speed_y = 0.5
    max_angvel = 1.0

    def update(self):
        robot_state = self._robot.get_state()
        self.command[0] = mix(self.command[0], robot_state.lxy[1] * self.max_speed_x, 0.2)
        self.command[1] = mix(
            self.command[1], -robot_state.lxy[0] * self.max_speed_y, 0.2
        )
        self.command[2] = mix(
            self.command[2], -robot_state.rxy[0] * self.max_angvel, 0.2
        )
        self.command[3] = 0.0

class FixedCommandForce(CommandManager):
    command_dim = 10

    setpoint_pos_b = np.array([1.0, 0.0, 0.0])
    yaw_diff = 0.0

    kp = 5.0
    kd = 3.0
    virtual_mass = 1.5

    def update(self):
        self.command[:2] = self.setpoint_pos_b[:2]
        self.command[2] = self.yaw_diff
        self.command[3:5] = self.kp * self.setpoint_pos_b[:2]
        self.command[5:8] = self.kd
        self.command[8] = self.kp * self.yaw_diff
        self.command[9] = self.virtual_mass
        
class JoyStickForce_xy_yaw(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_angvel = 0.7

    def __init__(self, robot: aliengo_py.Robot) -> None:
        
        super().__init__(robot)
        self._init_yaw = self._robot.get_state().rpy[2]
        self._target_yaw = 0.0
        self._angvel = 0.0
    
    def update(self):
        robot_state = self._robot.get_state()

        self._angvel = mix(self._angvel, -robot_state.rxy[0] * self.max_angvel, 0.2)
        self._target_yaw += self._angvel * 0.02

        self.yaw_diff = wrap_to_pi(self._target_yaw + self._init_yaw - robot_state.rpy[2])
        self.yaw_diff = 0
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()
    
class JoyStickForce_xy_kp(FixedCommandForce):
    max_pos_x_b = 1.0
    max_pos_y_b = 1.0
    max_kp = 10.0
    min_kp = 2.0

    def update(self):
        robot_state = self._robot.get_state()
        
        self.kp = (robot_state.rxy[1] + 1) / 2 * (self.max_kp - self.min_kp) + self.min_kp
        
        self.setpoint_pos_b[0] = mix(self.setpoint_pos_b[0], robot_state.lxy[1] * self.max_pos_x_b, 0.2)
        self.setpoint_pos_b[1] = mix(self.setpoint_pos_b[1], -robot_state.lxy[0] * self.max_pos_y_b, 0.2)
        
        super().update()


class ActionManager:

    action_dim: int

    def __init__(self, robot: Robot_py) -> None:
        self._robot = robot
        self.robot_cmd = AliengoCommand()
        self.action = np.zeros(self.action_dim)

    def update(self):
        pass

    def step(self, action: np.ndarray) -> None:
        raise NotImplementedError


class JointPositionAction(ActionManager):

    action_dim = 12

    kp: float = 60
    kd: float = 2

    def __init__(
        self,
        robot: Robot_py,
        clip_joint_targets: float = 1.6,
        alpha: float = 0.9,
        action_scaling: float = 0.5,
    ) -> None:
        super().__init__(robot)

        self.robot_cmd.jpos_des = default_joint_pos.tolist()
        self.robot_cmd.jvel_des = [0.0] * 12
        self.robot_cmd.kp = [self.kp] * 12
        self.robot_cmd.kd = [self.kd] * 12
        self.robot_cmd.tau_ff = [0.0] * 12

        self.clip_joint_targets = clip_joint_targets
        self.alpha = alpha
        self.action_scaling = action_scaling

    def step(self, action_sim: np.ndarray) -> np.ndarray:
        action_sim = action_sim.clip(-6, 6)
        self.action = mix(self.action, action_sim, self.alpha)
        self.jpos_target = (self.action * self.action_scaling) + default_joint_pos
        self.robot_cmd.jpos_des = self.jpos_target.tolist()
        # print(self.robot_cmd.jpos_des)
        self._robot.set_command(self.robot_cmd)
        return action_sim


class EnvBase:

    dt: float = 0.02

    def __init__(
        self,
        robot: Robot_py,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        self.robot = robot
        self.robot_state = AliengoState()

        self.command_manager = command_manager
        self.action_manager = action_manager

    def reset(self):
        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def apply_action(self, action: np.ndarray):
        raise NotImplementedError
        action = self.action_manager.step(action)
        self.action_buf[:, 1:] = self.action_buf[:, :-1]
        self.action_buf[:, 0] = action

        self.update()
        self.command_manager.update()
        return self.compute_obs()

    def update(self):
        """Update the environment state buffers."""
        raise NotImplementedError

    def compute_obs(self):
        """Compute the observation from the environment state buffers."""
        raise NotImplementedError

class Oscillator:
    def __init__(self, use_history: bool = False):
        self.use_history = use_history
        self.phi = np.zeros(4)
        self.phi[0] = np.pi
        self.phi[3] = np.pi
        self.phi_dot = np.zeros(4)
        self.phi_history = np.zeros((4, 4))

    def update_phis(self, command, dt=0.02):
        omega = np.pi * 4
        move = True #np.abs(command[:3]).sum() > 0.1
        if move:
            dphi = omega + self._trot(self.phi)
        else:
            dphi = self._stand(self.phi)
        self.phi_dot[:] = dphi
        self.phi = (self.phi + self.phi_dot * dt) % (2 * np.pi)
        self.phi_history = np.roll(self.phi_history, 1, axis=0)
        self.phi_history[0] = self.phi

    def _trot(self, phi: np.ndarray):
        dphi = np.zeros(4)
        dphi[0] = (phi[3] - phi[0])
        dphi[1] = (phi[2] - phi[1]) + ((phi[0] + np.pi - phi[1]) % (2 * np.pi))
        dphi[2] = (phi[1] - phi[2]) + ((phi[0] + np.pi - phi[2]) % (2 * np.pi))
        dphi[3] = (phi[0] - phi[3])
        return dphi

    def _stand(self, phi: np.ndarray, target=np.pi * 3 / 2):
        dphi = 2.0 * ((target - phi) % (2 * np.pi))
        return dphi

    def get_osc(self):
        if self.use_history:
            phi_sin = np.sin(self.phi_history)
            phi_cos = np.cos(self.phi_history)
        else:
            phi_sin = np.sin(self.phi)
            phi_cos = np.cos(self.phi)
        osc = np.concatenate([phi_sin, phi_cos, self.phi_dot], axis=-1)
        if self.use_history:
            osc = osc.reshape(-1)
        return osc


class FlatEnvLQZ(EnvBase):
    action_buf_steps = 3
    gyro_steps = 4
    gravity_steps = 3
    jpos_steps = 3
    jvel_steps = 3
    def __init__(
        self,
        robot: Robot_py,
        command_manager: CommandManager,
        action_manager: ActionManager,
    ) -> None:
        super().__init__(
            robot=robot,
            command_manager=command_manager,
            action_manager=action_manager,
        )
        
        self.plot = LivePlotClient(zmq_addr="tcp://192.168.123.55:5555")

        self.jpos_multistep = np.zeros((self.jpos_steps, 12))
        self.jvel_multistep = np.zeros((self.jvel_steps, 12))
        self.gyro_multistep = np.zeros((self.gyro_steps, 3))
        self.gravity_multistep = np.zeros((self.gravity_steps, 3))

        self.oscillator = Oscillator(use_history=False)
        self.command = np.zeros(22, dtype=np.float32)

        self.jpos_sim = np.zeros(12)
        self.jvel_sim = np.zeros(12)
        self.gyro = np.zeros(3)
        self.gravity = np.zeros(3)
        

        self.action_buf = np.zeros((action_manager.action_dim, self.action_buf_steps))
        
        self.projected_gravity_history = np.zeros((3, 1))
        self.projected_gravity = np.zeros(3)

        self.cmd_dim = self.command.shape[0]
        self.obs_dim = self.compute_obs().shape[-1]

        self.step_count = 0

    def update_command(self):
        self.command[:10] = self.command_manager.command[:]
        self.command[10:22] = self.oscillator.get_osc()
        # self.plot.send(self.command[10:22].tolist())

    def update(self):
        self.robot_state = self.robot.get_state()

        self.jpos_sim[:] = self.robot_state.jpos
        self.jvel_sim[:] = self.robot_state.jvel
        self.gyro[:] = self.robot_state.gyro
        self.gravity[:] = self.robot_state.projected_gravity
        
        self.plot.send(self.gravity.tolist())
        # print(self.gravity)

        self.jpos_multistep = np.roll(self.jpos_multistep, shift=1, axis=0)
        self.jpos_multistep[0] = self.jpos_sim
        self.jvel_multistep = np.roll(self.jvel_multistep, shift=1, axis=0)
        self.jvel_multistep[0] = self.jvel_sim
        self.gyro_multistep = np.roll(self.gyro_multistep, shift=1, axis=0)
        self.gyro_multistep[0] = self.gyro
        self.gravity_multistep = np.roll(self.gravity_multistep, shift=1, axis=0)
        self.gravity_multistep[0] = self.gravity

        rpy = R.from_euler("xyz", self.robot_state.rpy)
        gravity = rpy.inv().apply(np.array([0., 0., -1.]))
        self.projected_gravity = gravity

        self.command_manager.update()
        self.oscillator.update_phis(self.command_manager.command)
        self.update_command()

    def compute_obs(self):
        self.update()
        jpos_multistep = self.jpos_multistep.copy()
        # jpos_multistep[1:] = jpos_multistep[1:] - jpos_multistep[:-1]
        jvel_multistep = self.jvel_multistep.copy()
        # jvel_multistep[1:] = jvel_multistep[1:] - jvel_multistep[:-1]
        # print(self.jvel_multistep)
        obs = [
            self.command,
            # self.gyro_multistep.reshape(-1),
            self.gravity_multistep.reshape(-1),
            jpos_multistep.reshape(-1),
            jvel_multistep.reshape(-1),
            self.action_buf[:, :3].reshape(-1),
        ]
        self.obs = np.concatenate(obs, dtype=np.float32)
        return self.obs

    def apply_action(self, action_sim: np.ndarray = None):
        if action_sim is not None:
            self.action_manager.step(action_sim)
            self.action_buf[:, 1:] = self.action_buf[:, :-1]
            self.action_buf[:, 0] = action_sim
            # self.plot.send(action_sim.tolist())