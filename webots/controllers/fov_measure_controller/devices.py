from math import cos, sin


class CrazyflieDevices:
    def __init__(self, robot, timestep_ms):
        self.robot = robot
        self.timestep_ms = timestep_ms
        self._init_motors()
        self._init_sensors()
        self._prev_time = None
        self._prev_x = None
        self._prev_y = None

    def _init_motors(self):
        self.m1_motor = self.robot.getDevice("m1_motor")
        self.m1_motor.setPosition(float("inf"))
        self.m1_motor.setVelocity(-1)
        self.m2_motor = self.robot.getDevice("m2_motor")
        self.m2_motor.setPosition(float("inf"))
        self.m2_motor.setVelocity(1)
        self.m3_motor = self.robot.getDevice("m3_motor")
        self.m3_motor.setPosition(float("inf"))
        self.m3_motor.setVelocity(-1)
        self.m4_motor = self.robot.getDevice("m4_motor")
        self.m4_motor.setPosition(float("inf"))
        self.m4_motor.setVelocity(1)

    def _init_sensors(self):
        timestep = int(self.timestep_ms)
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(timestep)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(timestep)
        self.range_front = self.robot.getDevice("range_front")
        self.range_front.enable(timestep)
        self.range_left = self.robot.getDevice("range_left")
        self.range_left.enable(timestep)
        self.range_back = self.robot.getDevice("range_back")
        self.range_back.enable(timestep)
        self.range_right = self.robot.getDevice("range_right")
        self.range_right.enable(timestep)

    def read(self):
        time_now = self.robot.getTime()
        if self._prev_time is None:
            dt = max(self.timestep_ms / 1000.0, 1e-6)
        else:
            dt = max(time_now - self._prev_time, 1e-6)

        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        x_global, y_global, altitude = self.gps.getValues()

        if self._prev_x is None:
            v_x_global = 0.0
            v_y_global = 0.0
        else:
            v_x_global = (x_global - self._prev_x) / dt
            v_y_global = (y_global - self._prev_y) / dt

        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        camera_image = self.camera.getImage()
        camera_width = self.camera.getWidth()
        camera_height = self.camera.getHeight()

        snapshot = {
            "time": time_now,
            "dt": dt,
            "imu": {"roll": roll, "pitch": pitch, "yaw": yaw},
            "gyro": {"yaw_rate": yaw_rate},
            "gps": {"x": x_global, "y": y_global, "z": altitude},
            "velocity": {
                "vx": v_x,
                "vy": v_y,
                "vx_global": v_x_global,
                "vy_global": v_y_global,
            },
            "ranges": {
                "front": self.range_front.getValue() / 1000.0,
                "left": self.range_left.getValue() / 1000.0,
                "back": self.range_back.getValue() / 1000.0,
                "right": self.range_right.getValue() / 1000.0,
            },
            "camera": {
                "image": camera_image,
                "width": camera_width,
                "height": camera_height,
                "time": time_now,
            },
        }

        self._prev_time = time_now
        self._prev_x = x_global
        self._prev_y = y_global
        return snapshot

    def set_motor_speeds(self, motor_power):
        self.m1_motor.setVelocity(-motor_power[0])
        self.m2_motor.setVelocity(motor_power[1])
        self.m3_motor.setVelocity(-motor_power[2])
        self.m4_motor.setVelocity(motor_power[3])
