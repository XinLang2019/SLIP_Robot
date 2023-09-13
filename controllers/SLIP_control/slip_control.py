from controller import Robot
import numpy as np
import math
import matplotlib.pyplot as plt


LOADING = 0x00
COMPRESSION = 0x01
THRUST = 0x02
UNLOADING = 0x03
FLIGHT = 0x04

x_dot_plot = []
x_f_plot = []
x_b_plot = []
pitch_plot = []



# IMU数据类
class eulerAngleTypeDef:
    def __init__(self):
        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.


class SLIP_Robot:
    def __init__(self):
        self.robot = Robot()
        self.leg_length = 1.2  # leg length init
        self.r_threshold = 0.95
        self.curr_leg_length = 0.
        self.timestep = int(2)
        self.timeunit = 0.001

        # get device
        self.leg_motor = self.robot.getDevice("leg_motor")
        self.leg_length_sensor = self.robot.getDevice("leg_length_sensor")

        self.leg_joint_0 = self.robot.getDevice("leg_joint_0")
        self.joint_sensor = self.robot.getDevice("joint_sensor_0")

        self.touch_sensor = self.robot.getDevice("touch_sensor")
        self.robot_imu = self.robot.getDevice("imu")
        self.robot_gps = self.robot.getDevice("gps")

        # enable sensor
        self.leg_length_sensor.enable(self.timestep)
        self.joint_sensor.enable(self.timestep)
        self.touch_sensor.enable(self.timestep)
        self.robot_imu.enable(self.timestep)
        self.robot_gps.enable(self.timestep)

        self.touch_state = 0.
        self.curr_leg_length = 0.
        self.leg_length_dot = 0.
        self.curr_joint_angle = 0.
        self.curr_joint_dot = 0.
        self.pre_joint_angle = 0.
        self.pre_joint_dot = 0.
        self.stateMachine = FLIGHT

        self.curr_imu_angle = eulerAngleTypeDef()
        self.curr_imu_dot = eulerAngleTypeDef()
        self.pre_imu_angle = eulerAngleTypeDef()
        self.pre_imu_dot = eulerAngleTypeDef()

        self.pre_touch_sate = 0.
        self.stance_start_ms = 0.
        self.system_ms = 0.
        self.stance_end_ms = 0.
        self.Ts = 0.

        # {B}坐标系工作空间
        self.workPoint_B = np.array([0., 0., 0.])
        # {H}坐标系工作空间
        self.workPoint_H = np.array([0., 0., 0.])
        # {B}坐标系工作空间期望值
        self.workPoint_B_desire = np.array([0., 0., 0.])
        # {H}坐标系工作空间期望值
        self.workPoint_H_desire = np.array([0., 0., 0.])
        self.R_H_B = np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        # 从{H}坐标系到{B}坐标系的转换
        self.R_B_H = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        self.pre_x_dot = 0.
        self.x_dot = 0.
        self.body_pre_position = np.array([0., 0., 0.])
        self.Body_pre_vel = np.array([0., 0., 0.])
        self.Body_vel = np.array([0., 0., 0.])

    def easyMat_rotX(self, angle):
        a = angle  # * 3.141592654 / 180.0
        RotX_InitPdata = np.array([[1.0, 0.0, 0.0],
                                   [0.0, np.cos(a), -np.sin(a)],
                                   [0.0, np.sin(a), np.cos(a)]])
        return RotX_InitPdata

    # 绕Y轴旋转的3*3旋转矩阵方阵
    # Mat = RotY(angle)
    # 采用角度制
    def easyMat_rotY(self, angle):
        a = angle  # * 3.141592654 / 180.0
        RotY_InitPdata = np.array([[np.cos(a), 0.0, np.sin(a)],
                                   [0.0, 1.0, 0.0],
                                   [-np.sin(a), 0.0, np.cos(a)]])
        return RotY_InitPdata

    # 绕Z轴旋转的3*3旋转矩阵方阵
    # Mat = RotZ(angle)
    # 采用角度制
    def easyMat_rotZ(self, angle):
        a = angle  # * 3.141592654 / 180.0
        RotZ_InitPdata = np.array([[np.cos(a), -np.sin(a), 0.0],
                                   [np.sin(a), np.cos(a), 0.0],
                                   [0.0, 0.0, 1.0]])
        return RotZ_InitPdata

    # 求复合旋转矩阵
    # outRot=RotY(yaw)*RotZ(pitch)*RotX(roll);
    def easyMat_RPY(self, roll, pitch, yaw):
        RotX = self.easyMat_rotX(roll)
        RotY = self.easyMat_rotY(pitch)
        RotZ = self.easyMat_rotZ(yaw)

        rot_zx = np.dot(RotY, RotX)
        temp = np.dot(RotZ, rot_zx)
        return temp

    def set_spring_force(self, force):
        max_force = 1800
        if force > max_force:
            force = max_force
        elif force < -max_force:
            force = -max_force
        self.leg_motor.setForce(-force)

    def set_motor_torque(self, torque):
        max_force = 1800
        if torque > max_force:
            torque = max_force
        elif torque < -max_force:
            torque = -max_force
        self.leg_joint_0.setTorque(torque)

    def get_leg_length(self):
        r = self.leg_length_sensor.getValue()
        r = self.leg_length - r
        return r

    def get_touch_state(self):
        return self.touch_sensor.getValue()

    def get_length_dot(self):
        # 弹簧长度及其导数更新
        now_r = self.get_leg_length()
        now_r_dot = (now_r - self.curr_leg_length) / (0.016 * self.timestep)
        self.curr_leg_length = now_r
        self.leg_length_dot = self.leg_length_dot * 0.5 + now_r_dot * 0.5
        # print("length_dot", self.leg_length_dot)

    def get_joint_angle(self):
        return self.joint_sensor.getValue()

    def get_joint_dot(self):
        self.pre_joint_dot = self.curr_joint_dot
        curr_angle = self.get_joint_angle()
        curr_vel = (curr_angle - self.pre_joint_angle) / (self.timeunit * self.timestep)
        self.curr_joint_dot = curr_vel * 0.5 + self.pre_joint_dot * 0.5
        self.pre_joint_angle = curr_angle

    def get_IMU_Angle(self):
        data = self.robot_imu.getRollPitchYaw()
        eulerAngle = eulerAngleTypeDef()
        eulerAngle.roll = data[0]
        eulerAngle.pitch = data[1]
        eulerAngle.yaw = data[2]
        # print("pitch;", eulerAngle.pitch)
        return eulerAngle

    def get_imu_dot(self):
        # IMU，IMU导数，以及旋转矩阵更新
        now_IMU_dot = eulerAngleTypeDef()
        now_IMU = self.get_IMU_Angle()

        now_IMU_dot.roll = (now_IMU.roll - self.curr_imu_angle.roll) / (self.timeunit * self.timestep)  # X
        now_IMU_dot.pitch = (now_IMU.pitch - self.curr_imu_angle.pitch) / (self.timeunit * self.timestep)  # Z
        now_IMU_dot.yaw = (now_IMU.yaw - self.curr_imu_angle.yaw) / (self.timeunit * self.timestep)  # Y

        self.curr_imu_angle = now_IMU

        self.curr_imu_dot.roll = self.curr_imu_dot.roll * 0.5 + now_IMU_dot.roll * 0.5
        self.curr_imu_dot.pitch = self.curr_imu_dot.pitch * 0.5 + now_IMU_dot.pitch * 0.5
        self.curr_imu_dot.yaw = self.curr_imu_dot.yaw * 0.5 + now_IMU_dot.yaw * 0.5

    def update_last_Ts(self):

        if (self.pre_touch_sate == 0.) and (self.touch_state == 1.):
            self.stance_start_ms = self.system_ms

        if (self.pre_touch_sate == 1.) and (self.touch_state == 0.):
            self.stance_end_ms = self.system_ms
            self.Ts = (self.stance_end_ms - self.stance_start_ms)
        self.pre_touch_sate = self.touch_state

    def forward_kinematics(self, theta):
        x = self.curr_leg_length * np.sin(theta)
        y = 0
        z = - self.curr_leg_length * np.cos(theta)
        return np.array([x, y, z])

    # get gps date
    def get_body_pos(self):
        position = self.robot_gps.getValues()
        return position

    # get body vel
    def get_body_vel(self):

        now_position = self.robot_gps.getValues()
        now_vel = np.zeros(3)
        body_vel = np.zeros(3)
        for i in range(3):
            now_vel[i] = (now_position[i] - self.body_pre_position[i]) / (self.timeunit * self.timestep)
            body_vel[i] = now_vel[i] * 0.5 + self.Body_pre_vel[i] * 0.5
        self.body_pre_position = now_position
        self.Body_pre_vel = body_vel
        if self.stateMachine in (COMPRESSION, THRUST):
            self.Body_vel = body_vel

    def update_x_dot(self):
        # 正运动学
        self.workPoint_B = self.forward_kinematics(self.curr_joint_angle)
        # print("joint_angle", self.curr_joint_angle)
        # print("pos_b", self.workPoint_B)
        # 转换到{H}坐标系下
        pre_x = self.workPoint_H[0]
        self.workPoint_H = np.dot(self.R_H_B, self.workPoint_B)
        now_x = self.workPoint_H[0]
        # 求导
        now_x_dot = (now_x - pre_x) / (self.timeunit * self.timestep)
        # 滤波
        now_x_dot = self.pre_x_dot * 0.5 + now_x_dot * 0.5
        self.pre_x_dot = now_x_dot
        if self.stateMachine in (COMPRESSION, THRUST):
            self.x_dot = now_x_dot

    def update_state(self):
        self.system_ms += self.timeunit * self.timestep
        # print("time:", self.system_ms)
        self.update_last_Ts()
        # print("Ts:", self.Ts)
        self.touch_state = self.get_touch_state()

        self.get_length_dot()
        self.curr_leg_length = self.get_leg_length()

        self.get_joint_dot()
        self.curr_joint_angle = self.get_joint_angle()

        self.get_imu_dot()
        self.curr_imu_angle = self.get_IMU_Angle()

        self.R_H_B = self.easyMat_RPY(self.curr_imu_angle.roll, self.curr_imu_angle.pitch, self.curr_imu_angle.yaw)
        self.R_B_H = self.R_H_B.transpose()

        self.update_x_dot()
        self.get_body_vel()

        self.updateRobotStateMachine()

    def updateRobotStateMachine(self):
        if self.stateMachine == LOADING:
            if self.curr_leg_length < self.leg_length * self.r_threshold:
                self.stateMachine = COMPRESSION

        elif self.stateMachine == COMPRESSION:
            if self.leg_length_dot > 0:
                self.stateMachine = THRUST

        elif self.stateMachine == THRUST:
            if self.curr_leg_length > self.leg_length * self.r_threshold:
                self.stateMachine = UNLOADING

        elif self.stateMachine == UNLOADING:
            if not self.touch_state:
                self.stateMachine = FLIGHT

        elif self.stateMachine == FLIGHT:
            if self.touch_state:
                self.stateMachine = LOADING

    def robot_control(self):

        self.update_state()
        kp = 2000
        fz = kp * (self.leg_length - self.get_leg_length())
        if self.stateMachine == THRUST:
            fz += 100
            # print("fz:", fz)
        self.set_spring_force(fz)
        # k_pose_v = 9
        # k_pose_p = 360
        k_pose_p = 400
        k_pose_v = 20
        # 控制臀部扭矩力
        if self.stateMachine in [LOADING, UNLOADING]:
            self.set_motor_torque(0)
        elif self.stateMachine in [COMPRESSION, THRUST]:
            Tx = -k_pose_p * self.curr_imu_angle.pitch - k_pose_v * self.curr_imu_dot.pitch
            self.set_motor_torque(-Tx)
            print("--------------- stance -------------------")
            # print("Tx:", Tx)
        elif self.stateMachine == FLIGHT:
            r = self.curr_leg_length
            k_xz_dot = 0.072
            # k_leg_v = 0.45
            # k_leg_p = 1.2
            k_leg_v = 3
            k_leg_p = 10

            print("---------------- flight -------------------")
            # print("x_dot", self.x_dot)
            x_dot_plot.append(self.Body_vel[0])
            pitch_plot.append(self.curr_imu_angle.pitch)
            print("x_dot", self.Body_vel[0])
            # print("Ts:", self.Ts)

            x_f = self.Body_vel[0] * self.Ts / 2.0 + k_xz_dot * (self.Body_vel[0] - 0.1)
            # print("x_f", x_f)
            print("r:", r)
            z_f = -math.sqrt(r * r - x_f * x_f)
            x_f_plot.append(x_f)

            self.workPoint_H_desire[0] = x_f
            self.workPoint_H_desire[1] = 0.
            self.workPoint_H_desire[2] = z_f
            self.workPoint_B_desire = np.dot(self.R_B_H, self.workPoint_H_desire)
            # print("rhb:", self.R_B_H)

            x_f_B = self.workPoint_B_desire[0]
            x_b_plot.append(x_f_B)
            y_f_B = self.workPoint_B_desire[1]
            z_f_B = self.workPoint_B_desire[2]
            x_angle_desire = math.atan(x_f_B / z_f_B)

            x_angle = self.curr_joint_angle
            x_angle_dot = self.curr_joint_dot
            # x_angle_desire = self.curr_imu_angle.pitch - math.asin(x_f / r)
            # x_angle = self.curr_joint_angle
            # x_angle_dot = self.curr_joint_dot

            Tx = -k_leg_p * (x_angle - x_angle_desire) - k_leg_v * x_angle_dot
            self.set_motor_torque(Tx)


if __name__ == '__main__':

    slip_robot = SLIP_Robot()
    while slip_robot.robot.step(slip_robot.timestep) != -1:
        slip_robot.robot_control()

    fig = plt.figure()

    # 绘制空间曲线
    plt.subplot(2, 2, 1)
    plt.plot(x_dot_plot)
    plt.title("t-X_dot")
    plt.subplot(2, 2, 2)
    plt.plot(x_f_plot)
    plt.title("t-x_f")
    plt.subplot(2, 2, 3)
    plt.plot(x_b_plot)
    plt.title("t-x_b")
    plt.subplot(2, 2, 4)
    plt.plot(pitch_plot)
    plt.title("t-pitch")
    # 显示图形
    plt.show()


