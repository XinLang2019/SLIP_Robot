from controller import Robot
import numpy as np
import matplotlib.pyplot as plt
import math

LOADING = 0x00
COMPRESSION = 0x01
THRUST = 0x02
UNLOADING = 0x03
FLIGHT = 0x04

body_vel_plot = []
x_f_plot = []
x_angle_plot = []
x_angle_des_plot = []


# IMU数据类
class eulerAngleTypeDef:
    def __init__(self):
        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.


class Robot_Control:
    def __init__(self):
        self.robot = Robot()
        self.leg_length = 1.       # leg length init
        self.r_threshold = 0.95
        self.curr_leg_length = 1.  # curr leg length
        self.leg_length_dot = 0.
        self.r = 1.
        self.timestep = int(2)
        self.timeunit = 0.001

        # get device
        self.joint_motor = []
        self.joint_sensor = []
        self.joint_motor.append(self.robot.getDevice("joint_motor_0"))
        self.joint_motor.append(self.robot.getDevice("joint_motor_1"))
        self.joint_sensor.append(self.robot.getDevice("joint_sensor_0"))
        self.joint_sensor.append(self.robot.getDevice("joint_sensor_1"))

        self.touch_sensor = self.robot.getDevice("touch_sensor")
        self.robot_imu = self.robot.getDevice("imu")
        self.robot_gps = self.robot.getDevice("gps")

        # enable sensor
        self.joint_sensor[0].enable(self.timestep)
        self.joint_sensor[1].enable(self.timestep)
        self.touch_sensor.enable(self.timestep)
        self.robot_imu.enable(self.timestep)
        self.robot_gps.enable(self.timestep)

        # 关节角度 角速度 ，now & pre
        self.curr_joint_angle = np.zeros(2)
        self.curr_joint_dot = np.zeros(2)
        self.pre_joint_angle = np.array([0.5857, -1.1714])
        self.pre_joint_dot = np.zeros(2)
        self.stateMachine = FLIGHT
        self.touch_state = 0.

        # imu data now & pre
        self.curr_imu_angle = eulerAngleTypeDef()
        self.curr_imu_dot = eulerAngleTypeDef()
        self.pre_imu_angle = eulerAngleTypeDef()
        self.pre_imu_dot = eulerAngleTypeDef()

        # touch state now & pre , sim time and stance time
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
        # 从{B}坐标系到{H}坐标系的转换
        self.R_H_B = np.array([[1, 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        # 从{H}坐标系到{B}坐标系的转换
        self.R_B_H = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        # body forward velocity estimate
        self.body_pre_position = np.array([0., 0., 0.])
        self.body_pre_vel = np.array([0., 0., 0.])
        self.body_vel = np.array([0., 0., 0.])

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

    # set motor torque
    def set_motor_torque(self, joint, torque):
        max_torque = 1800
        if torque > max_torque:
            torque = max_torque
        elif torque < -max_torque:
            torque = -max_torque
        self.joint_motor[joint].setTorque(torque)

    def forward_kinematics(self, theta):
        a0 = 0.6
        a1 = 0.6
        x = -a0 * math.sin(theta[0]) - a1 * math.sin(theta[0] + theta[1])
        y = 0.
        z = -(a0 * math.cos(theta[0]) + a1 * math.cos(theta[0] + theta[1]))
        return np.array([x, y, z])

    def get_leg_length(self):
        angle_body = []
        for joint in range(2):
            angle_body.append(self.joint_sensor[joint].getValue())
        foot_pos_b = self.forward_kinematics(angle_body)
        r = math.sqrt(foot_pos_b[0]*foot_pos_b[0] + foot_pos_b[2]*foot_pos_b[2])
        return r

    def get_length_dot(self):
        # 弹簧长度及其导数更新
        now_r = self.get_leg_length()
        now_r_dot = (now_r - self.curr_leg_length) / (0.016 * self.timestep)
        self.curr_leg_length = now_r
        self.leg_length_dot = self.leg_length_dot * 0.5 + now_r_dot * 0.5

    def get_touch_state(self):
        return self.touch_sensor.getValue()

    def get_joint_angle(self):
        curr_joint_ang = np.zeros(2)
        for joint in range(2):
            curr_joint_ang[joint] = self.joint_sensor[joint].getValue()
        return curr_joint_ang

    def get_joint_dot(self):
        self.pre_joint_dot = self.curr_joint_dot
        curr_angle = self.get_joint_angle()
        curr_vel = np.zeros(2)
        for joint in range(2):
            curr_vel[joint] = (curr_angle[joint] - self.pre_joint_angle[joint]) / (self.timeunit * self.timestep)
            self.curr_joint_dot[joint] = curr_vel[joint] * 0.5 + self.pre_joint_dot[joint] * 0.5
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
            body_vel[i] = now_vel[i] * 0.5 + self.body_pre_vel[i] * 0.5
        self.body_pre_position = now_position
        self.body_pre_vel = body_vel
        if self.stateMachine in (COMPRESSION, THRUST):
            self.body_vel = body_vel

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
            if self.touch_state == 0.:
                self.stateMachine = FLIGHT

        elif self.stateMachine == FLIGHT:
            if self.touch_state == 1.:
                self.stateMachine = LOADING

    def update_state(self):
        self.system_ms += self.timeunit * self.timestep  # update sim time
        # print("time:", self.system_ms)
        # print("Ts:", self.Ts)
        self.touch_state = self.get_touch_state()        # update touch state
        self.update_last_Ts()                            # update stance time

        self.get_length_dot()  # update leg length
        self.curr_leg_length = self.get_leg_length()  # update leg length velocity

        self.get_joint_dot()                             # update joint velocity
        print("joint_dot", self.curr_joint_dot[0])
        self.curr_joint_angle = self.get_joint_angle()   # update joint angle

        self.get_imu_dot()                               # update imu velocity
        self.curr_imu_angle = self.get_IMU_Angle()       # update imu angle

        # update rotate matrix
        self.R_H_B = self.easyMat_RPY(self.curr_imu_angle.roll, self.curr_imu_angle.pitch, self.curr_imu_angle.yaw)
        self.R_B_H = self.R_H_B.transpose()

        self.get_body_vel()                              # update forward velocity
        self.updateRobotStateMachine()                   # update state phase

    def robot_control(self):
        self.update_state()

        kp = 2000
        fz = kp * (self.leg_length - self.curr_leg_length)
        # F_thrust = kp * (self.get_body_pos())
        # print("leg_length:", self.curr_leg_length)
        if self.stateMachine == THRUST:
            fz += 20
            # print("fz:", fz)
        Tz = fz * math.sqrt(0.6*0.6 - self.curr_leg_length*self.curr_leg_length/4)
        # self.set_motor_torque(0, 0)
        self.set_motor_torque(1, Tz)

        if self.stateMachine in [LOADING, UNLOADING]:
            print("-------------------- landing or unload ----------------------")
            self.set_motor_torque(0, 0)

        elif self.stateMachine in [COMPRESSION, THRUST]:
            print("-------------------- COMPRESSION,THRUST ----------------------")
            k_pose_p = 800
            k_pose_v = 40
            # print("pitch", self.curr_imu_angle.pitch)
            Tx = -k_pose_p * self.curr_imu_angle.pitch - k_pose_v * self.curr_imu_dot.pitch
            self.set_motor_torque(0, -Tx)

        elif self.stateMachine == FLIGHT:
            print("-------------------- flight ----------------------")
            self.r = 1.0
            print("r", self.r)
            k_xz_dot = 0.072
            k_leg_v = 3
            k_leg_p = 10

            body_vel_plot.append(self.body_vel[0])

            x_f = self.body_vel[0] * self.Ts / 2.0 + k_xz_dot * (self.body_vel[0] - 0.)
            x_f_plot.append(x_f)
            # print("r:", r)
            z_f = -math.sqrt(self.r * self.r - x_f * x_f)

            self.workPoint_H_desire[0] = x_f
            self.workPoint_H_desire[1] = 0.
            self.workPoint_H_desire[2] = z_f
            self.workPoint_B_desire = np.dot(self.R_B_H, self.workPoint_H_desire)
            # print("rhb:", self.R_B_H)

            x_f_B = self.workPoint_B_desire[0]
            y_f_B = self.workPoint_B_desire[1]
            z_f_B = self.workPoint_B_desire[2]
            x_angle_desire = math.atan(x_f_B / z_f_B)
            x_angle_des_plot.append(x_angle_desire)

            x_angle = self.curr_joint_angle[0] - math.acos(self.r / 1.2)
            x_angle_plot.append(x_angle)
            x_angle_dot = self.curr_joint_dot[0]

            Tx = -k_leg_p * (x_angle - x_angle_desire) - k_leg_v * x_angle_dot
            print("Tx", Tx)
            self.set_motor_torque(0, Tx)


if __name__ == '__main__':

    slip_robot = Robot_Control()
    while slip_robot.robot.step(slip_robot.timestep) != -1:
        slip_robot.robot_control()

    fig = plt.figure()
    # 绘制空间曲线
    plt.subplot(2, 2, 1)
    plt.plot(body_vel_plot)
    plt.title("t-body_v")
    plt.subplot(2, 2, 2)
    plt.plot(x_f_plot)
    plt.title("t-x_f")
    plt.subplot(2, 2, 3)
    plt.plot(x_angle_plot)
    plt.title("t-x_angle")
    plt.subplot(2, 2, 4)
    plt.plot(x_angle_des_plot)
    plt.title("t-x_angle_des")
    # 显示图形
    plt.show()