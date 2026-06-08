import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


class ForwardKinematics(Node):
    joint_positions: np.ndarray
    leg_ids: list[str] = ["leg_front_l", "leg_back_l", "leg_front_r", "leg_back_r"]

    def __init__(self):
        super().__init__("forward_kinematics")
        self.joint_subscription = self.create_subscription(
            JointState, "joint_states", self.listener_callback, 10
        )
        self.joint_subscription  # prevent unused variable warning

        self.position_publishers: dict[str, Publisher] = {
            leg_id: self.create_publisher(
                Float64MultiArray, f"{leg_id}_end_effector_position", 10
            )
            for leg_id in self.leg_ids
        }

        self.joint_positions = None
        timer_period = 0.02  # publish FK information and marker at 50Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.kp_publisher = self.create_publisher(
            Float64MultiArray, "/forward_kp_controller/commands", 10
        )
        self.kd_publisher = self.create_publisher(
            Float64MultiArray, "/forward_kd_controller/commands", 10
        )

        # Periodically set gains to 0 so legs go limp
        self.create_timer(0.1, self.publish_zero_gains)

    def publish_zero_gains(self):
        self.kp_publisher.publish(Float64MultiArray(data=[0.0] * 12))
        self.kd_publisher.publish(Float64MultiArray(data=[0.0] * 12))

    def listener_callback(self, msg):
        positions: dict[str, list[float]] = {}
        for leg_id in self.leg_ids:
            positions[leg_id] = [
                msg.position[msg.name.index(f"{leg_id}_{j + 1}")] for j in range(3)
            ]

        self.joint_positions = positions

    def forward_kinematics(
        self, thetas: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """
        thetas: dict of leg_id to joint angles (_1, _2, _3) for the leg.
        """

        def rotation_x(angle):
            # rotation about the x-axis implemented for you
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, np.cos(angle), -np.sin(angle), 0],
                    [0, np.sin(angle), np.cos(angle), 0],
                    [0, 0, 0, 1],
                ]
            )

        def rotation_y(angle):
            return np.array(
                [
                    [np.cos(angle), 0, np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1],
                ]
            )

        def rotation_z(angle):
            return np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0, 0],
                    [np.sin(angle), np.cos(angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

        def translation(x, y, z):
            return np.array(
                [
                    [1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1],
                ]
            )

        # theta is positive when the Z-axis points out of the BOTTOM (i.e. uncovered metal parts) of the BLDC motor.
        # Since we keep the frame orientation the same on both left and right sides, motors 1 and 3 will use negative angles on the left and positive angles on the right.
        out: dict[str, list[float]] = {}

        front_l = thetas["leg_front_l"]
        T_0_1 = (
            translation(0.07500, 0.0445, 0)
            @ rotation_x(1.57080)
            @ rotation_z(-front_l[0])
        )
        T_1_2 = (
            translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(+front_l[1])
        )
        T_2_3 = (
            translation(0, -0.0494, 0.0685)
            @ rotation_y(+1.57080)
            @ rotation_z(-front_l[2])
        )
        T_3_ee = translation(0.06231, -0.06216, -0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        out["leg_front_l"] = T_0_ee[:3, 3].copy()

        back_l = thetas["leg_back_l"]
        T_0_1 = (
            translation(-0.07500, 0.0445, 0)
            @ rotation_x(1.57080)
            @ rotation_z(-back_l[0])
        )
        T_1_2 = (
            translation(0, 0, -0.039) @ rotation_y(-1.57080) @ rotation_z(+back_l[1])
        )
        T_2_3 = (
            translation(0, -0.0494, 0.0685)
            @ rotation_y(+1.57080)
            @ rotation_z(-back_l[2])
        )
        T_3_ee = translation(0.06231, -0.06216, -0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        out["leg_back_l"] = T_0_ee[:3, 3].copy()

        front_r = thetas["leg_front_r"]
        T_0_1 = (
            translation(+0.07500, -0.0335, 0)
            @ rotation_x(1.57080)
            @ rotation_z(+front_r[0])
        )
        T_1_2 = (
            translation(0, 0, +0.039) @ rotation_y(-1.57080) @ rotation_z(+front_r[1])
        )
        T_2_3 = (
            translation(0, -0.0494, 0.0685)
            @ rotation_y(+1.57080)
            @ rotation_z(+front_r[2])
        )
        T_3_ee = translation(0.06231, -0.06216, +0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        out["leg_front_r"] = T_0_ee[:3, 3].copy()

        back_r = thetas["leg_back_r"]
        T_0_1 = (
            translation(-0.07500, -0.0335, 0)
            @ rotation_x(1.57080)
            @ rotation_z(+back_r[0])
        )
        T_1_2 = (
            translation(0, 0, +0.039) @ rotation_y(-1.57080) @ rotation_z(+back_r[1])
        )
        T_2_3 = (
            translation(0, -0.0494, 0.0685)
            @ rotation_y(+1.57080)
            @ rotation_z(+back_r[2])
        )
        T_3_ee = translation(0.06231, -0.06216, +0.018)
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee
        out["leg_back_r"] = T_0_ee[:3, 3].copy()

        return out

    def timer_callback(self):
        """Timer callback for publishing end-effector position."""
        if self.joint_positions is not None:
            end_effector_positions = self.forward_kinematics(self.joint_positions)

            for leg_id, ee in end_effector_positions.items():
                publisher = self.position_publishers[leg_id]
                position = Float64MultiArray()
                position.data = ee
                publisher.publish(position)


def main(args=None):
    rclpy.init(args=args)

    forward_kinematics = ForwardKinematics()

    rclpy.spin(forward_kinematics)


if __name__ == "__main__":
    main()
