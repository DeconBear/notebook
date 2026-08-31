# Lesson 09 — cmd_vel publisher（笔记: docs/09-速度控制cmd_vel.md）
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelPublisher(Node):
    """周期性发布速度指令（可先用 topic echo 观察，仿真课再接车）。"""

    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.declare_parameter('linear_x', 0.2)
        self.declare_parameter('angular_z', 0.3)
        self.declare_parameter('period', 0.1)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        period = self.get_parameter('period').value
        self.create_timer(period, self.timer_callback)
        self.get_logger().info('发布 /cmd_vel （geometry_msgs/Twist）')

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = float(self.get_parameter('linear_x').value)
        msg.angular.z = float(self.get_parameter('angular_z').value)
        self.pub.publish(msg)
        self.get_logger().info(
            f'cmd_vel: linear.x={msg.linear.x:.2f} angular.z={msg.angular.z:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
