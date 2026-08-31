# Lesson 06 — custom SensorStatus publisher（笔记: docs/06-自定义消息.md）
import math

import rclpy
from rclpy.node import Node
from lesson_interfaces.msg import SensorStatus


class SensorStatusPublisher(Node):
    def __init__(self):
        super().__init__('sensor_status_publisher')
        self.declare_parameter('device_id', 'board_a')
        self.device_id = self.get_parameter('device_id').value
        self.pub = self.create_publisher(SensorStatus, 'sensor_status', 10)
        self.t = 0.0
        self.create_timer(0.5, self.timer_callback)
        self.get_logger().info('发布自定义消息: /sensor_status')

    def timer_callback(self):
        msg = SensorStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'sensor_link'
        msg.device_id = self.device_id
        msg.temperature_c = 25.0 + 3.0 * math.sin(self.t)
        msg.voltage_v = 12.0 + 0.2 * math.sin(self.t * 0.7)
        msg.ok = msg.temperature_c < 40.0
        self.pub.publish(msg)
        self.get_logger().info(
            f'[{msg.device_id}] T={msg.temperature_c:.1f}C V={msg.voltage_v:.2f}V ok={msg.ok}'
        )
        self.t += 0.5


def main(args=None):
    rclpy.init(args=args)
    node = SensorStatusPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
