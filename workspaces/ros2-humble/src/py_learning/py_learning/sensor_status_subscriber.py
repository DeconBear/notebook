# Lesson 06 — custom SensorStatus subscriber（笔记: docs/06-自定义消息.md）
import rclpy
from rclpy.node import Node
from lesson_interfaces.msg import SensorStatus


class SensorStatusSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_status_subscriber')
        self.create_subscription(SensorStatus, 'sensor_status', self.callback, 10)

    def callback(self, msg: SensorStatus):
        status = '正常' if msg.ok else '告警'
        self.get_logger().info(
            f'收到 {msg.device_id}: {msg.temperature_c:.1f}C / {msg.voltage_v:.2f}V [{status}]'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SensorStatusSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
