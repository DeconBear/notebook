# Lesson 07 — SetLed service client（笔记: docs/07-自定义服务.md）
import sys

import rclpy
from rclpy.node import Node
from lesson_interfaces.srv import SetLed


class SetLedClient(Node):
    def __init__(self):
        super().__init__('set_led_client')
        self.cli = self.create_client(SetLed, 'set_led')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /set_led ...')

    def send(self, led_id: int, on: bool):
        req = SetLed.Request()
        req.led_id = led_id
        req.turn_on = on
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) != 3:
        print('用法: ros2 run py_learning set_led_client <led_id> <on|off>')
        print('示例: ros2 run py_learning set_led_client 1 on')
        rclpy.shutdown()
        return

    led_id = int(sys.argv[1])
    on = sys.argv[2].lower() in ('1', 'on', 'true', 'yes')
    node = SetLedClient()
    try:
        result = node.send(led_id, on)
        node.get_logger().info(f'success={result.success} message={result.message}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
