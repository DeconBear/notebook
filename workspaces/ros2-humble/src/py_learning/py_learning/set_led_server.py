# Lesson 07 — SetLed service server（笔记: docs/07-自定义服务.md）
import rclpy
from rclpy.node import Node
from lesson_interfaces.srv import SetLed


class SetLedServer(Node):
    def __init__(self):
        super().__init__('set_led_server')
        self.leds = {0: False, 1: False, 2: False}
        self.create_service(SetLed, 'set_led', self.callback)
        self.get_logger().info('服务已就绪: /set_led')

    def callback(self, request, response):
        if request.led_id not in self.leds:
            response.success = False
            response.message = f'未知 LED id={request.led_id}'
            self.get_logger().warn(response.message)
            return response

        self.leds[request.led_id] = request.turn_on
        state = 'ON' if request.turn_on else 'OFF'
        response.success = True
        response.message = f'LED{request.led_id} -> {state}'
        self.get_logger().info(f'{response.message} | 当前={self.leds}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SetLedServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
