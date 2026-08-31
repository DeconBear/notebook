# Lesson 02 — Service request/response（笔记: docs/02-Service请求应答.md）
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsServer(Node):
    """服务端：收到 a、b，返回 sum。"""

    def __init__(self):
        super().__init__('add_two_ints_server')
        # 服务名 'add_two_ints'：客户端必须连同名服务
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback,
        )
        self.get_logger().info('服务已就绪: add_two_ints')

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(
            f'收到请求: a={request.a}, b={request.b} -> sum={response.sum}'
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AddTwoIntsServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
