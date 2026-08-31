# Lesson 02 — Service request/response（笔记: docs/02-Service请求应答.md）
import sys

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class AddTwoIntsClient(Node):
    """客户端：发一次请求，打印结果后退出。"""

    def __init__(self):
        super().__init__('add_two_ints_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        # 等服务端上线（最多等一会儿）
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务 add_two_ints ...')

    def send_request(self, a, b):
        req = AddTwoInts.Request()
        req.a = a
        req.b = b
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 3:
        print('用法: ros2 run py_pubsub add_two_ints_client <a> <b>')
        print('示例: ros2 run py_pubsub add_two_ints_client 3 5')
        rclpy.shutdown()
        return

    a = int(sys.argv[1])
    b = int(sys.argv[2])

    node = AddTwoIntsClient()
    try:
        response = node.send_request(a, b)
        node.get_logger().info(f'结果: {a} + {b} = {response.sum}')
    except Exception as e:
        node.get_logger().error(f'调用失败: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
