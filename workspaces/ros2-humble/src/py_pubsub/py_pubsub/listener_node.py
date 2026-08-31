# Lesson 01 — Topic pub/sub（笔记: docs/01-Topic发布订阅.md）
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ListenerNode(Node):
    """订阅者：收到话题消息时打印。"""

    def __init__(self):
        super().__init__('listener')
        # 订阅同一话题 'chatter'，消息类型必须和发布者一致
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10,
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'收到: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = ListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
