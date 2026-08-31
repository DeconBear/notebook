# Lesson 01 — Topic pub/sub（笔记: docs/01-Topic发布订阅.md）
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerNode(Node):
    """发布者：定时往话题上发消息。"""

    def __init__(self):
        # 节点名：在系统里叫 'talker'，可用 ros2 node list 看到
        super().__init__('talker')
        # 创建发布者：消息类型 String，话题名 'chatter'，队列长度 10
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.i = 0
        # 每 0.5 秒调用一次 timer_callback
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = f'你好 ROS 2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'发布: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = TalkerNode()
    try:
        # 进入循环：处理定时器、回调等
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
