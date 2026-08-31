# Lesson 03 — Action with feedback（笔记: docs/03-Action长任务.md）
import sys

import rclpy
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionClient(Node):
    """Action 客户端：发 goal，打印 feedback，最后打印 result。"""

    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(self, Fibonacci, 'fibonacci')

    def send_goal(self, order):
        self.get_logger().info('等待 Action 服务端...')
        self._action_client.wait_for_server()

        goal_msg = Fibonacci.Goal()
        goal_msg.order = order
        self.get_logger().info(f'发送 goal: order={order}')

        send_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if not goal_handle.accepted:
            self.get_logger().error('goal 被拒绝')
            return None

        self.get_logger().info('goal 已接受，等待结果...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        return result_future.result()

    def feedback_callback(self, feedback_msg):
        seq = list(feedback_msg.feedback.sequence)
        self.get_logger().info(f'进度 feedback: {seq}')


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) != 2:
        print('用法: ros2 run py_pubsub fibonacci_action_client <order>')
        print('示例: ros2 run py_pubsub fibonacci_action_client 8')
        rclpy.shutdown()
        return

    order = int(sys.argv[1])
    node = FibonacciActionClient()
    try:
        wrapped = node.send_goal(order)
        if wrapped is None:
            return
        status = wrapped.status
        result = wrapped.result
        if status == GoalStatus.STATUS_SUCCEEDED:
            node.get_logger().info(f'最终结果: {list(result.sequence)}')
        else:
            node.get_logger().warn(f'未成功结束, status={status}')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
