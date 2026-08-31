# Lesson 03 — Action with feedback（笔记: docs/03-Action长任务.md）
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    """Action 服务端：按 order 生成斐波那契数列，过程中发 feedback。"""

    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )
        self.get_logger().info('Action 已就绪: fibonacci')

    def goal_callback(self, goal_request):
        self.get_logger().info(f'收到 goal: order={goal_request.order}')
        if goal_request.order < 0:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('收到取消请求')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        self.get_logger().info('开始执行...')
        order = goal_handle.request.order
        feedback_msg = Fibonacci.Feedback()
        sequence = []

        for i in range(order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('已取消')
                result = Fibonacci.Result()
                result.sequence = sequence
                return result

            if i < 2:
                sequence.append(i)
            else:
                sequence.append(sequence[i - 1] + sequence[i - 2])

            feedback_msg.sequence = sequence
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'feedback: {sequence}')
            time.sleep(0.5)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = sequence
        self.get_logger().info(f'完成: {sequence}')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = FibonacciActionServer()
    # Action 执行中还要处理 cancel/feedback，用多线程 executor 更稳妥
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
