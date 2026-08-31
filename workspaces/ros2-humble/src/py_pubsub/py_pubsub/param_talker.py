# Lesson 04 — Parameters（笔记: docs/04-Parameter参数.md）
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from std_msgs.msg import String


class ParamTalker(Node):
    """带参数的发布者：可用参数改前缀与发布周期。"""

    def __init__(self):
        super().__init__('param_talker')

        # 声明参数（名字、默认值）
        self.declare_parameter('message_prefix', '你好')
        self.declare_parameter('publish_period', 0.5)

        self.prefix = self.get_parameter('message_prefix').get_parameter_value().string_value
        self.period = self.get_parameter('publish_period').get_parameter_value().double_value

        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.i = 0
        self.timer = self.create_timer(self.period, self.timer_callback)

        # 运行中改参数时走这里
        self.add_on_set_parameters_callback(self.on_params_changed)
        self.get_logger().info(
            f'已启动: prefix="{self.prefix}", period={self.period}s'
        )

    def on_params_changed(self, params):
        for p in params:
            if p.name == 'message_prefix':
                self.prefix = p.value
                self.get_logger().info(f'参数更新: message_prefix="{self.prefix}"')
            elif p.name == 'publish_period':
                if p.value <= 0.0:
                    self.get_logger().warn('publish_period 必须 > 0，拒绝')
                    return SetParametersResult(successful=False)
                self.period = p.value
                self.timer.cancel()
                self.timer = self.create_timer(self.period, self.timer_callback)
                self.get_logger().info(f'参数更新: publish_period={self.period}s')
        return SetParametersResult(successful=True)

    def timer_callback(self):
        msg = String()
        msg.data = f'{self.prefix} ROS 2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'发布: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    node = ParamTalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
