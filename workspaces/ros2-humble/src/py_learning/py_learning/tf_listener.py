# Lesson 08 — TF2 listener（笔记: docs/08-TF2坐标变换.md）
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


class TfListenerDemo(Node):
    def __init__(self):
        super().__init__('tf_listener_demo')
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        try:
            t = self.buffer.lookup_transform(
                'base_link',
                'sensor_link',
                rclpy.time.Time(),
            )
            tr = t.transform.translation
            self.get_logger().info(
                f'sensor_link 相对 base_link: x={tr.x:.2f} y={tr.y:.2f} z={tr.z:.2f}'
            )
        except Exception as e:
            self.get_logger().warn(f'尚无 TF: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TfListenerDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
