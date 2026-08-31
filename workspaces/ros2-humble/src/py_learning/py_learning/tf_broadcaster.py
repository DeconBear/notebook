# Lesson 08 — TF2 static broadcaster（笔记: docs/08-TF2坐标变换.md）

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class SensorFrameBroadcaster(Node):
    """发布固定外参: base_link -> sensor_link。"""

    def __init__(self):
        super().__init__('sensor_frame_broadcaster')
        self.broadcaster = StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'sensor_link'
        # 传感器在车体前方 0.2m、上方 0.1m
        t.transform.translation.x = 0.2
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.1
        # 无旋转（单位四元数）
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.broadcaster.sendTransform(t)
        self.get_logger().info('已发布静态 TF: base_link -> sensor_link')


def main(args=None):
    rclpy.init(args=args)
    node = SensorFrameBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
