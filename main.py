import rosbag2_py
import pandas as pd
from pathlib import Path
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
import logging

# kinematics
from tf_transformations import euler_from_quaternion

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# パーサーを外付けするための関数の基本ストラクチャー
class MessageParser:
    def parse(self, msg, topic_name: str, result: dict):
        """
        メッセージを解析してデータに追加する基本関数
        実際の構造に対応してサブクラスで実装する
        """
        raise NotImplementedError("Subclasses should implement this method")

# PerceptionObjects用パーサー
class PerceptionObjectsParser(MessageParser):
    def parse(self, msg, topic_name: str, result: dict):
        header_time_stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        header_frame = msg.header.frame_id
        result.setdefault(topic_name, []) # prepare empty list if not exists
        for obj in msg.objects:
            base_data = {
                "timestamp": header_time_stamp,
                "frame_id": header_frame,
            }
            object_data = self.parse_objects(obj)
            result[topic_name].append({**base_data, **object_data})
    
    def parse_objects(self, object):
        kinematics = self.parse_kinematics(object.kinematics)
        classification = self.parse_classification(object.classification)
        shape = self.parse_shape(object.shape)
        other = {
            "object_id": self.parse_object_id(object.object_id) if hasattr(object, "object_id") else None,
            "existence_probability": object.existence_probability if hasattr(object, "existence_probability") else None,            
        }
        return {**kinematics, **classification, **shape, **other}
    
    def parse_object_id(self, id):
        # id is UUID message type, so convert it into string
        # list of int into string
        return ''.join(f'{byte:02x}' for byte in id.uuid)

    def parse_shape(self, msg):
        return {
            "shape_type": msg.type,
            "length": msg.dimensions.x,
            "width": msg.dimensions.y,
            "height": msg.dimensions.z,
            "footprints": msg.footprint,
        }

    def parse_kinematics(self, msg):
        msg_posewithcovariance = msg.pose_with_covariance if hasattr(msg, "pose_with_covariance") else msg.initial_pose_with_covariance
        msg_twistwithcovariance = msg.twist_with_covariance if hasattr(msg, "twist_with_covariance") else msg.initial_twist_with_covariance
        quaternion = (msg_posewithcovariance.pose.orientation.x, msg_posewithcovariance.pose.orientation.y,
                        msg_posewithcovariance.pose.orientation.z, msg_posewithcovariance.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]

        return {
            "pose_x": msg_posewithcovariance.pose.position.x,
            "pose_y": msg_posewithcovariance.pose.position.y,
            "pose_z": msg_posewithcovariance.pose.position.z,
            "yaw": yaw,
            "velocity_x": msg_twistwithcovariance.twist.linear.x,
            "velocity_y": msg_twistwithcovariance.twist.linear.y,
            # "velocity_z": msg_twistwithcovariance.twist.linear.z,
            "angular_vz": msg_twistwithcovariance.twist.angular.z,
            "orientation_availability": msg.orientation_availability if hasattr(msg, "orientation_availability") else None,
        }
    
    def parse_classification(self, msg):
        label_map = {
            0: "UNKNOWN",
            1: "CAR",
            2: "TRUCK",
            3: "BUS",
            4: "TRAILER",
            5: "MOTORCYCLE",
            6: "BICYCLE",
            7: "PEDESTRIAN"
        }
        # get the highest probability class
        highest_cls = None
        for cls in msg:
            if highest_cls is None or cls.probability > highest_cls.probability:
                highest_cls = cls
        return {
            "label": label_map[highest_cls.label] if highest_cls is not None else "UNKNOWN",
            "probability": highest_cls.probability if highest_cls is not None else 0.0
        }
        

# バッグファイルを開く関数
def open_reader(rosbag_uri: str, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=rosbag_uri, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    return reader

# 次のメッセージを読む関数
def read_next_msg(topic_name, data, type_map):
    try:
        msg_type = get_message(type_map[topic_name])
    except ModuleNotFoundError:
        logger.warning(f"Message type for topic {topic_name} not found.")
        return None

    try:
        msg = deserialize_message(data, msg_type)
    except Exception as e:
        logger.warning(f"Failed to deserialize message for topic {topic_name}: {e}")
        return None

    return msg

# バッグファイルからデータを抽出する関数
def extract_rosbag(rosbag_file: Path, target_topic: str, parser: MessageParser):
    reader = open_reader(str(rosbag_file))

    # トピックと型のマッピングを取得
    topics_and_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topics_and_types}

    # フィルタを適用
    storage_filter = rosbag2_py.StorageFilter(topics=[target_topic])
    reader.set_filter(storage_filter)

    result = {}

    while reader.has_next():
        topic_name, data, _ = reader.read_next()

        msg = read_next_msg(topic_name, data, type_map)
        if msg is None:
            continue

        if topic_name == target_topic:
            parser.parse(msg, topic_name, result)

    return result

# DataFrameに変換する関数
def make_data_frame(rosbag_file: Path, target_topic: str, parser: MessageParser):
    data = extract_rosbag(rosbag_file, target_topic, parser)

    if target_topic in data:
        df = pd.DataFrame(data[target_topic])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    else:
        logger.warning(f"No data found for topic {target_topic}")
        return pd.DataFrame()

# メイン処理
def main():
    rosbag_path = "~/Downloads/temp/tracking_eval/result_bag_0.db3" 
    rosbag_path = Path(rosbag_path).expanduser()
    target_topic = "/perception/object_recognition/objects"  # 抽出対象のトピック名

    logger.info(f"Processing ROS bag: {rosbag_path}, topic: {target_topic}")

    parser = PerceptionObjectsParser()
    df = make_data_frame(Path(rosbag_path), target_topic, parser)

    if not df.empty:
        output_path = "output.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
    else:
        logger.info("No data extracted.")

if __name__ == "__main__":
    main()
