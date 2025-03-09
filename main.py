import torch
import cv2
import time
import os
import hashlib
import csv
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
import threading

class ObjectDetector:
    """优化后的目标检测器，提升哈希计算和生命周期管理效率"""

    def __init__(self, weights='best.pt', device='cpu'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, device=device)
        self.model.eval()
        self.label_intervals = defaultdict(list)
        self.current_activations = defaultdict(dict)
        self.TIME_THRESHOLD = 5
        self.EXCLUDE_LABELS = ['face']

    def _generate_session_id(self, bbox):
        """优化后的快速哈希生成"""
        bbox_str = f"{int(bbox['xmin'])}_{int(bbox['ymin'])}_{int(bbox['xmax'])}_{int(bbox['ymax'])}"
        return hashlib.md5(bbox_str.encode(), usedforsecurity=False).hexdigest()

    def update_lifecycles(self, detections):
        """优化后的生命周期管理"""
        current_labels = set()
        current_time = time.time()
        active_objects = defaultdict(dict)

        # 预处理当前帧对象
        for _, row in detections.iterrows():
            label = row['name']
            if label in self.EXCLUDE_LABELS:
                continue

            current_labels.add(label)
            bbox = {
                'xmin': row['xmin'], 'ymin': row['ymin'],
                'xmax': row['xmax'], 'ymax': row['ymax']
            }
            obj_id = self._generate_session_id(bbox)
            active_objects[label][obj_id] = (row, bbox)

        # 批量更新生命周期
        for label in current_labels:
            # 移除过期对象
            expired_ids = set(self.current_activations[label].keys()) - set(active_objects[label].keys())
            for obj_id in expired_ids:
                del self.current_activations[label][obj_id]

            # 更新现有或新增对象
            for obj_id, (row, bbox) in active_objects[label].items():
                if obj_id in self.current_activations[label]:
                    index = self.current_activations[label][obj_id]
                    self.label_intervals[label][index]['end'] = current_time
                    self.label_intervals[label][index]['last_detected'] = current_time
                else:
                    self.label_intervals[label].append({
                        'start': current_time,
                        'end': current_time,
                        'first_detected': current_time,
                        'last_detected': current_time
                    })
                    self.current_activations[label][obj_id] = len(self.label_intervals[label]) - 1

        return current_labels


class DetectionRecorder:
    """记录器，增加写入缓冲"""
    def __init__(self, output_csv='detection_results.csv'):
        self.output_csv = output_csv
        self.file = None
        self.writer = None
        self.lock = threading.Lock()
        self.memory_cache = []  # 内存缓存
        self.columns = ['标签', '置信度', 'xmin', 'ymin', 'xmax', 'ymax', '时间']

    def __enter__(self):
        """支持上下文管理"""
        self._init_writer()
        return self
        
    def __exit__(self, *args):
        self.close()

    def _init_writer(self):
        """延迟初始化写入器"""
        if not self.file:
            if not os.path.exists(self.output_csv):
                open(self.output_csv, 'w').close()
            self.file = open(self.output_csv, 'a', newline='', encoding='utf-8', buffering=1)
            self.writer = csv.writer(self.file)
            if os.stat(self.output_csv).st_size == 0:
                self.writer.writerow(self.columns)

    def reset_cache(self):
        """新增方法：重置内存缓存"""
        with self.lock:
            self.memory_cache = []

    def get_live_data(self):
        """新增方法：获取实时数据"""
        with self.lock:
            return pd.DataFrame(self.memory_cache, columns=self.columns)

    def record(self, detection_row):
        with self.lock:
            self._init_writer()
            self.writer.writerow(detection_row)
            self.memory_cache.append(detection_row)
            
    def close(self):
        if self.file and not self.file.closed:
            self.file.flush()
            os.fsync(self.file.fileno())
            self.file.close()


class DetectionAnalyzer:
    """数据分析与可视化处理器"""

    def __init__(self):
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    def load_data(self, csv_path):
        """加载检测结果数据"""
        try:
            return pd.read_csv(csv_path)
        except UnicodeDecodeError:
            return pd.read_csv(csv_path, encoding='gbk')

    def calculate_stats(self, df):
        """计算精确的统计指标"""
        stats = defaultdict(lambda: {
            'count': 0,
            'durations': [],
            'detailed_intervals': [],
            'total_duration': 0,
            'avg_duration': 0
        })

        for label in df['标签'].unique():
            if pd.isna(label) or label == 'face':
                continue

            # 获取按时间排序的时间戳
            timestamps = df[df['标签'] == label]['时间'].apply(
                lambda x: pd.to_datetime(x).timestamp()
            ).sort_values().values

            # 合并连续时间段
            intervals = []
            for ts in timestamps:
                if not intervals:
                    intervals.append([ts, ts])
                else:
                    last_start, last_end = intervals[-1]
                    if ts - last_end <= 5:  # 同一对象消失后5秒内重新出现视为持续存在
                        intervals[-1][1] = ts
                    else:  # 新周期
                        intervals.append([ts, ts])

            # 计算统计指标
            durations = [end - start for start, end in intervals]
            stats[label]['count'] = len(intervals)
            stats[label]['durations'] = durations
            stats[label]['detailed_intervals'] = intervals
            stats[label]['total_duration'] = sum(durations)
            stats[label]['avg_duration'] = sum(durations) / len(durations) if durations else 0

        return stats

    def create_dashboard(self, stats):
        """创建交互式可视化仪表盘"""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'xy'}, {'type': 'polar'}],
                   [{'type': 'domain'}, {'type': 'xy'}]],
            subplot_titles=('出现次数统计', '特征雷达图',
                            '持续时间占比', '持续时间分布'),
            vertical_spacing=0.15
        )

        labels = [str(k) for k in stats.keys()]  # 确保标签为字符串

        # 柱状图（出现次数）
        fig.add_trace(
            go.Bar(
                x=labels,
                y=[v['count'] for v in stats.values()],
                marker_color=self.color_palette,
                name='出现次数'
            ), row=1, col=1
        )

        # 雷达图（综合特征）
        radar_categories = ['出现次数', '总持续时间', '平均持续时间']
        for idx, (label, data) in enumerate(stats.items()):
            fig.add_trace(go.Scatterpolar(
                r=[data['count'], data['total_duration'], data['avg_duration']],
                theta=radar_categories,
                fill='toself',
                name=str(label),
                line_color=self.color_palette[idx]
            ), row=1, col=2)

        # 饼图（持续时间占比）
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=[v['total_duration'] for v in stats.values()],
                hole=0.4,
                marker_colors=self.color_palette
            ), row=2, col=1
        )

        # 箱线图（持续时间分布）
        for idx, (label, data) in enumerate(stats.items()):
            fig.add_trace(go.Box(
                y=data['durations'],
                name=str(label),
                marker_color=self.color_palette[idx],
                boxpoints='outliers'
            ), row=2, col=2)

        # 更新布局
        fig.update_layout(
            height=1000,
            width=1200,
            title_text='检测结果综合分析',
            hovermode='closest',
            template='plotly_white',
            margin=dict(t=100, b=50),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.show()

class DetectionPipeline:
    """优化后的流水线控制器，提升帧处理效率"""
    def process_frame(self, frame):
        """供外部调用的帧处理方法"""
        # 转换颜色空间并推理
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.model(rgb_frame).pandas().xyxy[0]
        
        # 记录结果
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        for _, row in detections.iterrows():
            self.recorder.record([
                row['name'], row['confidence'],
                int(row['xmin']), int(row['ymin']),
                int(row['xmax']), int(row['ymax']),
                timestamp
            ])
        
        return detections
    
    def __init__(self, weights='best.pt', output_csv='detection_results.csv'):
        self.detector = ObjectDetector(weights=weights)
        self.recorder = DetectionRecorder(output_csv)
        self.analyzer = DetectionAnalyzer()
        self.target_fps = 30  # 提升目标帧率至30FPS
        self.last_processed = 0
        self.frame_counter = 0

    def _annotate_frame(self, frame, detections):
        """优化绘制逻辑"""
        label_count = defaultdict(int)
        h, w = frame.shape[:2]

        # 动态调整绘制尺寸
        font_scale = max(w, h) / 1500
        thickness = max(1, int(w / 500))

        for _, row in detections.iterrows():
            label = row['name']
            label_count[label] += 1
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])

            # 绘制检测框
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness)
            cv2.putText(frame, f"{label} {row['confidence']:.2f}",
                        (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 255, 0), thickness)

        # 显示统计信息
        stats_text = " | ".join([f"{k}:{v}" for k, v in label_count.items()])
        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.8, (255, 0, 0), thickness)
        help_text = "Press 'Q' to quit | Model: YOLOv5 | Labels: phone, drink, smoke"
        cv2.putText(frame, help_text, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return frame

    def run(self, source=0):
        """优化后的主循环，提升帧处理效率"""
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区堆积

        # 预热模型
        _ = self.detector.model(torch.zeros((1, 3, 640, 640)).float())

        while cap.isOpened():
            # 精确控制帧率
            now = time.time()
            if now - self.last_processed < (1.0 / self.target_fps):
                continue
            self.last_processed = now

            # 读取并预处理帧
            success, frame = cap.read()
            if not success:
                break

            # 转换颜色空间并推理
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.model(rgb_frame).pandas().xyxy[0]
            active_labels = self.detector.update_lifecycles(detections)

            # 记录结果
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            for _, row in detections.iterrows():
                self.recorder.record([
                    row['name'], row['confidence'],
                    int(row['xmin']), int(row['ymin']),
                    int(row['xmax']), int(row['ymax']),
                    timestamp
                ])

            # 优化显示逻辑
            self.frame_counter += 1
            if self.frame_counter % 2 == 0:  # 降低显示刷新率以提升性能
                display_frame = self._annotate_frame(frame.copy(), detections)
                cv2.imshow('Detection', cv2.resize(display_frame, (800, 600)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.recorder.close()
        self._generate_report()

    # 保持原有报告生成逻辑不变
    def _generate_report(self):
        df = self.analyzer.load_data(self.recorder.output_csv)
        stats = self.analyzer.calculate_stats(df)
        print("\n" + "=" * 50 + "\n检测结果:\n" + "=" * 50)
        for label, data in stats.items():
            print(f"\n■ 类别: {label}\n  出现次数: {data['count']}\n  总持续时间: {data['total_duration']:.1f}秒")
        self.analyzer.create_dashboard(stats)


if __name__ == "__main__":
    pipeline = DetectionPipeline(
        weights=r'D:\CV\code\2.16\yolo\weights\best.pt',
        output_csv='detection_results.csv'
    )
    pipeline.run(source=0)