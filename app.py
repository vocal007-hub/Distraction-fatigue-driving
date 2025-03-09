from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import cv2
import time
import base64
import threading
import os
import traceback
from datetime import datetime
from main import DetectionPipeline, DetectionAnalyzer
import pathlib

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局资源管理
detection_lock = threading.Lock()
counts_lock = threading.Lock()  # 新增计数器锁
stop_event = threading.Event()
current_cap = None
global_counts = {'phone': 0, 'smoke': 0, 'drink': 0}  # 新增全局计数器

pipeline = DetectionPipeline(
    weights=r'D:\CV\code\2.16\yolo\runs\train\exp\weights\best.pt',
    output_csv='detection_results.csv'
)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_frame(frame, detections):
    """在帧上绘制检测结果"""
    display_frame = frame.copy()
    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])

        cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{label} {confidence:.2f}", 
                   (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
    return display_frame 

def generate_detection_frames(source):
    global current_cap, stop_event

    # 初始化日志（显示检测开始）
    current_time = datetime.now().strftime("%H:%M:%S")
    socketio.emit('log_message', {'message': f'{current_time} 开始视频检测'})

    with detection_lock:
        try:
            # 初始化视频捕获
            pipeline.recorder.reset_cache()
            stop_event.clear()
            current_cap = cv2.VideoCapture(source)
            current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲区延迟
            frame_interval = 1/25  # 25 FPS控制

            while not stop_event.is_set():
                start_time = time.time()
                
                # 帧读取与校验
                success, frame = current_cap.read()
                if not success:
                    if isinstance(source, int):  # 摄像头模式重试逻辑
                        time.sleep(0.1)
                        continue
                    break  # 文件模式正常退出

                # 目标检测处理
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections = pipeline.process_frame(rgb_frame)
                
                # 帧处理与编码
                display_frame = process_frame(frame, detections)
                _, buffer = cv2.imencode('.jpg', display_frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # 实时统计计算
                df = pipeline.recorder.get_live_data()
                stats = DetectionAnalyzer().calculate_stats(df)
                current_counts = {
                    'phone': stats.get('phone', {}).get('count', 0),
                    'smoke': stats.get('smoke', {}).get('count', 0),
                    'drink': stats.get('drink', {}).get('count', 0)
                }

                # 发送数据到前端
                socketio.emit('video_frame', {'frame': frame_base64})
                socketio.emit('detection_update', current_counts)

                # 新增的日志逻辑（检测到具体标签时触发）
                current_log_time = datetime.now().strftime("%H:%M:%S")
                for label in ['phone', 'smoke', 'drink']:
                    if current_counts[label] > 0:
                        socketio.emit('log_message', {
                            'message': f'{current_log_time} 检测到{label}'
                        })

                # 精确帧率控制
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            # 异常处理与错误日志
            traceback.print_exc()
            error_time = datetime.now().strftime("%H:%M:%S")
            socketio.emit('error', {
                'message': f'{error_time} 视频处理异常: {str(e)}'
            })
        finally:
            # 资源清理
            if current_cap and current_cap.isOpened():
                current_cap.release()
            current_cap = None
            cv2.destroyAllWindows()

@app.route('/upload', methods=['POST'])
def handle_upload():
    """统一文件上传处理接口"""
    if 'file' not in request.files:
        return jsonify({'error': '未选择文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '空文件名'}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    # 图片处理逻辑
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(save_path)
        if frame is None:
            return jsonify({'error': '无效的图片文件'}), 400
            
        # 执行目标检测
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = pipeline.detector.model(rgb_frame).pandas().xyxy[0]
        
        # 处理并发送帧
        display_frame = process_frame(frame, detections)
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # 统计检测结果（使用英文标签）
        counts = {
            'phone': len(detections[detections['name'] == 'phone']),
            'smoke': len(detections[detections['name'] == 'smoke']),
            'drink': len(detections[detections['name'] == 'drink'])
        }

        socketio.emit('image_result', {
            'frame': frame_base64,
            'counts': counts
        })
        
        # 添加图片检测完成日志
        log_time = datetime.now().strftime("%H:%M:%S")
        socketio.emit('log_message', 
            {'message': f'{log_time} 图片检测完成'})
        
        return jsonify({'status': 'success'})  # 正确位于if代码块内
    
    # 视频文件返回路径（保持在if代码块外）
    return jsonify({'type': 'video', 'file_path': save_path})

@socketio.on('stop_detection')
def handle_stop_detection():
    """停止检测事件处理"""
    global stop_event, current_cap
    stop_event.set()
    
    # 生成报告的条件判断
    report_needed = os.path.exists('detection_results.csv') and \
                   os.stat('detection_results.csv').st_size > 0
    
    with detection_lock:
        if current_cap and current_cap.isOpened():
            current_cap.release()
            current_cap = None
    socketio.emit('log_message', {'message': f'{datetime.now().strftime("%H:%M:%S")} 生成检测报告'})
    # 生成最终统计报告
    if report_needed:
        try:
            analyzer = DetectionAnalyzer()
            df = analyzer.load_data('detection_results.csv')
            stats = analyzer.calculate_stats(df)
            
            report = {
                'labels': list(stats.keys()),
                'counts': [v['count'] for v in stats.values()],
                'total_durations': [v['total_duration'] for v in stats.values()],
                'avg_durations': [v['avg_duration'] for v in stats.values()],
                'durations_list': [v['durations'] for v in stats.values()]
            }
            socketio.emit('final_report', report)
            
        except Exception as e:
            traceback.print_exc()
            socketio.emit('error', {'message': f'报告生成失败: {str(e)}'})

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_detection')
def handle_start_detection(data=None):
    global stop_event
    source = data.get('path', 0) if data else 0
    stop_event.set()
    time.sleep(0.5)
    threading.Thread(target=generate_detection_frames, args=(source,)).start()

if __name__ == '__main__':
    socketio.run(app, debug=True, use_reloader=False)