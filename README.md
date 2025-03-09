# 1.文件结构说明：
程序文件实现了驾驶员分心检测的一web程序，app.py是启动程序，mian.py是外部文件包含加载yolo检查函数和根据检测保存的结果进行可视化展示（图表等形式），detection_results.csv是保存检测结果的外部文件，index.html则是浏览器加载的前端页面。
**启动流程：**
(1).**对于打开摄像头检测**，打开摄像头加载mian.py中的检测函数进行实时检测，此时浏览器页面显示带检测结果（检测框：标签、置信度）的实时视频，并基于保存的检测结果实时更新浏览器页面显示的标签数量，注意标签数量(不是按帧处理，是检测总数于后面生成的检测报告可视化结果中的标签总数一致，同时也和main.py终端输出的检测出的标签数量一致),当点击停止检测按钮，调用main.py的可视化函数以四种动态图显示保存的结果。

(2).上传文件分为照片和视频。照片是显示带带检测结果的图片以及实时更新浏览器页面显示的标签数量。

 对于上传视频文件与打开摄像头进行检测处理逻辑一致，即打开摄像头加载mian.py中的检测函数进行实时检测。

2.效果展示：

![image](https://github.com/user-attachments/assets/100caa5a-a06a-4b57-b153-f47161383f4f)
![image](https://github.com/user-attachments/assets/62ffaf6a-f238-4ee0-8d67-5a1e2aead4f1)
