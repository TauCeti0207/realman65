import cv2
import os
from pathlib import Path

def images_to_video(image_dir, output_video, fps=30, image_ext='.jpg'):
    """
    将图片合成为视频
    :param image_dir: 图片所在文件夹路径
    :param output_video: 输出视频路径
    :param fps: 视频帧率
    :param image_ext: 图片后缀（如.jpg、.png）
    """
    # 检查输入目录是否存在
    if not Path(image_dir).exists():
        print(f"图片目录不存在: {image_dir}")
        return
    
    # 创建输出目录（如果不存在）
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片路径并按名称排序
    image_paths = [
        str(file) for file in Path(image_dir).glob(f"*{image_ext}")
        if file.is_file()
    ]
    
    # 按文件名中的数字排序（处理0001.jpg、0002.jpg的顺序）
    def extract_number(filename):
        try:
            # 提取文件名中的数字部分（如0001.jpg -> 1）
            return int(Path(filename).stem)
        except ValueError:
            # 如果无法提取数字，使用文件名排序
            return Path(filename).stem
    
    image_paths.sort(key=extract_number)
    
    if not image_paths:
        print(f"在目录 {image_dir} 中未找到 {image_ext} 格式的图片！")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    # 读取第一张图片，获取尺寸
    first_image = cv2.imread(image_paths[0])
    if first_image is None:
        print(f"无法读取第一张图片: {image_paths[0]}")
        return
    height, width = first_image.shape[:2]
    print(f"图片尺寸: {width}x{height}")

    # 尝试多种编码器
    codecs_to_try = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'), 
        ('MJPG', '.avi'),
        ('X264', '.mp4')
    ]
    
    video_writer = None
    final_output = output_video
    
    for codec, ext in codecs_to_try:
        try:
            # 如果输出文件扩展名与编码器不匹配，修改扩展名
            if not final_output.endswith(ext):
                final_output = str(Path(output_video).with_suffix(ext))
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(
                final_output,
                fourcc,
                float(fps),  # 确保fps是浮点数
                (width, height)
            )
            
            if video_writer.isOpened():
                print(f"使用编码器: {codec}, 输出文件: {final_output}")
                break
            else:
                video_writer.release()
                video_writer = None
        except Exception as e:
            print(f"尝试编码器 {codec} 失败: {e}")
            if video_writer:
                video_writer.release()
                video_writer = None
    
    if video_writer is None or not video_writer.isOpened():
        print("所有编码器都无法创建视频写入对象！请检查OpenCV安装和编码器支持")
        return

    # 循环写入图片
    for idx, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        if image is None:
            print(f"跳过损坏的图片：{image_path}")
            continue
        
        # 确保图片尺寸一致
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        video_writer.write(image)
        print(f"已处理：{idx+1}/{len(image_paths)} 张图片", end='\r')

    # 释放资源
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"\n视频已保存到：{final_output}")

# 主程序
if __name__ == "__main__":
    # 配置参数（根据实际情况修改）
    IMAGE_DIR = r"/media/shui/Lexar/原始数据/dataset_12031652_temp/episode_0001/rgb"  # 图片文件夹路径
    OUTPUT_VIDEO = r"output.mp4"  # 输出视频路径
    FPS = 15  # 帧率

    # 检查输入路径
    print(f"输入目录: {IMAGE_DIR}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    
    # 调用函数
    images_to_video(IMAGE_DIR, OUTPUT_VIDEO, FPS)