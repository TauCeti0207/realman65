import os
import csv
import time
import threading
from bisect import bisect_left

# 可选：换成 cv2.imwrite；如果你已经在用 OpenCV，注释掉 PIL 相关，改用 cv2 即可
from PIL import Image
import numpy as np

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _np_to_pil(img):
    """把 numpy BGR/BGRA/RGB/GRAY 转成 PIL.Image 保存更通用。"""
    if isinstance(img, Image.Image):
        return img
    arr = np.asarray(img)
    if arr.ndim == 2:
        mode = "L"
        return Image.fromarray(arr, mode=mode)
    if arr.shape[2] == 3:
        # 暂不强制 BGR->RGB 转换，如你用的是 OpenCV(BGR)，可在此 arr[..., ::-1] 反转通道
        return Image.fromarray(arr[:, :, ::-1])  # 如果你的数组是 BGR，请保留这行
        # return Image.fromarray(arr)            # 如果你的数组本身就是 RGB，请改用这行
    if arr.shape[2] == 4:
        return Image.fromarray(arr[:, :, [2,1,0,3]])  # BGRA->RGBA
    return Image.fromarray(arr)

def _find_nearest_index(ts_list, t):
    """在递增时间戳列表 ts_list 中找到距离 t 最近的下标。"""
    pos = bisect_left(ts_list, t)
    if pos == 0:
        return 0
    if pos == len(ts_list):
        return len(ts_list) - 1
    before = pos - 1
    after = pos
    if abs(ts_list[after] - t) < abs(ts_list[before] - t):
        return after
    return before

def _save_job(pairs, out_dir, csv_name="pairs.csv", img_ext=".jpg"):
    """
    在独立线程中保存图片与 CSV。
    pairs: 列表[ (head_img, chest_img, t, idx) ]
    """
    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_frame", "chest_frame", "time"])
        for head_img, chest_img, t, idx in pairs:
            head_filename = f"head_{idx}{img_ext}"
            chest_filename = f"chest_{idx}{img_ext}"
            head_path = os.path.join(out_dir, head_filename)
            chest_path = os.path.join(out_dir, chest_filename)

            # 保存图片（PIL）
            _np_to_pil(head_img).save(head_path)
            _np_to_pil(chest_img).save(chest_path)

            # 写 CSV 一行
            writer.writerow([head_filename, chest_filename, t])

def _save_job_with_depth(pairs, out_dir, csv_name="pairs.csv"):
    """
    在独立线程中保存图片与 CSV。
    pairs: 列表[ (head_img, chest_img, t, idx) ]
    """
    _ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, csv_name)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head_rgb", "head_depth", "time"])
        for head_rgb, head_depth, t, idx, in pairs:
            head_rgb_filename = f"{idx:05d}.jpg"
            head_depth_filename = f"depth_{idx:05d}.png"
            head_rgb_path = os.path.join(out_dir, head_rgb_filename)
            head_depth_path = os.path.join(out_dir, head_depth_filename)

            # 保存图片（PIL）
            _np_to_pil(head_rgb[:,:,::-1]).save(head_rgb_path)
            Image.fromarray(head_depth).save(head_depth_path)

            # 写 CSV 一行
            writer.writerow([head_rgb_filename, head_depth_filename, t])

def persist_shortbuffer_pairs(
    head_frame_buffer, head_frame_timestep,
    chest_frame_buffer, chest_frame_timestep,
    out_dir="output",
    edge_margin=5,            # 去掉短 buffer 首尾的帧数
    max_time_diff=None,       # 可选：允许的最大时间差(秒)。例如 0.05；None 表示不做限制
    start_index=0,            # 保存文件起始序号，便于多批次不重名
    head_K=None,
):
    """
    以较短的 buffer 为主，丢弃其首尾 edge_margin 帧，
    再按时间戳在另一个 buffer 中找最近帧进行配对，启动线程保存并生成 CSV，
    最后清空四个 buffer。
    """
    # 选择短的一侧为 anchor
    if len(head_frame_buffer) <= len(chest_frame_buffer):
        anchor_name = "head"
        anchor_frames = head_frame_buffer
        anchor_ts = head_frame_timestep
        other_frames = chest_frame_buffer
        other_ts = chest_frame_timestep
    else:
        anchor_name = "chest"
        anchor_frames = chest_frame_buffer
        anchor_ts = chest_frame_timestep
        other_frames = head_frame_buffer
        other_ts = head_frame_timestep

    n_anchor = len(anchor_frames)
    if n_anchor == 0 or len(other_frames) == 0:
        # 没有可配对数据，直接清空并返回
        head_frame_buffer.clear()
        head_frame_timestep.clear()
        chest_frame_buffer.clear()
        chest_frame_timestep.clear()
        return None

    # 计算有效范围，防止首帧/尾帧找不到相近图片
    left = max(0, edge_margin)
    right = max(0, n_anchor - edge_margin)
    if right <= left:
        # edge_margin 太大，导致没有可用帧
        head_frame_buffer.clear()
        head_frame_timestep.clear()
        chest_frame_buffer.clear()
        chest_frame_timestep.clear()
        return None

    # 组装要保存的配对（拍个快照，避免清空后数据丢失）
    pairs = []
    # 预先复制时间戳（另一个 buffer 的）以便 bisect
    other_ts_list = list(other_ts)

    save_idx = start_index
    for i in range(left, right):
        t = anchor_ts[i]
        j = _find_nearest_index(other_ts_list, t)
        # 可选：过滤过大时间差
        if max_time_diff is not None and abs(other_ts_list[j] - t) > max_time_diff:
            continue

        if anchor_name == "head":
            head_img = anchor_frames[i]
            chest_img = other_frames[j]
        else:
            head_img = other_frames[j]
            chest_img = anchor_frames[i]

        # 为了避免原地复用导致后续修改影响到保存，可复制一份（视内存情况决定）
        if isinstance(head_img, np.ndarray):
            head_img = head_img.copy()
        if isinstance(chest_img, np.ndarray):
            chest_img = chest_img.copy()

        pairs.append((head_img, chest_img, t, save_idx))
        save_idx += 1

    # 启动保存线程
    th = threading.Thread(target=_save_job, args=(pairs, out_dir), daemon=False)
    th.start()
    print("saving")

    # 清空四个 buffer
    head_frame_buffer.clear()
    head_frame_timestep.clear()
    chest_frame_buffer.clear()
    chest_frame_timestep.clear()

    return th  # 如需可选地 join 等待：th.join()
