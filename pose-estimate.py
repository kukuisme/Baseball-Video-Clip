import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt

@torch.no_grad()  # 禁用梯度計算，加快推理速度
def run(poseweights="yolov7-w6-pose.pt",source="videoplayback.mp4",device='cpu',view_img=False,
        save_conf=False,line_thickness=3,hide_labels=False,hide_conf=True):

    frame_count = 0  # 計算幀數
    total_fps = 0  # 計算總FPS
    time_list = []  # 儲存處理時間的列表
    fps_list = []  # 儲存FPS的列表
    
    device = select_device(opt.device)  # 選擇設備 (CPU 或 GPU)
    half = device.type != 'cpu'  # 檢查是否使用 GPU

    model = attempt_load(poseweights, map_location=device)  # 加載模型
    _ = model.eval()  # 設置模型為推理模式
    names = model.module.names if hasattr(model, 'module') else model.names  # 獲取類別名稱

    if source.isnumeric():    
        cap = cv2.VideoCapture(int(source))  # 讀取攝像頭輸入
    else:
        cap = cv2.VideoCapture(source)  # 讀取影片檔案
   
    if not cap.isOpened():  # 檢查是否成功打開影片
        print('無法讀取影片，請檢查路徑')
        raise SystemExit()

    else:
        frame_width = int(cap.get(3))  # 獲取影片的寬度
        frame_height = int(cap.get(4))  # 獲取影片的高度

        # 初始化影片寫入器
        vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0] 
        resize_height, resize_width = vid_write_image.shape[:2]
        out_video_name = f"{source.split('/')[-1].split('.')[0]}"
        out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                            cv2.VideoWriter_fourcc(*'mp4v'), 30,
                            (resize_width, resize_height))

        while cap.isOpened():  # 迴圈處理影片直到結束
        
            print("正在處理第 {} 幀".format(frame_count + 1))

            ret, frame = cap.read()  # 獲取當前幀
            
            if ret:  # 如果成功讀取幀
                orig_image = frame  # 儲存原始幀
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image_ = image.copy()
                image = transforms.ToTensor()(image)  # 轉換為張量
                image = torch.tensor(np.array([image.numpy()]))
            
                image = image.to(device)  # 將圖片資料轉移到指定設備
                image = image.float()  # 轉換為浮點數精度
                start_time = time.time()  # 計算處理時間的起點
            
                with torch.no_grad():  # 推理過程
                    output_data, _ = model(image)

                output_data = non_max_suppression_kpt(output_data,  # 非最大值抑制
                                            0.25,  # 置信度閾值
                                            0.65,  # IoU 閾值
                                            nc=model.yaml['nc'],  # 類別數量
                                            nkpt=model.yaml['nkpt'],  # 關鍵點數量
                                            kpt_label=True)
            
                output = output_to_keypoint(output_data)  # 轉換為關鍵點格式

                im0 = image[0].permute(1, 2, 0) * 255  # 調整圖片格式為 [h, w, c]
                im0 = im0.cpu().numpy().astype(np.uint8)
                
                im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # 將格式轉回 BGR
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 正規化比例

                for i, pose in enumerate(output_data):  # 逐幀處理偵測結果
                    
                    if len(output_data):  # 如果有偵測結果
                        for c in pose[:, 5].unique():  # 列印結果
                            n = (pose[:, 5] == c).sum()  # 每類別的物件數量
                            print("當前幀的物件數量 : {}".format(n))
                        
                        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # 繪製邊界框與關鍵點
                            c = int(cls)  # 類別索引
                            kpts = pose[det_index, 6:]
                            label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                        line_thickness=opt.line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                        orig_shape=im0.shape[:2])

                end_time = time.time()  # 計算處理時間的結束點
                fps = 1 / (end_time - start_time)  # 計算 FPS
                total_fps += fps
                frame_count += 1
                
                fps_list.append(total_fps)  # 將 FPS 加入列表
                time_list.append(end_time - start_time)  # 將處理時間加入列表
                
                # 顯示結果
                if view_img:
                    cv2.imshow("YOLOv7 姿勢估計範例", im0)
                    cv2.waitKey(1)  # 等待 1 毫秒

                out.write(im0)  # 寫入處理後的影片幀

            else:
                break

        cap.release()
        # cv2.destroyAllWindows()
        avg_fps = total_fps / frame_count  # 計算平均 FPS
        print(f"平均 FPS: {avg_fps:.3f}")
        
        # 繪製 FPS 與處理時間的比較圖
        plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)

# 解析命令列參數
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='模型權重路徑')
    parser.add_argument('--source', type=str, default='videoplayback.mp4', help='影片路徑或 0（攝像頭）')
    parser.add_argument('--device', type=str, default='cpu', help='設備（cpu 或 gpu，如 0,1,2,3）')
    parser.add_argument('--view-img', action='store_true', help='顯示結果')
    parser.add_argument('--save-conf', action='store_true', help='儲存置信度到 --save-txt 標籤')
    parser.add_argument('--line-thickness', default=3, type=int, help='邊界框線條寬度（像素）')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隱藏標籤')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隱藏置信度')
    opt = parser.parse_args()
    return opt

# 繪製 FPS 與處理時間的比較圖
def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('時間 (秒)')
    plt.ylabel('FPS')
    plt.title('FPS 與時間比較圖')
    plt.plot(time_list, fps_list, 'b', label="FPS 與時間")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

# 主函式
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)  # 清理模型優化器以減少模型大小
    main(opt)
