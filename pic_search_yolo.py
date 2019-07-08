import requests
from tkinter import *
from PIL import Image, ImageTk
from tkinter import ttk
import threading
import shutil
import time
from pathlib import Path
from sys import platform
from models import *
from utils.datasets import *
from utils.utils import *
# cv2.imshow("image",img)
# cv2.waitKey()

class yolo:
    def __init__(self,path):
        with torch.no_grad():
            self.detect(
                'cfg/yolov3.cfg',
                'weights/yolov3.weights',
                path,
                img_size=32 * 13,
                conf_thres=0.5,
                nms_thres=0.45
            )

    def detect(
            self,
            cfg,
            weights,
            images,
            output='output',  # output folder
            img_size=416,
            conf_thres=0.3,
            nms_thres=0.45,
    ):
        device = torch_utils.select_device()
        if os.path.exists(output):
            shutil.rmtree(output)                                    # 清空重建文件夹
        os.makedirs(output)

        model = Darknet(cfg, img_size)                                                     # 初始化

        # 加载权重
        if weights.endswith('.pt'):  # pytorch模型
            if weights.endswith('yolov3.pt') and not os.path.exists(weights):
                if (platform == 'darwin') or (platform == 'linux'):
                    os.system('wget https://storage.googleapis.com/ultralytics/yolov3.pt -O ' + weights)
            model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
        else:  # weights模型
            load_darknet_weights(model, weights)

        model.to(device).eval()

        # 数据集
        dataloader = LoadImages(images, img_size=img_size)

                                                                        # class判断对象名，随机颜色
        classes = load_classes(parse_data_cfg('cfg/coco.data')['names'])
        colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(classes))]

        for i, (path, img, im0) in enumerate(dataloader):
            t = time.time()
            print('image %g/%g %s: ' % (i + 1, len(dataloader), path), end='')
            save_path = str(Path(output) / Path(path).name)

            # 开始检测
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            if ONNX_EXPORT:
                torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
                return
            pred = model(img)
            pred = pred[pred[:, :, 4] > conf_thres]  # remove boxes 小于阈值

            if len(pred) > 0:
                # 根据预测跑NMS
                detections = non_max_suppression(pred.unsqueeze(0), conf_thres, nms_thres)[0]

                # 将矩阵从416缩放到真实的图像大小
                scale_coords(img_size, detections[:, :4], im0.shape).round()

                # 输出结果
                unique_classes = detections[:, -1].cpu().unique()
                for c in unique_classes:
                    n = (detections[:, -1].cpu() == c).sum()
                    print('%g %ss' % (n, classes[int(c)]), end=', ')
                                                                                        # 画框和标签
                for x1, y1, x2, y2, conf, cls_conf, cls in detections:

                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=colors[int(cls)])

            dt = time.time() - t
            print('Done. (%.3fs)' % dt)
            self.out=im0
class sfs(ttk.Frame):
    i = 0
    thread = None
    thread_run = False
    Searchtxt = ''
    url = 'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word='

    def __init__(self, main):
        ttk.Frame.__init__(self, main)
        self.pho = ttk.Label(main)
        self.Txt=ttk.Entry(main)
        self.sdxx = ttk.Button(main, text="人工智障识别", command=self.yolo)
    #    self.wzu = ttk.Button(main, text="wzu识别", command=self.yolo)
        self.Next = ttk.Button(main, text=">>>", command=self.next)
        self.Prev = ttk.Button(main, text="<<<", command=self.prev)
        self.Search = ttk.Button(main, text="Search", command=self.search)
        self.labnow=ttk.Label(main,text='当前图片：')
        self.labload = ttk.Label(main,text='已加载：')

        main.geometry("700x800")
        main.title("人工智障图片搜索")
        self.Txt.place(x=200, y=10,w=300)
        self.Next.place(x=200, y=40)
        self.Prev.place(x=100, y=40)
        self.Search.place(x=505, y=8)
        self.pho.place(x=10, y=100)
        self.labnow.place(x=505, y=40)
        self.labload.place(x=600, y=8)
        self.sdxx.place(x=400, y=40)
    #    self.wzu.place(x=300, y=40)

    def fromweb(self):
        f = requests.get(self.url+self.Searchtxt).text
        print("搜索成功")
        self.pp = re.findall('"objURL":"(.*?)",', f, re.S)
        i = 0
        for e in self.pp:
            if i==1:
                self.i=0
                self.show()
            try:
                pic = requests.get(e, timeout=4)
            except :
                print('【错误】当前图片无法下载')
                continue
            with open('./images/' + str(i) + '.jpg', 'wb') as f:
                f.write(pic.content)
            try: ImageTk.PhotoImage(image=Image.open('./images/'+str(i)+'.jpg','r'))
            except: continue
            self.labload.config(text='已加载：'+str(i))
            i += 1
        print("读取成功")
        print(len(self.pp))
    def search(self):
        self.Searchtxt=self.Txt.get()
        self.add_thread=threading.Thread(target=self.fromweb,name='TT1')
        self.add_thread.setDaemon(True)
        self.add_thread.start()
    def getImg(self,img):
        # 太大图片的调小
        imgtk_change = ImageTk.PhotoImage(image=img)
        wide = imgtk_change.width()
        high = imgtk_change.height()
        if wide > 600 or high > 600:
            factor = min(600 / wide, 600 / high)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
        im = img.resize((wide, high), Image.ANTIALIAS)
        return im
    def getLocal(self):
        self.labnow.config(text='当前图片：' + str(self.i))
        images = Image.open('./images/'+str(self.i)+'.jpg','r')
        # 太大图片的调小
        imgtk_change = ImageTk.PhotoImage(image=images)
        wide = imgtk_change.width()
        high = imgtk_change.height()
        if wide > 600 or high > 600:
            factor = min(600 / wide, 600 / high)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
        im = images.resize((wide, high), Image.ANTIALIAS)
        return im
    def show(self):
        self.imgtk = ImageTk.PhotoImage(image=self.getLocal())
        self.pho.configure(imag=self.imgtk)
    def next(self):
        if (self.i < len(self.pp)-1):
            self.i += 1
        self.show()
    def prev(self):
        if(self.i>0):
            self.i -= 1
        self.show()
    def yolo(self):
        yl=yolo('./images/'+str(self.i)+'.jpg')
        roi = cv2.cvtColor(yl.out, cv2.COLOR_BGR2RGB)  # --------------cv转换
        images = Image.fromarray(roi)
        self.imgtk_roi = ImageTk.PhotoImage(image=self.getImg(images))
        self.pho.configure(image=self.imgtk_roi, state='enabled')  # --------------更新

        self.update_time = time.time()
        pass

main = Tk()
SFS = sfs(main)

main.mainloop()