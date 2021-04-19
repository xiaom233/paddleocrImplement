from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import cv2

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    m = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        if g >= b:
            h = ((g-b)/m)*60
        else:
            h = ((g-b)/m)*60 + 360
    elif mx == g:
        h = ((b-r)/m)*60 + 120
    elif mx == b:
        h = ((r-g)/m)*60 + 240
    if mx == 0:
        s = 0
    else:
        s = m/mx
    v = mx

    H = h / 2
    S = s * 255.0
    V = v * 255.0
    return H, S, V

def check(h,s,v):
    # 这是筛选黑白灰像素，效果不好
    #return (h>=0 and h<=180 and s>=0 and s<=255 and v >= 0 and v<= 46) or (h==0 and h<=180 and s>=0 and s<=43 and v >= 46 and v<= 220) or (h==0 and h<=180 and s>=0 and s<=30 and v >= 221 and v<= 255)
    # 这是筛选彩色像素，稍好
    return s>=43 and s<=255 and v >= 46 and v<= 255

# 这是输入的图像
img = Image.open("C:\\Users\\Hoven_Li\\Desktop\\服务外包\【A18】债券图表数据ocr检测与文本识别【同花顺】\\ocr测试集\检测框测试集\\img_50\\1691867488-061-001.png")

img_rgb = img.convert('RGB')

img_array = np.array(img)#把图像转成数组格式

shape = img_array.shape

print(shape[0])
print(shape[1])

dst = np.zeros((shape[0],shape[1],3))

for i in range(0,shape[0]):
    for j in range(0,shape[1]):
        value = img_array[i, j]
        hsv = rgb2hsv(value[0],value[1],value[2])
        if check(hsv[0],hsv[1],hsv[2]):
            dst[i,j] = (255,255,255)
        else:
            dst[i,j] = img_array[i,j]

#这是输出的图像
img2 = Image.fromarray(np.uint8(dst))
# img2.show(img2)
img2.save("test.png","png")
            
# 模型路径下必须含有model和params文件
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False)#det_model_dir='{your_det_model_dir}', rec_model_dir='{your_rec_model_dir}', rec_char_dict_path='{your_rec_char_dict_path}', cls_model_dir='{your_cls_model_dir}', use_angle_cls=True

img_path = 'C:\\Users\\Hoven_Li\\paddle\\test.png'
doc  = open('C:\\Users\\Hoven_Li\\paddle\\1691867488-061-001.txt','w',encoding = 'utf8')
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)
    for i in range(0,1):
        for j in range(0,4):
            for k in range (0,2):
                doc.write(str(int(line[i][j][k])))
                if(j==3 and k==1):
                    continue
                doc.write(",")
                #print("",file=doc,flush=False)
    doc.write("\n")
    
print(len(result))
doc.close()
# 显示结果
# from PIL import Image
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='D:/paddle_pp/PaddleOCR/doc/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('1691867488-061-001.jpg') #结果图片保存在代码同级文件夹中。
