import os
from paddleocr import PaddleOCR, draw_ocr
import paddlehub as hub
import logging
# 模型路径下必须含有model和params文件
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
ocr = PaddleOCR(use_angle_cls=True,use_gpu=False)
# #det_model_dir='{your_det_model_dir}', rec_model_dir='{your_rec_model_dir}', rec_char_dict_path='{your_rec_char_dict_path}', cls_model_dir='{your_cls_model_dir}', use_angle_cls=True
# ocr = hub.Module(name='chinese_ocr_db_crnn_mobile')
doc = open('C:\\Users\\Hoven_Li\\paddle\\image_info_paddle_log.txt','w',encoding='utf8')
file_dir = 'C:\\Users\\Hoven_Li\\paddle\\images_2k\\images_2'
for root, dirs, files in os.walk(file_dir):
    root = root.replace('\\','/')
    for file in files:
        doc.write(file+"\t")
        result = ocr.ocr(root+'/'+file,det=False)
        # logging.log(result[0])
        print(file+'\t'+result[0][0])
        doc.write(result[0][0]+'\n')
        print('置信度： '+str(result[0][1]))
        logging.info(msg=result[0][0])
        # print(result[0]['data'][0]['text'])
        # doc.write(result[0][0])
        # doc.write("\n")
        
    break
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
