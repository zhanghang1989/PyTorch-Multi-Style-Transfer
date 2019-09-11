#CyberFork Edited 2019-9-11 16:02:19
from main import evaluate
import argparse
import os

IMAGES_PATH = r'.\images\21styles' # 风格集地址
IMAGES_FORMAT = ['.png', '.jpg', '.JPG'] 
#获取N多风格素材
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]


contentName='apple.png'#想要被应用风格的原图
for styleName in image_names:
    args = argparse.Namespace(
        content_image='images/content/'+contentName,
        content_size=512, cuda=1, model='models/21styles.model', ngf=128,
        output_image='images/outputs/'+contentName+'-'+styleName, style_folder='images/21styles/',
        style_image='images/21styles/'+styleName, style_size=512, subcommand='eval',
        vgg_model_dir='models/')
    evaluate(args)
