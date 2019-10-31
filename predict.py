from utils import *
from config import *
import numpy as np
import argparse as parser
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from moviepy.editor import VideoFileClip
from load_data import *
tf.enable_eager_execution()
smooth = 1.

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2,3))
    den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def pipeline_final(img, is_video):
    channel = 1 if is_video else 4
    size = img.shape
    img = cv2.resize(img, dsize=(224, 224))
    img = np.array([img])
    t = model.predict([img, img])
    output_image = reverse_one_hot(t)
    out_vis_image = colour_code_segmentation(output_image, label_values)
    a = cv2.cvtColor(np.uint8(out_vis_image[0]), channel)
    b = cv2.cvtColor(np.uint8(img[0]), channel)
    added_image = cv2.addWeighted(a, 1, b, 1, channel)
    
    added_image = cv2.resize(added_image, dsize=(size[1],size[0]))

    return added_image


def pipeline_video(img):
    return pipeline_final(img, True)

def pipeline_img(img):
    return pipeline_final(img, False)

def process(media_dir, save_dir, model_dir):
    global model, label_values

    model = load_model(model_dir, custom_objects={'preprocess_input': preprocess_input, 'tversky_loss':tversky_loss,'dice_coef':dice_coef,'dice_coef_loss':dice_coef_loss})
    label_values, _, _ = get_label_values()

    try:
        img = load_image(media_dir)
        output = os.path.join(save_dir, 'output_image.png')
        img = pipeline_img(img)
        cv2.imwrite(output, img)
    except Exception as ex:
        output = os.path.join(save_dir, 'output_video.mp4')
        clip1 = VideoFileClip(media_dir)
        white_clip = clip1.fl_image(pipeline_video)
        white_clip.write_videofile(output, audio=False)

if __name__ == '__main__':

    if __name__ == "__main__":
        args = parser.ArgumentParser(description='Model prediction arguments')

        args.add_argument('-media', '--media_dir', type=str,
                          help='Media Directorium for prediction (mp4,png)')

        args.add_argument('-save', '--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                          help='Save Directorium')

        args.add_argument('-model', '--model_dir', type=str, default=PRETRAINED_MODEL_DIR,
                          help='Model Directorium')

        parsed_arg = args.parse_args()

        crawler = process(media_dir=parsed_arg.media_dir,
                          save_dir=parsed_arg.save_dir,
                          model_dir = parsed_arg.model_dir
                          )
