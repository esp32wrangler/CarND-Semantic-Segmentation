#!/usr/bin/env python3
import tensorflow as tf
import cv2
import numpy as np
import scipy
from moviepy.editor import VideoFileClip

GRAPH_FILE='./myfcnn.pb'
def load_graph(graph_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def frame_infer(clip, sess, input_tensor, output_tensor, keepprob_tensor, image_shape):
    def process_frame(image):
        print ("got image", image.shape)
        orig_size = image.shape[0:2]
        img = scipy.misc.imresize(image, image_shape)
        res = infer(sess, input_tensor, output_tensor, keepprob_tensor, image_shape, img)
        res = scipy.misc.imresize(res, orig_size)
        return res
    return clip.fl_image(process_frame)


def infer(sess, input_tensor, output_tensor, keepprob_tensor, image_shape, image):
    im_softmax = sess.run(
        [tf.nn.softmax(output_tensor)],
        {keepprob_tensor: 1.0, input_tensor: [image]})
    print (im_softmax[0][0].shape)
    # Splice out second column (road), reshape output back to image_shape
    im_softmax = im_softmax[0][0][:, :, 1]
    print("new shape", im_softmax.shape)
    # If road softmax > 0.5, prediction is road
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    print("seg shape", segmentation.shape)
    # Create mask based on segmentation to apply to original image
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)

    return street_im

def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    graph = load_graph(GRAPH_FILE)

    with tf.Session(graph=graph) as sess:

        input_tensor_name = 'image_input:0'
        output_tensor_name = "output_layer/BiasAdd:0"
        keep_prob_tensor_name = 'keep_prob:0'

        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        keepprob_tensor = graph.get_tensor_by_name(keep_prob_tensor_name)

        image = scipy.misc.imresize(scipy.misc.imread("test.png"), image_shape)


        outimg = infer(sess, input_tensor, output_tensor, keepprob_tensor, image_shape, image )

        scipy.misc.imsave("output.png", outimg)

        clip1 = VideoFileClip('./project_video.mp4').subclip(1, 1.5)
        projectClip = clip1.fx(frame_infer, sess, input_tensor, output_tensor, keepprob_tensor, (288,512))
        projectClip.write_videofile('./challenge_results.mp4', audio=False)


if __name__ == '__main__':
    run()
