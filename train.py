#!/usr/bin/env python

import argparse
from pathlib import Path

from os import environ as env
env['TF_CPP_MIN_LOG_LEVEL'] = '2'               # hide info & warnings
env['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'       # grow GPU memory as needed

import tensorflow as tf
import tensorflow_datasets as tfds
import nstesia as nst


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Dumoulin (2017) style transfer model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('style_image', nargs='+',
                        help='style image file(s)')

    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='content weight')
    parser.add_argument('--style_weight', type=float, default=1e-4,
                        help='style weight')
    parser.add_argument('--epochs', type=int, default=8,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')

    parser.add_argument('--data_dir', default='/tmp',
                        help='dataset directory - requires ~120gb')
    parser.add_argument('--saved_model', default='saved/model',
                        help='where to save the trained model.')

    return parser.parse_args()


def get_coco_ds(data_dir, batch_size):
    ds = tfds.load('coco/2014', split='train', data_dir=data_dir)
    ds = ds.map( lambda data: tf.cast(data['image'], dtype=tf.float32) )
    ds = ds.map( lambda image: tf.image.resize(image, [256,256]) )
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_model(
    style_image_files,
    content_weight, style_weight,
    image_ds, epochs,
):
    style_images = [ nst.io.load_image(file) for file in style_image_files ]

    style_ids = tf.data.Dataset.range(len(style_images)).repeat()
    train_ds = tf.data.Dataset.zip((image_ds,style_ids))

    model = nst.dumoulin_2017.StyleTransferModel(
        style_images,
        content_weight=content_weight,
        style_weight=style_weight,
    )
    model.compile(optimizer='adam')
    model.fit(train_ds, epochs=epochs)

    return model


if __name__ == '__main__':

    args = parse_args()

    image_ds = get_coco_ds(args.data_dir, args.batch_size)

    model = train_model(
        args.style_image,
        args.content_weight, args.style_weight,
        image_ds, args.epochs,
    )

    model.save(args.saved_model, save_traces=False)
