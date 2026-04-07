# coding=utf-8
import tensorflow as tf
import math


def lr_cold_start(lr_tensor, globalstep, lrcs_init_lr, lrcs_step):
  """
  Linear  interpolation increase learning rate from "lrcs_init_lr" to "lr_tensor"
  when global step increase from 0 to "lrcs_step"
  :param lr_tensor:
  :param globalstep:
  :param lrcs_init_lr:
  :param lrcs_step:
  :return: Tensor represents new learning rate
  """
  lrcs_init_lr = tf.convert_to_tensor(lrcs_init_lr, dtype=tf.float32)
  lrcs_step = tf.convert_to_tensor(lrcs_step, dtype=tf.float32)
  tlr = lrcs_init_lr + (lr_tensor - lrcs_init_lr) * tf.cast(globalstep, tf.float32) / lrcs_step
  tlr = tf.minimum(lr_tensor, tf.maximum(lrcs_init_lr, tlr))
  return tlr


def lr_warm_restart(lr_max, globalstep, lr_min, periodic_step):
  """
  From Paper "SGDR: Stochastic Gradient Descent with Warm Restarts"
  :param lr_max:
  :param globalstep:
  :param lr_min:
  :param periodic_step:
  :return: Tensor represents new learning rate
  """
  lr_min = tf.convert_to_tensor(lr_min)
  periodic_step = tf.convert_to_tensor(periodic_step, dtype=tf.int32)
  PI = tf.convert_to_tensor(math.pi)
  tlr = (lr_max + lr_min + (lr_max - lr_min) * tf.cos(
    PI * tf.cast(tf.mod(globalstep, periodic_step),tf.float32) / tf.cast(periodic_step,tf.float32))) * 0.5
  return tlr
