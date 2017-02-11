from r3_3 import R3
import  tensorflow as tf

full_height, full_width = 1080, 1920
num_blocks, num_objects = 100, 20

r3 = R3(full_height, full_width, num_objects, num_blocks)
train_step = r3.train_step