import os
import progressbar
import pdb
import random
import cv2 

############################Config###########################################
# path to waymo dataset "folder" (all .tfrecord files in that folder will be converted)
DATA_PATH = '/home/trail/datasets/Waymo/waymo/shuffle_test'
# path to save kitti dataset
SHUFFLED_PATH = '/home/trail/datasets/Waymo/waymo/shuffle_test/shuffled'
# as name
IMAGE_FORMAT = 'png'
# do not change
LABEL_FOLDER = '/label_0'
LABEL_ALL_FOLDER = '/label_all'
IMAGE_FOLDER = '/image_0'
CALIB_FOLDER = '/calib'
LIDAR_FOLDER = '/velodyne'
###############################################################################

class Shuffler:
    def __init__(self):
    	self.read_folder()
    	self.create_folders()
    	self.shuffled_names = self.file_names.copy()
    	random.shuffle(self.shuffled_names)
    	self.generate_prep_files(self.file_names, self.shuffled_names, self.file_number)


    def shuffle(self):
    	self.shuffle_images()
    	self.shuffle_text(LABEL_FOLDER)
    	self.shuffle_text(CALIB_FOLDER)
    	self.shuffle_pc()

    def shuffle_images(self):
    	fp_pairing = open(SHUFFLED_PATH + '/shuffle.txt','r')
    	lines = fp_pairing.readlines()
    	print(lines)
    	for line in lines:
            prev, new = line.replace('\n','').split('/')
            img=cv2.imread(DATA_PATH + IMAGE_FOLDER + '/'+prev+'.png')
            cv2.imwrite(SHUFFLED_PATH + IMAGE_FOLDER +'/'+new+'.png', img)

    def shuffle_text(self, target_folder):
    	fp_pairing = open(SHUFFLED_PATH + '/shuffle.txt','r')
    	lines = fp_pairing.readlines()
    	print(lines)
    	for line in lines:
            prev, new = line.replace('\n','').split('/')
            fp_original=open(DATA_PATH + target_folder + '/'+prev+'.txt','r')
            calib = fp_original.read()
            fp_original.close()
            fp_shuffled = open(SHUFFLED_PATH + target_folder + '/'+new+'.txt','w+')
            fp_shuffled.write(calib)
            fp_shuffled.close()

    def shuffle_pc(self):
    	fp_pairing = open(SHUFFLED_PATH + '/shuffle.txt','r')
    	lines = fp_pairing.readlines()
    	print(lines)
    	for line in lines:
            prev, new = line.replace('\n','').split('/')
            fp_original=open(DATA_PATH + LIDAR_FOLDER + '/'+prev+'.bin','rb')
            calib = fp_original.read()
            fp_original.close()
            fp_shuffled = open(SHUFFLED_PATH + LIDAR_FOLDER + '/'+new+'.bin','wb')
            fp_shuffled.write(calib)
            fp_shuffled.close()

    def read_folder(self):
    	self.file_number = 0
    	self.file_names = []
    	path, dirs, files = next(os.walk(DATA_PATH+IMAGE_FOLDER))
    	self.file_number = len(files)
    	self.file_names = files 

    def create_folders(self):
        if not os.path.exists(SHUFFLED_PATH):
            os.mkdir(SHUFFLED_PATH)
        if not os.path.exists(SHUFFLED_PATH + IMAGE_FOLDER):
            os.mkdir(SHUFFLED_PATH + IMAGE_FOLDER)
        if not os.path.exists(SHUFFLED_PATH + LABEL_FOLDER):
            os.mkdir(SHUFFLED_PATH + LABEL_FOLDER)
        if not os.path.exists(SHUFFLED_PATH + CALIB_FOLDER):
            os.mkdir(SHUFFLED_PATH + CALIB_FOLDER)
        if not os.path.exists(SHUFFLED_PATH + LIDAR_FOLDER):
            os.mkdir(SHUFFLED_PATH + LIDAR_FOLDER)


    def generate_prep_files(self, files, shuffled, file_num):
        fp_pairing = open(SHUFFLED_PATH + '/shuffle.txt', 'w+')
        pairing = ''
        for i in range(0,file_num):
        	pairing += files[i].replace('.png','') + '/'+ shuffled[i].replace('.png','') + '\n'
        fp_pairing.write(pairing)
        fp_pairing.close()


if __name__ == '__main__':
    shuffler = Shuffler()
    shuffler.shuffle()

