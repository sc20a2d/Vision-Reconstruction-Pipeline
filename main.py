import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import numpy as np
from tensorflow import keras

import skimage
import skimage.measure
import skimage.metrics
from sklearn import preprocessing
import cv2
import tensorflow as tf

import glob
import pickle
# from sklearn.decomposition import PCA, IncrementalPCA
from PIL import Image
from sklearn.model_selection import train_test_split
from keras import callbacks


# Change to a random integer
random_seed = 12345

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# Resize to resolution
resolution = 64


SID_import = __import__("Spike-Image-Decoder.SID")
SID = getattr(SID_import, "SID")


#@title Utility functions for data loading
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di


def get_fmri(fmri_dir, ROI) -> np.ndarray:
    """This function loads fMRI data into a numpy array for to a given ROI.
    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.
    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """


    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis = 1)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train

    subj = format(subject_number, '02')
    algo_dir = 'algonauts_2023_tutorial_data'
    data_dir = os.path.join(algo_dir, 'subj'+subj)
    fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    train_img_dir = os.path.join(data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(data_dir, 'test_split', 'test_images')


if __name__ == '__main__':
    print("Imports Successful.")

    # Load and grayscale images
    image_array = []
    for i in range(image_count):
        temp_img = cv2.imread(image_list[i])
        gray_frame = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
        image_array.append(cv2.resize(gray_frame, (resolution, resolution)))

    lh_fmri = lh_fmri[:image_count]
    rh_fmri = rh_fmri[:image_count]

    temp_combined_fmri = np.concatenate((lh_fmri, rh_fmri), axis=1)
    combined_fmri = []
    for i in range(image_count):
        fmri_for_image = temp_combined_fmri[i]
        mini_fmri_for_image = fmri_for_image[::5]
        combined_fmri.append(mini_fmri_for_image)

    video_dir = 'AlgonautsVideos268_All_30fpsmax'
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()

    frames_data = get_frames_from_videos(video_list)
    X_train = frames_data
    X_test = frames_data

    # X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    # X_test.reshape([X_test.shape[0], resolution, resolution, 1])

    print(X_train.shape)

    # tf.convert_to_tensor(X_train, dtype=tf.float32)
    # tf.convert_to_tensor(X_test, dtype=tf.float32)
    # tf.convert_to_tensor(Y_test, dtype=tf.float32)
    # tf.convert_to_tensor(Y_train, dtype=tf.float32)


    # X_train = np.asarray(X_train).astype(np.float32)
    # X_test = np.asarray(X_test).astype(np.float32)
    # Y_train = np.asarray(Y_train).astype(np.float32)
    # Y_test = np.asarray(Y_test).astype(np.float32)


    input_x = Input(shape = (ncell,))

    model_dense = SID.dense_decoder(ncell)
    model_ae = SID.AE((resolution, resolution, 1))

    dense_out = model_dense(input_x)
    ae_out = model_ae(dense_out)

    optimizer = keras.optimizers.Adam()


    end2end_model = Model(input_x, ae_out)
    end2end_model.summary()
    end2end_model.compile(loss = 'mse', optimizer = optimizer)

    weight_dir = 'e2e_model'
    result_dir = 'e2e_result'

    multiout_model = Model(input_x, [dense_out, ae_out])

    for i in range(10):
        end2end_model.fit(Y_train, X_train, batch_size = 10, epochs = 30, validation_data = (Y_test, X_test) )
         
        pred_dense, pred_ae = multiout_model.predict(Y_test)
        mse, psnr, ssim = SID.cal_performance(X_test, pred_ae)
        print('%dcell AE:\n\tmse:%f psnr:%f ssim:%f'%(ncell, mse, psnr, ssim))

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    end2end_model.save_weights(os.path.join(weight_dir, 'end2end_digit69_spk.h5'))


    # if not os.path.exists(result_dir):
    #    os.mkdir(result_dir)
    np.save(os.path.join(result_dir, 'end2end_digit69_spk.npy'), pred_ae)
    pred_dense_trn, pred_ae_trn = multiout_model.predict(Y_train)
    np.save(os.path.join(result_dir, 'end2end_digit69_spk_trn.npy'), pred_ae_trn)
    


    # # visualization the reconstructed images
    # X_reconstructed_mu = pred_ae
    # n = 10
    # for j in range(1):
    #     plt.figure(figsize=(12, 2))    
    #     for i in range(n):
    #         # display original images
    #         ax = plt.subplot(2, n, i +j*n*2 + 1)
    #         plt.imshow(np.rot90(np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    #         # display reconstructed images
    #         ax = plt.subplot(2, n, i + n + j*n*2 + 1)
    #         #plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)

    #     plt.show()
    #     plt.savefig('e2eRec_spk.png', dpi=300)

    # plt.close()