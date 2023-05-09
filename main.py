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

# Check original Spike-Image Decoder for non-modified cal_performance that works for TensorFlow 1
# https://github.com/jiankliu/Spike-Image-Decoder

def modified_cal_performance(src_imgs, dst_imgs):
    src_imgs = src_imgs.astype('float32')
    dst_imgs = dst_imgs.astype('float32')

    img_num = src_imgs.shape[0]
    all_mse = np.zeros(img_num)
    all_psnr = np.zeros(img_num)
    all_ssim = np.zeros(img_num)

    for i in range(img_num):
        all_mse[i] = skimage.metrics.mean_squared_error(src_imgs[i], dst_imgs[i])
        all_psnr[i] = skimage.metrics.peak_signal_noise_ratio(src_imgs[i], dst_imgs[i])
        all_ssim[i] = skimage.metrics.structural_similarity(src_imgs[i], dst_imgs[i], channel_axis = 2)

    return np.mean(all_mse), np.mean(all_psnr), np.mean(all_ssim)

# Heavily inspired by the original Spike-Image Decoder
# https://github.com/jiankliu/Spike-Image-Decoder

if __name__ == '__main__':
    print("Imports Successful.")

    # Setting directory strings and loading numpy array
    # Set for subjects 1-8
    subject_number = 1

    subj = format(subject_number, '02')
    algo_dir = 'algonauts_2023_tutorial_data'
    data_dir = os.path.join(algo_dir, 'subj'+subj)
    fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    train_img_dir = os.path.join(data_dir, 'training_split', 'training_images')
    test_img_dir  = os.path.join(data_dir, 'test_split', 'test_images')

    image_count = 2000

    image_list = glob.glob(train_img_dir + '/*.png')
    image_list.sort()

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


    x_data = np.array(image_array)
    # x_data.reshape([x_data.shape[0], resolution, resolution, 1])
    y_data = np.array(combined_fmri)


    # X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=random_seed)

    X_train = x_data[:1600]
    X_test = x_data[400:]
    Y_train = y_data[:1600]
    Y_test = y_data[400:]

    X_train = X_train.reshape([X_train.shape[0], resolution, resolution, 1])
    X_test = X_test.reshape([X_test.shape[0], resolution, resolution, 1])

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    ncell = Y_test.shape[1]

    # ## Normlization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))   
    Y_train = min_max_scaler.fit_transform(Y_train)     
    Y_test = min_max_scaler.transform(Y_test)

    input_x = tf.keras.Input(shape = (ncell,))

    model_dense = SID.dense_decoder(ncell)
    model_ae = SID.AE((resolution, resolution, 1))

    dense_out = model_dense(input_x)
    ae_out = model_ae(dense_out)

    optimizer = keras.optimizers.Adam()


    end2end_model = tf.keras.Model(input_x, ae_out)
    end2end_model.summary()
    end2end_model.compile(loss = 'mse', optimizer = optimizer)

    weight_dir = 'e2e_model'
    result_dir = 'e2e_result'

    multiout_model = tf.keras.Model(input_x, [dense_out, ae_out])

    num_iterations = 7

    # earlystopping = callbacks.EarlyStopping(monitor="val_loss",
    #                                     mode="min", patience=5, start_from_epoch = 30,
    #                                     restore_best_weights=True)

    for i in range(num_iterations):
        print("iteration: %d"%(i + 1))
        history = end2end_model.fit(Y_train, X_train, 
                          batch_size = 16, epochs = 125,
                          validation_data = (Y_test, X_test),
                          callbacks=[] )
         
        pred_dense, pred_ae = multiout_model.predict(Y_test)

        mse, psnr, ssim = modified_cal_performance(X_test, pred_ae)
        print('\nTesting: AE:\tmse:%f psnr:%f ssim:%f'%(mse, psnr, ssim))
        mse, psnr, ssim = modified_cal_performance(X_train, pred_ae)
        print('\nTraining: AE:\tmse:%f psnr:%f ssim:%f'%(mse, psnr, ssim))

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    end2end_model.save_weights(os.path.join(weight_dir, 'model_weights.h5'))

    np.save(os.path.join(result_dir, 'test_predictions.npy'), pred_ae)
    pred_dense_trn, pred_ae_trn = multiout_model.predict(Y_train)
    np.save(os.path.join(result_dir, 'train_predictions.npy'), pred_ae_trn)
    
    # visualization the reconstructed images
    X_reconstructed_mu = pred_ae
    n = 8
    for j in range(1):
        plt.figure(figsize=(12, 2))    
        for i in range(n):
            # display original images
            ax = plt.subplot(2, n, i +j*n*2 + 1)
            plt.imshow((np.fliplr(X_test[i+j*n].reshape(resolution ,resolution ))),cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # display reconstructed images
            ax = plt.subplot(2, n, i + n + j*n*2 + 1)
            #plt.imshow(np.rot90(np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='hot')
            plt.imshow((np.fliplr(X_reconstructed_mu[i+j*n].reshape(resolution ,resolution ))),cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.show()
    print("Done!")
    plt.close()
    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()