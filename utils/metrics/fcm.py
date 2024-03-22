import skfuzzy as fuzz
import numpy as np
import cv2


def change_color_fuzzycmeans(cluster_membership, clusters):
    return [clusters[np.argmax(pix)] for pix in cluster_membership.T]


def fcm(img_np_gray):
    img_shape = img_np_gray.shape
    img_np_gray_flatten = img_np_gray.reshape((img_shape[0] * img_shape[1], 1))
    cluster = 2
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img_np_gray_flatten.T,
                                                     cluster,
                                                     2,
                                                     error=0.005,
                                                     maxiter=1000,
                                                     init=None,
                                                     seed=42)
    new_img = np.array(change_color_fuzzycmeans(u, cntr))
    fuzzy_img = np.reshape(new_img, img_shape).astype(np.uint8)
    return fuzzy_img


def calc_CR(gt, pred_img):
    img_shape = pred_img.shape
    fcm_gt = fcm(gt)
    fcm_pred = fcm(pred_img)
    ret, bin_gt = cv2.threshold(fcm_gt,
                                np.max(fcm_gt) - 1, 1, cv2.THRESH_BINARY)
    ret, bin_pred = cv2.threshold(fcm_pred,
                                  np.max(fcm_pred) - 1, 1, cv2.THRESH_BINARY)
    # print(bin_gt)
    assert bin_gt.shape == bin_pred.shape, f"pred_img shape is not match with GT, GT shape:{gt.shape}, pred_img shape:{pred_img.shape}"
    # print(np.array(cv2.bitwise_xor(bin_gt, bin_pred)))
    n_miss = np.sum(cv2.bitwise_xor(bin_gt, bin_pred))
    # print("n_miss:", n_miss)
    n_hit = img_shape[0] * img_shape[1] - n_miss
    # print("n_hit:", n_hit)
    CR = float(n_hit) / (n_miss + n_hit)
    # print("CR:", CR)
    return CR


def fcm_batch(img_np_gray_batch):
    img_shape = img_np_gray_batch.shape
    img_np_gray_batch_flatten = img_np_gray_batch.reshape(
        (img_shape[0], img_shape[1] * img_shape[2]))
    cluster = 2
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img_np_gray_batch_flatten,
                                                     cluster,
                                                     2,
                                                     error=0.005,
                                                     maxiter=1000,
                                                     init=None,
                                                     seed=42)
    new_imgs = np.array(change_color_fuzzycmeans(u, cntr))
    fuzzy_imgs = np.reshape(new_imgs, img_shape).astype(np.uint8)
    return fuzzy_imgs


def calc_CR_batch(gt, pred_img):
    img_shape = pred_img.shape
    fcm_gt = fcm_batch(gt)
    fcm_pred = fcm_batch(pred_img)
    # print("fcm_gt.shape:", fcm_gt.shape)
    ret, bin_gt = cv2.threshold(fcm_gt,
                                np.max(fcm_gt) - 1, 1, cv2.THRESH_BINARY)
    ret, bin_pred = cv2.threshold(fcm_pred,
                                  np.max(fcm_pred) - 1, 1, cv2.THRESH_BINARY)
    # print(bin_gt)
    assert bin_gt.shape == bin_pred.shape, f"pred_img shape is not match with GT, GT shape:{gt.shape}, pred_img shape:{pred_img.shape}"
    # print(np.array(cv2.bitwise_xor(bin_gt, bin_pred)).shape)
    n_miss = np.sum(cv2.bitwise_xor(bin_gt, bin_pred))
    # print("n_miss:", n_miss)
    n_hit = img_shape[0] * img_shape[1] * img_shape[2] - n_miss
    # print("n_hit:", n_hit)
    CR = float(n_hit) / (n_miss + n_hit)
    # print("CR:", CR)
    return CR


# test
if __name__ == '__main__':
    img = cv2.imread(r"test.png", 0)
    print(img.shape)
    img2 = cv2.imread(r"test2.png", 0)
    img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    # print(img2.shape)
    # cv2.namedWindow("Image")
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    print(img.shape, img2.shape)
    print(np.max(img), np.min(img))
    print(np.max(img2), np.min(img2))
    calc_CR(img, img2)
