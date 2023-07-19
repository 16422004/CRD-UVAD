import pickle
import os
import numpy as np
import sklearn.metrics
from sklearn.metrics import roc_auc_score, confusion_matrix,roc_curve
import sys
from utils import scorebinary, anomap
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def eval_p(itr, dataset, predict_dict, logger, save_path, args, plot=False, zip=False, inference=False):
    global label_dict_path
    if inference:
        save_root = './inference_result'
    else:
        save_root = './result'
    label_dict_path = '{}/{}/GT'.format(args.dataset_path,args.dataset_name)

    with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:
        frame_label_dict = pickle.load(f)
        if type(frame_label_dict) is list:
            frame_label_dict = np.array(frame_label_dict)
    with open(file=os.path.join(label_dict_path, 'video_label.pickle'), mode='rb') as f:
        video_label_dict = pickle.load(f)
        if type(video_label_dict) is list:
            video_label_dict = np.array(video_label_dict)
    all_predict_np = np.zeros(0)
    all_label_np = np.zeros(0)
    normal_predict_np = np.zeros(0)
    normal_label_np = np.zeros(0)
    abnormal_predict_np = np.zeros(0)
    abnormal_label_np = np.zeros(0)
    for k, v in predict_dict.items():
        if video_label_dict[k] == [1.]:
            frame_labels = frame_label_dict[k]
            new_len = int(len(frame_labels) / 16)
            v = v[:new_len]
            predict_dict[k] = v
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            abnormal_predict_np = np.concatenate((abnormal_predict_np, v.repeat(16)))
            abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels[:len(v.repeat(16))]))
        elif video_label_dict[k] == [0.]:
            frame_labels = frame_label_dict[k]
            new_len = int(len(frame_labels) / 16)
            v = v[:new_len]
            predict_dict[k] = v
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            normal_predict_np = np.concatenate((normal_predict_np, v.repeat(16)))
            normal_label_np = np.concatenate((normal_label_np, frame_labels[:len(v.repeat(16))]))

    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
    eer,auc,fpr,tpr = calculate_eer(all_label_np,all_predict_np)
    binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)
    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_label_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count

    abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
    binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
    fnr = fn / (fn+tp)
    abnormal_ano_false_alarm = fp / (fp + tn)

    print('Iteration: {} AUC_score_all_video is {}'.format(itr, all_auc_score))
    print('Iteration: {} EER is {}'.format(itr, eer))
    print('Iteration: {} AUC_score_abnormal_video is {}'.format(itr, abnormal_auc_score))
    print('Iteration: {} ano_false_alarm_all_video is {}'.format(itr, all_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_normal_video is {}'.format(itr, normal_ano_false_alarm))
    print('Iteration: {} ano_false_alarm_abnormal_video is {}'.format(itr, abnormal_ano_false_alarm))
    print('Iteration: {} fnr is {}'.format(itr, fnr))
    if plot:
        anomap(predict_dict, frame_label_dict, save_path, itr, save_root, zip)
        draw_ROC(auc,fpr,tpr, os.path.join(save_root,save_path),itr)
    if logger:
        logger.log_value('Test_AUC_all_video', all_auc_score, itr)
        logger.log_value('Test_AUC_abnormal_video', abnormal_auc_score, itr)
        logger.log_value('Test_false_alarm_all_video', all_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_normal_video', normal_ano_false_alarm, itr)
        logger.log_value('Test_false_alarm_abnormal_video', abnormal_ano_false_alarm, itr)
    if os.path.exists(os.path.join(save_root,save_path)) == 0:
        os.makedirs(os.path.join(save_root,save_path))
    with open(file=os.path.join(save_root, save_path, 'result.txt'), mode='a+') as f:
        f.write('itration_{}_AUC_Score_all_video is {}\n'.format(itr, all_auc_score))
        f.write('itration_{}_EER is {}\n'.format(itr, eer))
        f.write('itration_{}_AUC_Score_abnormal_video is {}\n'.format(itr, abnormal_auc_score))
        f.write('itration_{}_ano_false_alarm_all_video is {}\n'.format(itr, all_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_normal_video is {}\n'.format(itr, normal_ano_false_alarm))
        f.write('itration_{}_ano_false_alarm_abnormal_video is {}\n'.format(itr, abnormal_ano_false_alarm))

def calculate_eer(y,y_score):
    fpr,tpr,thresholds = roc_curve(y,y_score,pos_label=1)
    eer = brentq(lambda x:1. - x - interp1d(fpr,tpr)(x), 0., 1.)
    auc = sklearn.metrics.auc(fpr,tpr)
    thresh = interp1d(fpr,thresholds)(eer)
    return eer, auc, fpr,tpr

def draw_ROC(auc, fpr, tpr, save_path, itr):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(auc), lw=2)
    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # save roc curve
    np.savez(r'{}/plot/ROC.npz'.format(save_path), auc, fpr, tpr)
    plt.savefig(os.path.join(save_path, 'plot', 'itr_{}'.format(itr)))
    plt.close()
