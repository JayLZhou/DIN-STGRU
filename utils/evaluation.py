import os
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')
EVALUATION_THRESHOLDS = (20, 30, 40,50)
np.seterr(divide='ignore',invalid='ignore')

class EmptyEvaluator(object):
    def __init__(self):
        pass
    def evaluate(self, gt, pred):
        pass
    def done(self):
        pass
    
class Evaluator(object):
    def __init__(self, save_path, seq=20):
        self.metric = {}
        for threshold in EVALUATION_THRESHOLDS:
            self.metric[threshold] = {
                "pod": np.zeros((seq, ), np.float32),
                "far": np.zeros((seq, ), np.float32),
                "csi": np.zeros((seq, ), np.float32),
                "hss": np.zeros((seq, ), np.float32)
            }
        self.seq = seq
        self.save_path = save_path
        self.total = 0
        self.losses = {
            'mae': 0,
            'mse': 0,
            'psnr': 0,
            'ssim': 0
        }
        print(self.metric.keys())

    def get_metrics(self, gt, pred, threshold):
        b_gt = gt > threshold
        b_pred = pred > threshold
        b_gt_n = np.logical_not(b_gt)
        b_pred_n = np.logical_not(b_pred)
        summation_axis = (1, 2)

        
        hits = np.logical_and(b_pred, b_gt).sum(axis=summation_axis)
        misses = np.logical_and(b_pred_n, b_gt).sum(axis=summation_axis)
        false_alarms = np.logical_and(b_pred, b_gt_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(b_pred_n, b_gt_n).sum(axis=summation_axis)

        a = hits
        b = false_alarms
        c = misses
        d = correct_negatives

        pod = a / (a + c)
        # pod = np.divide(a, (a+c), out=np.zeros_like(a), where=(a+c)!=0)
        far = b / (a + b)
        # far = np.divide(a, (a+b), out=np.zeros_like(a), where=(a+b)!=0)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        # self.check(pod, a, b, c, d)
        # self.check(far, a, b, c, d)
        # self.check(csi, a, b, c, d)
        # self.check(hss, a, b, c, d)
        pod[pod == np.inf] = 0
        pod = np.nan_to_num(pod)
        far[far == np.inf] = 0
        far = np.nan_to_num(far)
        csi[csi == np.inf] = 0
        csi = np.nan_to_num(csi)
        hss[hss == np.inf] = 0
        hss = np.nan_to_num(hss)
        return pod, far, csi, hss

    def check(self, data, a, b, c, d):
        nans = np.argwhere(np.isnan(data))
        infs = np.argwhere(np.isinf(data))
        if len(nans) != 0 or len(infs) != 0:
            print("no!")
            # print(data.reshape(1, -1))
            # print("hits", a, "far", b, "misses", c, "TF", d)

    def evaluate(self, gt, pred):
        batch_size = gt.shape[0]
        for threshold in EVALUATION_THRESHOLDS:
            for i in range(batch_size):
                self.total += 1
                pod, far, csi, hss = self.get_metrics(gt[i], pred[i], threshold)
                self.metric[threshold]["pod"] += pod
                self.metric[threshold]["far"] += far
                self.metric[threshold]["csi"] += csi
                self.metric[threshold]["hss"] += hss

    def done(self):
        thresholds = EVALUATION_THRESHOLDS
        pods = []
        fars = []
        csis = []
        hsss = []
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # draw line chart
        for threshold in thresholds:
            metrics = self.metric[threshold]
            pod = metrics["pod"] / (self.total / len(thresholds))
            pods.append(np.average(pod))
            far = metrics["far"] / (self.total / len(thresholds))
            fars.append(np.average(far))
            csi = metrics["csi"] / (self.total / len(thresholds))
            csis.append(np.average(csi))
            hss = metrics["hss"] / (self.total / len(thresholds))
            hsss.append(np.average(hss))

            x = list(range(len(pod)))
            plt.plot(x, pod, "r--", label='pod')
            plt.plot(x, far, "g--", label="far")
            plt.plot(x, csi, "b--", label="csi")
            plt.plot(x, hss, "k--", label="hss")
            for a, p, f, cs, h in zip(x, pod, far, csi, hss):
                plt.text(a, p + 0.005, "%.4f" % p, ha='center', va='bottom', fontsize=7)
                plt.text(a, f + 0.005, "%.4f" % f, ha='center', va='bottom', fontsize=7)
                plt.text(a, cs + 0.005, "%.4f" % cs, ha='center', va='bottom', fontsize=7)
                plt.text(a, h + 0.005, "%.4f" % h, ha='center', va='bottom', fontsize=7)

            plt.title(f"Threshold {threshold}")
            plt.xlabel("Time step")
            plt.ylabel("Rate")
            plt.legend()
            plt.gcf().set_size_inches(4.8 + (4.8 * self.seq // 10), 4.8)
            plt.savefig(os.path.join(save_path, f"{threshold}.jpg"))
            plt.clf()
        # draw bar chart
        x = np.array(range(len(thresholds)))
        total_width, n = 0.8, 4
        width = total_width / n
        plt.bar(x, pods, width=width, label='pod', fc='r')
        plt.bar(x + 0.2, fars, width=width, label='far', fc='g', tick_label=thresholds)
        plt.bar(x + 0.4, csis, width=width, label='csi', fc='b')
        plt.bar(x + 0.6, hsss, width=width, label='hss', fc='k')
        for a, p, f, cs, h in zip(x, pods, fars, csis, hsss):
            plt.text(a, p + 0.005, "%.4f" % p, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.2, f + 0.005, "%.4f" % f, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.4, cs + 0.005, "%.4f" % cs, ha='center', va='bottom', fontsize=7)
            plt.text(a + 0.6, h + 0.005, "%.4f" % h, ha='center', va='bottom', fontsize=7)
        plt.xlabel("Thresholds")
        plt.ylabel("Rate")
        plt.title(f"Average metrics")
        plt.legend()
        plt.gcf().set_size_inches(9.6, 4.8)
        plt.savefig(os.path.join(save_path, f"average.jpg"))
        plt.clf()

        res = ",".join(list(map(str, csis)))
        return res
    def record(self, _mae, _mse, _ssim, _psnr):
        self.losses['mae'] += _mae
        self.losses['mse'] += _mse
        self.losses['ssim'] += _ssim
        self.losses['psnr'] += _psnr        