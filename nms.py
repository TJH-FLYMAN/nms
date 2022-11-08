import numpy as np

bboxes = np.array([[100, 100, 210, 210, 0.72],
                   [250, 250, 420, 420, 0.8],
                   [220, 220, 320, 330, 0.92],
                   [100, 100, 210, 210, 0.72],
                   [230, 240, 325, 330, 0.81],
                   [220, 230, 315, 340, 0.9]])

def nms(iou_thresh=0.5,conf_threash=0.5):
    x1,y1,x2,y2,confidence= bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3],bboxes[:,-1]
    area = np.abs(x2-x1) * np.abs(y2-y1)
    print(area)
    indices = confidence.argsort()[::-1]
    # print(confidence.argsort()) [0 3 1 4 5 2]
    # print(indices)              [2 5 4 1 3 0]
    keep = []
    while indices.size > 0:
        idx_self, idx_other = indices[0], indices[1:] #0.92 ...
        if confidence[idx_self] < conf_threash:
            break
        keep.append(idx_self)
        # print(x1[idx_self])  220
        # print(x1[idx_other]) [220. 230. 250. 100. 100.]
        xx1, yy1 = np.maximum(x1[idx_self], x1[idx_other]), np.maximum(y1[idx_self], y1[idx_other])
        xx2, yy2 = np.minimum(x2[idx_self], x2[idx_other]), np.minimum(y2[idx_self], y2[idx_other])
        w, h = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)
        intersection = w * h
        # 计算并集(两个面积和-交集)
        union = area[idx_self] + area[idx_other] - intersection
        iou = intersection / union
        # 只保留iou小于等于阈值的元素
        keep_idx = np.where(iou <= iou_thresh)[0]
        indices = indices[keep_idx + 1]
    return np.array(keep)
if __name__ == '__main__':
    print(nms())
