from analysis import *
import torch
import numpy as np

def result_struct(detections,h, w,all_boxes,OPIXray_CLASSES):
    class_scores_dict = series2dict(OPIXray_CLASSES)
    class_coordinate_dict = series2dict(OPIXray_CLASSES)
    class_correct_scores = series2dict(OPIXray_CLASSES)
    print(w,h)

    for index, j in enumerate(range(1, detections.size(1))):
        # class now
        present_class = OPIXray_CLASSES[index]
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        # print("boxes:", boxes)
        boxes[:, 0] *= w
        boxes[:, 2] *= w
        boxes[:, 1] *= h
        boxes[:, 3] *= h
        # print("after boxes:",boxes)
        # print(boxes.cpu().numpy())
        scores = dets[:, 0].cpu().numpy()
        scores_list = scores.tolist()
        class_scores_dict[present_class] = scores_list
        # print("scores:", scores_list)
        cls_dets = np.hstack((boxes.cpu().numpy(),
                              scores[:, np.newaxis])).astype(np.float32,
                                                             copy=False)
        all_boxes[j][0] = cls_dets

        class_correct_scores = max_class(class_scores_dict)
        class_coordinate_dict[present_class] = boxes.cpu().numpy().tolist()[:len(class_correct_scores[present_class])]
        # for item in cls_dets:
        # print(item)
        # print(item[5])
        # if item[4] > thresh:
        # print(item)
        # chinese = labelmap[j - 1] + str(round(item[], 2))
        # print(chinese+'det\n\n')
        # if chinese[0] == 'knife':
        # chinese = 'knife' + chinese[6:]
        # cv2.rectangle(im_det, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
        # cv2.putText(im_det, chinese, (int(item[0]), int(item[1]) - 5), 0, 0.6, (0, 0, 255), 2)


    return class_correct_scores, class_coordinate_dict
