from pathlib import Path

from data import OPIXray_CLASSES
from detection_draw import *
from test import *
from utils.analysis import *


class OPIXrayDetectionSingle(OPIXrayDetection):
    def __init__(self, img_path, *kargs, **kwargs):
        super(OPIXrayDetectionSingle, self).__init__(*kargs, **kwargs)
        self.img_path = Path(img_path)
        self.ids.append(self.img_path.stem)


def test_net_single_img(save_folder, net, cuda, dataset, transform, top_k,
                        im_size=300, thresh=0.05):
    num_images = len(dataset)
    '''
    all detections are collected into:
    all_boxes[cls][image] = N x 5 array of detections in
    (x1, y1, x2, y2, score)
    
    '''

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')
    # if(k==0):
    # img = x.int().cpu().squeeze().permute(1,2,0).detach().numpy()
    # cv2.imwrite('edge_s.jpg',img)
    #    x = self.edge_conv2d(x)
    # rgb_im = rgb_im.int().cpu().squeeze().permute(1,2,0).detach().numpy()
    # cv2.imwrite('rgb_im.jpg', rgb_im)
    # for i in range(6):
    #    im = Image.fromarray(edge_detect[i]*255).convert('L')
    #    im.save(str(i)+'edge.jpg')
    # x = self.edge_conv2d.edge_conv2d(x)
    # else:
    # for i in range(num_images):
    i = 0
    im, gt, h, w, og_im = dataset.pull_item(i)
    # img = im.int().cpu().squeeze().permute(1, 2, 0).detach().numpy()
    # cv2.imwrite('/mnt/SSD/results/orgin'+str(i)+'.jpg', img)
    # im_saver = cv2.resize(im[(a2,a1,0),:,:].permute((a1,a2,0)).numpy(), (w,h))
    im = im.type(torch.FloatTensor)
    # im_det = og_im.copy()
    # im_gt = og_im.copy()

    # print(im_det)
    x = Variable(im.unsqueeze(0))
    print(x.shape)
    if args.cuda:
        x = x.cuda()
    _t['im_detect'].tic()
    detections = net(x).data
    detect_time = _t['im_detect'].toc(average=False)

    # skip j = 0, because it's the background class
    # //
    # //
    # print("detections:", detections.size(1))
    class_scores_dict = series2dict(OPIXray_CLASSES)
    class_coordinate_dict = series2dict(OPIXray_CLASSES)
    class_correct_scores = series2dict(OPIXray_CLASSES)

    for index, j in enumerate(range(1, detections.size(1))):
        # class now
        present_class = OPIXray_CLASSES[index]
        dets = detections[0, j, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.size(0) == 0:
            continue
        boxes = dets[:, 1:]
        print("boxes:",boxes)
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
        all_boxes[j][i] = cls_dets

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
        real = 0
        if gt[0][4] == 99:
            real = 0
        else:
            real = 1

        for item in gt:
            if real == 0:
                print('this pic dont have the obj:', dataset.ids[i])
                break

    print(class_correct_scores, class_coordinate_dict)

    return class_correct_scores, class_coordinate_dict, og_im

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    # evaluate_detections(all_boxes, output_dir, dataset)


if __name__ == '__main__':
    # EPOCHS = [45]
    # EPOCHS = [40,45,50, 55, 60, 65, 70, 75, 80,85,90,95,100,105,110,115,120,125,130,135,140,145]
    # EPOCHS = [130,135,140,145,150,155,160,165,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,255]
    # EPOCHS = [90, 95, 100, 105, 110, 115, 120, 125]
    # EPOCHS = [255]
    # print(EPOCHS)
    # for EPOCH in EPOCHS:
    from torchsummary import summary
    reset_args()

    parser.add_argument('--image',
                        default='OPIXray_Dataset/train/train_image/009069.jpg', type=str,
                        help='image file path to inference')
    args = parser.parse_args()
    sys.argv = []
    # load net
    num_classes = len(labelmap) + 1  # +a1 for background
    if args.cuda:


        net = build_ssd('test', 300, num_classes)  # initialize SSD
        net.load_state_dict(torch.load(args.trained_model))
        print('cuda')
    else:
        net = build_ssd('test', 300, num_classes, mode='cpu')
        net.load_state_dict(torch.load(args.trained_model, map_location="cpu"))
        print('no cuda')
    net.eval()
    # print('Finished loading model!')
    # load data
    dataset = OPIXrayDetectionSingle(img_path=args.image, root=args.OPIXray_root,
                                     # BaseTransform(300, dataset_mean),
                                     target_transform=OPIXrayAnnotationTransform(), phase='train')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
        # evaluation

    result = test_net_single_img(args.save_folder, net, args.cuda, dataset,
                                 None, args.top_k, 300,
                                 thresh=args.confidence_threshold)
    draw_with_coordinate(*result)
