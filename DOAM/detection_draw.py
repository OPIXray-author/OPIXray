import cv2

COLOR_CONFIG = {
    'Folding_Knife': (255, 255, 0)
    , 'Straight_Knife': (0, 255, 0)
    , 'Scissor': (0, 0, 255)
    , 'Utility_Knife': (255, 0, 255)
    , 'Multi-tool_Knife': (255, 0, 0),
}


def draw_with_coordinate(class_correct_scores: dict, class_coordinate_dict: dict, og_im, color_config=COLOR_CONFIG):
    for cls, scores in class_correct_scores.items():
        if scores:
            for index, score in enumerate(scores):
                coordinate = tuple(map(int, class_coordinate_dict[cls][index]))
                first_point = (coordinate[0], coordinate[1])
                last_point = (coordinate[2], coordinate[3])
                cv2.rectangle(og_im, first_point, last_point, color_config[cls], 2)
                # 在矩形框上方绘制该框的名称
                text_point = ((coordinate[0], coordinate[1] - 4 if coordinate[1] - 4 > 0 else coordinate[1]))
                cv2.putText(og_im, "{0},score:{1}".format(cls, "%.2f" % score), text_point, cv2.FONT_HERSHEY_COMPLEX,
                            fontScale=1, color=color_config[cls],
                            thickness=2)
    cv2.imshow("image", og_im)
    cv2.waitKey(0)
