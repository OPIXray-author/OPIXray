def max_class(input: dict,threshold=0.5):
    """

    Args:
        input: a dict of class and scores.
        threshold:

    Returns: >threshold scores

    """
    output = dict()
    for cls, scores in input.items():
        temp_list = []
        if scores:
            for score in scores:
                if score>=threshold:
                    temp_list.append(score)
        output[cls] = temp_list
    return output


def series2dict(input):
    return  dict((idx, None) for idx in input)


if __name__ == '__main__':
    test_dict = {'a':[0.014652923680841923],'c':[0.012250889092683792, 0.011581446044147015],'b':[0.91553694,0.01644069]}
    print(max_class(test_dict))
    print(series2dict(('Folding_Knife', 'Straight_Knife','Scissor','Utility_Knife','Multi-tool_Knife',)))
