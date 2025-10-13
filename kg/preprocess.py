def preprocess(scene_list, idx):
    script_dict = {}
    script_dict['id'] = idx
    script_dict['chapters'] = []

    elem_dict = {}
    elem_dict['index'] = 1
    elem_dict['text'] = scene_list
    elem_dict['summaries'] = ""

    script_dict['chapters'].append(elem_dict)

    return script_dict