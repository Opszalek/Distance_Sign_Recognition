import os
import pandas as pd
import yaml
import warnings

path_to_yolo_folder = '../Junk/test_v2.v1i.yolov9'


def load_yaml(path):
    with open(os.path.join(path, 'data.yaml'), 'r') as file:
        data = yaml.safe_load(file)
    return data


def return_class_names(prime_service):
    if not prime_service['names'][-1] == 'sign-text':
        raise Exception('sign-text is not in the list')
    return prime_service['names']


def return_labels_files(path, extension):
    try:
        paths = os.listdir(path + '/' + extension + '/labels')
    except FileNotFoundError:
        warning_message = f'No such directory: {path}/{extension}/labels\n' \
                          f'Please check the path to the labels folder.\n' \
                          f'Or create {extension}/labels folder and put the labels there.'
        warnings.warn(warning_message, category=UserWarning)
        return None
    return paths


def check_labels(labels, extension):
    if labels is None:
        warning_message = (f'Labels for {extension} are not found '
                           f'and will not be refactored.')
        warnings.warn(warning_message, category=UserWarning)
        return False
    return True


def read_data(path):
    return pd.read_csv(path, header=None, sep=' ')


def refactor_labels(data, class_names, sign_class_index):
    new_data = pd.DataFrame()
    text_labels = []
    for index, row in data.iterrows():
        if row[0] == sign_class_index:
            new_data = new_data.append(row)
        else:
            text_labels.append(class_names[int(row[0])])
    return new_data, text_labels


def save_text_labels(text_labels, extension, file):
    if not os.path.exists(os.path.join(path_to_yolo_folder, extension, 'text_labels')):
        os.makedirs(os.path.join(path_to_yolo_folder, extension, 'text_labels'))
    path = os.path.join(path_to_yolo_folder, extension, 'text_labels', f'{file}.txt')
    with open(path, 'w') as file:
        for label in text_labels:
            file.write(label + '\n')


def overwrite_labels(data, path):
    data.to_csv(path, header=False, index=False, sep=' ')


def label_manager(files, class_names, extension):
    sign_class_index = class_names.index('sign-text')
    if check_labels(files, extension):
        for file in files:
            path = os.path.join(path_to_yolo_folder, extension, 'labels', file)
            data = read_data(path)
            new_data, text_labels = refactor_labels(data, class_names, sign_class_index)
            save_text_labels(text_labels, extension, file[:-4])
            overwrite_labels(new_data, path)
    else:
        pass


def main():
    data = load_yaml(path_to_yolo_folder)
    class_names = return_class_names(data)

    train_labels = return_labels_files(path_to_yolo_folder, 'train')
    valid_labels = return_labels_files(path_to_yolo_folder, 'valid')
    test_labels = return_labels_files(path_to_yolo_folder, 'test')

    label_manager(train_labels, class_names, 'train')
    label_manager(valid_labels, class_names, 'valid')
    label_manager(test_labels, class_names, 'test')


if __name__ == '__main__':
    main()
