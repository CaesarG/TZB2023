from pathlib import Path
import scipy.io as sciio
import numpy as np
from sklearn.utils import Bunch


# Load images in structured directory like it's sklearn sample dataset
def load_image_files(container_path, REAL_IMAG):
    image_dir = Path(container_path)
    # folders is the list of folders each conains a category of data
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    # _______________________________________________________
    # 去掉annotations文件夹
    # folders = folders[:-1]
    # _______________________________________________________
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []

    for i, direc in enumerate(folders):
        # print(i)
        print(i, categories[i])
        # print(direc)
        for file in direc.iterdir():
            mat_data = sciio.loadmat(file)
            # img = np.abs(mat_data['frame_Ev'])
            raw_data = []
            if REAL_IMAG:
                # raw_data.extend(np.real(mat_data['frame_Ev']).flatten())
                # raw_data.extend(np.imag(mat_data['frame_Ev']).flatten())
                # raw_data.extend(np.real(mat_data['frame_Eh']).flatten())
                # raw_data.extend(np.imag(mat_data['frame_Eh']).flatten())
                # raw_data = np.abs(raw_data)
                raw_data.extend(np.abs(mat_data['frame_Ev']).flatten())
                # raw_data.extend((np.angle(mat_data['frame_Ev'].flatten())))
                raw_data.extend(((np.angle(mat_data['frame_Ev'].flatten()) / np.pi + 1) * 100))
                raw_data.extend(np.abs(mat_data['frame_Eh'].flatten()))
                # raw_data.extend((np.angle(mat_data['frame_Eh']).flatten()))
                raw_data.extend(((np.angle(mat_data['frame_Eh'].flatten()) / np.pi + 1) * 100))
                # print(Angle(mat_data['frame_Ev'].flatten()))
                # break
            else:
                raw_data.extend(np.abs(mat_data['frame_Ev']).flatten())
                raw_data.extend(np.abs(mat_data['frame_Eh']).flatten())
            flat_data.append(raw_data)
            # images.append(img)
            target.append(categories[i])
        # break
    # return [flat_data,target,categories,descr]
    flat_data = np.array(flat_data, dtype='float32')
    target = np.array(target)
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 DESCR=descr)


def LOAD_IMAGE(REAL_IMAG):
    image_dataset1 = load_image_files('.\dataRCS', REAL_IMAG)
    image_dataset2 = load_image_files('.\dataRCS2', REAL_IMAG)
    # flat_data = np.concatenate([image_dataset1.data, image_dataset2.data])
    # target = np.concatenate([image_dataset1.target, image_dataset2.target])
    # categories = np.concatenate([image_dataset1.categories, image_dataset2.categories])
    # descr = image_dataset1.descr
    categories = image_dataset1.target_names
    categories.extend(image_dataset2.target_names)
    print("fucked")
    return Bunch(data=np.concatenate([image_dataset1.data, image_dataset2.data]),
                 target=np.concatenate([image_dataset1.target, image_dataset2.target]),
                 target_names=categories,
                 DESCR=image_dataset1.DESCR)
