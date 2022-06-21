import os, sys
import os.path
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def dataloader(csv_path="ytvos.csv", E=0):
    filenames = open(csv_path).readlines()

    frame_all = [filename.split(',')[0].strip() for filename in filenames]
    startframe = [int(filename.split(',')[1].strip()) for filename in filenames]
    nframes = [int(filename.split(',')[2].strip()) for filename in filenames]

    all_index = np.arange(len(nframes))
    np.random.shuffle(all_index)

    refs_train = []

    for index in all_index:
        ref_num = 2

        # compute frame index (ensures length(image set) >= random_interval)
        refs_images =[]
        n_frames = nframes[index]
        
        frame_interval = np.random.choice([2,5,8],p=[0.4,0.4,0.2])
#         factors = [36, 16, 12, 6, 4.5, 3]
#         frame_interval = np.random.choice([int(n_frames//factor) for factor in factors],p=[0.2,0.2,0.1,0.2,0.2,0.1])
        
        start_frame = startframe[index]
        frame_indices = np.arange(start_frame, start_frame+n_frames, max(frame_interval,2))  # start from startframe
        total_batch, batch_mod = divmod(len(frame_indices), ref_num)
        if batch_mod > 0:
            frame_indices = frame_indices[:-batch_mod]
        frame_indices_batches = np.split(frame_indices, total_batch)
        for batches in frame_indices_batches:
            # ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
            #               for frame in [max(start_frame,batches[0]-30)]+ list(batches)]
            ref_images = [os.path.join(frame_all[index], '{:05d}.jpg'.format(frame))
                          for frame in list(batches)]
            refs_images.append(ref_images)

        refs_train.extend(refs_images)

    return refs_train[:69600], frame_all

if __name__ == '__main__':
    x = dataloader()
