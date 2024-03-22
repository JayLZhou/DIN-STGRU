import imageio


def create_gif(image_paths, gif_path, duration=1):
    frames = []
    for imp in image_paths:
        frames.append(imageio.imread(imp))
    imageio.mimsave(gif_path, frames, 'GIF', duration=duration)

if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from util import cmp_fuc
    from functools import cmp_to_key
    results_path = './results'
    model_names = [i for i in os.listdir(results_path) if i[-4:] != 'json']
    for modeln in model_names:
        print("modeln:", modeln)
        model_path = os.path.join(results_path, modeln)
        sample_dirs = os.listdir(model_path)
        for d in tqdm(sample_dirs):
            sample_path = os.path.join(model_path, d)
            img_paths = [os.path.join(sample_path, i) for i in os.listdir(sample_path) if i[-3:] == 'png']
            img_paths.sort(key=cmp_to_key(cmp_fuc))
            gif_path = os.path.join(sample_path, 'predict.gif')
            create_gif(img_paths, gif_path)
            
    
