import glob
import os

import imageio


GENERATED_SAMPLES_DIR = "./generated_samples/"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True,
        help="Name of the run which generated the source images.")
    parser.add_argument('--output', '-o', type=str, default="",
        help="Path to output GIF. Defaults to ./generated_samples/(RUN_NAME).gif")
    args = parser.parse_args()

    gif_file = args.output
    if len(gif_file) == 0:
        gif_file = os.path.join(GENERATED_SAMPLES_DIR, "{}.gif".format(args.name))

    with imageio.get_writer(gif_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(GENERATED_SAMPLES_DIR, "{}_epoch*".format(args.name)))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
