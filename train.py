import argparse
import os
import pandas as pd
from wgan import WGAN
from generator import Generator
from discriminator import Discriminator
from image_sampler import ImageSampler
from noise_sampler import NoiseSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', type=str)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--nb_epoch', '-e', type=int, default=1000)
    parser.add_argument('--noise_dim', '-nd', type=int, default=128)
    parser.add_argument('--height', '-ht', type=int, default=128)
    parser.add_argument('--width', '-wd', type=int, default=128)
    parser.add_argument('--save_steps', '-ss', type=int, default=10)
    parser.add_argument('--visualize_steps', '-vs', type=int, default=10)
    parser.add_argument('--lambda', '-l', type=float, default=10., dest='lmbd')
    parser.add_argument('--initial_steps', '-is', type=int, default=20)
    parser.add_argument('--initial_critics', '-sc', type=int, default=20)
    parser.add_argument('--normal_critics', '-nc', type=int, default=5)
    parser.add_argument('--model_dir', '-md', type=str, default="./params")
    parser.add_argument('--result_dir', '-rd', type=str, default="./result")
    parser.add_argument('--noise_mode', '-nm', type=str, default="uniform")

    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # output config to csv
    config_path = os.path.join(args.result_dir, "config.csv")
    dict_ = vars(args)
    df = pd.DataFrame(list(dict_.items()), columns=['attr', 'status'])
    df.to_csv(config_path, index=None)

    input_shape = (args.height, args.width, 3)

    image_sampler = ImageSampler(args.image_dir, target_size=input_shape[:2])
    noise_sampler = NoiseSampler(args.noise_mode)

    generator = Generator(args.noise_dim, is_training=True)
    discriminator = Discriminator(input_shape, is_training=True, normalization='layer')

    wgan = WGAN(generator, discriminator, lambda_=args.lmbd, is_training=True)

    wgan.fit(image_sampler.flow(args.batch_size), noise_sampler, nb_epoch=args.nb_epoch,
             result_dir=args.result_dir, model_dir=args.model_dir,
             save_steps=args.save_steps, visualize_steps=args.visualize_steps,
             initial_steps=args.initial_steps, initial_critics=args.initial_critics,
             normal_critics=args.normal_critics)


if __name__ == '__main__':
    main()