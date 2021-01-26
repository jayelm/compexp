from settings import OUTPUT_FOLDER
from subprocess import call
from glob import glob

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Clean experiment file",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "extensions",
        nargs="*",
        default=["csv", "npy", "npz", "mmap", "pkl"],
        help="Which extensions to remove",
    )

    args = parser.parse_args()

    for ext in args.extensions:
        files = glob(f"{OUTPUT_FOLDER}/*.{ext}")
        if files:
            cmd = ["rm", *files]
            print(" ".join(cmd))
            call(cmd)
