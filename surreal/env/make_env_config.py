from surreal.env import make_env
from surreal.session import Config
import surreal.utils as U
import pickle
import sys
import os


def main():
    output_pipe = sys.stdout
    with open(os.devnull, 'w') as redirect:
        sys.stdout = redirect
        env_config = U.from_pickle_hex(sys.argv[1])
        _, env_config = make_env(env_config)
        redirect.close()
        sys.stdout = output_pipe
    sys.stdout.buffer.write(pickle.dumps(env_config))


if __name__ == "__main__":
    main()
