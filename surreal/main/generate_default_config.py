import shutil
import pkg_resources
import surreal.utils as U


def main():
    print('Generating default config for surreal at ~/.surreal.yml')
    default_path = U.f_expand('~/.surreal.yml')
    U.move_with_backup(default_path)

    fname = pkg_resources.resource_filename('surreal', 'sample_surreal.yml')
    shutil.copyfile(fname, default_path)


if __name__ == "__main__":
    main()
