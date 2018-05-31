# OpenAI Gym Mujoco

Troubleshoots Gym 0.10.5 and `mujoco-py` 1.50.1.56

The first time you import mujoco on Ubuntu, it will trigger a Cython build. Make sure you have:

```bash
sudo apt-get install libglew-dev
```

Also you need to have in path
```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/[user]/.mujoco/mjpro150/bin:/usr/lib/nvidia-384
```

If you see "patchelf not found", use the solution:
https://github.com/openai/mujoco-py/issues/147

```bash
sudo add-apt-repository ppa:jamesh/snap-support
sudo apt-get update
sudo apt install patchelf
```
