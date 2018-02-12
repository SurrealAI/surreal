import os
import time

from .wrapper import Wrapper
import imageio

from multiprocessing import Process, Queue

# File taken from 5d4757b9d2789c34a1120f4c6e4452e4cb7a11fc by Zihua

def save_video(frame_queue, filename, fps):
    '''
    Target function for video process. Opens up file with path and uses library imageio
    Args:
        frame_queue: a queue of frames to be store. If the frame is None, the capture is over
        filename: filename to which the capture is to be stored
        fps: framerate.
    Note that Queue.get() is a blocking function and thus will hang until new frames are
    added to frame_queue
    '''
    writer = imageio.get_writer(filename, fps=fps)
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        writer.append_data(frame)
    writer.close()

class VideoWrapper(Wrapper):
    '''
    Environment Wrappers for automatically rendering and saving test runs.
    Attributes:
        public attributes
            env (Env): environment to be wrapped
            capture_interval (int): number of episodes between captures
                default to 10
            frame_interval (int): number of frames between each recorded frame
                default to 10
            fps (int): frame rate
                default to 30
            ext (str): file extention. Either .gif or .mp4 depending on input
                default to .mp4
            save_dir (str): directory to which the files are to be saved
                default to './snaps'
            max_videos (int): maximum number of videos allowed in a directory
                default to 10
        helper attributes
            num_eps (int): number of episodes executed
            num_steps (int): number of steps executed
            is_recording (bool): whether the on going episode is being recorded
            path_queue (Queue): to keep track of saved videos to maintain max number of captures
            num_paths (int): number of paths stored in path_queue. necessary because Queue.qsize()
                             is not implemented on Mac.
            video_process (Process): separate process that writes images to file
            video_queue (Queue): queue of frames to be writen to file
    '''

    def __init__(self, env, env_config, frame_interval = 10, fps =30, use_gif = False):
        '''
        Constructor for VideoWrapper. also creates the save directory if not present
        Args:
            env (Env): environment to be wrapped
            capture_interval (int): number of episodes between captures
            frame_interval (int): number of frames between each recorded frame
            fps (int): frame rate
            save_dir (str): directory to which the files are to be saved
            max_videos (int): maximum number of videos allowed in a directory
            use_gif (bool): boolean flag to use either gif or mp4
        '''

        super().__init__(env)
        self.env = env

        self.max_videos = env_config.video.max_videos
        self.capture_interval = env_config.video.record_every
        self.frame_interval   = frame_interval
        self.fps = fps

        self.ext = '.gif' if use_gif else '.mp4'
        self.save_dir = env_config.video.save_directory

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.num_eps = 0
        self.num_steps = 0
        self.is_recording = False
        self.path_queue = Queue()
        self.num_paths = 0 #work around, qsize() not implemented on mac

    def _reset(self, **kwargs):
        '''
        Overwrites reset method. in addition to reseting the environment,
        this method also manages the video process and when to start writing video
        or gif to file
        '''
        self.num_steps = 0
        if self.num_eps % self.capture_interval == 0:
            self.video_queue = Queue()
            self.is_recording = True

            path = self.save_dir + 'video_eps_{}{}'.format(self.num_eps, self.ext)
            if self.num_paths >= self.max_videos:
                dep_path = self.path_queue.get()
                os.remove(dep_path)
                self.num_paths -= 1

            self.path_queue.put(path)
            self.video_process = Process(target=save_video,
                                         args=(self.video_queue, path, self.fps))
            self.video_process.start()
            self.num_paths += 1

        state = self.env.reset(**kwargs)
        return state

    def _step(self, action):
        '''
        Overwrites _step function. In addition to taking an action,
        if the video is recording and its time to capture a frame, the
        frame is rendered using 'rgb_array' mode and is put into the video_queue.
        If video capture is over, stop_recording is called.
        '''
        state, step_reward, terminal, info = self.env.step(action)
        self.num_steps += 1

        if self.is_recording and self.num_eps % self.frame_interval == 0:
            ob = self.render(mode = 'rgb_array')
            self.video_queue.put(ob)

        if terminal and self.is_recording:
            self.stop_record()

        if terminal:
            self.num_eps += 1

        return state, step_reward, terminal, info

    def stop_record(self):
        '''
        stops recording and wait for the video_process to finish writing to file and join
        First puts a None (End of Video) frame into the frame queue and wait for writing
        process to terminate.
        '''
        self.video_queue.put(None)
        self.video_process.join()
        self.is_recording = False
        print('finished recording video {}'.format(self.num_eps))
