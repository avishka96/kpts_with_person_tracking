import ffmpeg
import os
import sys
import argparse

def change_fps(vid_loc, fps):
    stream = ffmpeg.input(vid_loc)
    stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, str(os.path.splitext(vid_loc)[0])+'_20fps.mp4')
    ffmpeg.run(stream)

if __name__ == '__main__':
    dir = 'inference'
    for root, dirs, files in os.walk(r'{}'.format(dir)):
        for file in files:
            if file.endswith('.mp4'):
                vidpath = os.path.abspath(os.path.join(root, file))
                print("Downsampling {}".format(vidpath))
                change_fps(vidpath, 20)
                print("Generating pose video for {}".format(vidpath))
                command = "python kpts_obj_tracking.py --source {}".format(str(os.path.splitext(vidpath)[0])+'_20fps.mp4')
                os.system(command)





