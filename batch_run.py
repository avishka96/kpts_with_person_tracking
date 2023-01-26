import ffmpeg
import os
import sys
import argparse

def change_fps(vid_loc, fps, format):
    stream = ffmpeg.input(vid_loc)
    stream = ffmpeg.filter(stream, 'fps', fps=fps, round='up')
    stream = ffmpeg.output(stream, str(os.path.splitext(vid_loc)[0])+'_{}fps{}'.format(fps,format))
    ffmpeg.run(stream)

if __name__ == '__main__':
    dir = str(sys.argv[1])
    dwn_fps = 15
    vid_count = 0
    for root, dirs, files in os.walk(r'{}'.format(dir)):
        for file in files:
            ext = os.path.splitext(str(file))[1]
            if (ext == '.mp4' or ext == '.avi') and ('{}fps'.format(dwn_fps) not in str(file)):
                vidpath = os.path.relpath(os.path.join(root, file))
                print("[*] Video file = {}".format(vidpath))
                print("[1/2] Downsampling video to {} fps".format(dwn_fps))
                change_fps(vidpath, dwn_fps, ext)
                print("[2/2] Generating skeletons")
                command = "python kpts_obj_tracking.py --source {}".format(str(os.path.splitext(vidpath)[0])+'_{}fps{}'.format(dwn_fps, ext))
                os.system(command)
                print("[*] Skeletons generated\n")
                vid_count += 1
    print("[*] Skeletons generated for {} videos".format(vid_count))
    print("[*] Process Completed")
