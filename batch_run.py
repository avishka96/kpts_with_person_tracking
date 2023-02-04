import os
import sys
import argparse


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
                fps_cmd = "ffmpeg -i {} -c:v libx264 -crf 0 -filter:v fps={} {}".format(vidpath, str(dwn_fps), str(os.path.splitext(vidpath)[0])+'_{}fps{}'.format(str(dwn_fps), ext))
                os.system(fps_cmd)
                print("[2/2] Generating skeletons")
                py_cmd = "python kpts_obj_tracking.py --source {}".format(str(os.path.splitext(vidpath)[0])+'_{}fps{}'.format(dwn_fps, ext))
                os.system(py_cmd)
                print("[*] Skeletons generated\n")
                vid_count += 1
    print("[*] Skeletons generated for {} videos".format(vid_count))
    print("[*] Process Completed")
