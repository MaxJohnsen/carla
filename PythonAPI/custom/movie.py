from moviepy.editor import ImageSequenceClip
import argparse
import os
import glob
from pathlib import Path
 
IMAGE_EXT = ['jpeg', 'gif', 'png', 'jpg']
 
 
def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'folder_path',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()
 
    folder_path = Path(args.folder_path)
    for image_folder in glob.glob(args.folder_path + "/*"):
 
        #convert file folder into list firltered for image file types
        image_list = sorted([os.path.join(image_folder + "/imgs", image_file)
                            for image_file in os.listdir(image_folder +"/imgs")])
       
        image_list = [image_file for image_file in image_list if os.path.splitext(image_file)[1][1:].lower() in IMAGE_EXT]
 
        forward_center = []
        hq1 = []
        hq2 = []
 
        for img in image_list:
            if "forward_center" in img:
                forward_center.append(img)
            elif "hq_record1" in img:
                hq1.append(img)
            elif "hq_record2" in img:
                hq2.append(img)
 
        if len(forward_center)>0:
            print("Creating video for forward center images, FPS={}".format(args.fps))
            clip_fc = ImageSequenceClip(forward_center, fps=args.fps)
            clip_fc.write_videofile(image_folder+"/forward_center.mp4")
        else:
            print("WARNING: No forward center images")
 
        if len(hq1)> 0:
            print("Creating video for hq_1 images, FPS={}".format(args.fps))
            clip_hq1 = ImageSequenceClip(hq1, fps=args.fps)
            clip_hq1.write_videofile(image_folder+"/hq_rec1.mp4")
        else:
            print("WARNING: No hq1 images")
 
        if len(hq2)>0:
            print("Creating video for hq_2 images, FPS={}".format(args.fps))
            clip_hq2 = ImageSequenceClip(hq2, fps=args.fps)
            clip_hq2.write_videofile(image_folder+"/hq_rec2.mp4")
        else:
            print("WARNING: No hq2 images")
       
 
 
if __name__ == '__main__':
    main()