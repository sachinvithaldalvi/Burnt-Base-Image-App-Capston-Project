import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

import pafy #retrive metadata 
import cv2

from skimage.metrics import structural_similarity as ssim

######################################################
# DEFINE FUNCTION TO GET VIDEO FRAMES
def get_video_frames(url):
	'''
	Create frame generator object from a youtube url.
	------------------------------------------------------
	return frame generator object and total number of frames.
	'''
	video = pafy.new(url)
	best = video.getbest(preftype="mp4")
	cam = cv2.VideoCapture(best.url)
	# number of frames in video
	num_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
	#fps = cam.get(cv2.CAP_PROP_FPS) # Gets the frames per second
	return cam, num_frames

####################################################
# DEFINE FUNCTION TO CALCULATE STRUCTURE SIMILARITY
def struc_sim(target, img):
	'''
	Calculate structure similarity between 2 images.
	-------------------------------------------------
	param: target: numpy array image in "RGB"
	param: img: numpy array image in "RGB"
	'''

	# Downsizing the target - see how can I keep the aspect Ratio the same 
	target = cv2.resize(target, (406, 720), interpolation=cv2.INTER_AREA)
	
	# target image size
	target_dim = (target.shape[1], target.shape[0])
	#st.write(f'{traget_dim}')
	# resize img to match target size
	img = cv2.resize(img, target_dim, interpolation=cv2.INTER_AREA)
	# convert target image to gray scale
	target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
	img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return ssim(target_gray, img_gray)

# DEFINE FUNCTION TO SHOW 2 IMAGES SIDE BY SIDE
def show_2_imgs(target, img=None):
	'''
	Target and img NEED to be RGB numpy array.
	'''
	fig, axes = plt.subplots(1,2)
	axes[0].imshow(target)
	axes[0].set_title("Target")
	if img is not None:
		axes[1].imshow(img, )
		axes[1].set_title("Match")
	st.pyplot(fig)

#############################################################
# LOAD TARGET IMAGE
st.sidebar.write("# Load image from local")
img_target = st.sidebar.file_uploader(label="Upload image", 
	type=["jpg", "jpeg", "png"],
	key="i")
	
# if user upload a target image
if img_target is not None:
	img_target = Image.open(img_target)

	# convert target image to RGB and put in numpy array format.
	img_target_RGB = np.array(img_target.convert("RGB"))


	# convert image to RGB
	#"https://www.youtube.com/watch?v=r3iIy5m2Emc",
	url = ["https://www.youtube.com/watch?v=VvL5Q7YVyWM","https://www.youtube.com/watch?v=J-V8fffpgvw"]

	# loop through video links
	for url in url:
		frame_generator, num_frames = get_video_frames(url)
		st.write(f"Video url: {url} \n\n **Total in video: {num_frames} frames**")


		# LOOP to search for match frame
		threshold = 0.40
		found = False
		for i in range(num_frames):
			# current frame
			_, frame = frame_generator.read()
			#convert current frame from BGR to RGB format
			#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# Future release How to extract one frame of a video every N seconds to an image?
			# check if the current frame is similar to the target above a threshold
			# Here I only check every 10 frames to save time.
			if i%60==0 and struc_sim(img_target_RGB, frame) >= threshold:
			#if struc_sim(img_target_RGB, frame) >= threshold:
				# show the ssim value
				st.write(f"Video url: {url} \n\n **Total in video: {num_frames} frames**")
				st.text(f'**Found match !**\
						\nFrame number: {i+1}\
						\nSSIM: {struc_sim(img_target_RGB, frame):.4}')
				found = True
				# plot the target and frame side by side.
				show_2_imgs(img_target_RGB, frame)
				break
		if found == True:
			break
		else:
			st.write("Target Not Found in this video")

	# if no match found:
	if not found:
		st.write("### Target image is NOT FOUND in avialable video links")
		show_2_imgs(img_target_RGB, None)