## Notes:
- ConvNets have been imporotant in:
	- object recognition (Russakovsky et al., 2015) 
	- scene classification (Zhou et al., 2014)
	- object detection (Grishick,2015)
	- semantic segmentation (Long et al., 2015)
	- image captioning (Karpathy & Fei-Fei, 2015)
- need massive amount of labeled data
- main limitation: 
	-  intensive manual labeling effort (expensive-infeasible)
- results increase intrests in: 
	- self-supervised learning
- tested methods:
	- colorize gray scale images (Zhang et al., 2016)
	- predict relative postioin of image patches (larsson et al.
	2016)
	- predict egomotion between consecutive frames (Argrawal et al. 2015)
	- clustering based methods
	- reconstructoin based methods
	- learing gernative provabilistic models
- different things that were done:
	- give representation that is invariabnt to geometric and
	chromatic transformations
	- predict a camera transfomation in a video
- why only 4 rotation:
	- reasons that effect us:
		- no ambiguity
	- reasons that don't effect us:
		- scaling/translation would not work in normal images
		- no artifacts in 90 degree rotations
		- fast implimentation(transpose and flip)

