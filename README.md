. Single Shot Detector (SSD) 
7.1 Introduction 
In recent times, there is more emphasis on research in a field of object detection. Object detection is quite different than image detection as we can predict the location of the object in an image in object detection. The current state of art uses various methods for object detection and improved classification. Recent research is based on Faster R-CNN architecture which comprises high computational cost which is not feasible option of embedded systems. For Faster RCNN we need to have two shot detection, one for generating region proposals and another for object detection. Faster R-CNN has 7 FPS (frames per second). 
7.2 Concept: - 
Earlier object detection designs used two independent stages: a classifier to identify the sorts of items in the suggested regions, and a region proposal network to accomplish object localisation. These can be exceedingly expensive to compute, making them unsuitable for realworld, time-sensitive applications. Single-shot models enable the deployment of less hardware while encapsulating both localization and detection duties in a single forward sweep of the network. 
SSD only need images and ground truth boxes as an input, and it will detect the object with confidence score for classification. SSD uses the series of convolution to learn different features through an input. 
  
Fig 7.1:- Object detection principle of SSD [16] 
SSD divide the image into number of symmetrical grids and feature maps.In convulution network we will evaluate different types of aspect ratio in different scales at each of the loaction on grids.The different scales are used to map the object with different size and shapes (e.g. 16×16, 8×8).for each of the default boxes we will calculate the confidence score and location coordinate for each default boxes.The boxes which have highest probality in confidence of having certain class is examined by using non max suppresion.The samples below certain confidence level is referred as negatives and rest are positives.The model is then improved by using the gradient decent algorithm.  
Input parameters and Bounding boxes:- 
The SSD model has two types i.e. SSD-300, SSD-512.The suffixes just shows the input image size for SSD architecture.As in a project we are using SSD 300, we have imput size of 300×300. Bounding boxes represent the postion of the object in 2D plane with the labels.The bounding boxes can be represnted in different formats e.g. PASCAL VOC,YOLO.  
In pascal voc format the boundary coordinates is represented by using X-Y coordinates (X_min,Y_min,X_max,Y_max).In YOLO format the boundary coordinates is represented by centre parameter  and size of an an image (Cx,Cy,w,h). 
 
Intersection Over Union:- 
The term IOU is used to describe the amount of overlap between two boxes.It is also used to express the accuracy of a detector and also called as Jaccard Index. If the bounding box aligns with object perfectly then IOU will be one and vice versa. IOU more than 0.90 will be considered perfect in practical. 
 
 
7.3 SSD Model :-  
SSD is purely convolutional network which we converted into three parts of convolutions i.e. Base convolution, Auxillary convolution and prediction convolution.Base convolution are used to extract low level features maps by using standerd convolution layer.Auxillary convolution are used extraxt the higher features maps which are added top of the first convolution where prediction convolution used to identify and locate the object. 
 
Base convolution:- 
VGG-16 architecture is used as base convolution which works well for lower level features extraction.As name suggest VGG 16 consist 16 convolutional layers for the feature extraction.It consist of 13 convolutional layers and 3 flatten layers.As for SSD, apart from the SSD developed by standerd research paper we converted last 3 fully conncted layers into convolutional layers. 
In VGG 16 we are using 3×3 kernal for feature extraction with padding of 1.To reduce the non linearity we are using activation function RELU. To adjust the dimenstion of the image we are using maxpooling with stride of 2. In 3rd pooling layer we are using celling function to avoid any float value of channel (75/2 = 32.5). Any fully connected layer can be converted to an equivalent convolutional layer simply by reshaping its parameters [17]. We are converting last 3 fully connected layers into convolutional layers.  
 
Auxiliary Convolution: - 
Auxiliary convolution will give additional high-level feature and it staked above the base convolution layer. The feature map cumulatively smaller than the last feature map. We are using 4 layers of convolutional layer with 3×3 kernel. 
 
Priors and Aspect Ratio: 
Before creating the prediction convolution, it is very important to understand about the different prediction boxes and its aspect ratio. Object detection is very diverse that it can be at any shape at any position with infinite number of possibilities. We link multiple default boxes for each feature map on top of base network. Each feature map will calculate 4 offset distance with confidence score of the class. In our project we have three classes i.e., Car, Number Plate, background. 
 
In Defining the Priors authors recommend the following arguments: 
•	The Priors should be applied to low level feature and high-level feature. Priors are manually generated, although they are carefully selected based on the dimensions and shapes of the ground truth objects in our dataset. We also take into consideration variation in position by inserting these priors at each site conceivable in a feature map. [16] 
•	If the feature map has a scale of ‘s’ its area is ‘ s² ’.The largest feature map has a scale of 0.1 means it occupies the 10% size of an image. The scale of default boxes is calculated by following formula, 
 
 
Where K~(1:M) and minimum scale (Smin) is 0.2 and maximum scale(Smax) is 0.9 and all the layers are equally spaced from each other. [16] 
 
•	Priors will be in the ratios of 1:1, 2:1, and 1:2 for all feature maps. Conv7, Conv8_2, and Conv9_2 intermediate feature maps will likewise have priors with the ratios 3:1 and 1:3. All feature maps will also have an additional previous with a 1:1 aspect ratio and a scale equal to the geometric mean of the scales of the current and following feature maps. [16] 
 
 
 
Visualizing the default boxes: 
From defined scales and aspect ratio we can calculate dimension of default boxes. 
W = S×  𝒓𝒂𝒕𝒊𝒐 
𝑺
H = 
 
 ![image](https://user-images.githubusercontent.com/91695139/198104916-051d155e-59d2-46f8-a4e0-d97678bd56de.png)

 
 
 
 
 
Fig 7.7:- Conceptual Representation of default boxes on convolution 9_2 with 0.55 aspect ratio 
Prediction Convolution: -  
At each location for each default box and for every feature map we want to predict Cx, Cy,w,h (Refer Fig 2. Yolo format) and the confidence of score of each classes for each bounding box. 
The prediction convolution layer predicts 8732 boxes of for each location with the confidence score of classes. Most of prediction boxes are corresponds to object class 2 termed as negative classes. We are going to separate these classes before training. 
M×N×Number of channel  	 	 	 Offsets  	 	 	    Car     plate       
No object 

 	 	 	 
 	 	 	 
 	 	 	 
 	 	 	 
	 
Matching the prediction: - 
Each prior is used as a rough starting point, and we then determine how much adjustment is necessary to get a more precise prediction for a bounding box. We therefore need a method to assess or quantify it if each anticipated bounding box deviates somewhat from a previous and our objective is to calculate this deviation. 
 Blue box represents the ground truth box where red box shows the first prediction. So now we must find the deviation from the actual boxes and then adjust our weights to learn new weights. 
 
Fig 7.7:- Test Figure [Carissma Dataset] 
7.4 Training steps: - 
•	Discover the Jaccard overlaps between the N ground truth objects and the 8732 priors. This will be an 8732-by-N tensor. Each of the 8732 priors should be matched with the object that it has the most overlap with. 
•	A prior is a negative match if it is matched with an object that has a Jaccard overlap of less than 0.5 since it cannot be considered positive match. Since we have thousands of priors, most of them will test false for an object. 
•	A few priors, on the other hand, will actually strongly overlap (more than 0.5) with an item considered as positives.They are successful matches. 
•	Now we have matched 8732 prior boxes with default boxes and we now have a boxes which contains an object.So now we can move forward and train the data which actually contain the object. 
•	So we are learning the positive boxes with his classes by regressing over the boundingboxes. 
7.5 Compute Losses: 
Loss is computed by regreesing the positively matched boxes with corresponding ground truth boxes.So we are matching the offsets (cx,cy,w,h) with GT boxes coordinates and hence we computing the losses. 
The loss is averaged smooth L1 loss between the offsets and GT coordinates. 
𝟏
𝑳𝒐𝒔𝒔 =   (∑
𝑵𝒑𝒐𝒔𝒕𝒊𝒗𝒆𝒔	𝑳𝟏 𝑳𝒐𝒔𝒔) 
𝑷𝒐𝒔𝒕𝒊𝒗𝒆𝒔
  
 
 

Priors	GT1	GT2	GT3
Bbox 1	0	0.8	0
Bbox2	0	0	0
Bbox3	0.2	0	0.1
Bbox 4	0	0	0.56
	 
	Fig 7.8 :- IOU Calculation  	 	                      Fig 7.8:- Class identification 
 
Positive-  	It has an IOU greater than 0.5 with class selection (Car, Numberplate) Negative-  	It has an IOU less than 0.5. 
Background- 	No object 
 
 
