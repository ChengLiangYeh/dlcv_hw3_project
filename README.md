# Deep Learning for Computer Vision HW3
## Introduction
In HW3, we will use CelebA datasets of human face and digit images to implement the models of both VAE and GAN for generating human face images, and the model of DANN for classifying digit images from different domains.

<p align="center">
  <img width="853" height="500" src="http://mmlab.ie.cuhk.edu.hk/projects/CelebA/intro.png">
</p>

Datasets Citation:
@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}

## HW3 Project 1 ― VAE
A variational autoencoder (VAE) is a type of neural network that learns to reproduce its input, and also map data to latent space. it is worth noting that a VAE model can generate samples by sampling from the latent space.

In this project, done list:
- Implement a VAE model and train it on a face dataset. 
- Compare the original images and reconstructed images from VAE model.(calculate KL divergence and mean square error) 
- Randomly generate 32 images by sampling latent vectors from an normal distribution.
- Visualize the latent space by mapping the latent vectors of the test images to 2D space (by tSNE) and color 
them with respect to an attribute (e.g., gender and hair color). 

## HW3 Project 2 ― GAN
A generative adversarial network (GAN) is a deep learning method in which two neural networks (generator and discriminator) compete with each other to become 
more accurate in their predictions.

In this project, done list:
- Build and train a Generator and a Discriminator in GAN model.
- Samples 32 noise vectors from normal distribution and input them into a Generator to generate 32 images.

## HW3 Project 3 ― DANN
In this project, done list:
- Implement DANN on digits datasets (USPS, MNIST-M and SVHN) and consider the following 3 scenarios: 
	
	(Source domain → Target domain)
	
	1. USPS → MNIST-M

	2. MNIST-M → SVHN

	3. SVHN → USPS
- Compute the accuracy on target domain, while the model is trained on source domain only. (lower bound)
- Compute the accuracy on target domain, while the model is trained on source and target domain. (domain adaptation)
- Compute the accuracy on target domain, while the model is trained on target domain only. (upper bound)
- Visualize the latent space by mapping the testing images to 2D space (with t-SNE) and use different colors to indicate data of (a) different digit classes 
0-9 and (b) different domains.

## HW3 Project 4 ― Improved UDA model
Implement improved model on digits datasets(USPS, MNIST-M and SVHN) for unsupervised domain adaptation.

## Dataset
### A subset of human face dataset: 

- size = 64 * 64 * 3

- 40000 training images

- 13 out for 40 attributes.

### Digits datasets:

USPS: 
- 7291/2007 (training set/testing)
- classes = 0~9
- size = 28 * 28 * 1

MNIST-M:
- 60000/10000 (training set/testing)
- classes = 0~9
- size = 28 * 28 * 3

SVHN:
- 73257/26032 (training set/testing)
- classes = 0~9
- size = 28 * 28 * 3


Contact me for Dataset or [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html]). 

Email: chengliang.yeh@gmail.com

## Models
- VAE(variational autoencoder)
![1](./pic/VAE.png)

- GAN(generative adversarial network)
![1](./pic/GAN.png)

## Evaluation
To evaluate your UDA models in Problems 3 and 4, you can run the evaluation script provided in the starter code by using the following command.

    python3 hw3_eval.py $1 $2

 - `$1` is the path to your predicted results (e.g. `hw3_data/digits/mnistm/test_pred.csv`)
 - `$2` is the path to the ground truth (e.g. `hw3_data/digits/mnistm/test.csv`)

Note that for `hw3_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.

| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

# Submission Rules
### Deadline
109/12/01 (Tue.) 02:00 AM (GMT+8)

### Late Submission Policy
You have a three-day delay quota for the whole semester. Once you have exceeded your quota, the credit of any late submission will be deducted by 30% each day.

Note that while it is possible to continue your work in this repository after the deadline, **we will by default grade your last commit before the deadline** specified above. If you wish to use your quota or submit an earlier version of your repository, please contact the TAs and let them know which commit to grade.

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

### Submission Format
Aside from your own Python scripts and model files, you should make sure that your submission includes *at least* the following files in the root directory of this repository:
 1.   `hw3_<StudentID>.pdf`  
The report of your homework assignment. Refer to the "*Grading*" section in the slides for what you should include in the report. Note that you should replace `<StudentID>` with your student ID, **NOT** your GitHub username.
 2.   `hw3_p1.sh`  
The shell script file for running your VAE model. This script takes as input a path and should output your generated image in problem 1-4 in the given path.
 3.   `hw3_p2.sh`  
The shell script file for running your GAN model. This script takes as input a path and should output your generated image in problem 2-2 in the given path.
 4.   `hw3_p3.sh`  
The shell script file for running your DANN model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.
 5.   `hw3_p4.sh`  
The shell script file for running your improved UDA model. This script takes as input a folder containing testing images and a string indicating the target domain, and should output the predicted results in a `.csv` file.

We will run your code in the following manner:

    bash ./hw3_p1.sh $1
    bash ./hw3_p2.sh $1
    bash ./hw3_p3.sh $2 $3 $4
    bash ./hw3_p4.sh $2 $3 $4

-   `$1` is the path to your output generated images (Problem 1-4 and 2-2) (e.g. hw3/VAE/fig1_4.png or hw3/GAN/fig2_2.png ).
-   `$2` is the directory of testing images in the **target** domain (e.g. `hw3_data/digits/mnistm/test`).
-   `$3` is a string that indicates the name of the target domain, which will be either `mnistm`, `usps` or `svhn`. 
	- Note that you should run the model whose *target* domain corresponds with `$3`. For example, when `$3` is `mnistm`, you should make your prediction using your "USPS→MNIST-M" model, **NOT** your "MNIST-M→SVHN" model.
-   `$4` is the path to your output prediction file (e.g. `hw3_data/digits/mnistm/test_pred.csv`).

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to the requirments.txt for more details.

You can run the following command to install all the packages listed in the requirements.txt:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

### Remarks
- If your model is larger than GitHub’s maximum capacity (100MB), you can upload your model to another cloud service (e.g. Dropbox). However, your shell script files should be able to download the model automatically. For a tutorial on how to do this using Dropbox, please click [this link](https://goo.gl/XvCaLR).
- **DO NOT** hard code any path in your file or script, and the execution time of your testing code should not exceed an allowed maximum of **10 minutes**.
- **Please refer to HW3 slides for details about the penalty that may incur if we fail to run your code or reproduce your results.**

# Q&A
If you have any problems related to HW3, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under hw3 FAQ section in FB group
