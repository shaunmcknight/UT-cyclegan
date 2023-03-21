# UT cyclegan
A image-to-image generative model which introduces task specific modifications to CycleGAN, including an additional loss function, to learn the mapping from physics-based simulations of defect indications to experimental indications in resulting ultrasound images. The resulting synthetic dataset resulted in an improved F1 classification score on experimental ultrasonic images of 0.45 (from 0.394 to 0.843), when training a CNN classifier on simulated and GAN generated data respectively.


Example simulated defect:

![image](https://user-images.githubusercontent.com/71640417/223427399-f2470a2e-a70b-4074-8b8e-963c278bccfb.png)


Example experimental response:

![image](https://user-images.githubusercontent.com/71640417/223427454-39b883a9-fa3c-454f-b102-016a923d0b37.png)



Introduction of a mid activation map loss to encourage accurate defect reconstruction.

![image](https://user-images.githubusercontent.com/71640417/223427211-148c4e1f-77ac-457b-9bdf-5d113823c19f.png)



Example of GAN generated responses and respective simulated response:

![image](https://user-images.githubusercontent.com/71640417/226600685-9bd91f48-129e-46d2-84c4-9b922a9ffbda.png)

