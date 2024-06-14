# 2D image Morphing

#### i. meshMorph1.py

- User interface for mesh morphing between two human faces
- Basic operation and process animation
- Triangle-based mesh morphing

<img src="/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/meshMorph_2-8249746.png" alt="meshMorph_2" style="zoom:33%;" />

<img src="/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/meshMorph_1.png" alt="meshMorph_1" style="zoom:33%;" />





#### ii. meshMorph2.py  //failed

- Mesh morphing within rectangles
- User-adjustable mesh points

<img src="/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/meshMorph2_1-8250399.png" alt="meshMorph2_1" style="zoom:40%;" />

==> better if mesh be more dense and have proper rectangles mapping



#### iii. fieldMorph.py

- UI for field morphing  
- warping from user-defined feature lines

![Screenshot 2024-06-13 at 11.30.02](/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/Screenshot 2024-06-13 at 11.30.02.png)

==> better if more feature lines 



#### iv. faceChange.py

> examele : python faceChange.py img1.jpg dance.mp4
>
> -->the program will ouput dance_output.mp4

- Replace faces in a video with a given face image

- Output the modified video
- Supports only low-resolution face images
- Mesh warping technique

![img1](/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/img1.png)  +   <img src="/Users/johnsonhsu1999/Desktop/mdNotes/the_photos/faceChange_2.png" alt="faceChange_2" style="zoom:16%;" />

=  <img src="/Users/johnsonhsu1999/Desktop/CGfinal(new)/faceChange_1.png" alt="faceChange_1" style="zoom: 30%;" />



**Ref** : T. Beier and S. Neely, "Feature-based image metamorphosis," *Computer Graphics*, vol. 26, no. 2, pp. 1-10, 1992.

**Img src** : https://youtu.be/lJZMUQp8EKo?si=ZIPFPFLQkFwybzcw
