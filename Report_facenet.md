# FaceNet: A Unified Embedding for Face Recognition and Clustering
## Principles
-  FaceNet is a face recognition pipeline that learns mapping from faces to a position in a multidimensional space where the distance between points directly correspond to a measure of face similarity
- FaceNet strives for an embedding `f(x)`, from an image x into a feature space, such that the **squared distance between all faces**, independent of imaging conditions, of the same identity is **small**, whereas the squared distance between a pair of face images from different identities is large
- Building on the previous work on FaceNet, our solution is formulated in three stages:
    1. **Pre-processing** — a method used to take a set of images and convert them all to a uniform format — in our case, a square image containing just a person’s face.
    2. **Embedding** — a process, fundamental to the way FaceNet works, which learns representations of faces in a multidimensional space where distance corresponds to a measure of face similarity.
    3. **Classification** — the final step which uses information given by the embedding process to separate distinct faces.

- Deep Convolutional Networks:

|layer|size-in|size-out|kernel|param|FLPS|
|-----|-------|--------|------|-----|-----|
|conv1|220×220×3| 110×110×64| 7×7×3, 2| 9K |115M
|pool1| 110×110×64| 55×55×64| 3×3×64, 2| 0|
|rnorm1| 55×55×64| 55×55×64| |0|
|conv2a| 55×55×64| 55×55×64| 1×1×64, 1| 4K| 13M
|conv2| 55×55×64| 55×55×192| 3×3×64, 1| 111K |335M
|rnorm2| 55×55×192| 55×55×192| |0|
|pool2| 55×55×192| 28×28×192| 3×3×192, 2| 0|
|conv3a| 28×28×192| 28×28×192| 1×1×192, 1| 37K |29M
|conv3| 28×28×192| 28×28×384| 3×3×192, 1| 664K |521M
|pool3| 28×28×384| 14×14×384| 3×3×384, 2| 0|
|conv4a| 14×14×384| 14×14×384| 1×1×384, 1| 148K |29M
|conv4| 14×14×384| 14×14×256| 3×3×384, 1| 885K| 173M
|conv5a| 14×14×256| 14×14×256| 1×1×256, 1| 66K |13M
|conv5| 14×14×256| 14×14×256| 3×3×256, 1| 590K |116M
|conv6a| 14×14×256| 14×14×256| 1×1×256, 1 |66K |13M
|conv6| 14×14×256| 14×14×256| 3×3×256, 1| 590K |116M
|pool4| 14×14×256| 7×7×256| 3×3×256, 2 |0|
|concat| 7×7×256| 7×7×256 ||0|
|fc1| 7×7×256| 1×32×128| maxout p=2 |103M |103M
|fc2| 1×32×128| 1×32×128| maxout p=2 |34M |34M
|fc7|128| 1×32×128| 1×1×128 524K| 0.5M|
|L2| 1×1×128| 1×1×128|| 0|
|total| | ||140M| 1.6B

## Pros and cons of FaceNet
- Advantage: In the traditional methods of face extraction, face extraction was designed by computer vision researchers who expertise in computer vision methods. However, because of the limited amount of dataset these researchers can access, those traditional methods have their own disadvantages when researchers cannot cover all features to classify correctly the dataset. So FaceNet can utilize the dataset which was originally trained on to generate more efficient feature extraction method. After grabbing the feature extraction vectors, we can utilize Support Vector Machines to classify the label of each face

- Disadvantage: Speed of running is still inferior to traditional methods of face extraction, but it obtains more superior performance than its variants.