I got the MTMMC dataset (MTMMC: A Large-Scale Real-World Multi-Modal Camera Tracking Benchmark). The problem is that I have no idea on how to use it, the dataset structure be like:

```

MTMMC

|-kaist_mtmdc_train.json

|-kaist_mtmdc_val.json

|-train

||-train

|||-s01

||||-c01

|||||-gt

||||||-gt.txt

|||||-rgb

||||||-000000.jpg

||||||-000001.jpg

||||||-000002.jpg

...until c16

||||-c02

|||||-gt

||||||-gt.txt

|||||-rgb

||||||-000000.jpg

||||||-000001.jpg

||||||-000002.jpg

... until c16

|||-s10

||||-c01

|||||-gt

||||||-gt.txt

|||||-rgb

||||||-000000.jpg

||||||-000001.jpg

||||||-000002.jpg

...until c16

...

|-val

... Similar to the test

```



Here is some example of

gt.txt:

```

0000,063,971.20,135.70,21.3,79.1

0000,018,943.30,143.50,21.33,67.6

0000,011,1036.10,336.60,43.4,136.9

0000,051,1054.31,458.96,62.0,186.38

0000,022,1084.30,426.80,63.57,182.1

0000,067,864.80,456.10,71.6,180.2

0000,081,301.90,709.20,76.1,215.86

0000,029,966.60,616.40,86.8,225.8

0000,130,910.92,133.10,12.88,76.98

0001,063,970.88,135.70,21.2,79.1

0001,018,942.82,143.34,21.33,67.6

0001,011,1036.10,336.60,45.3,132.7

0001,051,1050.95,458.96,67.0,188.81

0001,022,1084.30,428.00,63.57,182.7

0001,067,863.10,456.10,66.5,180.2

0001,081,301.90,709.20,76.1,215.86

0001,029,964.10,616.40,88.6,225.8

0001,130,910.92,133.10,13.02,76.98

0002,063,970.56,135.70,21.1,79.1

0002,018,942.34,143.18,21.33,67.6

```

kaist_mtmdc_train.json:

```
{
    "annotations": [
       {
          "area": 1466.0622,
          "bbox": [
             871.25,
             263.0,
             22.41,
             65.42
          ],
          "category_id": 1,
          "frame_id": 0,
          "id": 0,
          "image_id": 0,
          "instance_id": 0,
          "iscrowd": 0,
          "video_id": 0
       }
    ],
    "categories": [
      {
         "id": 1,
         "name": "person",
         "supercategory": "person"
      },
      {
         "id": 2,
         "name": "background",
         "supercategory": "background"
      }
   ],
   "images": [
    {
        "camera_id": 0,
        "file_name": "val/s14/c01/rgb/000000.jpg",
        "frame_id": 0,
        "height": 1080,
        "id": 0,
        "scenario_id": 0,
        "video": "val/s14/c01",
        "video_id": 0,
        "width": 1920
     }
    ],
    "info": {
      "contributor": "kaist_rcv",
      "description": "kaist_mtmdc",
      "version": "1.0",
      "year": "2022"
   },
   "videos": [
    {
        "camera_id": 0,
        "frame_range": 7362,
        "height": 1080,
        "id": 0,
        "name": "val/s14/c01",
        "scenario_id": 0,
        "width": 1920
     }
    ]
}
```



For now, I know that:

```

Understanding and Utilizing the MTMMC Dataset for Multi-Target Multi-Camera Tracking

Multi-target multi-camera tracking (MTMC) stands as a pivotal task within the field of computer vision, holding substantial importance across a spectrum of applications including visual surveillance, the analysis of crowd behavior, and the detection of anomalous activities . The ability to accurately identify and track individuals as they move through environments monitored by multiple cameras over time is crucial for gaining a comprehensive understanding of dynamic scenes. To facilitate advancements in this domain, the Multi-Target Multi-Modal Camera Tracking Benchmark (MTMMC) dataset has been introduced as a significant resource . This large-scale, real-world benchmark aims to overcome the limitations of existing datasets, which often consist of synthetically generated data or recordings captured within controlled laboratory settings . The MTMMC dataset offers a more realistic and challenging platform for studying multi-camera tracking under diverse real-world complexities . It encompasses long video sequences captured by 16 multi-modal cameras across two distinct environments – a campus and a factory – and includes variations in time, weather, and season . Furthermore, the dataset provides an additional input modality through spatially aligned and temporally synchronized RGB and thermal cameras, which has been shown to enhance the accuracy of multi-camera tracking . With its large scale, encompassing 3,669 unique person identities across approximately 3 million frames, the MTMMC dataset presents a significant opportunity for advancing research in person detection, re-identification, and multiple object tracking . The user in this context has gained access to the MTMMC dataset but seeks guidance on its utilization, specifically regarding the methods for loading and processing this complex data. The provided dataset structure serves as the initial point of reference for addressing this challenge.   



The MTMMC dataset is organized in a hierarchical structure, with the top-level directory named MTMMC. Within this root directory, the user has identified two primary JSON files: kaist_mtmdc_train.json and kaist_mtmdc_val.json. These files serve as the main annotation repositories for the training and validation splits of the dataset, respectively. Alongside these JSON files are two crucial subdirectories: train and val. These directories contain the actual image data and further annotations organized by scene and camera. Specifically, within the train and val folders, the data is structured into subdirectories denoted by sXX, where XX represents a scene or sequence identifier (e.g., s01, s10). Each scene directory further contains subdirectories named cYY, where YY indicates a specific camera identifier within that scene (e.g., c01, c16). This nested structure effectively separates the data captured from different cameras and across various recording scenarios, reflecting the multi-camera nature of the dataset. Inside each camera directory (sXX/cYY/), two essential folders are present: gt and rgb. The rgb folder contains the sequence of individual video frames captured by that specific camera in the form of JPEG image files. These files are typically named with numerical sequences, such as 000000.jpg, 000001.jpg, 000002.jpg, and so forth, where the number likely corresponds to the frame number within the video sequence. The gt folder, located alongside the rgb folder, contains a text file named gt.txt. This file holds the ground truth annotations for the objects present in the corresponding video frames found in the rgb folder. The organization of the dataset into scenes and camera views, with synchronized RGB imagery and ground truth annotations, is fundamental for tasks that involve analyzing individual camera perspectives or understanding the relationships between different camera views within a multi-camera network.



The MTMMC dataset employs two primary formats for its annotations: the gt.txt file found within each camera's ground truth directory and the COCO-style JSON files at the top level of the dataset. Understanding the structure and content of each of these formats is crucial for effectively utilizing the dataset.



The gt.txt file, present in the gt folder of each camera view, provides frame-level annotations for the objects detected in the corresponding RGB video sequence. Each line within this comma-separated text file represents an annotation for a single object instance in a specific frame . Based on the example provided by the user and the conventions established in similar multi-object tracking datasets like those following the MOTChallenge format, the values in each line can be interpreted as follows: The first value, such as 0000, indicates the Frame ID, representing the specific frame number within the video sequence to which the annotation refers. This is likely a zero-based index. The second value, for instance 063, 018, or 011, represents the Object ID. This is a unique identifier assigned to each tracked object (presumably a person in this context) within the video sequence captured by a particular camera. This ID remains consistent across all frames where the same individual is visible, enabling the tracking of their movement within that camera's view. The subsequent four floating-point numbers define the bounding box of the object in the image frame. 971.20, 943.30, 1036.10, etc., represent the Bounding Box X-coordinate, which is the x-coordinate of the top-left corner of the bounding box in pixels. 135.70, 143.50, 336.60, etc., denote the Bounding Box Y-coordinate, the y-coordinate of the top-left corner. 21.3, 21.33, 43.4, etc., specify the Bounding Box Width in pixels, and 79.1, 67.6, 136.9, etc., represent the Bounding Box Height in pixels. This gt.txt format provides essential information for tasks such as evaluating the performance of single-camera tracking algorithms. The unique object IDs allow for the assessment of how well an algorithm can maintain the identity of individuals across frames within a single camera's perspective.   

```
## ENV per scenario
Factory: s01, s10, s11, s13, s16, s18, s20. Use only c09, c12, c13, c16
Campus: s35, s36, s38, s39, s42, s47. Use only c01, c02, c03, c05