## <span id="page-0-0"></span>Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations

Shivansh Patel<sup>1</sup> Shraddhaa Mohan<sup>1</sup> Hanlin Mai<sup>1</sup><br>Unnat Jain<sup>2\*</sup> Svetlana Lazebnik<sup>1\*</sup> Yunzhu Li<sup>3\*</sup> Svetlana Lazebnik<sup>1</sup>\*

<sup>1</sup>UIUC <sup>2</sup>UC Irvine <sup>3</sup>Columbia University

<https://rigvid-robot.github.io/>

## Abstract

*This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks—such as pouring, wiping, and mixing—purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive realworld evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation.*

#### 1. Introduction

Videos offer a rich and expressive source of training data for robotic manipulation, and numerous methods have successfully leveraged them for supervision. Such methods typically fall into two categories: (1) Learning from publicly available large-scale datasets of real-world videos [\[9,](#page-10-0) [13,](#page-10-1) [22,](#page-10-2) [36,](#page-11-0) [106,](#page-14-0) [128\]](#page-15-0), and (2) Imitation of specific demonstrations captured under controlled conditions that closely

match the execution setting [\[8,](#page-10-3) [21,](#page-10-4) [55,](#page-12-0) [65,](#page-12-1) [70,](#page-12-2) [114\]](#page-14-1). Unfortunately, both of these strategies come with challenges that limit broad deployment. Large-scale video datasets often introduce domain gaps [\[36,](#page-11-0) [122,](#page-14-2) [137\]](#page-15-1) and require adaptation to specific robot embodiments and tasks [\[9,](#page-10-0) [86\]](#page-13-0). On the other hand, video-based imitation involves laborious data collection that must ensure close alignment in viewpoints, morphologies, and interaction modalities [\[7,](#page-10-5) [8,](#page-10-3) [26,](#page-11-1) [106\]](#page-14-0).

Motivated by recent advances in video generation, we explore a potentially new paradigm: can a single generated video, synthesized to exactly match our input environment and task description, be used as the sole source of supervision for robotic manipulation? Recently released models like SORA [\[16\]](#page-10-6) and Kling [\[1\]](#page-10-7) have demonstrated impressive capabilities in producing realistic-seeming videos from language and image inputs. At the same time, it has been shown that such videos can suffer from distorted object geometries [\[74,](#page-12-3) [132\]](#page-15-2), physically implausible interactions [\[82,](#page-13-1) [127\]](#page-15-3), and unrealistic scene dynamics [\[11,](#page-10-8) [39\]](#page-11-2). Consequently, while the idea of synthesizing video demonstrations is enticing, its usefulness in the robotics setting is yet to be convincingly established. Prior work incorporating video generation into robotics typically relies on additional supervision, such as task-specific training [\[30\]](#page-11-3) or fine-tuning on offline robot trajectory datasets [\[14,](#page-10-9) [15\]](#page-10-10). By contrast, we ask whether a robot can perform real-world manipulation tasks solely by imitating generated videos *without any additional supervision or task-specific training*.

To this end, we introduce Robots Imitating Generated Videos (RIGVid), a framework that connects video generation models to real-world robotic execution. Fig. [1](#page-1-0) gives an oveview of the method. Given an input RGB-D image of the scene and a free-form language command (e.g., "pour water on the plant"), we use a state-of-the-art video diffusion model to generate a candidate video of the task being performed. The generated video is not guaranteed to accurately

<sup>\*</sup>denotes equal advising.

<span id="page-1-1"></span><span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1. RIGVid overview. Given an initial scene image and depth, we generate a video conditioned on a language command. A VLMbased automatic filtering step (not shown) can be used to reject videos that fail to follow the prompt. A monocular depth estimator recovers depth for each frame of the generated video, and these depth maps are combined with the corresponding RGB frames to produce 6D Object Pose Trajectory. After grasping, the trajectory is retargeted to the robot for execution.

follow the language command – but we show that a VLM can be used to automatically filter out unsuccessful generations with high accuracy. Next, we estimate per-frame depth on the video, segment the manipulated object, and track its *6D object pose trajectory* across the frames using the FoundationPose tracker [\[120\]](#page-14-3). While this tracker relies on a precomputed object mesh, preliminary experiments (App. [C\)](#page-16-0) indicate that our method is compatible with mesh-free approaches, though their inference speed is currently infeasible for real-time deployment. Finally, the extracted 6D object pose trajectory is retargeted to the robot for execution. The retargeting process only requires the transformation between the end-effector and the object, so it can be easily applied across platforms. During deployment, RIGVid performs real-time object tracking and dynamically adjusts the robot's actions to handle disturbances and execution-time variations, promoting robust and adaptive behavior.

We evaluate RIGVid on four real-world manipulation tasks: pouring water, lifting a lid, placing a spatula on a pan, and sweeping trash. These tasks span diverse manipulation challenges, including a range of depth variation (minimal in pouring vs. significant in lifting), thin and partially occluded objects (spatula, sweeping brush), and diverse object geometries and actions. Our results show that, when paired with our filtering mechanism, generated videos are as effective as human videos for visual imitation, enabling robots to act entirely from synthetic supervision. Moreover, the performance of RIGVid improves with video quality, suggesting a favorable trend where advances in generative models directly translate to stronger manipulation capabilities.

The main downside of video generation is its substantial computational cost. Also, on a representational level, one may wonder whether predicting video pixels is wasteful, and whether we should instead predict a more compact and minimal representation that can be efficiently translated to an executable trajectory. One example of this philosophy is the recent ReKep method [\[49\]](#page-11-4), which uses a VLM to generate relational keypoint constraints from a task description and then solves for a 6D trajectory given these constraints. We compare our approach to ReKep and demonstrate that video generation does, in fact, perform substantially better than the generation of a more sparse and high-level representation. Next, given a generated video, one may ask whether 6D object-level tracking is necessary, given its upfront requirement of an object mesh. To address this question, we compare against a broad range of alternative tracking approaches — sparse point tracking [\[15\]](#page-10-10), dense optical flow [\[60\]](#page-12-4), 3D feature fields [\[58\]](#page-12-5), and generated goal supervision [\[14\]](#page-10-9) — and show consistently higher success rates.

In summary, our key contributions are: (1) We propose a framework that enables robots to perform open-world manipulation tasks without any real-world demonstrations only generated videos. (2) We show high-quality generated videos perform on par with real videos for robotic imitation, establishing that synthetic data can serve as an effective substitute for real data in this domain. (3) We demonstrate that our combination of video generation and 6D trajectory extraction outperforms a wide variety of competing stateof-the-art methods based on VLMs, point tracking, optical flow, feature fields, and generated-goal supervision.

## 2. Related Work

Direct Imitation from Videos. This seeks to match visual states in demonstration videos to those of the learner, without requiring expert action labels or robot state information [\[8,](#page-10-3) [26,](#page-11-1) [34,](#page-11-5) [46,](#page-11-6) [55,](#page-12-0) [58,](#page-12-5) [93,](#page-13-2) [103,](#page-14-4) [104,](#page-14-5) [106,](#page-14-0) [112,](#page-14-6) [114,](#page-14-1) [115,](#page-14-7) [124,](#page-14-8) [129\]](#page-15-4). While effective, this approach demands paired demonstrations in the same setting. A common strategy is to leverage visual correspondences—tracks [\[15\]](#page-10-10) or optical flow [\[5,](#page-10-11) [35,](#page-11-7) [125\]](#page-14-9)—to infer object trajectories. Bharadhwaj *et al*. [\[15\]](#page-10-10) predicts object tracks and uses PnP to recover poses for closed-loop task execution. Dense descriptor learning [\[33,](#page-11-8) [113,](#page-14-10) [138\]](#page-15-5) has proven powerful for handling variations in object geometry and appearance. Kerr <span id="page-2-1"></span>*et al*. [\[58\]](#page-12-5) recover object part trajectories from monocular videos using feature fields. Crucially, these methods rely on demonstrations collected under conditions closely matching the target task. In contrast, our method removes this requirement by generating task and scene-conditioned videos.

Imitation from Offline Videos. This paradigm alleviates the need for paired demonstrations by leveraging offline video data, and has consequently attracted significant attention [\[12,](#page-10-12) [22,](#page-10-2) [32,](#page-11-9) [60,](#page-12-4) [77,](#page-13-3) [80,](#page-13-4) [90,](#page-13-5) [91,](#page-13-6) [96,](#page-13-7) [100–](#page-13-8) [102,](#page-14-11) [104,](#page-14-5) [107,](#page-14-12) [114,](#page-14-1) [131,](#page-15-6) [136\]](#page-15-7). Many works focus on learning affordance models from internet-scale video datasets [\[9,](#page-10-0) [10,](#page-10-13) [22,](#page-10-2) [27,](#page-11-10) [52,](#page-11-11) [54,](#page-12-6) [68,](#page-12-7) [69,](#page-12-8) [81,](#page-13-9) [90,](#page-13-5) [96,](#page-13-7) [100,](#page-13-8) [108,](#page-14-13) [130\]](#page-15-8). Here, affordances are defined as scene-centric predictions of where and how an agent can interact, often captured as contact-point heatmaps and short motion trajectories that can be translated into robot actions. For example, Bahl *et al*. [\[9\]](#page-10-0) learn from large-scale human videos to output dense contact maps and trajectory waypoints, which downstream imitation, exploration, or reinforcement modules can transform into executable robot motions. However, these methods suffer from domain gap between training videos and task-specific environments, and require additional mechanisms to obtain task-conditioned affordances. In contrast, our method does not explicitly predict affordances, but instead relies on generated, task- and scene-specific generated videos for imitation.

Video Generation for Robotics. Video generation has emerged as a promising avenue for robotics [\[3,](#page-10-14) [4,](#page-10-15) [14,](#page-10-9) [29,](#page-11-12) [30,](#page-11-3) [72,](#page-12-9) [72,](#page-12-9) [126,](#page-15-9) [135\]](#page-15-10). A common limitation of these is their reliance on robot data, either to train the video generation model [\[72,](#page-12-9) [110\]](#page-14-14), or to train policies [\[14\]](#page-10-9), or both [\[3,](#page-10-14) [29,](#page-11-12) [30\]](#page-11-3). Bharadhwaj *et al*. [\[14\]](#page-10-9) leverages tracks on generated videos to condition policy learning. Albaba *et al*. [\[4\]](#page-10-15) uses generated videos to compute rewards for RL training. The most closely related work is Liang *et al*. [\[72\]](#page-12-9), which executes robotic tasks by tracking tools attached to the robot's end effector. While effective, their method relies on 1,822 human-collected robot demonstrations for four tasks, and is confined to tasks executable only by tools. In contrast, our approach requires no such data collection. Instead of tools, our method tracks objects, enabling a significantly broader range of manipulation tasks without using any robot data.

6D Pose Estimation and Tracking. Instance-level object pose tracking methods fall into two main categories: modelbased and model-free. Model-based approaches [\[19,](#page-10-16) [43,](#page-11-13) [44,](#page-11-14) [62,](#page-12-10) [63,](#page-12-11) [84,](#page-13-10) [88,](#page-13-11) [105\]](#page-14-15) require a 3D CAD model and typically estimate pose by constructing 2D-3D correspondences and solving the PnP problem  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$  $[66, 88, 111, 117, 118]$ . In contrast, model-free methods [\[17,](#page-10-17) [42,](#page-11-15) [45,](#page-11-16) [67,](#page-12-13) [78,](#page-13-12) [89,](#page-13-13) [109\]](#page-14-19) rely on multiple reference images instead of an explicit 3D mesh. Recent advances in vision foundation models and large datasets have enabled zero-shot methods [\[6,](#page-10-18) [19,](#page-10-16) [63,](#page-12-11) [76,](#page-12-14) [87\]](#page-13-14), which extend to unseen objects and categories but still lag behind instance-level methods in performance. We employ FoundationPose [\[120\]](#page-14-3), a versatile instance-level tracking method that supports model-based pose tracking. Notably, it does not require any instancespecific fine-tuning. Our choice is guided by its state-ofthe-art performance and real-time execution speed, both of which are crucial for ensuring robustness against disturbances during execution.

Motion Retargeting for Object Manipulation. Early work in learning from demonstration established the foundation for object-centric motion retargeting [\[18,](#page-10-19) [38,](#page-11-17) [51,](#page-11-18) [79,](#page-13-15) [85,](#page-13-16) [95\]](#page-13-17). More recently, deep learning-based retargeting methods have emerged [\[24,](#page-10-20) [25,](#page-10-21) [41\]](#page-11-19), with some incorporating object-centric representations to bridge the gap between the demonstrator and the robot [\[58,](#page-12-5) [70,](#page-12-2) [121\]](#page-14-20). Many approaches have applied retargeting to humanoid robots [\[47,](#page-11-20) [61,](#page-12-15) [73,](#page-12-16) [83,](#page-13-18) [94\]](#page-13-19). Other works have extended these techniques to dexterous manipulation [\[64,](#page-12-17) [97\]](#page-13-20). Like most prior work, we assume a fixed transformation between the end-effector and the object. While motion retargeting has traditionally relied on human demonstrations, RIGVid eliminates this dependency by leveraging generated videos.

## 3. Our Method: Robots Imitating Generated Videos

An overview of our method is shown in Fig. [1.](#page-1-0) It takes as inputs the initial scene RGB image, its corresponding depth map, and a free-form human command. Our goal is to predict the robot's 6DoF end-effector trajectory. This section describes the key steps of RIGVid: (1) Generate a scene and task-conditioned video and predict its corresponding depth using a monocular depth estimator (Sec. [3.1\)](#page-2-0); (2) Compute 6D pose rollout via an object pose tracker (Sec. [3.2\)](#page-3-0); (3) Grasp the object and retarget the pose trajectory to the robot, and execute the resulting trajectory (Sec. [3.3\)](#page-3-1).

#### <span id="page-2-0"></span>3.1. Generating Videos and Corresponding Depth

Since the generated videos may not necessarily follow the language command or have other issues, we need an automatic filtering mechanism to discard inaccurate generations. We found that we can do the filtering reliably by prompting a VLM – specifically, GPT-4o  $[2]$  – to assess whether the generated video depicts a successful execution of the command. As image input to GPT-4o, we sample four evenly spaced frames in the video and concatenate them vertically to create a video summary. The VLM determines whether the action described in the command is performed by a visible hand. App. [B](#page-16-1) provides the full prompt used for filtering and examples of video summaries with their corresponding VLM responses. If the response is negative, we regenerate the video and repeat the process for up to five attempts. If all attempts fail, we default to the final attempt.

<span id="page-3-3"></span>As input to the downstream tracking step, we also need to predict the depth for the generated video, using the predictor from Ke *et al*. [\[56\]](#page-12-18). One complication is that the estimated depth values are not grounded in real-world units, but subject to a scale and shift ambiguity [\[40\]](#page-11-21). Consistent with prior works adopting depth estimators in visionbased robotics [\[23,](#page-10-23) [37\]](#page-11-22), we compute an affine scale-andshift transformation, aligning the predicted depth in the first frame with the initial real depth map around the active ob-ject (discussed in Sec. [3.2\)](#page-3-0). This transformation is then applied to the entire predicted video to resolve the ambiguity.

#### <span id="page-3-0"></span>3.2. Identifying Active Object Mask and 6D Object Pose Trajectory

To extract 6D pose rollout, we first identify the active object—the one being manipulated in the generated video. A binary mask for this object in the initial RGB image is essential for object tracking and determining which object to grasp. Given the initial image and the task command, we prompt GPT-4o to identify the object most likely to be manipulated. We then ground the predicted object category into a bounding box using Grounding DINO [\[75\]](#page-12-19), and further refine this into a segmentation mask using SAM-2 [\[99\]](#page-13-21).

Once the active object is localized by the mask, we track it across the generated video using the scaled predicted depth. This yields the 6D pose rollout. Tracking objects in videos is a rich area of research, and we experimented with several 6D pose trackers [\[63,](#page-12-11) [119,](#page-14-21) [120\]](#page-14-3). For real-world deployment, we found FoundationPose [\[120\]](#page-14-3) to perform the best. It requires an object mesh, which we pre-compute using BundleSDF [\[119\]](#page-14-21). For this, we record a short RGBD video where the object is held and rotated in front of the camera to capture all sides. While straightforward, this process constrains our method to settings where a mesh can be precomputed. Nonetheless, as shown in App. [C,](#page-16-0) our method is also compatible with mesh-free approaches—BundleSDF can jointly reconstruct and track the object—but current inference speeds make these alternatives infeasible for realtime use. To ensure stable and realistic motion during execution, we apply an averaging filter to smooth abrupt pose changes, particularly in rotation. Additional details on this smoothing step are provided in App. [D.](#page-17-0)

#### <span id="page-3-1"></span>3.3. Object to Robot Motion Retargeting

Once the object trajectory is obtained, we first grasp the object. We use an off-the-shelf grasper, AnyGrasp [\[31\]](#page-11-23), to identify and execute the highest-scoring grasp within a defined boundary around the active object mask. After grasping, we retarget its trajectory to the robot's end-effector. Since the object remains firmly grasped, we assume a fixed transformation between the robot's end-effector and the object. This transformation is obtained by composing two rigid-body transforms: (1) the pose of the object relative

![](_page_3_Figure_6.jpeg)

Figure 2. Re-targeting RIGVid to a robot trajectory. Assuming a fixed transformation between the end-effector and the object after grasping, the 6D Object Pose Trajectory (*orange arrow*) is re-targeted to the robot (*blue arrow*). This formulation is embodiment agnostic and can be transferred to a different robot.

to the gripper at the moment it is grasped and (2) the offset between the gripper and the robot's end-effector. By combining these two components, we obtain a single transformation from the end-effector to the object.

The corresponding end-effector trajectory is obtained by applying the fixed end-effector-to-object transformation to the object's pose along the entire trajectory. This formulation ensures that the retargeted 6D pose rollout follows the object's motion while maintaining a stable grasp. These are executed on the physical robot, enabling it to reproduce the object's movement as observed in the generated video. A key strength of this approach is that it is robot-agnostic. Specifically, to accommodate a different robot or gripper, only the end-effector to the object transformation needs to be updated to reflect the new end-effector configuration.

![](_page_3_Picture_10.jpeg)

Figure 3. RIGVid is robust to perturbations. A human pushes the robot during execution (image 1), causing the object to deviate from the planned trajectory. When the deviation is detected (image 2), the robot backtracks to the last successfully executed trajectory point (image 3) and then resumes the planned motion (image 4).

#### <span id="page-3-2"></span>3.4. Closed Loop Execution

A core strength of our approach is its ability to operate in a closed-loop manner, enabling robust execution despite disturbances or unexpected changes during task execution.

![](_page_4_Picture_0.jpeg)

Figure 4. Evaluation tasks. We evaluate RIGVid on everyday manipulation tasks of varying difficulty.

During deployment, the system continuously tracks the object's 6D pose in real time using FoundationPose to update the robot's end-effector trajectory as the task progresses. This feedback allows the robot to dynamically adjust its motions: if the object deviates from the planned trajectory due to external perturbations, such as a human pushing the robot or a slip after grasping, the system detects the discrepancy by comparing the current object pose to the precomputed trajectory. If the detected deviation exceeds a threshold of 3 cm in position or 20 degrees in orientation, the robot backtracks to the last successfully executed trajectory point and resumes execution from there (Fig. [3\)](#page-3-2). This recovery mechanism enables RIGVid to maintain stable task execution, realigning and successfully completing the manipulation task despite perturbations. Additional examples of robustness to perturbations are provided in App. [H.](#page-18-0)

## 4. Experiments

This section presents our experimental evaluation. We describe the robot setup, manipulation tasks, and evaluation protocol (Sec. [4.1\)](#page-4-0). Then assess the impact of video generation models and filtering strategies on downstream robotic performance (Sec. [4.2\)](#page-4-1). Next, we compare RIGVid to SOTA VLM-based trajectory prediction method that allows zero-shot execution (Sec. [4.3\)](#page-6-0), and to alternative tracking approaches for trajectory extraction (Sec. [4.4\)](#page-6-1). Finally, we test generalization across embodiments, extensions to new tasks, and robustness to real-world disturbances (Sec. [4.5\)](#page-9-0).

#### <span id="page-4-0"></span>4.1. Robot Setup, Tasks, and Evaluation

We conduct experiments on an xArm7 robot arm with a stationary Orbbec Femto Bolt camera, positioned next to the robot to capture RGBD observations. We evaluate our method on four everyday manipulation tasks, which are illustrated in Fig. [4.](#page-4-2) These span a diverse range of robotic challenges, and their descriptions are as follows:

1. Pouring water requires the robot to position and tilt a watering can above a plant. While the depth of the can relative to the camera remains largely constant, successful execution demands a smooth trajectory spanning the pick-up, transport, and pouring phases. A trial is considered successful if the watering can's spout is positioned above the plant at the end of the execution.

- <span id="page-4-2"></span>2. Lifting a lid requires the robot to lift a pot lid. Unlike pouring, where the camera is viewing the scene from the side, the camera here is looking down towards the pot. As a result, this task involves significant variation in object depth, as the lid moves closer to the camera during execution. Success is achieved if the lid is no longer in contact with the pot at the end of the trial.
- 3. Placing a spatula on a pan requires the robot to place the head of a spatula into a pan. The spatula has a thin, elongated geometry and is often partially occluded during manipulation, which presents a challenge for object tracking, particularly for non-meshbased approaches. The task is considered successful if the spatula's head is in the pan at the end of execution.
- 4. Sweeping trash requires the robot to sweep trash into a dustpan. This task is especially challenging as it combines the need for precise positioning to align the head of the sweeping brush with the trash, along with all the challenges encountered from the placing task. A trial is successful if the trash is touching the base of the dustpan at the end of the execution.

Task success is determined via human judgment based on the above criteria, though the procedure could be readily automated with a VLM. The initial setup configuration is fixed across trials of the same task, and each trial has a different generated video. All baselines use the same videos for consistent comparison and reporting.

#### <span id="page-4-1"></span>4.2. Quality and Filtering of Generated Videos

As discussed in Sec. [3.1,](#page-2-0) we experimented with Sora, Kling v1.5, and Kling v1.6 for video generation. We also compared different video filtering strategies. Next, we present our key empirical findings.

*How do different video generation models compare for robotic imitation?* Sora is known for creating visually impressive and cinematic videos. Unfortunately, these videos often prioritize aesthetics over following the human command. As seen in the top row of Fig. [5,](#page-5-0) Sora frequently alters the camera viewpoint, changes object positions, or even swaps out objects mid-sequence. This lack of scene and object consistency makes Sora poorly suited for imitation. Kling v1.5 places more emphasis on following language instructions, generally preserves the original scene, and correctly depicts the target object. Nonetheless, it is still prone

<span id="page-5-2"></span>![](_page_5_Picture_0.jpeg)

Figure 5. Qualitative comparison of video generation for three models. Sora (top) drastically alters the scene layout and object size. Kling v1.5 (middle) does not fully follow the prompt (water not poured over the plant) and exhibits physically implausible behaviors (water pouring out of the top of the kettle but not the spout). Kling v1.6 (bottom) produces the most consistent and realistic result.

to physically implausible behaviors and command following failures. In the second row of Fig. [5,](#page-5-0) the teapot is not positioned over the plant and the water pours out from the top, not the spout (in other failure cases, nothing at all happens in the video, and the command is not even attempted). By contrast, Kling v1.6 (bottom row of Fig. [5\)](#page-5-0) has greatly improved command following and physical plausibility, proving to be the most reliable video generator for us. More examples of generated videos are shown in App. Fig. [21.](#page-22-0)

![](_page_5_Figure_3.jpeg)

<span id="page-5-1"></span>Figure 6. Filtering statistics. Kling V1.6 videos have the highest pass rate, demonstrating more accurate adherence to language commands.

*What are the filtering statistics for different video generation models?* Confirming the trends described above, Fig. [6](#page-5-1) reports the pass rates of each model across our four tasks for the GPT-4o filter described in Sec. [3.1.](#page-2-0) Sora fails all tasks 100% of the time. Kling v1.5 does better, successfully passing pouring 52.6% of the time, lifting 27.7%, placing 4%, and sweeping 2%. Kling V1.6 shows a substantial improvement across tasks, passing pouring 83%, lifting 66%, placing 55%, and sweeping 45% of the time. We noticed that, particularly for the placing and sweeping tasks, even Kling V1.6 frequently generated videos in which the command was not followed. In many cases, the video remained static, and no action was performed.

<span id="page-5-0"></span>*How accurate is VLM-based video filtering, and are there any simpler alternatives?* In Tab. [1,](#page-6-2) we report Pearson correlation coefficients between filtering metrics and human judgments of generation correctness. Our VLM-based filtering achieves strong agreement with human ratings across all tasks, with high correlation values. Most errors made by the VLM-based filter are false negatives—occasionally discarding usable videos, but almost never passing an incorrect one. We also explore the most relevant metrics for our case from a recent benchmark suite for evaluating video generation quality and instruction following, VBench++ [\[50\]](#page-11-24): (i) video-text consistency measuring the alignment between the command and the generated video  $[116]$ , and (ii) imageto-video (I2V) subject consistency which evaluates whether subjects present in the input image persist throughout the video [\[20\]](#page-10-24). These metrics correlate only weakly or inconsistently with task success and are not reliable for filtering.

*Does higher video quality lead to better robot performance?* To quantify this, Fig. [7](#page-6-3) plots RIGVid 's task success across five video sources: unfiltered Sora, unfiltered Kling v1.5, unfiltered Kling v1.6, filtered Kling v1.6, and real human demonstration videos. For each source, we use 10 videos per task. We observe a clear trend: as video quality improves, so does success rate. Sora's unfiltered videos lead to 0% success rate, Kling v1.5 performs better, and Kling v1.6 gives the highest results among all generated videos. Filtering dramatically improves reliability: after discarding failed generations using our automatic approach, success rate with filtered Kling v1.6 videos improves from 80% to 100% on pouring, from 60% to 80% on lifting, from 50% to 90% on placing, and from 20% to 70% on sweeping.

*Can generated videos replace real videos for imitation?* The results in Fig. [7](#page-6-3) indicate that, when using filtered Kling v1.6 videos, RIGVid 's performance is similar to that achieved with real human demonstration videos. This find-

| <b>Filtering Metrics</b>       | <b>Pour Water</b> | Lift Lid | <b>Place Spatula</b> | Sweep Trash | Average |
|--------------------------------|-------------------|----------|----------------------|-------------|---------|
| <b>Video-text Consistency</b>  | 0.06              | 0.47     | 0.70                 | 0.11        | 0.34    |
| <b>I2V Subject Consistency</b> | 0.35              | 0.58     | $-0.09$              | 0.63        | 0.37    |
| <b>Querying GPT o1</b>         | 0.91              | 0.91     | 0.91                 | 0.66        | 0.84    |

<span id="page-6-5"></span><span id="page-6-2"></span>Table 1. Comparison of video filtering metrics. Pearson correlation coefficients measure each metric's effectiveness in assessing whether a generated video follows the language command. Querying GPT o1 proves to be most effective.

![](_page_6_Figure_2.jpeg)

Figure 7. RIGVid performance vs. video quality. The dashed lines separate performance on generated videos from real videos. Kling V1.6 produces most reliable videos and leads to highest RIGVid success. Filtered videos perform on par with real ones. UF denotes unfiltered and F denotes filtered.

ing suggests that, at current model quality, generated videos are already sufficient for visual imitation, substantially reducing the need for manual data collection.

*What causes failure of imitation given high-quality videos?* Aside from one case where the object slipped out of the gripper, all failures on filtered Kling v1.6 videos are attributed to errors in monocular depth estimation. These errors result in inaccurate 6D trajectories and lead to tracking failures. Notably, similar depth estimation issues are also observed in real videos, suggesting that the limitation lies in the depth model itself. App. [I](#page-19-0) provides a detailed analysis of failure cases with qualitative examples.

#### <span id="page-6-0"></span>4.3. RIGVid vs. VLM-Based Trajectory Prediction

Video generation is computationally expensive, prompting the question of whether more efficient alternatives can enable robot manipulation without any demonstrations. VLMs offer one potential alternative by predicting simplified task abstractions—goal states [\[48\]](#page-11-25), constraints [\[49\]](#page-11-4), or reward functions [\[92\]](#page-13-22)—without generating full visual sequences, making them cheaper in computation and infer-

![](_page_6_Figure_8.jpeg)

<span id="page-6-4"></span>Figure 8. RIGVid vs. ReKep Success Rates. RIGVid outperforms SOTA VLM-based trajectory prediction method ReKep.

<span id="page-6-3"></span>ence time. We take the state-of-the-art ReKep [\[49\]](#page-11-4) method as a representative of this line of work, and compare against it in Fig. [8.](#page-6-4) In our comparison, RIGVid achieves 85% vs. ReKep's 50% success over four tasks. App. [F](#page-18-1) illustrates ReKep's failures, which we attribute to inaccurate keypoint predictions. This comparison suggests that, for our tasks and experimental setup, VLM-generated abstractions are compact and may lack the rich, necessary details for successful robot execution. Thus, despite its higher cost, video generation provides crucial supervision rather than being a wasteful expense in these settings.

While this result highlights, for our tasks and setup, the additional detail in generated videos supports more reliable execution than the current VLM-based alternative, it does not rule out the possibility that future or alternative VLMbased approaches could close this gap. Our findings suggest that, at present, video generation can provide richer supervision for manipulation compared to this specific VLM-based method, despite its higher computational cost.

#### <span id="page-6-1"></span>4.4. Comparison to Alternative Trajectory Extraction Methods

Next, we investigate the best way to extract trajectory information from videos for the purpose of visual imitation. To this end, we adapted several competitive methods that use different types of tracking tos imitate a video without any demonstrations. Summary of all the methods is shown in Tab. [2.](#page-7-0) For each method, we describe its inputs and outputs, core approach, our modifications, and the motivation for its inclusion (additional details can be found in App. [E\)](#page-17-1).

Track2Act [\[15\]](#page-10-10) (Tracks-Based). This method takes an RGBD image of the initial scene, and a single goal image that specifies the desired final configuration. Since we have

<span id="page-7-2"></span><span id="page-7-0"></span>

| Method                             | Inputs                                                                                                            | Intermediate Repr.                                  | Salient Method Characteristic                                                                                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Track2Act<br><b>AVDC</b><br>4D-DPM | Initial RGBD, goal image<br>Initial RGBD, task desc., mask, generated video<br>3D Gaussian field, generated video | 2D point tracks<br>Optical flow<br>3D feature field | Only needs initial and goal image; no intermediate frames<br>Dense flow over full video for trajectory optimization<br>3D field tracking (NeRF-like); needs $360^{\circ}$ video |
| Gen2Act                            | Initial RGBD, task desc., mask, generated video                                                                   | 2D point tracks                                     | Sparse tracks from generated video for pose estimation                                                                                                                          |

Table 2. Summary of trajectory extraction baselines. Each baseline processes the same generated videos to extract object trajectories for robot execution, but differs in inputs, intermediate representations, and the way correspondences are established.

![](_page_7_Figure_2.jpeg)

Figure 9. Comparative evaluation of trajectory extraction methods. RIGVid consistently achieves higher success rates across all four tasks; relative improvements are higher as tasks become harder (*i.e*., from left to right).

no other way to get the goal image, we set it to the last frame of the generated video. Using only this pair of images, Track2Act uses a learned model to predict a dense grid of 2D point tracks, producing pixel-level correspondences between the initial and goal image. These tracks are then lifted to 3D using the depth map from the initial frame and converted into a sequence of 3D object poses via the Perspective-n-Point (PnP) algorithm. We do not finetune their track prediction network, and do not use their residual policy. Track2Act is an attractive alternative as it uses a dedicated track prediction network that operates solely on the start and goal images, without requiring any intermediate frames. However, its main drawback is that the learned track prediction network may not generalize to all scenarios, as evidenced by our experiments and qualitative results.

AVDC [\[60\]](#page-12-4) (Flow-Based). Given an initial RGBD image, task description, and active object mask, AVDC predicts object motion by first generating a task-conditioned video and then computing optical flow between frames. This optical flow is used in an optimization process to reconstruct the object trajectory. In our adaptation, we substitute AVDC's original video generator with our improved model, while preserving all downstream processing. Unlike Track2Act, AVDC leverages optical flow across the entire video, giving it dense temporal correspondences at every pixel and thus many more cues for tracking. It is attractive because it offers a denser input for object tracking. Nevertheless, it is sensitive to cumulative errors in flow estimation, which can degrade the accuracy of the resulting object trajectories.

4D-DPM [\[58\]](#page-12-5) (Feature Field-Based). This method takes a 3D Gaussian splatting field of the object and a real video of the task, and outputs object trajectories over time. A fea<span id="page-7-1"></span>ture field, similar to NeRF representations, is a continuous mapping from 3D space to high-dimensional feature vectors that capture both geometry and appearance. By aligning the feature field with individual video frames, the method can estimate object trajectory across the video. To build the field, 4D-DPM requires a separate video where the object is captured from all sides. In our adaptation, since 4D-DPM expects a real human demonstration video, we instead use a generated video as the task input video. We modify this method from tracking object part poses to tracking single objects. This approach is compelling because it applies semantic, feature-based reasoning to track objects, capturing entire object structure from video, without relying on explicit correspondences. However, it tends to produce unstable tracking in our experiments, limiting its practicality.

Gen2Act [\[14\]](#page-10-9) (Generated Goal-Based). Gen2Act takes as input an RGBD image of the scene and a task description, and outputs robot actions predicted by a learned policy. In the original formulation, the extracted tracks on the generated video were used to supervise behaviour-cloning on a large offline robotics dataset. In our adaptation, we do not use any policy learning. Instead, we extract object tracks from the generated video and directly estimate object poses from these tracks using the initial depth image. This removes any dependence on expert demonstration data or learned policies. Gen2Act is notable for leveraging sparse correspondences extracted from the generated video, enabling task-relevant object motion to be tracked and retargeted without requiring explicit actions. However, when large portions of the object become occluded or undergo significant rotations, many of the tracking points are lost, resulting in too few correspondences to estimate object

<span id="page-8-1"></span>pose accurately and ultimately causing the tracking to fail.

Fig. [9](#page-7-1) shows that RIGVid achieves a success rate of 85.0%, compared to 67.5% for Gen2Act and considerably lower rates for all other baselines. This margin grows with more complex tasks. Methods such as Track2Act (7.5%), AVDC (32.5%), and 4D-DPM (35.0%) rely on point tracks or optical flow, but their performance remains limited—especially as object rotations or occlusions are severe. Gen2Act, which combines video generation with point-based tracking, closes part of the gap but consistently struggles when large portions of the object become untrackable. In contrast, RIGVid's use of a structured 6D object pose trajectory enables robust execution across all tasks, accounting for the 17.5% improvement over Gen2Act. This advantage persists when more powerful tracking models like CoTracker3 [\[53\]](#page-12-20) are used, as shown in App. [G.](#page-18-2)

Looking at the task-wise breakdown in Fig. [9,](#page-7-1) RIGVid maintains high success rates even as object depth varies significantly (such as in the lifting task) or when the objects are thin, small, or partially occluded (such as in placing a spatula or sweeping trash). Other methods frequently struggle in these settings, often failing to recover accurate object trajectories when objects become partially hidden or change distance rapidly. The advantage of RIGVid is most pronounced on the most challenging tasks: for both spatula placement and sweeping, RIGVid achieves success rates 20–25% above the next best baseline. These results suggest that the structured 6D pose trajectory not only enables robust tracking under depth changes and occlusion but also scales to scenarios where correspondence methods fail.

Visualizing the outputs in Fig. [10](#page-8-0) for the same generated video, we observe the intermediate predictions and resulting robot executions produced by each method. For Track2Act, the predicted tracks diverge from the true object path, leading to failed execution. Often, the track2act track prediction does not follow the true motion paths, which is the primary source of errors in our experiments. AVDC generates reasonable optical flow in individual frames, but when summed across the entire video, the resulting trajectory is often physically implausible, and the execution fails. We often found that this summing up of object flow across the video leads to small errors that accumulate over the entire video, re-

![](_page_8_Figure_5.jpeg)

<span id="page-8-0"></span>Figure 10. Analyzing intermediate visual representations. Only Gen2Act and our 6D Object Pose Trajectory can correctly track the position and rotation of the watering can, leading to a successful execution. Check the description in the main paper for detailed discussions of the failure modes of the alternative methods.

<span id="page-9-3"></span>![](_page_9_Picture_0.jpeg)

Figure 11. RIGVid's embodiment-agnostic capabilities and examples on solving complex, open-world tasks. RIGVid can readily work on ALOHA setup [\[134\]](#page-15-11) as shown on top left. On the bottom left, RIGVid is retargeted to the bimanual ALOHA setup. On the right, it generates trajectories for diverse manipulation tasks—including wiping, mixing, and ironing—without using any physical demonstrations.

sulting in faulty object location across the video. Gen2Act yields plausible tracks and leads to successful manipulation. We often found that tracks were accurate, and the resulting trajectory after PnP was also accurate. 4D-DPM exhibits inconsistent tracking performance. While it accurately follows the object in certain segments, the example shown reveals incorrect tracking during the first half of the episode, which ultimately causes the rollout to fail. We often found that the tracking was unstable and very jerky. In contrast, the 6D object pose trajectories produced by RIGVid remain stable throughout the episode and closely match the actual object motion, resulting in successful execution.

#### <span id="page-9-0"></span>4.5. Testing Generalization

Embodiment-Agnostic Transfer. We test RIGVid's generalizability to another embodiment by deploying it on the ALOHA robot for the pouring task (Fig. [11,](#page-9-1) top left). On this setup, it achieves 80% success, compared to 100% on our default xArm setup.<sup>[1](#page-9-2)</sup> RIGVid also generalizes to a bimanual setup, simultaneously placing a pair of shoes into a box using both arms (Fig. [11,](#page-9-1) bottom left).

Extensions to Additional Tasks. Besides our four main focus tasks, we also obtained promising preliminary results on a larger variety of diverse and challenging manipulation tasks shown in Fig. [11](#page-9-1) (right). These tasks include wiping, mixing, and ironing, uprighting a ketchup bottle, unplugging a charger, and rotating a spoon to spill beans. Notably, the latter three tasks involve extreme rotations, which RIGVid can handle successfully.

#### 5. Conclusions

We introduced Robots Imitating Generated Videos (RIGVid), the first method for robotic manipulation that

<span id="page-9-1"></span>works without demonstrations —- no teleoperation, no human demonstration, or expert policy rollouts. By leveraging recent advances in generative vision models and 6D pose estimation, RIGVid enables robots to execute complex tasks entirely from generated video. We extract 6D Object Pose Trajectory from the generated videos and retarget it to the robot, demonstrating a data-efficient approach to robotic skill acquisition. Our analysis shows a clear correlation between video quality and task success: as generation improves, RIGVid approaches real demo performance. Additionally, our comparisons with SOTA VLM-based zero-shot manipulation methods confirm that leveraging dense visual and temporal cues from generated videos yields much more reliable performance. We also show that RIGVid significantly outperforms competing trajectory extraction methods across a diverse set of visual imitation tasks, and demonstrate the robustness of our approach to environmental disturbances. Our work represents a step toward enabling robots to learn from the vast visual knowledge in generative models, reducing reliance on costly and time-consuming real-world data collection.

#### 6. Acknowledgement

We thank the members of the RoboPIL lab, and UIUC vision and robotics labs for their valuable discussions and feedback. Unnat would like to especially acknowledge Chen Bao, Homanga Bharadhwaj, Shikhar Bahl, and friends at CMU and Skild for their insightful conversations on learning from videos. We also thank Justin Kerr for his assistance in reproducing the 4D-DPM baseline. This work is partially supported by the Toyota Research Institute (TRI), the Sony Group Corporation, Google, Dalus AI, and an Amazon Research Award, Fall 2024. This article solely reflects the opinions and conclusions of its authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors.

<span id="page-9-2"></span><sup>&</sup>lt;sup>1</sup>The slight performance drop stems primarily from camera calibration challenges, as ALOHA's arms yield less accurate pose estimates.

#### References

- <span id="page-10-7"></span>[1] Kling ai. <https://www.klingai.com/>, 2024. Accessed: 2024-02-10. [1](#page-0-0)
- <span id="page-10-22"></span>[2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023. [3](#page-2-1)
- <span id="page-10-14"></span>[3] Anurag Ajay, Seungwook Han, Yilun Du, Shuang Li, Abhi Gupta, Tommi Jaakkola, Josh Tenenbaum, Leslie Kaelbling, Akash Srivastava, and Pulkit Agrawal. Compositional foundation models for hierarchical planning. *Advances in Neural Information Processing Systems*, 36: 22304–22325, 2023. [3](#page-2-1)
- <span id="page-10-15"></span>[4] Mert Albaba, Chenhao Li, Markos Diomataris, Omid Taheri, Andreas Krause, and Michael Black. Nil: No-data imitation learning by leveraging pre-trained video diffusion models. *arXiv preprint arXiv:2503.10626*, 2025. [3](#page-2-1)
- <span id="page-10-11"></span>[5] Max Argus, Lukas Hermann, Jon Long, and Thomas Brox. Flowcontrol: Optical flow based visual servoing. In *2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 7534–7541. IEEE, 2020. [2](#page-1-1)
- <span id="page-10-18"></span>[6] Philipp Ausserlechner, David Haberger, Stefan Thalhammer, Jean-Baptiste Weibel, and Markus Vincze. Zs6d: Zero-shot 6d object pose estimation using vision transformers. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 463–469. IEEE, 2024. [3](#page-2-1)
- <span id="page-10-5"></span>[7] Arpit Bahety, Priyanka Mandikal, Ben Abbatematteo, and Roberto Martín-Martín. Screwmimic: Bimanual imitation from human videos with screw space projection. *arXiv preprint arXiv:2405.03666*, 2024. [1](#page-0-0)
- <span id="page-10-3"></span>[8] Shikhar Bahl, Abhinav Gupta, and Deepak Pathak. Human-to-robot imitation in the wild. *arXiv preprint arXiv:2207.09450*, 2022. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-10-0"></span>[9] Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, and Deepak Pathak. Affordances from human videos as a versatile representation for robotics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13778–13790, 2023. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-10-13"></span>[10] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. *Advances in Neural Information Processing Systems*, 35:24639–24654, 2022. [3](#page-2-1)
- <span id="page-10-8"></span>[11] Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. *arXiv preprint arXiv:2406.03520*, 2024. [1](#page-0-0)
- <span id="page-10-12"></span>[12] Leonardo Barcellona, Andrii Zadaianchuk, Davide Allegro, Samuele Papa, Stefano Ghidoni, and Efstratios Gavves. Dream to manipulate: Compositional world models empowering robot imitation learning with imagination. *arXiv preprint arXiv:2412.14957*, 2024. [3](#page-2-1)

- <span id="page-10-1"></span>[13] Homanga Bharadhwaj, Abhinav Gupta, Shubham Tulsiani, and Vikash Kumar. Zero-shot robot manipulation from passive human videos. *arXiv preprint arXiv:2302.02011*, 2023. [1](#page-0-0)
- <span id="page-10-9"></span>[14] Homanga Bharadhwaj, Debidatta Dwibedi, Abhinav Gupta, Shubham Tulsiani, Carl Doersch, Ted Xiao, Dhruv Shah, Fei Xia, Dorsa Sadigh, and Sean Kirmani. Gen2act: Human video generation in novel scenarios enables generalizable robot manipulation. *arXiv preprint arXiv:2409.16283*, 2024. [1,](#page-0-0) [2,](#page-1-1) [3,](#page-2-1) [8,](#page-7-2) [18](#page-17-2)
- <span id="page-10-10"></span>[15] Homanga Bharadhwaj, Roozbeh Mottaghi, Abhinav Gupta, and Shubham Tulsiani. Track2act: Predicting point tracks from internet videos enables diverse zero-shot robot manipulation, 2024. [1,](#page-0-0) [2,](#page-1-1) [7,](#page-6-5) [18](#page-17-2)
- <span id="page-10-6"></span>[16] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. [1](#page-0-0)
- <span id="page-10-17"></span>[17] Ming Cai and Ian Reid. Reconstruct locally, localize globally: A model free method for object pose estimation. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3153–3163, 2020. [3](#page-2-1)
- <span id="page-10-19"></span>[18] Sylvain Calinon. A tutorial on task-parameterized movement learning and retrieval. *Intelligent service robotics*, 9: 1–29, 2016. [3](#page-2-1)
- <span id="page-10-16"></span>[19] Andrea Caraffa, Davide Boscaini, Amir Hamza, and Fabio Poiesi. Freeze: Training-free zero-shot 6d pose estimation with geometric and vision foundation models. *European Conference on Computer Vision (ECCV)*, 2024. [3](#page-2-1)
- <span id="page-10-24"></span>[20] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In *Proceedings of the International Conference on Computer Vision (ICCV)*, 2021. [6,](#page-5-2) [21](#page-20-0)
- <span id="page-10-4"></span>[21] Elliot Chane-Sane, Cordelia Schmid, and Ivan Laptev. Learning video-conditioned policies for unseen manipulation tasks. In *2023 IEEE International Conference on Robotics and Automation (ICRA)*, pages 909–916. IEEE, 2023. [1](#page-0-0)
- <span id="page-10-2"></span>[22] Matthew Chang, Arjun Gupta, and Saurabh Gupta. Semantic visual navigation by watching youtube videos. In *NeurIPS*, 2020. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-10-23"></span>[23] Matthew Chang, Theophile Gervet, Mukul Khanna, Sriram Yenamandra, Dhruv Shah, So Yeon Min, Kavit Shah, Chris Paxton, Saurabh Gupta, Dhruv Batra, Roozbeh Mottaghi, Jitendra Malik, and Devendra Singh Chaplot. Goat: Go to any thing. *arXiv preprint arXiv:2311.06430*, 2023. [4](#page-3-3)
- <span id="page-10-20"></span>[24] Xuxin Cheng, Yandong Ji, Junming Chen, Ruihan Yang, Ge Yang, and Xiaolong Wang. Expressive whole-body control for humanoid robots. *arXiv preprint arXiv:2402.16796*, 2024. [3](#page-2-1)
- <span id="page-10-21"></span>[25] Sungjoon Choi, Matthew KXJ Pan, and Joohyung Kim. Nonparametric motion retargeting for humanoid robots on shared latent space. In *Robotics: science and systems*, 2020. [3](#page-2-1)

- <span id="page-11-1"></span>[26] Sudeep Dasari and Abhinav Gupta. Transformers for oneshot visual imitation. In *Conference on Robot Learning*, pages 2071–2084. PMLR, 2021. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-11-10"></span>[27] Sudeep Dasari, Mohan Kumar Srirama, Unnat Jain, and Abhinav Gupta. An unbiased look at datasets for visuomotor pre-training. In *Conference on Robot Learning*, pages 1183–1198. PMLR, 2023. [3](#page-2-1)
- <span id="page-11-26"></span>[28] Carl Doersch, Yi Yang, Dilara Gokay, Pauline Luc, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ross Goroshin, João Carreira, and Andrew Zisserman. Bootstap: Bootstrapped training for tracking-any-point. *arXiv preprint arXiv:2402.00847*, 2024. [18](#page-17-2)
- <span id="page-11-12"></span>[29] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. *arXiv preprint arXiv:2310.10625*, 2023. [3](#page-2-1)
- <span id="page-11-3"></span>[30] Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. *Advances in Neural Information Processing Systems*, 36, 2024. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-11-23"></span>[31] Hao-Shu Fang, Chenxi Wang, Hongjie Fang, Minghao Gou, Jirong Liu, Hengxu Yan, Wenhai Liu, Yichen Xie, and Cewu Lu. Anygrasp: Robust and efficient grasp perception in spatial and temporal domains. *IEEE Transactions on Robotics*, 2023. [4](#page-3-3)
- <span id="page-11-9"></span>[32] Chelsea Finn, Tianhe Yu, T. Zhang, P. Abbeel, and Sergey Levine. One-shot visual imitation learning via metalearning. In *CoRL*, 2017. [3](#page-2-1)
- <span id="page-11-8"></span>[33] Peter R Florence, Lucas Manuelli, and Russ Tedrake. Dense object nets: Learning dense visual object descriptors by and for robotic manipulation. *arXiv preprint arXiv:1806.08756*, 2018. [2](#page-1-1)
- <span id="page-11-5"></span>[34] Zipeng Fu, Qingqing Zhao, Qi Wu, Gordon Wetzstein, and Chelsea Finn. Humanplus: Humanoid shadowing and imitation from humans. *arXiv preprint arXiv:2406.10454*, 2024. [2](#page-1-1)
- <span id="page-11-7"></span>[35] Chongkai Gao, Haozhuo Zhang, Zhixuan Xu, Zhehao Cai, and Lin Shao. Flip: Flow-centric generative planning for general-purpose manipulation tasks. *arXiv preprint arXiv:2412.08261*, 2024. [2](#page-1-1)
- <span id="page-11-0"></span>[36] Shenyuan Gao, Siyuan Zhou, Yilun Du, Jun Zhang, and Chuang Gan. Adaworld: Learning adaptable world models with latent actions. *arXiv preprint arXiv:2503.18938*, 2025. [1](#page-0-0)
- <span id="page-11-22"></span>[37] Theophile Gervet, Soumith Chintala, Dhruv Batra, Jitendra Malik, and Devendra Singh Chaplot. Navigating to objects in the real world. *Science Robotics*, 2023. [4](#page-3-3)
- <span id="page-11-17"></span>[38] Michael Gleicher. Retargetting motion to new characters. In *Proceedings of the 25th annual conference on Computer graphics and interactive techniques*, pages 33–42, 1998. [3](#page-2-1)
- <span id="page-11-2"></span>[39] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation. *arXiv preprint arXiv:2505.00337*, 2025. [1](#page-0-0)
- <span id="page-11-21"></span>[40] Richard Hartley and Andrew Zisserman. *Multiple view geometry in computer vision*. Cambridge university press, 2003. [4](#page-3-3)

- <span id="page-11-19"></span>[41] Tairan He, Zhengyi Luo, Wenli Xiao, Chong Zhang, Kris Kitani, Changliu Liu, and Guanya Shi. Learning humanto-humanoid real-time whole-body teleoperation. In *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 8944–8951. IEEE, 2024. [3](#page-2-1)
- <span id="page-11-15"></span>[42] Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, and Xiaowei Zhou. Onepose++: Keypointfree one-shot object pose estimation without cad models. *Advances in Neural Information Processing Systems*, 35: 35103–35115, 2022. [3](#page-2-1)
- <span id="page-11-13"></span>[43] Yisheng He, Wei Sun, Haibin Huang, Jianran Liu, Haoqiang Fan, and Jian Sun. Pvn3d: A deep point-wise 3d keypoints voting network for 6dof pose estimation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11632–11641, 2020. [3](#page-2-1)
- <span id="page-11-14"></span>[44] Yisheng He, Haibin Huang, Haoqiang Fan, Qifeng Chen, and Jian Sun. Ffb6d: A full flow bidirectional fusion network for 6d pose estimation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 3003–3013, 2021. [3](#page-2-1)
- <span id="page-11-16"></span>[45] Yisheng He, Yao Wang, Haoqiang Fan, Jian Sun, and Qifeng Chen. Fs6d: Few-shot 6d pose estimation of novel objects. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6814– 6824, 2022. [3](#page-2-1)
- <span id="page-11-6"></span>[46] Cheng-Chun Hsu, Bowen Wen, Jie Xu, Yashraj Narang, Xiaolong Wang, Yuke Zhu, Joydeep Biswas, and Stan Birchfield. Spot: Se (3) pose trajectory diffusion for objectcentric manipulation. *arXiv preprint arXiv:2411.00965*, 2024. [2](#page-1-1)
- <span id="page-11-20"></span>[47] Kai Hu, Christian Ott, and Dongheui Lee. Online human walking imitation in task and joint space based on quadratic programming. In *2014 IEEE International Conference on Robotics and Automation (ICRA)*, pages 3458–3464. IEEE, 2014. [3](#page-2-1)
- <span id="page-11-25"></span>[48] Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. Voxposer: Composable 3d value maps for robotic manipulation with language models. *arXiv preprint arXiv:2307.05973*, 2023. [7](#page-6-5)
- <span id="page-11-4"></span>[49] Wenlong Huang, Chen Wang, Yunzhu Li, Ruohan Zhang, and Li Fei-Fei. Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation. *arXiv preprint arXiv:2409.01652*, 2024. [2,](#page-1-1) [7](#page-6-5)
- <span id="page-11-24"></span>[50] Ziqi Huang, Fan Zhang, Xiaojie Xu, Yinan He, Jiashuo Yu, Ziyue Dong, Qianli Ma, Nattapol Chanpaisit, Chenyang Si, Yuming Jiang, Yaohui Wang, Xinyuan Chen, Ying-Cong Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench++: Comprehensive and versatile benchmark suite for video generative models. *arXiv preprint arXiv:2411.13503*, 2024. [6,](#page-5-2) [21,](#page-20-0) [22](#page-21-0)
- <span id="page-11-18"></span>[51] Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, and Tao Chen. Motiongpt: Human motion as a foreign language. *Advances in Neural Information Processing Systems*, 36:20067–20079, 2023. [3](#page-2-1)
- <span id="page-11-11"></span>[52] Yuanchen Ju, Kaizhe Hu, Guowei Zhang, Gu Zhang, Mingrun Jiang, and Huazhe Xu. Robo-abc: Affordance generalization beyond categories via semantic correspondence for

robot manipulation. In *European Conference on Computer Vision*, pages 222–239. Springer, 2024. [3](#page-2-1)

- <span id="page-12-20"></span>[53] Nikita Karaev, Iurii Makarov, Jianyuan Wang, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Cotracker3: Simpler and better point tracking by pseudolabelling real videos. *arXiv preprint arXiv:2410.11831*, 2024. [9](#page-8-1)
- <span id="page-12-6"></span>[54] Siddharth Karamcheti, Suraj Nair, Annie S Chen, Thomas Kollar, Chelsea Finn, Dorsa Sadigh, and Percy Liang. Language-driven representation learning for robotics. *arXiv preprint arXiv:2302.12766*, 2023. [3](#page-2-1)
- <span id="page-12-0"></span>[55] Simar Kareer, Dhruv Patel, Ryan Punamiya, Pranay Mathur, Shuo Cheng, Chen Wang, Judy Hoffman, and Danfei Xu. Egomimic: Scaling imitation learning via egocentric video. *arXiv preprint arXiv:2410.24221*, 2024. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-12-18"></span>[56] Bingxin Ke, Dominik Narnhofer, Shengyu Huang, Lei Ke, Torben Peters, Katerina Fragkiadaki, Anton Obukhov, and Konrad Schindler. Video depth without video models, 2024. [4](#page-3-3)
- <span id="page-12-21"></span>[57] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. *ACM Trans. Graph.*, 42(4):139–1, 2023. [18](#page-17-2)
- <span id="page-12-5"></span>[58] Justin Kerr, Chung Min Kim, Mingxuan Wu, Brent Yi, Qianqian Wang, Ken Goldberg, and Angjoo Kanazawa. Robot see robot do: Imitating articulated object manipulation with monocular 4d reconstruction. *arXiv preprint arXiv:2409.18121*, 2024. [2,](#page-1-1) [3,](#page-2-1) [8,](#page-7-2) [18](#page-17-2)
- <span id="page-12-22"></span>[59] Chung Min Kim, Mingxuan Wu, Justin Kerr, Ken Goldberg, Matthew Tancik, and Angjoo Kanazawa. Garfield: Group anything with radiance fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 21530–21539, 2024. [18](#page-17-2)
- <span id="page-12-4"></span>[60] Po-Chen Ko, Jiayuan Mao, Yilun Du, Shao-Hua Sun, and Joshua B Tenenbaum. Learning to act from actionless videos through dense correspondences. *arXiv preprint arXiv:2310.08576*, 2023. [2,](#page-1-1) [3,](#page-2-1) [8,](#page-7-2) [18](#page-17-2)
- <span id="page-12-15"></span>[61] Scott Kuindersma, Robin Deits, Maurice Fallon, Andrés Valenzuela, Hongkai Dai, Frank Permenter, Twan Koolen, Pat Marion, and Russ Tedrake. Optimization-based locomotion planning, estimation, and control design for the atlas humanoid robot. *Autonomous robots*, 40:429–455, 2016. [3](#page-2-1)
- <span id="page-12-10"></span>[62] Yann Labbé, Justin Carpentier, Mathieu Aubry, and Josef Sivic. Cosypose: Consistent multi-view multi-object 6d pose estimation. In *Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16*, pages 574–591. Springer, 2020. [3](#page-2-1)
- <span id="page-12-11"></span>[63] Yann Labbé, Lucas Manuelli, Arsalan Mousavian, Stephen Tyree, Stan Birchfield, Jonathan Tremblay, Justin Carpentier, Mathieu Aubry, Dieter Fox, and Josef Sivic. Megapose: 6d pose estimation of novel objects via render & compare. In *Proceedings of the 6th Conference on Robot Learning (CoRL)*, 2022. [3,](#page-2-1) [4,](#page-3-3) [20](#page-19-1)
- <span id="page-12-17"></span>[64] Arjun S Lakshmipathy, Jessica K Hodgins, and Nancy S Pollard. Kinematic motion retargeting for contact-

rich anthropomorphic manipulations. *arXiv preprint arXiv:2402.04820*, 2024. [3](#page-2-1)

- <span id="page-12-1"></span>[65] Marion Lepert, Jiaying Fang, and Jeannette Bohg. Phantom: Training robots without robots using only human videos. *arXiv preprint arXiv:2503.00779*, 2025. [1](#page-0-0)
- <span id="page-12-12"></span>[66] Vincent Lepetit, Francesc Moreno-Noguer, and Pascal Fua. Ep n p: An accurate o (n) solution to the p n p problem. *International journal of computer vision*, 81:155–166, 2009. [3](#page-2-1)
- <span id="page-12-13"></span>[67] Fu Li, Shishir Reddy Vutukur, Hao Yu, Ivan Shugurov, Benjamin Busam, Shaowu Yang, and Slobodan Ilic. Nerfpose: A first-reconstruct-then-regress approach for weaklysupervised 6d object pose estimation. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 2123–2133, 2023. [3](#page-2-1)
- <span id="page-12-7"></span>[68] Gen Li, Deqing Sun, Laura Sevilla-Lara, and Varun Jampani. One-shot open affordance learning with foundation models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 3086– 3096, 2024. [3](#page-2-1)
- <span id="page-12-8"></span>[69] Gen Li, Nikolaos Tsagkas, Jifei Song, Ruaridh Mon-Williams, Sethu Vijayakumar, Kun Shao, and Laura Sevilla-Lara. Learning precise affordances from egocentric videos for robotic manipulation. *arXiv preprint arXiv:2408.10123*, 2024. [3](#page-2-1)
- <span id="page-12-2"></span>[70] Jinhan Li, Yifeng Zhu, Yuqi Xie, Zhenyu Jiang, Mingyo Seo, Georgios Pavlakos, and Yuke Zhu. Okami: Teaching humanoid robots manipulation skills through single video imitation. In *8th Annual Conference on Robot Learning*, 2024. [1,](#page-0-0) [3](#page-2-1)
- <span id="page-12-23"></span>[71] Zhen Li, Zuo-Liang Zhu, Ling-Hao Han, Qibin Hou, Chun-Le Guo, and Ming-Ming Cheng. Amt: All-pairs multifield transforms for efficient frame interpolation. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. [21](#page-20-0)
- <span id="page-12-9"></span>[72] Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, and Carl Vondrick. Dreamitate: Real-world visuomotor policy learning via video generation. *arXiv preprint arXiv:2406.16862*, 2024. [3](#page-2-1)
- <span id="page-12-16"></span>[73] Yuwei Liang, Weijie Li, Yue Wang, Rong Xiong, Yichao Mao, and Jiafan Zhang. Dynamic movement primitive based motion retargeting for dual-arm sign language motions. In *2021 IEEE International Conference on Robotics and Automation (ICRA)*, pages 8195–8201. IEEE, 2021. [3](#page-2-1)
- <span id="page-12-3"></span>[74] Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model. *arXiv preprint arXiv:2408.16767*, 2024. [1](#page-0-0)
- <span id="page-12-19"></span>[75] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. *arXiv preprint arXiv:2303.05499*, 2023. [4](#page-3-3)
- <span id="page-12-14"></span>[76] Xingyu Liu, Gu Wang, Ruida Zhang, Chenyangguang Zhang, Federico Tombari, and Xiangyang Ji. Unopose: Unseen object pose estimation with an unposed rgb-d reference image. *arXiv preprint arXiv:2411.16106*, 2024. [3](#page-2-1)

- <span id="page-13-3"></span>[77] YuXuan Liu, Abhishek Gupta, Pieter Abbeel, and Sergey Levine. Imitation from observation: Learning to imitate behaviors from raw video via context translation. In *2018 IEEE International Conference on Robotics and Automation (ICRA)*, pages 1118–1125. IEEE, 2018. [3](#page-2-1)
- <span id="page-13-12"></span>[78] Yuan Liu, Yilin Wen, Sida Peng, Cheng Lin, Xiaoxiao Long, Taku Komura, and Wenping Wang. Gen6d: Generalizable model-free 6-dof object pose estimation from rgb images. In *European Conference on Computer Vision*, pages 298–315. Springer, 2022. [3](#page-2-1)
- <span id="page-13-15"></span>[79] Zhengyi Luo, Jinkun Cao, Kris Kitani, Weipeng Xu, et al. Perpetual humanoid control for real-time simulated avatars. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 10895–10904, 2023. [3](#page-2-1)
- <span id="page-13-4"></span>[80] Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In *Conference on Robot Learning*, pages 879– 893. PMLR, 2018. [3](#page-2-1)
- <span id="page-13-9"></span>[81] Russell Mendonca, Shikhar Bahl, and Deepak Pathak. Structured world models from human videos. *arXiv preprint arXiv:2308.10901*, 2023. [3](#page-2-1)
- <span id="page-13-1"></span>[82] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models learn physical principles from watching videos? *arXiv preprint arXiv:2501.09038*, 2025. [1](#page-0-0)
- <span id="page-13-18"></span>[83] Shinichiro Nakaoka, Atsushi Nakazawa, Fumio Kanehiro, Kenji Kaneko, Mitsuharu Morisawa, and Katsushi Ikeuchi. Task model of lower body motion for a biped humanoid robot to imitate human dances. In *2005 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 3157–3162. IEEE, 2005. [3](#page-2-1)
- <span id="page-13-10"></span>[84] Van Nguyen Nguyen, Thibault Groueix, Mathieu Salzmann, and Vincent Lepetit. Gigapose: Fast and robust novel object pose estimation via one correspondence. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9903–9913, 2024. [3](#page-2-1)
- <span id="page-13-16"></span>[85] Scott Niekum, Sarah Osentoski, George Konidaris, and Andrew G Barto. Learning and generalization of complex tasks from unstructured demonstrations. In *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems*, pages 5239–5246. IEEE, 2012. [3](#page-2-1)
- <span id="page-13-0"></span>[86] Abby O'Neill, Abdul Rehman, Abhinav Gupta, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, et al. Open x-embodiment: Robotic learning datasets and rt-x models. *arXiv preprint arXiv:2310.08864*, 2023. [1](#page-0-0)
- <span id="page-13-14"></span>[87] Evin Pınar Örnek, Yann Labbé, Bugra Tekin, Lingni Ma, Cem Keskin, Christian Forster, and Tomáš Hodaň. Foundpose: Unseen object pose estimation with foundation features. *European Conference on Computer Vision (ECCV)*, 2024. [3](#page-2-1)
- <span id="page-13-11"></span>[88] Kiru Park, Timothy Patten, and Markus Vincze. Pix2pose: Pixel-wise coordinate regression of objects for 6d pose estimation. In *Proceedings of the IEEE/CVF international conference on computer vision*, pages 7668–7677, 2019. [3](#page-2-1)

- <span id="page-13-13"></span>[89] Keunhong Park, Arsalan Mousavian, Yu Xiang, and Dieter Fox. Latentfusion: End-to-end differentiable reconstruction and rendering for unseen object pose estimation. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10710–10719, 2020. [3](#page-2-1)
- <span id="page-13-5"></span>[90] Austin Patel, Andrew Wang, Ilija Radosavovic, and Jitendra Malik. Learning to imitate object interactions from internet videos. *arXiv preprint arXiv:2211.13225*, 2022. [3](#page-2-1)
- <span id="page-13-6"></span>[91] Dhruvesh Patel, Hamid Eghbalzadeh, Nitin Kamra, Michael Louis Iuzzolino, Unnat Jain, and Ruta Desai. Pretrained language models as visual planners for human assistance. In *ICCV*, 2023. [3](#page-2-1)
- <span id="page-13-22"></span>[92] Shivansh Patel, Xinchen Yin, Wenlong Huang, Shubham Garg, Hooshang Nayyeri, Li Fei-Fei, Svetlana Lazebnik, and Yunzhu Li. A real-to-sim-to-real approach to robotic manipulation with vlm-generated iterative keypoint rewards. *arXiv preprint arXiv:2502.08643*, 2025. [7](#page-6-5)
- <span id="page-13-2"></span>[93] Deepak Pathak, Parsa Mahmoudieh, Guanghao Luo, Pulkit Agrawal, Dian Chen, Yide Shentu, Evan Shelhamer, Jitendra Malik, Alexei A Efros, and Trevor Darrell. Zero-shot visual imitation. In *Proceedings of the IEEE conference on computer vision and pattern recognition workshops*, pages 2050–2053, 2018. [2](#page-1-1)
- <span id="page-13-19"></span>[94] Luigi Penco, Nicola Scianca, Valerio Modugno, Leonardo Lanari, Giuseppe Oriolo, and Serena Ivaldi. A multimode teleoperation framework for humanoid loco-manipulation: An application for the icub robot. *IEEE Robotics & Automation Magazine*, 26(4):73–82, 2019. [3](#page-2-1)
- <span id="page-13-17"></span>[95] Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. Amp: Adversarial motion priors for stylized physics-based character control. *ACM Transactions on Graphics (ToG)*, 40(4):1–20, 2021. [3](#page-2-1)
- <span id="page-13-7"></span>[96] Georgy Ponimatkin, Martin Cífka, Tomáš Souček, Médéric Fourmy, Yann Labbé, Vladimir Petrik, and Josef Sivic. 6d object pose tracking in internet videos for robotic manipulation. *arXiv preprint arXiv:2503.10307*, 2025. [3](#page-2-1)
- <span id="page-13-20"></span>[97] Yuzhe Qin, Yueh-Hua Wu, Shaowei Liu, Hanwen Jiang, Ruihan Yang, Yang Fu, and Xiaolong Wang. Dexmv: Imitation learning for dexterous manipulation from human videos. In *European Conference on Computer Vision*, pages 570–587. Springer, 2022. [3](#page-2-1)
- <span id="page-13-23"></span>[98] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. [21](#page-20-0)
- <span id="page-13-21"></span>[99] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos, 2024. [4](#page-3-3)
- <span id="page-13-8"></span>[100] Juntao Ren, Priya Sundaresan, Dorsa Sadigh, Sanjiban Choudhury, and Jeannette Bohg. Motion tracks: A unified representation for human-robot transfer in few-shot imitation learning. *arXiv preprint arXiv:2501.06994*, 2025. [3](#page-2-1)

- [101] Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan Schaal, Sergey Levine, and Google Brain. Time-contrastive networks: Self-supervised learning from video. In *2018 IEEE international conference on robotics and automation (ICRA)*, pages 1134–1141. IEEE, 2018.
- <span id="page-14-11"></span>[102] Pratyusha Sharma, Lekha Mohan, Lerrel Pinto, and Abhinav Gupta. Multiple interactions made easy (mime): Large scale demonstrations data for imitation. *arXiv:1810.07121*, 2018. [3](#page-2-1)
- <span id="page-14-4"></span>[103] Pratyusha Sharma, Deepak Pathak, and Abhinav Gupta. Third-person visual imitation learning via decoupled hierarchical controller. *Advances in Neural Information Processing Systems*, 32, 2019. [2](#page-1-1)
- <span id="page-14-5"></span>[104] Junyao Shi, Zhuolun Zhao, Tianyou Wang, Ian Pedroza, Amy Luo, Jie Wang, Jason Ma, and Dinesh Jayaraman. Zeromimic: Distilling robotic manipulation skills from web videos. *arXiv preprint arXiv:2503.23877*, 2025. [2,](#page-1-1) [3](#page-2-1)
- <span id="page-14-15"></span>[105] Ivan Shugurov, Fu Li, Benjamin Busam, and Slobodan Ilic. Osop: A multi-stage one shot object pose estimation framework. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6835– 6844, 2022. [3](#page-2-1)
- <span id="page-14-0"></span>[106] Aravind Sivakumar, Kenneth Shaw, and Deepak Pathak. Robotic telekinesis: Learning a robotic hand imitator by watching humans on youtube. *arXiv preprint arXiv:2202.10448*, 2022. [1,](#page-0-0) [2](#page-1-1)
- <span id="page-14-12"></span>[107] Laura Smith, Nikita Dhawan, Marvin Zhang, Pieter Abbeel, and Sergey Levine. Avid: Learning multi-stage tasks via pixel-level translation of human videos. *arXiv preprint arXiv:1912.04443*, 2019. [3](#page-2-1)
- <span id="page-14-13"></span>[108] Mohan Kumar Srirama, Sudeep Dasari, Shikhar Bahl, and Abhinav Gupta. Hrp: Human affordances for robotic pretraining. *arXiv preprint arXiv:2407.18911*, 2024. [3](#page-2-1)
- <span id="page-14-19"></span>[109] Jiaming Sun, Zihao Wang, Siyu Zhang, Xingyi He, Hongcheng Zhao, Guofeng Zhang, and Xiaowei Zhou. Onepose: One-shot object pose estimation without cad models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6825– 6834, 2022. [3](#page-2-1)
- <span id="page-14-14"></span>[110] Yihong Sun, Hao Zhou, Liangzhe Yuan, Jennifer J Sun, Yandong Li, Xuhui Jia, Hartwig Adam, Bharath Hariharan, Long Zhao, and Ting Liu. Video creation by demonstration. *arXiv preprint arXiv:2412.09551*, 2024. [3](#page-2-1)
- <span id="page-14-16"></span>[111] Jonathan Tremblay, Thang To, Balakumar Sundaralingam, Yu Xiang, Dieter Fox, and Stan Birchfield. Deep object pose estimation for semantic robotic grasping of household objects. *arXiv preprint arXiv:1809.10790*, 2018. [3](#page-2-1)
- <span id="page-14-6"></span>[112] Eugene Valassakis, Georgios Papagiannis, Norman Di Palo, and Edward Johns. Demonstrate once, imitate immediately (dome): Learning visual servoing for one-shot imitation learning. In *2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 8614–8621. IEEE, 2022. [2](#page-1-1)
- <span id="page-14-10"></span>[113] Mel Vecerik, Carl Doersch, Yi Yang, Todor Davchev, Yusuf Aytar, Guangyao Zhou, Raia Hadsell, Lourdes Agapito, and Jon Scholz. Robotap: Tracking arbitrary points for few-shot

visual imitation. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 5397–5403. IEEE, 2024. [2](#page-1-1)

- <span id="page-14-1"></span>[114] Chen Wang, Linxi Fan, Jiankai Sun, Ruohan Zhang, Li Fei-Fei, Danfei Xu, Yuke Zhu, and Anima Anandkumar. Mimicplay: Long-horizon imitation learning by watching human play. *arXiv preprint arXiv:2302.12422*, 2023. [1,](#page-0-0) [2,](#page-1-1) [3](#page-2-1)
- <span id="page-14-7"></span>[115] Jianren Wang, Kangni Liu, Dingkun Guo, Xian Zhou, and Christopher G Atkeson. One-shot video imitation via parameterized symbolic abstraction graphs. *arXiv preprint arXiv:2408.12674*, 2024. [2](#page-1-1)
- <span id="page-14-22"></span>[116] Yi Wang, Yinan He, Yizhuo Li, Kunchang Li, Jiashuo Yu, Xin Ma, Xinhao Li, Guo Chen, Xinyuan Chen, Yaohui Wang, et al. Internvid: A large-scale video-text dataset for multimodal understanding and generation. In *The Twelfth International Conference on Learning Representations*, 2023. [6,](#page-5-2) [21](#page-20-0)
- <span id="page-14-17"></span>[117] Justin Wasserman, Karmesh Yadav, Girish Chowdhary, Abhinav Gupta, and Unnat Jain. Last-mile embodied visual navigation. In *Conference on Robot Learning*, pages 666– 678. PMLR, 2023. [3](#page-2-1)
- <span id="page-14-18"></span>[118] Justin Wasserman, Girish Chowdhary, Abhinav Gupta, and Unnat Jain. Exploitation-guided exploration for semantic embodied navigation. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 2901– 2908. IEEE, 2024. [3](#page-2-1)
- <span id="page-14-21"></span>[119] Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Muller, Alex Evans, Dieter Fox, Jan Kautz, and Stan Birchfield. Bundlesdf: Neural 6-dof tracking and 3d reconstruction of unknown objects. *CVPR*, 2023. [4,](#page-3-3) [17](#page-16-2)
- <span id="page-14-3"></span>[120] Bowen Wen, Wei Yang, Jan Kautz, and Stan Birchfield. Foundationpose: Unified 6d pose estimation and tracking of novel objects. *arXiv preprint arXiv:2312.08344*, 2023. [2,](#page-1-1) [3,](#page-2-1) [4,](#page-3-3) [20](#page-19-1)
- <span id="page-14-20"></span>[121] Albert Wu, Ruocheng Wang, Sirui Chen, Clemens Eppner, and C Karen Liu. One-shot transfer of long-horizon extrinsic manipulation through contact retargeting. In *2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pages 13891–13898. IEEE, 2024. [3](#page-2-1)
- <span id="page-14-2"></span>[122] Annie Xie, Lisa Lee, Ted Xiao, and Chelsea Finn. Decomposing the generalization gap in imitation learning for visual robotic manipulation. In *2024 IEEE International Conference on Robotics and Automation (ICRA)*, pages 3153– 3160. IEEE, 2024. [1](#page-0-0)
- <span id="page-14-23"></span>[123] Haofei Xu, Jing Zhang, Jianfei Cai, Hamid Rezatofighi, and Dacheng Tao. Gmflow: Learning optical flow via global matching. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 8121–8130, 2022. [18](#page-17-2)
- <span id="page-14-8"></span>[124] Mengda Xu, Zhenjia Xu, Cheng Chi, Manuela Veloso, and Shuran Song. Xskill: Cross embodiment skill discovery. In *Conference on robot learning*, pages 3536–3555. PMLR, 2023. [2](#page-1-1)
- <span id="page-14-9"></span>[125] Mengda Xu, Zhenjia Xu, Yinghao Xu, Cheng Chi, Gordon Wetzstein, Manuela Veloso, and Shuran Song. Flow as the cross-domain manipulation interface. *arXiv preprint arXiv:2407.15208*, 2024. [2](#page-1-1)

- <span id="page-15-9"></span>[126] Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. *arXiv preprint arXiv:2310.06114*, 1(2):6, 2023. [3](#page-2-1)
- <span id="page-15-3"></span>[127] Xindi Yang, Baolu Li, Yiming Zhang, Zhenfei Yin, Lei Bai, Liqian Ma, Zhiyong Wang, Jianfei Cai, Tien-Tsin Wong, Huchuan Lu, et al. Vlipp: Towards physically plausible video generation with vision and language informed physical prior. *arXiv e-prints*, pages arXiv–2503, 2025. [1](#page-0-0)
- <span id="page-15-0"></span>[128] Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Bill Yuchen Lin, et al. Latent action pretraining from videos. *arXiv preprint arXiv:2410.11758*, 2024. [1](#page-0-0)
- <span id="page-15-4"></span>[129] Tianhe Yu, Chelsea Finn, Annie Xie, Sudeep Dasari, Tianhao Zhang, Pieter Abbeel, and Sergey Levine. One-shot imitation from observing humans via domain-adaptive metalearning. *arXiv preprint arXiv:1802.01557*, 2018. [2](#page-1-1)
- <span id="page-15-8"></span>[130] Chengbo Yuan, Chuan Wen, Tong Zhang, and Yang Gao. General flow as foundation affordance for scalable robot learning. *arXiv preprint arXiv:2401.11439*, 2024. [3](#page-2-1)
- <span id="page-15-6"></span>[131] Kevin Zakka, Andy Zeng, Pete Florence, Jonathan Tompson, Jeannette Bohg, and Debidatta Dwibedi. Xirl: Crossembodiment inverse reinforcement learning. In *Conference on Robot Learning*, pages 537–546. PMLR, 2022. [3](#page-2-1)
- <span id="page-15-2"></span>[132] Qihang Zhang, Shuangfei Zhai, Miguel Angel Bautista, Kevin Miao, Alexander Toshev, Joshua Susskind, and Jiatao Gu. World-consistent video diffusion with explicit 3d modeling. *arXiv preprint arXiv:2412.01821*, 2024. [1](#page-0-0)
- <span id="page-15-12"></span>[133] Zhengyou Zhang. A flexible new technique for camera calibration. *IEEE Transactions on pattern analysis and machine intelligence*, 22(11):1330–1334, 2000. [18,](#page-17-2) [19](#page-18-3)
- <span id="page-15-11"></span>[134] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. *arXiv preprint arXiv:2304.13705*, 2023. [10](#page-9-3)
- <span id="page-15-10"></span>[135] Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li, Siyuan Zhou, Yilun Du, and Chuang Gan. Tesseract: Learning 4d embodied world models. 2025. [3](#page-2-1)
- <span id="page-15-7"></span>[136] Huayi Zhou, Ruixiang Wang, Yunxin Tai, Yueci Deng, Guiliang Liu, and Kui Jia. You only teach once: Learn oneshot bimanual robotic manipulation from video demonstrations. *arXiv preprint arXiv:2501.14208*, 2025. [3](#page-2-1)
- <span id="page-15-1"></span>[137] Jiaming Zhou, Teli Ma, Kun-Yu Lin, Zifan Wang, Ronghe Qiu, and Junwei Liang. Mitigating the human-robot domain discrepancy in visual pre-training for robotic manipulation. *arXiv preprint arXiv:2406.14235*, 2024. [1](#page-0-0)
- <span id="page-15-5"></span>[138] Junzhe Zhu, Yuanchen Ju, Junyi Zhang, Muhan Wang, Zhecheng Yuan, Kaizhe Hu, and Huazhe Xu. Densematcher: Learning 3d semantic correspondence for category-level manipulation from a single demo. *arXiv preprint arXiv:2412.05268*, 2024. [2](#page-1-1)

# Appendix

<span id="page-16-2"></span>We structure the supplement into the following subsections:

- [A](#page-16-3) Details on best practices for video generation.
- [B](#page-16-1) Overview of prompt and examples of video summaries with GPT responses used for video filtering.
- [C](#page-16-0) Results and discussion on our method's mesh-free object tracking version.
- [D](#page-17-0) Details on reducing noise in 6D pose rollouts for stable and realistic motion.
- [E](#page-17-1) Adaptation and implementation details of baseline methods.
- [F](#page-18-1) Comprehensive example of Rekep Predictions and Execution.
- [G](#page-18-2) Discussion of limitations of Tracking using point tracks.
- [H](#page-18-0) Elaboration on our method's robustness.
- [I](#page-19-0) Thorough analysis of errors caused by depth estimation.
- [J](#page-19-2) Discussion regarding the choice between the use of MegaPose and FoundationPose, focusing on trajectory stability.
- $K$  Additional analysis of generated videos and human demos using VBench++ metrics.

## <span id="page-16-3"></span>A. Best Practices for Video Generation

We found that the following practices lead to reliable video generation: (1) having a clean background without visual distractions, (2) minimizing the number of distractor objects in the scene, (3) ensuring objects are reasonably large and viewed from a natural, human-like perspective, (4) ensuring there is one clearly identifiable task that can be performed, (5) using simple and concise text prompts, and (6) setting the relevance factor to 0.7 with the negative prompt "fast motion" led to the most reliable video generations.

## <span id="page-16-1"></span>B. Prompting for Video Filtering and Filtering **Statistics**

The prompt for GPT o1-based filtering is shown in Figure [12.](#page-16-4) We provide GPT o1 with the prompt, a video summary—created by vertically concatenating evenly sampled frames from the video—and the language command (e.g., "pour water"). GPT o1 then responds with "Yes" or "No" to indicate whether the task is successfully performed.

## <span id="page-16-0"></span>C. Mesh-Free Object Tracking

We experiment with a mesh-free object tracking version of our method. Specifically, we use BundleSDF [\[119\]](#page-14-21), which jointly performs 6-DoF object tracking and reconstruction

![](_page_16_Picture_19.jpeg)

## Instructions *Same as previous prompt. Instructions omitted for simplicity.* ## Query Query Task: Place the spatula on the pan Query Image: Prompt

GPT o1 Response

![](_page_16_Picture_22.jpeg)

<span id="page-16-4"></span>1) Hand present? Yes, a hand is visible holding the spatula.<br>2) New object introduced? No, there is no new object introduced; it is still the original<br>spatula and the original pan.<br>3) Reasoning: Although we see a hand hold to the pan. action\_done: No

Figure 12. Examples of prompting GPT o1 to filter generated videos. We sample frames from the generated video and prompt GPT o1 to assess whether the specified task is performed successfully in the video. The top example passes the filtering, while the bottom does not.

from RGBD observations. For the *pouring* task, we evaluate our method using trajectories obtained via BundleSDF over 10 trials and observe a success rate of (90%), matching our default tracking setup. While the BundleSDF paper reports real-time capabilities, we found that its official implementation takes approximately 30 minutes to process each video in practice, which limits its applicability for real-time deployment. In contrast, our default tracker operates in realtime, enabling closed-loop execution and recovery from disturbances as discussed in Sec. [4.5.](#page-9-0) While the BundleSDF paper reports real-time capabilities, we observed significantly higher runtimes in practice with the official implementation. We expect that future advances in model-free tracking will address these efficiency bottlenecks, allowing for real-time mesh-free deployment.

## <span id="page-17-2"></span><span id="page-17-0"></span>D. Smoothing Object Trajectories

To reduce noise and jitter in the estimated object poses, we apply a moving average filter with a fixed sliding window (centered on each point) to the position and orientation components. Translations are smoothed independently along each axis, while orientation is processed similarly after converting from quaternions to rotation vectors. This approach mitigates abrupt changes, resulting in a more stable and realistic object trajectory with smoother transitions.

#### <span id="page-17-1"></span>E. Description of Baselines

Track2Act [\[15\]](#page-10-10): We adapt Track2Act's procedure to our setup preserving its core idea of object-centric trajectory estimation from point tracks. Track2Act generates a future interaction plan by predicting 2D point trajectories (using a DiT-based diffusion model) between an initial image and a goal image, then recovers a sequence of 3D object transforms via Perspective-and-Point (PnP) [\[133\]](#page-15-12).

To integrate this into our pipeline, we use their published checkpoint but modify the input formulation–while the initial image remains identical to our real camera's view, the goal image is taken from the last frame of a generated video rather than being physically captured. We then use PnP on the predicted point tracks along with the initial depth image to estimate the object's rigid motion across frames, thereby defining the end-effector trajectory. We use interpolation between consecutive poses because Track2Act generates only a sparse set of frames, and denser sampling is needed for smooth trajectory estimation and execution. However, we exclude Track2Act's closed-loop residual policy correction, focusing solely on open-loop 6D object-pose estimation and execution. This adaptation allows us to directly evaluate how well a vision-based, open-loop approach generalizes to our setting without additional corrections.

AVDC [\[60\]](#page-12-4): The AVDC approach models action trajectories by synthesizing a task-driven video (using a trained text-conditioned video generation model) and using optical flow from GMFlow [\[123\]](#page-14-23) to estimate dense pixel correspondences. It then reconstructs 3D object motion using an optimization step that refines pose estimates based on the tracked flow and depth information. To improve robustness, AVDC also includes a replanning mechanism that reexecutes the pipeline when predicted motion stagnates.

Since the trained text-conditioned video generation model did not generalize well to our setup, we use the same generated video as in other experiments to ensure a fair comparison. While we do not employ AVDC's replanning strategy, we predict object poses using a similar optimization framework based on flow and depth information.

4D-DPM [\[58\]](#page-12-5): 4D-DPM is designed to track 3D motion of articulated object parts from a single video. It constructs a 3D Gaussian splatting [\[57\]](#page-12-21) representation of the scene to capture object features, then applies GARField [\[59\]](#page-12-22) to cluster the Gaussians into discrete object components. In our adaptation, we modify this to operate on entire objects rather than individual parts. Specifically, we set the clustering parameters to treat the object as a single entity, ensuring that motion estimation is performed at the object level rather than segmenting it into multiple parts. This allows us to track and execute trajectories for the whole object.

Gen2Act [\[14\]](#page-10-9): Gen2Act introduces a video-conditioned policy learning framework that first generates a human video using a video generation model from a scene image and a task description. It then extracts object tracks using BootsTAP [\[28\]](#page-11-26), and trains a policy using behavior cloning with an auxiliary track prediction loss and offline robot demonstrations. At inference, Gen2Act uses the generated video and the learned policy to predict robot actions. Our approach presents a simplified adaptation of this frame-

![](_page_17_Figure_10.jpeg)

<span id="page-17-3"></span>Figure 13. ReKep's output for the pouring task and the resulting robot execution (top-right). The VLM predicts to grasp at keypoint 1, move keypoint 8 above 15 and 7 during transport, and above 15 and 4 for pouring—leading to failed execution.

<span id="page-18-3"></span>work that removes the need for behavior cloning, and offline demonstrations. Instead of using the extracted tracks as an auxiliary loss, we directly process them for pose estimation. To recover 3D object positions, we leverage an initial depth image corresponding to the scene image, allowing us to obtain depth values for the extracted 2D tracks. We apply RANSAC filtering to remove outlier track points and then use the Perspective-n-Point (PnP) [\[133\]](#page-15-12) to estimate the object's 6DoF pose. This adaptation preserves the core idea of leveraging video and track-based signals while eliminating the need for supervised policy learning.

## <span id="page-18-1"></span>F. ReKep Predictions and Executions

A detailed example of ReKep's keypoint and VLM predictions for pouring task is shown in Fig. [13.](#page-17-3) The VLM first predicts grasping the watering can at keypoint 1. For the transport phase, it instructs moving keypoint 8 above keypoint 15, while keeping its height above keypoint 7. For the pouring action, keypoint 8 remains above 15 (to place the spout over the plant) and above keypoint 4 (to induce tilting). The resulting robot execution fails. We attribute most ReKep failures to inaccurate keypoint predictions, as shown in Fig. [14.](#page-18-4) In the lid image, no keypoint appears at the lid handle. In the placing task, keypoints cluster around pan corners. For the sweeping task, the keypoints are generally well-placed, and executions succeeded. Suboptimal initial keypoints lead to inaccurate downstream VLM predictions.

## <span id="page-18-2"></span>G. Limitation of Tracking with Point Tracks

All point tracks fail under extreme rotations, as initially visible points often become occluded. This is a fundamental limitation of any correspondence-based tracking method relying solely on visible surface features. We show this failure in Fig. [15.](#page-18-5) As the object rotates, most initial points are lost, resulting in insufficient 2D-3D correspondences to solve a stable PnP problem. This degrades pose estimation quality, leading to large drift or abrupt jumps in estimated object motion. Such instability cascades into execution errors, often causing the robot to fail the task altogether. As a result, both variants of Gen2Act—despite stronger tracking backbones like CoTracker—still fail under large out-of-plane rotations. In contrast, RIGVid's model-based 6D tracking handles these situations more robustly, as it uses full-object geometry and SE(3) filtering to maintain stable trajectories.

## <span id="page-18-0"></span>H. Additional Robustness Examples

Examples of RIGVid's robustness are shown in Fig. [16.](#page-19-3) In the first row, the robot grasps the object, but due to a misaligned grasp, the object rotates unexpectedly. The robot compensates by rotating it back to the correct orientation and then resumes the planned trajectory, completing the task successfully. In the bottom row, a human perturbs the object

![](_page_18_Picture_7.jpeg)

Figure 14. Examples of ReKep's Keypoint Locations. The keypoint placements are often suboptimal, except for sweeping task, where the keypoints are reasonable.

<span id="page-18-4"></span>![](_page_18_Picture_9.jpeg)

<span id="page-18-5"></span>Figure 15. Gen2Act with BootsTAP, CoTracker, and RIGVid. Blue points denote the tracked points used for PnP; red points represent the reprojected 3D points. For a good PnP solution, these should align, as seen in the first frame. For Gen2Act, the blue points drift significantly from the red ones in later frames, indicating failure in pose estimation due to tracking loss, which leads to failed robot execution.

during execution while it is held by the robot. RIGVid detects the resulting change in the relative transformation and automatically re-aligns the object before continuing. When the human intervenes a second time, RIGVid again corrects the deviation, resulting in successful task completion.

<span id="page-19-1"></span>![](_page_19_Picture_0.jpeg)

Figure 16. Additional examples of RIGVid's robustness. In the top row, RIGVid recovers from a faulty initial grasp by reorienting the object before continuing execution. In the bottom row, it corrects for external disturbances on the object when a human pushes it mid-execution, realigning and successfully completing the task.

#### <span id="page-19-0"></span>I. Errors from Depth Estimation

![](_page_19_Figure_3.jpeg)

Figure 17. Impact of Depth Estimation Errors on RIGVid performance. Errors in monocular depth estimation result in worse performance of generated and real videos. RIGVid achieves perfect success across all tasks with real videos and real depth.

In Fig. [17,](#page-19-4) we isolate the impact of depth estimation errors. Robot executions on real videos with real depth (captured using an RGBD camera) achieve a 100% success rate, whereas executions from real videos with generated depth result in an 85% average success. Similarly, executions from Kling V1.6-generated videos with generated depth also achieve 85% success, suggesting that the primary source of error lies in monocular depth estimation. Upon inspection, we observe two common undesirable behaviors in the predicted depth: inaccurate depth values and temporal flickering. An example of inaccurate depth is shown in Fig. [18.](#page-19-5) In the generated video, when the spatula is brought close to the camera, the depth changes by only 6.8 cm, which is visibly inconsistent with the video and likely much smaller than the real-world change. Inaccuracies also occur in real videos, as shown in the figure—the head of the spatula is estimated to be far from the camera, despite appearing close, revealing another failure mode in monocular depth estimation. Flickering is shown in Fig. [19.](#page-20-2) Although the position of the watering can relative to the camera remains nearly unchanged across three consecutive frames, the estimated depth varies significantly. The zoomed-in region on the right shows the can appearing much whiter than on the left, indicating a substantial change in predicted depth. The average depth of the can

<span id="page-19-5"></span><span id="page-19-3"></span>(a) Generated Video

![](_page_19_Picture_7.jpeg)

![](_page_19_Figure_8.jpeg)

![](_page_19_Picture_9.jpeg)

Figure 18. Errors in Monocular Depth Estimation. In the generated video (top), the depth of the spatula changes only slightly despite a large visual change. In the real video (bottom), the spatula's head is predicted to lie farther away, contradicting the visual appearance.

<span id="page-19-4"></span>changes from 40.1 cm to 38.2 cm–a 1.9 cm difference over just 0.066 seconds–which is physically implausible for the generated video. We find similar flickering behavior in real videos as well, where the depth changes from 43.2 cm to 40.9 cm in the given example–a 2.3 cm difference. Since errors in the generated depth are the main source of failure, we also tested removing it entirely by estimating object pose directly from the RGB frames of the generated video using MegaPose. However, this approach leads to even more unstable and noisy trajectories, as detailed in App. [J.](#page-19-2)

## <span id="page-19-2"></span>J. Choice between MegaPose and Foundation-Pose

We compare trajectory stability from MegaPose [\[63\]](#page-12-11) and FoundationPose [\[120\]](#page-14-3) by computing the translational and rotational RMS jitter. For each method, we apply a Gaussian smoothing filter ( $\sigma = 2$  frames) to the raw SE(3) pose sequences, compute the residual between original and smoothed trajectories, and then calculate:

$$
\text{jitter}_{\text{trans}} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \|\Delta \mathbf{t}_t\|^2}, \quad \text{jitter}_{\text{rot}} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} \theta_t^2},
$$

where  $\Delta t_t$  is the translational residual at frame t, and  $\theta_t$ is the angular magnitude (in radians) of the relative rotation  $R^{-1}_{\text{smooth}}R_{\text{raw}}$ , converted to degrees. Metrics are aver<span id="page-20-2"></span><span id="page-20-0"></span>(a) Generated Video **Avg Depth: 40.1 cm Avg Depth: 38.227 cm**

(b) Real Video **Avg Depth: 43.275 cm Avg Depth: 40.969 cm**

Figure 19. Flickering in Depth Prediction. We show three consecutive frames of the video and its corresponding predicted depth. The depth of the watering can change noticeably across frames—appearing significantly whiter in the third frame despite minimal actual motion. We observe this behavior in both generated and real videos.

aged over ten pouring trajectories from generated videos.

MegaPose yields an average translational RMS jitter of 0.0045m and rotational RMS jitter of 37.47°, whereas FoundationPose achieves 0.0029m translational and 14.31° rotational jitter. This demonstrate that FoundationPose produces significantly smoother and more stable trajectories. Additionally, it allows for real-time tracking during the execution, making RIGVid robust to external disturbances.

## <span id="page-20-1"></span>K. Comparing Video Generative Models

To further assess video quality, we report VBench++ [\[50\]](#page-11-24) metrics in Table [3](#page-20-3) and explain them below. The numbers in the table are scaled  $100\times$  for easier interpretation. We collect these metrics on 40 randomly selected and unfiltered videos per model, 10 for each of the four tasks. Kling v1.6 outperformed the other models on most metrics but performed similarly or worse in video-text consistency and dynamic degree. Human evaluations discussed in Sec. [4.2](#page-4-1) suggest that the video-text consistency and I2V subject consistency are not reliable indicators of whether a generated video correctly follows a given command. Sora scored high on dynamic degree, likely due to its tendency to drastically alter the scene, resulting in exceptionally large motions. Generated videos from these models and their corresponding metrics are shown in Fig. [20](#page-21-1) and further details on these metrics can be found the next section.

#### VBench++ Metric Definitions:

• Subject Consistency. Subject consistency describes whether subjects' appearance remain consistent, which is computed by DINOv1 [\[20\]](#page-10-24) similarities across video frames. • Background Consistency. Background temporal consistency by CLIP [\[98\]](#page-13-23) similarities across frames.

• Motion Smoothness. Evaluates smoothness of videos by utilizing video frame interpolation model AMT [\[71\]](#page-12-23).

• Dynamic Degree. Describes whether the video contains large motions as a binary metric.

• Aesthetic Quality. Human perceived artistic and beauty value such as photo-realism, layout and color harmony.

• Imaging Quality. Assesses the presence of distortion in a video, such as noisiness, blurriness, and over-exposure.

• Video-Text Consistency. Text-to-video alignment score calculated by ViCLIP [\[116\]](#page-14-22).

• I2V Subject Consistency. Similarity between subjects in input image and each video frame, as well as similarity between consecutive frames. Features are extracted from DINOv1 [\[20\]](#page-10-24).

<span id="page-20-3"></span>

| <b>Metrics</b>                 | <b>Video Generation Models</b> | Human      |       |              |
|--------------------------------|--------------------------------|------------|-------|--------------|
|                                | Kling V1.6                     | Kling V1.5 | Sora  | <b>Demos</b> |
| <b>Subject Consistency</b>     | 96.34                          | 91.66      | 83.09 | 94.91        |
| <b>Background Consistency</b>  | 96.64                          | 93.97      | 89.34 | 95.00        |
| <b>Motion Smoothness</b>       | 99.68                          | 99.57      | 99.06 | 99.51        |
| Dynamic Degree                 | 52.5                           | 57.5       | 70.0  | 80.0         |
| <b>Aesthetic Quality</b>       | 51.75                          | 49.77      | 46.22 | 49.30        |
| <b>Imaging Quality</b>         | 72.80                          | 71.48      | 68.68 | 72.52        |
| Video-Text Consistency         | 22.01                          | 22.61      | 21.42 | 21.57        |
| <b>I2V Subject Consistency</b> | 97.88                          | 95.96      | 89.09 | 97.89        |

Table 3. Video generation quality metrics for real human demonstration videos and different models. Higher values indicate better quality. Kling v1.6 performs comparably to or surpasses other models on most metrics.

![](_page_20_Figure_18.jpeg)

## **SORA**

<span id="page-21-0"></span>![](_page_21_Picture_1.jpeg)

VT Const : 0.267 I2V Subj. Const : 0.887 Subj. Const : 0.808

VT Const : 0.221 I2V Subj. Const : 0.792 Subj. Const : 0.746

VT Const : 0.208 I2V Subj. Const : 0.930 Subj. Const : 0.915

VT Const : 0.218 I2V Subj. Const : 0.977 Subj. Const : 0.839

VT Const : 0.244 I2V Subj. Const : 0.989 Subj. Const : 0.936

VT Const : 0.195 I2V Subj. Const : 0.978 Subj. Const : 0.731

VT Const : 0.231 I2V Subj. Const : 0.989 Subj. Const : 0.982

VT Const : 0.201 I2V Subj. Const : 0.865 Subj. Const : 0.965

![](_page_21_Picture_10.jpeg)

Kling AI v1.6

Kling AI v1.5

![](_page_21_Picture_12.jpeg)

VT Const : 0.217 I2V Subj. Const : 0.995 Subj. Const : 0.975

VT Const : 0.208 I2V Subj. Const : 0.964 Subj. Const : 0.969

VT Const: 0.245 I2V Subj. Const: 0.9965 Subj. Const: 0.965

<span id="page-21-1"></span>VT Const : 0.188 I2V Subj. Const : 0.955 Subj. Const : 0.951

Figure 20. Qualitative Comparison of Different Video Generative Models. Videos from the three video generation models are shown using evenly sampled frames, along with VBench++ [\[50\]](#page-11-24) metrics: video-text consistency, image-to-video subject consistency, and subject consistency. Kling v1.6 scores highest on these metrics, followed by Kling v1.5 and then Sora.

<span id="page-22-0"></span>![](_page_22_Picture_0.jpeg)

Figure 21. Qualitative comparison of video generation. Sora-generated videos often alter the scene layout and objects. Kling V1.5 produces more plausible results but includes physically implausible elements. Kling V1.6 better preserves scene fidelity and closely follows the human command.