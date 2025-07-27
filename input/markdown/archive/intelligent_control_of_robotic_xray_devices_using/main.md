# Intelligent Control of Robotic X-ray Devices using a Language-promptable Digital Twin

Benjamin D. Killeen<sup>1\*</sup>, Anushri Suresh<sup>1</sup>, Catalina Gomez<sup>1</sup>, Blanca Íñigo<sup>1</sup>, Christopher Bailey<sup>2</sup>, Mathias Unberath<sup>1</sup>

1\*Laboratory for Computational Sensing and Robotics, Johns Hopkins University, Baltimore, 21218, MD, USA.

<sup>2</sup>Department of Interventional Radiology, Johns Hopkins University, Baltimore, 212187, MD, USA.

\*Corresponding author(s). E-mail(s): killeen@jhu.edu; Contributing authors: asures13@jhu.edu; cgomezc1@jhu.edu; binigo2@jhu.edu; Christopher.Bailey@jhmi.edu; unberath@jhu.edu;

#### Abstract

Purpose: Natural language offers a convenient, flexible interface for controlling robotic C-arm X-ray systems, making advanced functionality and controls easily accessible. However, enabling language interfaces requires specialized AI models that interpret X-ray images to create a semantic representation for languagebased reasoning. The fixed outputs of such AI models fundamentally limits the functionality of language controls that users may access. Incorporating flexible and language-aligned AI models that can be prompted through language control facilitates more flexible interfaces for a much wider variety of tasks and procedures.

Methods: Using a language-aligned foundation model for X-ray image segmentation, our system continually updates a patient digital twin based on sparse reconstructions of desired anatomical structures. This allows for multiple autonomous capabilities, including visualization, patient-specific viewfinding, and automatic collimation from novel viewpoints, enabling complex language control commands like "Focus in on the lower lumbar vertebrae."

Results: In a cadaver study, multiple users were able to visualize, localize, and collimate around structures from across the torso region using only verbal commands to control a robotic X-ray system, with 84% end-to-end success. In post hoc analysis of randomly oriented images, our patient digital twin was able to localize 35 commonly requested structures from a given image to within

 $51.68 \pm 30.84$  mm, which enables localization and isolation of the object from arbitrary orientations.

Conclusion: Overall, we show how intelligent robotic X-ray systems can incorporate physicians' expressed intent directly. Existing foundation models for intra-operative X-ray image analysis exhibit certain failure modes. Nevertheless, our results suggest that as these models become more capable, they can facilitate highly flexible, intelligent robotic C-arms.

Keywords: Image-guided surgery, foundation models, large language models, voice user interfaces, segment anything, fluoroscopy

# 1 Introduction

Natural language offers an appealing interface for commanding robotic X-ray devices in surgery, allowing physicians to express their imaging needs rather than execute them manually [\[1\]](#page-9-0). When equipped with artificial intelligence (AI) models capable of analyzing intra-operative images, voice-controlled robotic C-arms effectively become intelligent assistants for image-guided surgery, with the potential to reduce radiation exposure [\[2\]](#page-9-1), avoid complications [\[3\]](#page-9-2), streamline procedures [\[4\]](#page-9-3), and improve overall patient outcomes. So far, however, intelligent systems for X-ray image-guided surgery have relied on specialized, task-specific models with fixed outputs, limiting their general application. Meanwhile, voice interfaces for general robotics are rapidly accelerating due to the rise of multi-modal foundation models [\[5\]](#page-9-4), which are characterized by large-scale training and generalizability for a wide range of downstream tasks [\[6\]](#page-9-5). Although foundation models have been developed for the X-ray domain [\[7–](#page-10-0) [9\]](#page-10-1), they are generally limited to diagnostic chest images, and it remains unclear how to incorporate them into a voice-user interface for commanding robotic C-arms more generally.

Here, we leverage an X-ray foundation model to support intelligent capabilities in a voice-controlled robotic C-arm, including visualization, collimation, and patientspecific viewfinding. Our approach continually updates a patient digital twin using images acquired during surgery. A large language model (LLM) interprets spoken commands, like "focus on the lower lumbar vertebrae" and extracts the intended action. Low-level actions, like "roll over 30 degrees" are converted directly to joint movements by the LLM, while high-level actions are associated with a language prompt, *i.e.* "lower lumbar vertebrae," potentially based on past commands. The LLM then sends the action and prompt to the digital twin, which uses a multi-modal segment-anything model (FluoroSAM [\[9\]](#page-10-1)) to analyze past images and aggregate 3D information about the desired anatomy. This enables future acquisitions from unseen viewpoints to have appropriate collimation for the desired anatomy, limiting radiation exposure while still ensuring the structure remains in view. This also allows for automatic viewfinding, where the robotic C-arm can adjust its position and orientation toward a desired anatomy, based on the current understanding.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

![](_page_2_Figure_1.jpeg)

(d) Patient-specific Viewfinding (c) Visualization (e) 3D collimation Fig. 1: We present a natural language interface for commanding robotic X-ray devices using a multi-modal foundation model for X-ray imaging. Our approach, which we demonstrate in a real-time cadaver study (a), uses a large language model to parse the desired action and suitable prompt from the spoken input. The digital twin (b) uses FluoroSAM [\[9\]](#page-10-1) to segment anatomies based on the prompt, supporting real-time visualization (c), patient-specific viewfinding (d), and 3D collimation.

We demonstrate our fully integrated system in a cadaver study with spoken prompts and in post-hoc analysis of randomly acquired X-ray images registered to a full-body CT. Our results demonstrate the robustness of the voice interface in realtime, with a success rate of 83.54%, and the accuracy of the digital twin in aligning with ground truth structures in the CT. In *post hoc* analysis of 1990 possible image subsets, the digital twin yields an average  $51.68 \pm 30.84$  mm localization error, with a 3D bounding box precision and recall of 0.26 and 0.70 for tested prompts, respectively. Our work paves the way for general-purpose AI surgical assistants in image-guided surgery, supporting a wide range of high-level capabilities by incorporating.

![](_page_2_Picture_4.jpeg)

## 2 Related Work

Previous works have explored integrating natural language into robotic systems, given the promise of more intuitive and flexible interactions with users, particularly through voice and text commands [\[10\]](#page-10-2). Such systems may demand real-time language understanding, interpretation, and generation for inference while operating and executing commands constrained by real-world physics and the environmental context [\[11\]](#page-10-3). Recent developments in foundation models are appealing for robotic applications considering the wide range of downstream tasks they can be adapted to and their ability to process multimodal data [\[5\]](#page-9-4). The combination of existing language models with vision-based models enables the development of robots that can engage in conversations with users through textual interfaces [\[12\]](#page-10-4) or real-time instructions [\[13\]](#page-10-5) for object navigation, modifying robot trajectories [\[14,](#page-10-6) [15\]](#page-10-7), and fine-grained manipulation [\[16\]](#page-10-8). Importantly, these language-conditioned models ground user text queries to the robot's visual observations of its environment, generating actions within the robot's capabilities.

Likewise, foundation models in medicine can interpret multimodal data and offer communication via multiple modalities (text, visualizations, etc.), enabling novel human-machine interactions and improved generalization abilities [\[17\]](#page-10-9). The development of medical foundation models has been particularly successful in radiology, especially for chest X-rays, where abundant image-text paired data facilitates finetuning of general-domain models. These models can leverage existing image-based analysis methods [\[18\]](#page-10-10) and clinical large language models (e.g., BioGPT [\[19\]](#page-10-11)) to train vision-language assistants with conversational abilities [\[20\]](#page-10-12) and successful at both image interpretation and textual understanding [\[7\]](#page-10-0), among a wide range of tasks identified through collaborative work with medical experts [\[21\]](#page-11-0). Segmentation is another fundamental task in medical image analysis where foundation models have shown improvements with respect to specialized models. Existing medical Segment Anything models [\[8,](#page-10-13) [22\]](#page-11-1) follow the fine-tuning strategy of a general-domain foundation model for segmentation on image-mask pairs covering multiple imaging modalities and disease types. While promptable segmentation models can adjust to different use cases and users' needs, point-based prompts can be ambiguous and bounding boxes require expert input. Instead, text prompts enable broader accessibility, especially in scenarios like surgery where voice commands (mapped to text) can streamline operating room (OR) workflows [\[23\]](#page-11-2). FluoroSAM [\[9\]](#page-10-1) supports text-only prompting for segmentation of anatomical structures in X-rays, for which scalable data generation in simulation was needed [\[4,](#page-9-3) [24,](#page-11-3) [25\]](#page-11-4).

Despite multiple efforts leveraging the appealing features and flexibility of multimodal foundation models, it remains unclear how these benefits translate beyond diagnostic tasks to interventional procedures. In particular, how to include languagebased interactions in interventional radiology is crucial for developing smart OR assistants grounded on the complexities of X-ray image-guided procedures.

## 3 Methods

Our approach consists of a command interpretation system and a digital twin that supports multiple high-level capabilities using a language-aligned foundation model. Following [\[1\]](#page-9-0), we define a communication protocol for an LLM to interpret natural language as machine-readable actions. The user gives natural commands like "Show me the right lung," and the LLM returns a machine-readable action, in this case a highlight action with the text-prompt "right lung." In our experiments, we used a lavalier microphone clipped to the user's lead apron to record speech, with a mute button to prevent cross-talk. In practice, we envision the microphone being unmuted using a foot pedal or other sterility-preserving mechanism. Live speech-to-text is accomplished via OpenAI Whisper. Using low-level commands, such as "roll over 30 degrees" the user can adjust independent axes and take an X-ray, as described in [\[1\]](#page-9-0). High-level commands trigger one of three new high-level actions using language-aligned foundation model: visualization (highlight), collimation (collimate), and patient-specific viewfinding (view). Visualization is a straightforward function for displaying, in real time, the segmentation of a given anatomy on the current image. Collimation and patientspecific viewfinding rely on the digital twin to localize the desired structure in 3D. For example, the portion of the instruction that specifies the view action reads:

A 'view' action requests a specific view:

action ;view;view˙name;prompt

where 'view˙name=–ap—lateral—current˝'

and 'prompt' is derived from the user input to be used as a language prompt for a segmentation model. 'prompt' should be a concise description of the desired anatomy or structure. If no anatomy is specified , use 'prompt=current'.

In total, our instruction set and examples are 1655 words (4356 tokens) total. This is supported by the maximum input token length of the GPT-4o (128,000) model used in our experiments, and results in an LLM latency of less than 1 second. This is small compared to the robotic movement and acquisition times of the Loop-X, which are typically on the order of 10 or more seconds. For safety reasons, the Loop-X requires users to confirm these actions using a physical button; however, during our experiments, the user never initiated actions directly from the Brainlab user interface. This physical confirmation is a specific requirement for the Loop-X device, but future systems might remove confirmation for non-radiation actions.

### 3.1 A Digital Twin Based on a Language-aligned Foundation Model

We describe a highly flexible patient digital twin that incorporates casually acquired images to reconstruct desired anatomical structures in 3D. This is accomplished through text-prompted segmentation of past-structures using FluoroSAM [\[9\]](#page-10-1), a segment-anything model for X-ray imaging that uses language to disambiguate overlapping structures. Traditionally, segment-anything models (SAMs) predict a valid mask for a given prompt, which may be a point, mask, or 2D bounding box. FluoroSAM incorporates a CLIP [\[26,](#page-11-5) [27\]](#page-11-6) embedding of an anatomy description. We

examine the use of FluoroSAM without additional point prompts, using only the extracted prompt from the LLM to obtain a segmentation.

Formally, let  $f(\mathbf{u}, I, t)$  denote the value of the FluoroSAM logits at  $\mathbf{u} \in \mathbb{R}^2$  for image I with prompt t. Let  $I_0 \in \mathbb{R}^{H_0 \times W_0}$  and  $P_0 = K[R_0 | t_0]$  denote the image being examing as well as its projection matrix relative to a fixed coordinate system, such as the optical marker in Fig. [1.](#page-2-0) Following [\[3\]](#page-9-2), to support 3D reconstruction we identify a set of images that have a minimum acute angle of 30°. If two images are available from similar viewpoints  $(< 10°$  angle), we take the most recent, so as to maintain an up-todate digital twin. Practically, this is reasonable given the routine workflows in many image-guided procedures, which alternate between views such as the anteroposterior (AP) and lateral [\[4,](#page-9-3) [28\]](#page-11-7). If such images are not available, such as at the start of the procedure, our system still supports visualization, collimation, and translation of the imaging center based on  $f(I_0, t)$ . Let  $I_0, I_1, \ldots, I_{n-1}$  denote the images identified from the history, including the current image.  $n$  is typically between 2 and 5, given the constraints on image selection. We define the sparse reconstruction of the anatomy as the set of 3D points where the mean value of  $f$  is at least 0.5, and the point is present in at least 2 masks:

$$
\mathbf{X} = \left\{ \mathbf{x} \in \mathbb{R}^3 \; \middle| \; \vec{0} \le \mathbf{P}_0 \tilde{\mathbf{x}} < \begin{bmatrix} W_0 \\ H_0 \end{bmatrix} \; \text{and} \; \left| \mathcal{I}_\mathbf{x} \right| \ge 2 \; \text{and} \; \frac{1}{\left| \mathcal{I}_\mathbf{x} \right|} \sum_{i \in \mathcal{I}_\mathbf{x}} f\left( \mathbf{P}_i \tilde{\mathbf{x}}, I_i, t \right) \right\} \tag{1}
$$

where  $\mathcal{I}_{\mathbf{x}} = \{i \mid f(\mathbf{P}_i \tilde{\mathbf{x}}, I_i, t) > 0.5\}$  is the set of images for which a point **x** is in the mask. This is computed on GPU by backprojecting each heatmap value through a patient volume with 3 mm isocentric spacing, which is sufficient given the 0.3 mm detector pixel size on our device. Fig. [1b](#page-2-0) shows sparse reconstructions of several prompts over the torso, including "lower lumbar vertebrae" from the given images. As can be observed, the quality of these reconstructions depends in large part on the quality and consistency of the segmentation mask across images, but we show in our experiments that they are sufficient to locate the centroids of observed structures, and they enable significant collimation of desired structures. Our system supports viewfinding based on AP and lateral orientations, as well as previous viewpoints.

## 4 Experiments

We evaluate our approach in terms of three aspects, based on a cadaveric imaging study using a Brainlab Loop-X robotic X-ray device. First, we examine the 2D performance of FluoroSAM, which was trained using digitally reconstructed radiographs, on real images. Since the current version of FluoroSAM was designed with point-based prompts in mind, it has notable failure modes when using text-only prompting. Second, we evaluate the accuracy of our 3D digital twin for prompts where FluoroSAM achieves a sufficient DICE score. Finally, we examine the real-world usability of the fully integrated system for spoken prompts across a variety of anatomies and views. The cadaveric specimen included the torso section from the mid-femur to the T2 vertebra, not including the arms, from a 60 year-old female donor with a living height of 157 cm and living weight of 50 kg. The specimen was thawed at  $4^{\circ}$ C for 6 days prior to the study. All fluoroscopic images were acquired with navigation relative to a fixed patient array. To obtain complete ground truth masks, we stitched together 4 navigated cone-beam CT images acquired with the Loop-X immediately following the study and project organ segmentations [\[29\]](#page-11-8) onto each image. These are combined according to the specifications of each prompt, such as by joining L3, L4, and L5 for the "lower lumbar vertebrae." 80 X-ray images were acquired over the course of the study, of which 46 were from unique viewpoints. The specimen was not moved throughout the study, aside from slight soft tissue deformation due to settling.

#### 4.1 FluoroSAM Performance

We evaluate FluoroSAM's performance using prompts obtained from an attending interventional radiologist that describe primary structures in the torso and pelvic region. Overall, FluoroSAM is able to segment large structures based on text-only prompting, but it fails to localize small, repeating structures and small organs. This is reasonable, since small organs are generally not visible without contrast agent. Moreover, a single image may not provide enough visual context to distinguish between similar vertebrae levels (0.04 DICE) or rib bones (0.02 DICE) without additional point prompting, as in [\[9\]](#page-10-1). However, we do observe reasonable performance when segmenting larger structures, such the vertebrae  $(0.78 \pm 0.15)$ , vertebrae sections  $(0.72 \pm 0.18)$ lumbar and  $0.64 \pm 0.20$  $0.64 \pm 0.20$  $0.64 \pm 0.20$  thoracic), the ribcage  $(0.54 \pm 0.12)$ , etc. Fig. 2 shows the DICE score for the top 40 prompts, which we use in our study, and example predictions. These include organs which are in the base TotalSegmentator [\[29\]](#page-11-8) classes from which FluoroSAM was trained, as well as groups of organs never seen before, like "lower lumbar vertebrae," which we define as L3 - L5. Table [A1](#page-12-0) details the DICE score and centroid error for the top 40 prompts used. We also note that FluoroSAM is able to localize many structures, with an average centroid error of  $58.87 \pm 50.29$  mm among the top 40 DICE prompts. The detector size is 430 mm square.

#### 4.2 Digital Twin Reconstruction

We evaluate our digital twin in terms of its ability to localize and isolate desired structures in a given image, using randomly selected images as secondary shots. Using each of the 46 unique images in our study as the primary image, we randomly sample additional images from  $n = 2$  to  $n = 5$ , totaling 1990 unique subsets of images with common structures for which FluoroSAM's DICE score exceeds 0.3, to isolate the ability of the digital twin to localize structures given a reasonable segmentation. We ignore the acquisition time for the purpose of evaluation. We evaluate the centroid error of the desired anatomy relative to its overlap with the "current" image, in alignment with physicians' expectations. For these anatomy, our system is able to localize desired structures in 3D to within  $51.68 \pm 30.84$ , among the top 40 prompts. Some structures, like individual femur pones and gluteous muscle groups, are more easily localized, with a centroid error of less than 40 mm, while others, like the liver, pose more of a challenge, possibly due to their asymmetric shape. Table [A2](#page-13-0) in the supplement details these results.

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 2: Performance and example masks for text-only prompting with FluoroSAM. Although certain classes struggle without additional point-based prompts, the model correctly localizes many structures based on CLIP [\[26,](#page-11-5) [27\]](#page-11-6) embeddings of natural language prompts, extracted by the LLM protocol, including unseen prompts like "lower lumbar vertebrae."

On the Loop-X, 3D collimation is achieved by defining a bounding box in the patient coordinate system which all future acquisitions should collimate to. We evaluate our digital twin's value for collimation in terms of the bounding box statistics in the patient coordinate system of the post-study CT. The bounding box recall provides a measure of how much of the desired anatomy is not present in the collimation volume, while the precision measures how tightly our digital twin is able to isolate the desired anatomy. Over the tested prompts, the average precision was  $0.26 \pm 0.22$  and the average recall was  $0.70 \pm 0.26$ , indicating high coverage of the desired structure and reasonable isolation even when cropped tightly. Fig. [3](#page-8-0) shows the performance for each prompt tested.

#### 4.3 Real-time Cadaver Study

In a real-time study, our approach was able to visualize, localize, and collimate to anatomical structures based solely on voice control with a high success rate. Success was evaluated based on the end-to-end success, including failures due to poor transcription or movement constraints. Over 158 prompts,  $38 / 46 (82.6%)$  visualization actions,  $17 / 28 (60.7%)$  collimations, and  $10 / 12 (83.33%)$  were successful. Other lowlevel actions, like "Take a shot" or "Roll over 30 degrees" accounted for the remainder

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>![](_page_8_Figure_1.jpeg)

Fig. 3: The reconstruction error of the digital twin in terms of localization and collimation of desired structures. We observe a tendency toward better localization and isolation for structures which FluoroSAM segments more easily.

of the prompts. Of these, 40 were spoken by a person with an Indian accent and 118 by a person with an American accent, with similar success rates of 87.5% and 82.2%, respectively. We observed some accent-specific transcription errors of medical words, like "Collimate" as "column 8" when spoken in isolation. Fig. [1a](#page-2-0) and the video supplement show the study, which had an overall success rate of 83.54%.

## 5 Discussion and Conclusion

There are notable limitations in the approach outlined here. As we observed in our study, the performance of the foundation model, FluoroSAM, is severely limited when used without additional point prompts. Although we identified a number of anatomical structures in the torso region which it was able to localize effectively, a more capable model would lend itself to a wider variety of tasks and anatomies. Additionally, speechto-text systems exhibit notable bias toward users with different accents and languages, as we observed in our study, and make transcription errors when lacking the medical

context. Mechanical solutions, like a foot-pedal mute switch, may reduce the need to differentiate intentional commands from other communications among the surgical team, but it is an inelegant solution. As LLMs become faster and more capable, they are exhibiting a greater understanding of users' intent in a more general context, including when the system is being addressed. Moving forward, there is a need for more detailed evaluation of potential human-robot interactions in the operating room. This is to better understand physicians' needs and how they may express them throughout a given procedure. Initial efforts have collected data for surgical workflow analysis that includes system commands by medical personnel to operate devices in the OR [\[30\]](#page-11-9), but patient privacy concerns prevent open data sharing. Additionally, data from current ORs is fundamentally limited because it contains surgeon-technologist interactions, but validation is needed before intelligent systems can be deployed in patient care. One avenue, which promises to enhance both data collection and human-centered design, is the use of virtual reality environments for image-guided procedures [\[31\]](#page-11-10). These allow surgeons and technologists to work through communication challenges with intelligent assistance systems in a more accessible, scalable way.

We have shown that a language-aligned foundation model can be used to facilitate flexible control of a robotic C-arm device. As images are acquired in the course of a procedure, our approach continually updates a digital twin to allow visualization, collimation, and viewfinding for desired anatomies based on natural language prompting. In a cadaver study, the fully integrated system successfully interprets commands and performs requested actions.

### References

- <span id="page-9-0"></span>[1] Killeen, B.D., et al.: Take a shot! Natural language control of intelligent robotic X-ray systems in surgery. Int. J. CARS 19(6), 1165–1173 (2024) [https://doi.org/](https://doi.org/10.1007/s11548-024-03120-3) [10.1007/s11548-024-03120-3](https://doi.org/10.1007/s11548-024-03120-3)
- <span id="page-9-1"></span>[2] Kausch, L., et al.: Toward automatic C-arm positioning for standard projections in orthopedic surgery. Int. J. CARS  $15(7)$ , 1095–1105 (2020) [https://doi.org/10.](https://doi.org/10.1007/s11548-020-02204-0) [1007/s11548-020-02204-0](https://doi.org/10.1007/s11548-020-02204-0)
- <span id="page-9-2"></span>[3] Killeen, B.D., et al.: An autonomous X-ray image acquisition and interpretation system for assisting percutaneous pelvic fracture fixation. Int. J. CARS 18(7), 1201–1208 (2023) <https://doi.org/10.1007/s11548-023-02941-y>
- <span id="page-9-3"></span>[4] Killeen, B.D., et al.: Pelphix: Surgical Phase Recognition from X-Ray Images in Percutaneous Pelvic Fixation. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2023, pp. 133–143. Springer, Cham, Switzerland (2023). [https://doi.org/10.1007/978-3-031-43996-4](https://doi.org/10.1007/978-3-031-43996-4_13) 13
- <span id="page-9-4"></span>[5] Kawaharazuka, K., et al.: Real-world robot applications of foundation models: a review. Adv. Rob. (2024)
- <span id="page-9-5"></span>[6] Bommasani, R., et al.: On the Opportunities and Risks of Foundation Models.

arXiv (2021) <https://doi.org/10.48550/arXiv.2108.07258> [2108.07258](https://arxiv.org/abs/2108.07258)

- <span id="page-10-0"></span>[7] Chen, Z., et al.: Chexagent: Towards a foundation model for chest x-ray interpretation. arxiv 2024. arXiv preprint arXiv:2401.12208
- <span id="page-10-13"></span>[8] Ma, J., et al.: Segment anything in medical images. Nat. Commun. 15(654), 1–9 (2024) <https://doi.org/10.1038/s41467-024-44824-z>
- <span id="page-10-1"></span>[9] Killeen, B.D., et al.: FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation. arXiv (2024) <https://doi.org/10.48550/arXiv.2403.08059> [2403.08059](https://arxiv.org/abs/2403.08059)
- <span id="page-10-2"></span>[10] Tellex, S., et al.: Robots that use language. Annual Review of Control, Robotics, and Autonomous Systems  $3(1)$ , 25–55 (2020)
- <span id="page-10-3"></span>[11] Vemprala, S.H., et al.: Chatgpt for robotics: Design principles and model abilities. IEEE Access (2024)
- <span id="page-10-4"></span>[12] Dai, Y., et al.: Think, act, and ask: Open-world interactive personalized robot navigation. In: 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 3296–3303 (2024). IEEE
- <span id="page-10-5"></span>[13] Lynch, C., et al.: Interactive language: Talking to robots in real time. IEEE Robotics and Automation Letters (2023)
- <span id="page-10-6"></span>[14] Martinez-Baselga, D., et al.: Hey robot! personalizing robot navigation through model predictive control with a large language model. arXiv preprint arXiv:2409.13393 (2024)
- <span id="page-10-7"></span>[15] Bucker, A., et al.: Reshaping robot trajectories using natural language commands: A study of multi-modal data alignment using transformers. In: 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 978–984 (2022). IEEE
- <span id="page-10-8"></span>[16] Shridhar, M., et al.: Cliport: What and where pathways for robotic manipulation. In: Conference on Robot Learning, pp. 894–906 (2022). PMLR
- <span id="page-10-9"></span>[17] Moor, M., et al.: Foundation models for generalist medical artificial intelligence. Nature 616(7956), 259–265 (2023)
- <span id="page-10-10"></span>[18] Gao, C., et al.: Synthetic data accelerates the development of generalizable learning-based algorithms for X-ray image analysis. Nat. Mach. Intell. 5, 294–308 (2023) <https://doi.org/10.1038/s42256-023-00629-1>
- <span id="page-10-11"></span>[19] Luo, R., et al.: Biogpt: generative pre-trained transformer for biomedical text generation and mining. Briefings in bioinformatics 23(6), 409 (2022)
- <span id="page-10-12"></span>[20] Li, C., et al.: Llava-med: Training a large language-and-vision assistant for

biomedicine in one day. Advances in Neural Information Processing Systems 36 (2024)

- <span id="page-11-0"></span>[21] Yildirim, N., et al.: Multimodal healthcare ai: identifying and designing clinically relevant vision-language applications for radiology. In: Proceedings of the CHI Conference on Human Factors in Computing Systems, pp. 1–22 (2024)
- <span id="page-11-1"></span>[22] Shen, Y., et al.: Fastsam3d: An efficient segment anything model for 3d volumetric medical images. arXiv preprint arXiv:2403.09827 (2024)
- <span id="page-11-2"></span>[23] Kim, J.H., et al.: Development of a Smart Hospital Assistant: integrating artificial intelligence and a voice-user interface for improved surgical outcomes. In: Proceedings Volume 11601, Medical Imaging 2021: Imaging Informatics for Healthcare, Research, and Applications vol. 11601, pp. 159–170. SPIE, ??? (2021). <https://doi.org/10.1117/12.2580995>
- <span id="page-11-3"></span>[24] Killeen, B.D., et al.: In silico simulation: a key enabling technology for nextgeneration intelligent surgical systems. Prog. Biomed. Eng. 5(3), 032001 (2023) <https://doi.org/10.1088/2516-1091/acd28b>
- <span id="page-11-4"></span>[25] Unberath, M., et al.: DeepDRR – A Catalyst for Machine Learning in Fluoroscopy-Guided Procedures. In: Medical Image Computing and Computer Assisted Intervention – MICCAI 2018, pp. 98–106. Springer, Cham, Switzerland (2018). [https://doi.org/10.1007/978-3-030-00937-3](https://doi.org/10.1007/978-3-030-00937-3_12) 12
- <span id="page-11-5"></span>[26] Radford, A., et al.: Learning Transferable Visual Models From Natural Language Supervision. arXiv (2021) <https://doi.org/10.48550/arXiv.2103.00020> [2103.00020](https://arxiv.org/abs/2103.00020)
- <span id="page-11-6"></span>[27] Wang, Z., et al.: MedCLIP: Contrastive Learning from Unpaired Medical Images and Text. arXiv (2022) <https://doi.org/10.48550/arXiv.2210.10163> [2210.10163](https://arxiv.org/abs/2210.10163)
- <span id="page-11-7"></span>[28] Opfermann, J.D., et al.: Feasibility of a cannula-mounted piezo robot for imageguided vertebral augmentation: Toward a low cost, semi-autonomous approach. In: 2021 IEEE 21st International Conference on Bioinformatics and Bioengineering (BIBE), pp. 1–8 (2021). <https://doi.org/10.1109/BIBE52308.2021.9635356>
- <span id="page-11-8"></span>[29] Wasserthal, J., et al.: TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence (2023)
- <span id="page-11-9"></span>[30] Demir, K.C., et al.: Pocap corpus: A multimodal dataset for smart operating room speech assistant using interventional radiology workflow analysis. In: International Conference on Text, Speech, and Dialogue, pp. 464–475 (2022). Springer
- <span id="page-11-10"></span>[31] Killeen, B.D., et al.: Stand in surgeon's shoes: virtual reality cross-training to enhance teamwork in surgery. Int. J. CARS  $19(6)$ , 1213–1222 (2024) [https://doi.](https://doi.org/10.1007/s11548-024-03138-7) [org/10.1007/s11548-024-03138-7](https://doi.org/10.1007/s11548-024-03138-7)

# <span id="page-12-0"></span>Appendix A Additional Results

| Prompt                     | <b>DICE</b>       | Centroid Error (mm) |
|----------------------------|-------------------|---------------------|
| "Vertebrae"                | $0.78 \pm 0.15$   | $21.80 \pm 24.04$   |
| "Lumbar vertebrae"         | $0.72 \pm 0.18$   | $29.12 \pm 35.00$   |
| "Lower lumbar vertebrae"   | $0.68 \pm 0.19$   | $37.57 \pm 46.94$   |
| "Right lung"               | $0.67 \pm 0.22$   | $42.18 \pm 35.16$   |
| "Femurs"                   | $0.66 \pm 0.17$   | $43.45 \pm 29.83$   |
| "Thoracic vertebrae"       | $0.64 \pm 0.20$   | $51.16 \pm 43.23$   |
| "Left femur bone"          | $0.64 \pm 0.26$   | $41.39 \pm 57.24$   |
| "Lungs"                    | $0.64 \pm 0.16$   | $45.96 \pm 28.75$   |
| "Spinal cord"              | $0.64 \pm 0.24$   | $35.47 \pm 32.48$   |
| "Left lung"                | $0.57\,\pm\,0.22$ | $41.86 \pm 17.36$   |
| "Ribs"                     | $0.54 \pm 0.12$   | $50.62 \pm 25.30$   |
| "Pelvis"                   | $0.53 \pm 0.16$   | $67.35 \pm 41.86$   |
| "Left ribs"                | $0.53 \pm 0.12$   | $44.15 \pm 29.74$   |
| "Left scapula bone"        | $0.53 \pm 0.24$   | $38.03 \pm 40.84$   |
| "Right femur bone"         | $0.51 \pm 0.33$   | $66.77 \pm 77.62$   |
| "Left half of the pelvis"  | $0.51 \pm 0.13$   | $69.75 \pm 42.68$   |
| "Right half of the pelvis" | $0.50 \pm 0.21$   | $66.92 \pm 46.42$   |
| "Right scapula bone"       | $0.49 \pm 0.15$   | $40.72 \pm 22.21$   |
| "Right ribs"               | $0.48 \pm 0.21$   | $54.42 \pm 50.28$   |
| "Left autochthon"          | $0.47 \pm 0.24$   | $52.93 \pm 40.06$   |
| "Small bowel"              | $0.42 \pm 0.20$   | $71.95 \pm 48.55$   |
| "Right autochthon"         | $0.42 \pm 0.25$   | $54.40 \pm 36.07$   |
| "Upper lumbar vertebrae"   | $0.42 \pm 0.18$   | $77.13 \pm 35.25$   |
| "Right gluteus maximus"    | $0.40 \pm 0.23$   | $73.43 \pm 78.34$   |
| "Left gluteus minimus"     | $0.38 \pm 0.23$   | $45.63 \pm 44.00$   |
| "Left gluteus maximus"     | $0.35 \pm 0.21$   | $72.05 \pm 36.12$   |
| "Sacrum"                   | $0.35 \pm 0.19$   | $62.85 \pm 63.14$   |
| "Upper thoracic vertebrae" | $0.35 \pm 0.22$   | $141.87 \pm 66.56$  |
| "Urinary bladder"          | $0.34 \pm 0.16$   | $37.76 \pm 30.95$   |
| "Heart"                    | $0.33 \pm 0.19$   | $58.55 \pm 24.90$   |
| "Left gluteus medius"      | $0.32 \pm 0.19$   | $72.97 \pm 61.26$   |
| "Lower thoracic vertebrae" | $0.27 \pm 0.19$   | $159.99 \pm 63.95$  |
| "L5 vertebra bone"         | $0.25 \pm 0.25$   | $50.71 \pm 48.80$   |
| "Right clavicle bone"      | $0.25 \pm 0.20$   | $46.99 \pm 59.10$   |
| "L3 vertebra bone"         | $0.21 \pm 0.21$   | $53.86 \pm 41.80$   |
| "Right kidney"             | $0.19 \pm 0.09$   | $51.06 \pm 23.15$   |
| "Right gluteus minimus"    | $0.19 \pm 0.19$   | $62.42 \pm 41.89$   |
| "Left clavicle bone"       | $0.17 \pm 0.18$   | $33.27 \pm 24.40$   |
| "Sternum bone"             | $0.17 \pm 0.20$   | $70.62 \pm 38.32$   |
| "L4 vertebra bone"         | $0.16 \pm 0.15$   | $49.40 \pm 38.92$   |
| "Kidneys"                  | $0.16 \pm 0.09$   | $83.09 \pm 30.16$   |
| Avg.                       | $0.47 \pm 0.25$   | $58.87 \pm 50.29$   |

Table A1: FluoroSAM Single-image Performance

<span id="page-13-0"></span>

| Prompt                     | 3D Centroid Error (mm) | <b>B.</b> Box Precision | <b>B.</b> Box Recall |
|----------------------------|------------------------|-------------------------|----------------------|
| "Right femur bone"         | $27.50 \pm 28.07$      | $0.17 \pm 0.22$         | $0.77 \pm 0.23$      |
| "Right gluteus medius"     | $31.83 \pm 8.97$       | $0.33 \pm 0.39$         | $0.58 \pm 0.41$      |
| "Left gluteus minimus"     | $32.44 \pm 15.87$      | $0.24 \pm 0.18$         | $0.51 \pm 0.20$      |
| "Right gluteus minimus"    | $32.60 \pm 7.19$       | $0.12 \pm 0.07$         | $0.70 \pm 0.27$      |
| "Lumbar vertebrae"         | $35.28 \pm 28.23$      | $0.30 \pm 0.25$         | $0.84 \pm 0.20$      |
| "Urinary bladder"          | $35.29 \pm 19.17$      | $0.25 \pm 0.24$         | $0.44 \pm 0.31$      |
| "Spinal cord"              | $36.35 \pm 26.84$      | $0.24 \pm 0.22$         | $0.74 \pm 0.29$      |
| "Vertebrae"                | $37.77 \pm 22.04$      | $0.32 \pm 0.22$         | $0.82 \pm 0.18$      |
| "Left femur bone"          | $37.97 \pm 30.08$      | $0.18 \pm 0.17$         | $0.80 \pm 0.24$      |
| "Lower lumbar vertebrae"   | $40.27 \pm 26.86$      | $0.26 \pm 0.23$         | $0.81 \pm 0.24$      |
| "Thoracic vertebrae"       | $43.25 \pm 30.24$      | $0.25 \pm 0.25$         | $0.80 \pm 0.25$      |
| "Heart"                    | $45.40 \pm 19.16$      | $0.05 \pm 0.06$         | $0.81 \pm 0.30$      |
| "Left gluteus medius"      | $47.11 \pm 24.19$      | $0.28 \pm 0.26$         | $0.68 \pm 0.22$      |
| "Right gluteus maximus"    | $49.00 \pm 20.64$      | $0.20 \pm 0.16$         | $0.62 \pm 0.23$      |
| "Sacrum"                   | $49.19 \pm 31.53$      | $0.25 \pm 0.14$         | $0.46 \pm 0.17$      |
| "Right half of the pelvis" | $49.53 \pm 29.42$      | $0.29 \pm 0.30$         | $0.65 \pm 0.25$      |
| "Right lung"               | $51.20 \pm 25.85$      | $0.34 \pm 0.23$         | $0.58 \pm 0.22$      |
| "Small bowel"              | $53.43 \pm 24.87$      | $0.26 \pm 0.19$         | $0.60 \pm 0.22$      |
| "Upper lumbar vertebrae"   | $53.67 \pm 28.25$      | $0.14 \pm 0.18$         | $0.93 \pm 0.15$      |
| "Left half of the pelvis"  | $53.98 \pm 18.94$      | $0.21 \pm 0.09$         | $0.73 \pm 0.21$      |
| "Left autochthon"          | $54.92 \pm 19.04$      | $0.17 \pm 0.22$         | $0.64 \pm 0.27$      |
| "Lungs"                    | $55.24 \pm 34.91$      | $0.35 \pm 0.28$         | $0.54 \pm 0.29$      |
| "Femurs"                   | $58.49 \pm 33.84$      | $0.10 \pm 0.08$         | $0.87 \pm 0.19$      |
| "Pelvis"                   | $58.82 \pm 34.77$      | $0.29 \pm 0.19$         | $0.79 \pm 0.21$      |
| "Lower thoracic vertebrae" | $61.17 \pm 26.53$      | $0.23 \pm 0.21$         | $0.57 \pm 0.26$      |
| "Colon"                    | $62.85 \pm 18.92$      | $0.33 \pm 0.22$         | $0.50 \pm 0.18$      |
| "Upper thoracic vertebrae" | $64.14 \pm 16.54$      | $0.25 \pm 0.04$         | $0.62 \pm 0.19$      |
| "Right autochthon"         | $64.55 \pm 28.27$      | $0.23 \pm 0.17$         | $0.57 \pm 0.24$      |
| "Left lung"                | $64.90 \pm 28.98$      | $0.36 \pm 0.26$         | $0.52 \pm 0.32$      |
| "Left ribs"                | $69.80 \pm 33.14$      | $0.35 \pm 0.26$         | $0.68 \pm 0.24$      |
| "Left scapula bone"        | $70.56 \pm 58.55$      | $0.41 \pm 0.33$         | $0.48 \pm 0.29$      |
| "Right ribs"               | $71.77 \pm 44.00$      | $0.32 \pm 0.29$         | $0.59 \pm 0.25$      |
| "Ribs"                     | $76.16 \pm 34.18$      | $0.35 \pm 0.21$         | $0.73 \pm 0.24$      |
| "Left gluteus maximus"     | $78.10 \pm 29.58$      | $0.14 \pm 0.08$         | $0.73 \pm 0.17$      |
| "Liver"                    | $89.05 \pm 18.40$      | $0.16 \pm 0.14$         | $0.85 \pm 0.18$      |
| Avg.                       | $51.68 \pm 30.84$      | $0.26 \pm 0.22$         | $0.70 \pm 0.26$      |

Table A2: 3D Digital Twin Reconstruction for Selected Prompts