# Toward Lung Ultrasound Automation: Fully Autonomous Robotic Longitudinal and Transverse Scans Along Intercostal Spaces

Long Le[i](https://orcid.org/0000-0001-9999-7311 )<sup>o</sup>[,](https://orcid.org/0000-0002-0835-3770) Yingbai Hu<sup>o</sup>, [Zix](https://orcid.org/0000-0003-1495-3278)in[g](https://orcid.org/0009-0006-7510-9563) Jiang<sup>o</sup>, Juzheng Mia[o,](https://orcid.org/0000-0003-4455-0808) [X](https://orcid.org/0000-0003-4455-0808)iao Luo, Yu Zhang<sup>o</sup>, Qiong Wang<sup>o</sup>, Shujun Wang<sup>®</sup>, *Member, IEEE*, Zheng Li<sup>®</sup>, *Senior Member, IEEE*, and Pheng-Ann Heng<sup>1</sup>[,](https://orcid.org/0000-0003-3055-5034) *Senior Member, IEEE* 

*Abstract***—Lung ultrasound scanning is essential for diagnosing lung diseases. The scan effectiveness critically depends on both longitudinal and transverse scans through intercostal spaces to reduce rib shadowing interference, as well as maintaining the probe perpendicular to pleura for pathological artifact generation. Achieving this level of scan quality often depends heavily on the experience of doctors. Robotic ultrasound scanning shows promise, but currently lacks a direct path planning method for intercostal scanning, and probe orientation does not consider imaging differences between lungs and solid organs. In this paper, we aim to fully automate two fundamental operations in lung ultrasound scanning: longitudinal and transverse scans. We propose pioneering path planning methods along intercostal spaces and innovative solutions for adaptive probe posture adjustment using real-time pleural line feedback, specifically addressing the unique characteristics of lung ultrasound scanning. This ensures the acquisition of high-quality, diagnostically meaningful ultrasound images. In addition, we develop a robotic lung ultrasound system to validate the proposed methods. Extensive experimental results on two volunteers and a chest phantom confirm the efficacy of our methods, and demonstrate the system's feasibility in automated lung ultrasound examinations. Our work lays a solid foundation for automated robotic complete lung scanning.**

*Index Terms***—Robotic ultrasound system, lung ultrasound, medical automation.**

Received 29 October 2024; revised 28 December 2024; accepted 25 January 2025. Date of publication 12 March 2025; date of current version 13 May 2025. This article was recommended for publication by Associate Editor V. Iacovacci and Editor P. Dario upon evaluation of the reviewers' comments. This work was supported in part by the Research Grants Council of the Hong Kong Special Administrative Region, China, under Project T45-401/22-N; in part by the Hong Kong Innovation and Technology Fund under Grant GHP/080/20SZ; in part by the Regional Joint Fund of Guangdong (Guangdong–Hong Kong–Macao Research Team Project) under Grant 2021B1515130003; in part by the NSFC Key Project under Grant U23A20391; in part by the NSFC-RGC Joint Research Scheme under Grant N\_CUHK447/24; in part by the Collaborative Research Fund under Grant C4042-23GF; in part by the CUHK Group Research Scheme; and in part by the Multiscale Medical Robotics Center. *(Long Lei, Yingbai Hu, and Zixing Jiang contributed equally to this work.) (Corresponding authors: Shujun Wang; Zheng Li.)*

This work involved human subjects or animals in its research. Approval of all ethical and experimental procedures and protocols was granted by Institutional Review Board of the Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences under Application No. YSB-2020-Y0902.

Please see the Acknowledgment section of this article for the author affiliations.

This article has supplementary downloadable material available at https://doi.org/10.1109/TMRB.2025.3550663, provided by the authors.

Digital Object Identifier 10.1109/TMRB.2025.3550663

# <span id="page-0-2"></span><span id="page-0-1"></span><span id="page-0-0"></span>I. INTRODUCTION

**L** UNG Ultrasound (LUS) allows physicians to quickly and safely obtain immediate images of a patient's lungs. It has been shown to be superior to bedside chest X-rays and comparable to chest CT in the diagnosis of various pleural and pulmonary conditions, such as pediatric pneumonia [\[1\]](#page-12-0) and COVID-19 [\[2\]](#page-12-1), [\[3\]](#page-12-2). Compared to CT scans or chest X-rays, ultrasound equipment is generally more affordable, portable, and user-friendly, allowing rapid evaluations at the bedside. This is particularly valuable in emergency, critical care, and resource-limited settings, such as remote or rural areas. Moreover, lung ultrasound does not carry the risk of ionizing radiation, making it suitable for repeated use, especially in pregnant women, children, and critically ill patients who require frequent monitoring [\[4\]](#page-12-3). To obtain adequate ultrasound image samples that are minimally obstructed by the ribs, a lung ultrasound examination requires longitudinal and transverse scans along the intercostal spaces (ICS) for each hemithorax [\[5\]](#page-12-4), as shown in Fig. [1.](#page-1-0) Unlike ultrasound examinations for other organs, a diseased lung is identified primarily from a healthy lung through artifacts, including A-lines and Blines [\[6\]](#page-12-5). The A-lines are several hyperechoic lines parallel to a pleural line, they are healthy lung ultrasound signs. When lung tissue becomes diseased and the gas in the lung is partially replaced by substances that can conduct ultrasound waves, such as exudate and blood, ultrasound waves can reach deeper tissues and form a hyperechoic beam perpendicular to the pleural line, called the B-lines. These A-lines and B-lines can only be fully produced when ultrasound pulses are emitted vertically onto the pleura [\[7\]](#page-12-6), [\[8\]](#page-12-7), allowing the probe to receive the strongest echoes from the pleura. Therefore, maintaining the probe perpendicular to the pleura is essential to acquire diagnostically meaningful ultrasound images.

<span id="page-0-6"></span><span id="page-0-5"></span><span id="page-0-4"></span><span id="page-0-3"></span>During a lung ultrasound scan, sonographers or clinicians must apply appropriate pressure with the probe on the patient's chest and move it in specific patterns to examine different parts of the lungs for possible pathologies, diagnosing based on the resulting images [\[9\]](#page-12-8). Scanning techniques can vary significantly between doctors, and diagnostic accuracy often depends on their level of experience. In addition, manual scanning exposes doctors directly to the patient's environment, increasing the risk of infection. Prolonged manual scanning can also lead to muscle fatigue and decreased concentration,

2576-3202 © 2025 IEEE. All rights reserved, including rights for text and data mining, and training of artificial intelligence and similar technologies. Personal use is permitted, but republication/redistribution requires IEEE permission.

See https://www.ieee.org/publications/rights/index.html for more information.<br>Authorized licensed use limited to: Johns Hopkins University. Downloaded on July 26,2025 at 19:42:58 UTC from IEEE Xplore. Restrictions apply.

![](_page_1_Figure_1.jpeg)

<span id="page-1-0"></span>Fig. 1. Illustration of lung ultrasound scan protocol. Lung ultrasound examination requires both longitudinal and transverse scans along the intercostal spaces, with the probe maintained perpendicular to the pleural.

which can compromise diagnostic accuracy. An autonomous robotic lung ultrasound system (RLUS) could standardize and automate the scanning process, reducing the physical burden and infection risks for doctors, allowing them to focus on diagnosis [\[10\]](#page-12-9), [\[11\]](#page-12-10). This would benefit both doctors and patients, particularly in remote areas.

<span id="page-1-2"></span>In this paper, we focus on achieving full autonomy in the two fundamental operations of lung ultrasound: longitudinal and transverse scans. This represents a significant step toward fully autonomous robotic lung ultrasound scan for the entire lung ultrasound examinations. By addressing the unique characteristics of lung ultrasound, we have developed solutions for path planning along the intercostal spaces for both types of scans. In addition, we also solved the problem of adaptive probe posture adjustment using real-time pleural line feedback. These advances ensure the acquisition of a sufficient number of high-quality ultrasound images that are diagnostically meaningful. To our knowledge, this is the first work that enables robotic lung ultrasound scanning to automatically follow intercostal spaces while optimizing probe posture based on pleural line feedback. The main contributions of this paper are as follows.

- 1) We propose a patient-specific lung ultrasound scan path planning method that enables longitudinal scan along the intercostal spaces by extracting intercostal centerlines from a standard human model and adapting them to individual patient surface data. This is the first work to plan lung ultrasound scanning paths along the intercostal spaces.
- 2) We propose a method for locating actual intercostal centerlines through pleural line segmentation and reconstruction based on the results of longitudinal scans, and further plan the transverse scan paths, ensuring that Authorized licensed use limited to: Johns Hopkins University. Downloaded on July 26,2025 at 19:42:58 UTC from IEEE Xplore. Restrictions apply.

transverse scans obtain ultrasound images unobstructed by the ribs.

- 3) We propose an ultrasound probe posture control method based on real-time pleural line feedback servo for the first time, both eliminating the possibility of the target pleural line deviating from the imaging field of view during the longitudinal scan and ensuring the full generation of pathological artifacts for obtaining diagnostically meaningful ultrasound images during the overall scanning process.
- 4) A robotic lung ultrasound system is developed and the scan workflow is standardized. Experiments conducted on a volunteer and a chest phantom validate the performance of the proposed methods and the system.

The rest of this article is organized as follows. Section [II](#page-1-1) reviews related work. Section [III](#page-2-0) introduces the system setup and calibration method. Section [IV](#page-3-0) presents the longitudinal scan path planning method. Section [V](#page-5-0) presents the segmentation of pleural lines from ultrasound images, reconstruction of the intercostal centerlines to plan transverse scan paths, online probe posture adjustment and compliant motion control methods. Section [VI](#page-7-0) presents the experiments and results. Finally, Section [VII](#page-11-0) concludes this article.

## <span id="page-1-6"></span><span id="page-1-5"></span><span id="page-1-4"></span><span id="page-1-3"></span>II. RELATED WORK

## <span id="page-1-1"></span>*A. Robotic Lung Ultrasound Scan System*

<span id="page-1-7"></span>Existing autonomous robotic lung ultrasound systems can be mainly divided into two categories: Imaging only in several specific locations [\[12\]](#page-12-11), [\[13\]](#page-12-12), [\[14\]](#page-12-13), [\[15\]](#page-12-14) and scanning of the lung zones along specific trajectories [\[16\]](#page-12-15), [\[17\]](#page-12-16), [\[18\]](#page-12-17). Systems of the first type usually follow the 8-point Point-of-Care Ultrasound (POCUS) [\[19\]](#page-12-18) or Bedside Lung Ultrasound in Emergency (BLUE) [\[20\]](#page-12-19) protocols, or the 10-point BLUEplus [\[21\]](#page-12-20) protocol, and automatically obtain ultrasound images at these positions for diagnosis. Although this method is efficient, the scanning range is limited and there is a risk of missed diagnosis [\[22\]](#page-12-21). This issue is particularly highlighted for the COVID-19, where the viral pneumonia is characterized by multiple discrete interstitial B-lines in the lung [\[2\]](#page-12-1). When inflammatory lesions do not appear at these specific locations, the missed diagnosis occurs.

<span id="page-1-12"></span><span id="page-1-11"></span><span id="page-1-10"></span><span id="page-1-9"></span><span id="page-1-8"></span>Scanning the lung zones can avoid missed diagnoses. Just like the robotic full-coverage ultrasound scan for other organs, such as the breast  $[23]$ , the lumbar  $[24]$ , and the abdominal organs [\[25\]](#page-12-24), this type of RLUS also adopts a two-step work-flow [\[12\]](#page-12-11), [\[16\]](#page-12-15), [\[18\]](#page-12-17). The first step is to plan a scan trajectory consisting of a series of probe positions and attitudes based on the patient's body surface point cloud, which is obtained from preoperative magnetic resonance imaging (MRI) or computed tomography (CT) images [\[26\]](#page-12-25) or a depth camera. The robot then holds the probe and performs an ultrasound scan along the planned trajectory. During this process, the probe posture is usually fine-tuned in real time on the basis of the actual contact force and image feedback information to obtain high-quality ultrasound images and ensure scan security. However, existing RLUSs cannot achieve scanning along intercostal spaces nor consider the impact of probe posture on artifact generation.

## *B. Ultrasound Scan Path Planning*

Currently, various path planning methods for lung ultrasound have been proposed. Suligoj et al. [\[16\]](#page-12-15) first evenly divide a plane grid according to the scan boundary and the probe size, and then generate the actual curve scan path by projecting these grid points onto the skin surface. Tan et al. [\[18\]](#page-12-17) slice the body surface point set at a certain interval along the cranial-caudal direction, and use a 3rd degree nonuniform rational B-splines (NUBRS) curve-fitting method to obtain the scan path. Since these path planning methods do not account for the obstruction of imaging caused by anatomical structures such as ribs, they can result in images that are meaningless for diagnosis  $[27]$  and may interfere with the doctor's judgment. Jiang et al. [\[28\]](#page-12-27) propose a nonrigid registration method between the thoracic cartilage point sets extracted from a CT template and a patient's ultrasound images, to map planned scan paths from a generic atlas to the individual patient. However, additional ultrasound scan of cartilages is required to reconstruct their morphology for registration, which means that a long preparation time is needed before the formal ultrasound scanning.

## *C. Probe Posture Control*

<span id="page-2-4"></span>Most ultrasound scan robots use the normal direction of the skin surface at the contact point as the central axis direction of the probe [\[12\]](#page-12-11), [\[13\]](#page-12-12), [\[16\]](#page-12-15), [\[23\]](#page-12-22), [\[24\]](#page-12-23), [\[29\]](#page-12-28), [\[30\]](#page-12-29), [\[31\]](#page-12-30), [\[32\]](#page-12-31), which is considered to ensure the optimal acoustic coupling between the transducer and the body, thus providing a clear visualization of pathological clues [\[33\]](#page-12-32). Before scanning, the normal direction of the skin is often calculated directly on the basis of the point cloud of the patient's body surface [\[13\]](#page-12-12), [\[16\]](#page-12-15), [\[23\]](#page-12-22), [\[24\]](#page-12-23), [\[29\]](#page-12-28). During the scan, Jiang et al. [\[30\]](#page-12-29), [\[31\]](#page-12-30) identify the normal direction of the body surface based on the change in the contact force between the probe and the tissue during its fan-shaped movement. Ma et al. [\[12\]](#page-12-11) calculate the rotation adjustment values to perpendicularity based on the distances to the skin detected by multiple distance sensors installed around the probe. Some works also optimize probe posture online based on the coverage level or imaging angle of the target anatomical structures [\[27\]](#page-12-26), [\[34\]](#page-12-33), [\[35\]](#page-12-34). Jiang et al. [\[34\]](#page-12-33) align the probe to the normal direction of the target blood vessel to accurately measure its diameter. However, there are differences in the thickness of the fat and muscle layers at different locations in the chest, the normal direction of the body surface is not always perpendicular to the pleura, resulting in possible failure to produce A- and B-lines. Furthermore, no one has yet directly used the pleural lines in ultrasound images as feedback to adjust the probe posture.

## <span id="page-2-6"></span>III. SYSTEM SETUP AND METHOD OVERVIEW

## <span id="page-2-0"></span>*A. System Setup*

Based on the need for lung ultrasound, we developed a robotic lung ultrasound system, as shown in Fig. [2.](#page-2-1) The hardware of the developed RLUS mainly includes a 6 degree-of-freedom (DoF) lightweight robot (UR5, Universal Robots, Denmark), an ultrasound imaging system (Clover 60,

![](_page_2_Figure_9.jpeg)

<span id="page-2-3"></span><span id="page-2-2"></span><span id="page-2-1"></span>Fig. 2. Robotic lung ultrasound scan system setup and involved coordinate transformations.

Wisonic, China) with a convex array probe (C5-1, Wisonic, China), a frame grabber (OK\_VGA41A-4E+, JoinHope Image, China), a 6-axis force/torque (F/T) sensor (M3733C, Sunrise Instruments, China), an RGB-D camera (RealSense D435i, Intel, USA), and a host computer. The ultrasound probe is connected to the end flange of the robot by a customized fixture via the F/T sensor, and the fixture has a symmetrical structure that can automatically align the axis of the probe, sensor, and end flange of the robot. The F/T sensor measures the real-time contact force between the probe and the subject's skin. The camera obtains the point cloud of the subject's body surface, fixed at the end of the robot through a 3Dprinted clamp. The frame grabber is plugged into the host motherboard and collects ultrasound images from the imaging system through its HDMI port.

<span id="page-2-5"></span>The software of the developed robotic system mainly includes two modules: an ultrasound image capture and analysis (USCA) module, and a robot control module. The USCA module is used mainly to continuously capture ultrasound images, segment pleural lines based on a deep learning model, and analyze the results of pleural line segmentation. This main program of this module is built with the Qt toolkit (Qt 5.15.2), the deep learning model is deployed as a backend server with the Flask framework (Flask 2.2.5), and the two parts communicate using the POST method of the HTTP protocol. The robot control module is mainly used for system calibration, obtaining body surface point cloud data, path planning, intercostal centerline reconstruction, and robot motion simulation and control under a hybrid forceposition framework. This module is developed with the Robot Operating System (ROS2 Humble). The USCA module sends the ultrasound image analysis results to the robot control module via the GET method of the HTTP protocol. For the robotic system, the control frequency for probe posture adjustment based on image feedback is 10 Hz, while the compliance motion control frequency based on force feedback is 125 Hz.

#### *B. System Calibration*

The coordinate systems and their transformations in the robotic system are also illustrated in Fig. [2.](#page-2-1) The frames {*RB*}, {*RE*}, {*C*}, {*US*} represent the robot base, robot end flange, camera and ultrasound image coordinate systems, respectively. {*P*} represents the coordinate system of the ultrasound probe, its origin is located at the intersection of the central axis of the probe and the imaging surface, and its direction is consistent with the {*RE*}. The homogeneous transformation  ${}^{RE}_{P}T \in SE(3)$  is obtained based on the geometry of the probe and its fixture. Transformation  $_{US}^{RE}T \in SE(3)$  is obtained by the ultrasound probe calibration using a N-wire phantom [\[36\]](#page-12-35). The transformation  ${}^{RE}_{C}T \in SE(3)$  can be obtained by the eye-in-hand calibration with a calibration board [\[23\]](#page-12-22), before which the camera is calibrated to obtain its internal parameters and align the RGB image with the depth image to provide accurate colorized point cloud of the subject's body surface. Transformation  $_{RE}^{RB}T \in SE(3)$  is obtained by robot forward kinematics. Based on these transformations, the subject's body surface point cloud can be mapped to the robot base coordinate system by the transformation

$$
{}^{RB}_{C}T = {}^{RB}_{RE}T \cdot {}^{RE}_{C}T, \qquad (1)
$$

the subject's internal anatomy can be mapped to the robot base coordinate system by the transformation

$$
{}_{US}^{RB}T = {}_{RE}^{RB}T \cdot {}_{US}^{RE}T, \qquad (2)
$$

and the target probe pose in the robot base coordinate system can be obtained by the transformation

$$
{}^{RB}_{P}T = {}^{RB}_{RE}T \cdot {}^{RE}_{P}T.
$$
 (3)

The gravity compensation of the F/T sensor is performed by identifying gravity and its center of the probe and fixture [\[37\]](#page-12-36), to realize an accurate perception of the contact force/torque between the probe and the subject's skin.

## IV. LONGITUDINAL SCAN PATH PLANNING

<span id="page-3-0"></span>In this paper, we propose an intercostal space centerline estimation method based on a standard human body model and the subject's body surface point cloud. On this basis, an ultrasound scanning path planning method along the intercostal spaces is developed. As shown in Fig. [3,](#page-4-0) we first extract the intercostal space centerlines from a standard human body model, then map the extracted intercostal space centerlines from the human body model to the subject according to the coordinate systems defined by the navel and nipples on the body surface, and finally project them to the subject's body surface to complete the path planning along the intercostal spaces.

# *A. Intercostal Centerline Extraction From Standard Human Body Model*

In this work, a standard adult male body model from the Zygote Body atlas (Zygote Media Group, Inc., American) is used. As shown in Fig.  $3(a)$  $3(a)$ , this model includes the skin and a complete bony framework of the thorax. Each rib and its connected costal cartilage (CC) are individually

<span id="page-3-2"></span><span id="page-3-1"></span>**Algorithm 1:** Intercostal Centerlines Extraction **Input:** Adjacent rib-cartilage meshes:  $\{RC_i, i = 1, 2.\}$ left, right boundary and search interval distance of search plane: *lb*,*rb*,*sd*. **Output:** Intercostal centerline: *centerline*. // define function **Function** ComputeSection(*mesh*, *plane*)**:**  $|$  **section**  $\leftarrow \emptyset$ ; **for** *face* in *mesh* **do for** *edge* in *face* **do if** *edge* intersects *plane* **then**  $\bullet$  | | | *point* ← IntersectionPoint(*edge*, *plane*);  $\vert$   $\vert$   $\vert$  Add *point* to *section*; **return** *section*. // main algorithm *centerline* ← ∅; *searchPlanes* ← GenerateSearchPlanes(*lb*,*rb*,*sd*); **for** *plane* in *searchPlanes* **do**  $12 \mid \text{section}_1 \leftarrow \text{ComputeSection}(RC_1, \text{plane});$  $\vert$  *section*<sub>2</sub> ← ComputeSection( $RC_2$ , plane); **if** Length(*section*<sub>1</sub>) > 0 and Length(*section*<sub>2</sub>) > 0 **then** *center* ← AveragePoints(*section*<sup>1</sup> ∪ *section*2); Add *center* to *centerline*; *centerline* ← Smooth(*centerline*); **return** *centerline*.

<span id="page-3-3"></span>represented as a polygonal mesh file mainly composed of a series of quadrilateral faces. For example, Quadrilateral *ABCD* in Fig. [3\(](#page-4-0)b) is a face composed of the four vertices *A*, *B*, *C*, and *D*. Here, we merge the meshes of each rib and its adjacent costal cartilage, and refer to this combined structure as a ribcartilage (RC) mesh. The origin of the human body model coordinate system {*M*} is located on the midsagittal plane, with the X-axis pointing to the anatomical left of the body, Y-axis superior, and Z-axis anterior.

Based on the bony framework of the thorax, the intercostal centerlines are extracted using the **Algorithm** [1](#page-3-1) as follows: Given adjacent rib-cartilage meshes, first, define a search plane parallel to the Y-O-Z plane of the coordinate system {*M*}, which moves from the middle to the sides of the body model at equal intervals. Then, at each search position, calculate the sections of rib-cartilage meshes intersected by this search plane. Next, determine the intercostal center by averaging the coordinates of these section points. By parameterizing these intercostal centers across various positions of the search plane using cumulative chord lengths [\[38\]](#page-12-37) and fitting them with cubic B-splines, we ultimately obtain a smooth intercostal centerline. The extracted four intercostal centerlines of the right chest are shown in Fig. [3\(](#page-4-0)c).

# <span id="page-3-4"></span>*B. Subject-Personalized Intercostal Centerline Estimation*

After extracting the intercostal space centerlines from the standard human body model, they need to be mapped onto

![](_page_4_Figure_1.jpeg)

<span id="page-4-0"></span>Fig. 3. Schematic diagram of longitudinal scan path planning method along the intercostal spaces. Starting from the (a) thorax skeleton framework of a standard human body model, the (b) plane-mesh intersection point search method is used to extract the (c) intercostal centerlines. Then, define the (d) human model skin frame and the (e) subject's skin frame using the nipples and the navel as landmarks, and map the intercostal centerlines extracted from the human model to the actual examined subject to estimate the (f) subject-personalized intercostal centerlines. Finally, a (g) normal vector matching method is used to project the estimated intercostal centerlines to the skin point cloud to obtain the (h) scan paths along the intercostal spaces.

the subject based on the body shape characteristics of the standard human body model and the subject, so as to obtain the subject's personalized intercostal space centerlines.

<span id="page-4-1"></span>*1) Body Surface Landmark Positioning:* Just like in the works [\[39\]](#page-12-38), [\[40\]](#page-12-39), we choose the nipples and navel on the surface of the skin as landmarks to describe the individual shape of the body. For the standard human body model, the positions of its nipple and navel centers are obtained by querying the corresponding vertex coordinates. For the subject, after obtaining the 2D RGB image, the image is first converted from the BGR color space to the HSV (Hue, Saturation, Value) color space, and the skin area is segmented by setting the HSV ranges, which are determined by the patient's skin tone. Then, in order to improve the robustness of nipple and navel positioning under size uncertainty, a multiscale template matching method is adopted. That is, the template image is scaled in multiple scales, the template matching is performed on each scale, and the result with the highest matching degree is selected as the target position. In particular, to eliminate the influence of brightness changes, normalized cross-correlation (NCC) is used to measure the degree of matching of the template image and the target image at the current position. Finally, according to the internal parameters of the camera, the landmarks detected in the RGB image are mapped to the point cloud data, and their positions in the camera coordinate system are obtained.

*2) Skin Coordinate System Definition:* Based on the detected body surface landmarks, the skin coordinate systems of the standard human body model and the subject, {*MS*} and

 ${S}{S}$ , are defined in the same way. As shown in  $3(d-e)$  $3(d-e)$ , the origin of the skin coordinate system is set as the midpoint of the line connecting the two nipples, with the X-axis pointing from the right nipple to the left nipple, and the Y-axis pointing from the navel to the origin. For the human body model, since its skin is symmetric and the plane where the navel and nipples are located is parallel to the X-O-Y plane of the coordinate system  $M$ , the model coordinate system  $\{M\}$  and its skin coordinate system {*MS*} have the same direction, with only an offset in the origin. For the subject, let  ${}^{C}P_{LN} \in \mathbb{R}^3$ ,  ${}^{C}P_{RN} \in$  $\mathbb{R}^3$ , and  $^C P_{\text{NA}} \in \mathbb{R}^3$  respectively represent the positions of the centers of the left nipple, right nipple and navel in the camera coordinate system  $\{C\}$ , then, under the  $\{C\}$ , the origin of the coordinate system {*SS*} can be obtained by

$$
{}^{C}\boldsymbol{O}_{SS} = \left({}^{C}\boldsymbol{P}_{LN} + {}^{C}\boldsymbol{P}_{RN}\right)/2, \tag{4}
$$

the unit vector along the X-axis of the coordinate system {*SS*} can be obtained by

$$
{}^{C}x_{SS} = \left({}^{C}P_{LN} - {}^{C}P_{RN}\right) / \Vert {}^{C}P_{LN} - {}^{C}P_{RN}\Vert. \tag{5}
$$

Taking into the landmark positioning errors, the unit vector along the Y-axis of the coordinate system {*SS*} is obtained by

$$
{}^{C}\mathbf{y}_{SS} = \frac{{}^{C}\mathbf{y}_{SS}^{0} - \left({}^{C}\mathbf{y}_{SS}^{0} \cdot {}^{C}\mathbf{x}_{SS}\right) \cdot {}^{C}\mathbf{x}_{SS}}{\|{}^{C}\mathbf{y}_{SS}^{0} - \left({}^{C}\mathbf{y}_{SS}^{0} \cdot {}^{C}\mathbf{x}_{SS}\right) \cdot {}^{C}\mathbf{x}_{SS}\|},
$$
\n
$$
{}^{C}\mathbf{y}_{SS}^{0} = {}^{C}\mathbf{O}_{SS} - {}^{C}\mathbf{P}_{NA}.
$$
\n(7)

Finally, the rotation transformation from the subject skin coordinate system to the camera coordinate system can be obtained by

$$
{}_{SS}^{C} \boldsymbol{R} = ({}^{C} \boldsymbol{x}_{SS}, {}^{C} \boldsymbol{y}_{SS}, {}^{C} \boldsymbol{x}_{SS} \times {}^{C} \boldsymbol{y}_{SS}) \in SO(3). \tag{8}
$$

*3) Mapping Intercostal Centerlines From Standard Human Body Model to Subject:* Due to the same definition of the coordinate system, the intercostal centerlines in the coordinate system {*MS*} are mapped to the coordinate system {*SS*} without rotation and translation transformations. Taking into account the body size difference between the actual subject and the standard human model, scaling is respectively performed in the X- and Y-directions during mapping, while the Z-axis coordinate remains unchanged. Let  ${}^M P_{LN} \in \mathbb{R}^3$ ,  ${}^M P_{RN} \in \mathbb{R}^3$ , and  ${}^M P_{NA} \in \mathbb{R}^3$  respectively represent the left nipple, right nipple and navel of the standard human body model, then, the scaling matrix from the human body model to the subject is calculated by

$$
{}_{MS}^{SS}\boldsymbol{D} = \begin{pmatrix} scaleX & 0 & 0 \\ 0 & scaleY & 0 \\ 0 & 0 & 1 \end{pmatrix}, \tag{9}
$$

scaleX = 
$$
\| {}^{C}P_{LN} - {}^{C}P_{RN}\| / \| {}^{M}P_{LN} - {}^{M}P_{RN}\|,
$$
 (10)  
 $\| {}^{C}Q_{LN} - {}^{C}P_{LN}\| / \| {}^{M}P_{LN} - {}^{M}P_{RN}\|,$ 

$$
scaleY = \frac{(^{C}\textbf{O}_{SS} - ^{C}\textbf{P}_{NA}) \cdot ^{C}\textbf{y}_{SS}}{||^{M}\textbf{O}_{MS} - ^{M}\textbf{P}_{NA}||},
$$
\n(11)

$$
{}^{M}\mathbf{O}_{MS} = ({}^{M}\mathbf{P}_{LN} + {}^{M}\mathbf{P}_{RN})/2. \tag{12}
$$

Based on these transformations, the intercostal centerlines extracted from the thorax skeleton of the standard human model can be mapped to the camera coordinate system by [\(13\)](#page-5-1) to finally estimate the subject's personalized intercostal centerlines, as shown in Fig.  $3(f)$  $3(f)$ , which are combined with the subject's body surface point cloud data to provide a basis for scanning path planning along the intercostal spaces.

<span id="page-5-1"></span>
$$
{}^{C}\boldsymbol{P}_{i,j} = {}^{C}_{SS}\boldsymbol{R} \cdot {}^{SS}_{MS}\boldsymbol{D} \cdot ({}^{M}\boldsymbol{P}_{i,j} - {}^{M}\boldsymbol{O}_{MS}) + {}^{C}\boldsymbol{O}_{SS}, \qquad (13)
$$

where  ${}^{M}P_{i,j}$  is the *j*-th point on the *i*-th intercostal centerline extracted from the human body model, and  ${}^{C}P_{i,j}$  is its position mapped to the camera coordinate system.

## <span id="page-5-2"></span>*C. Path Planning Along Intercostal Spaces*

In order to obtain the ultrasound scanning paths along the intercostal spaces, a normal vector matching method is proposed to project the intercostal centerlines to the subject's body surface point cloud. Unlike conventional approaches that project 2D planar paths directly onto the surface of a 3D point cloud, our method takes into account the normal direction of the candidate points within the point cloud, thereby enhancing the accuracy of the planned paths along the intercostal spaces.

As shown in Fig. [3\(](#page-4-0)g), given an intercostal point *P* and a body surface point cloud  $Q$ , we first identify the point  $q_{\text{nearest}}$ in the point cloud that minimizes the distance to *P*,

$$
\boldsymbol{q}_{\text{nearest}} = \arg\min_{\boldsymbol{q}_i \in \mathcal{Q}} \|\boldsymbol{P} - \boldsymbol{q}_i\|,\tag{14}
$$

next, establish a neighborhood  $N$  around  $q_{\text{nearest}}$  with radius *R*,

$$
\mathcal{N} = \{ \boldsymbol{q}_j \in \mathcal{Q} \mid \|\boldsymbol{q}_j - \boldsymbol{q}_{\text{nearest}}\| \le R \},\tag{15}
$$

finally, within the neighborhood, find the point  $q_{\text{target}}$  that maximizes the cosine of the angle between the vector from *P* to the point and the surface normal vector  $N_i$  at the point,

$$
\boldsymbol{q}_{\text{target}} = \arg \max_{\boldsymbol{q}_j \in \mathcal{N}} \frac{(\boldsymbol{q}_j - \boldsymbol{P}) \cdot N_j}{\|\boldsymbol{q}_j - \boldsymbol{P}\|},\tag{16}
$$

thus,  $q_{\text{target}}$  is determined as the optimal projection point of the intercostal point *P* on the point cloud *Q*.

Using this method, we obtain a series of path points along the intercostal spaces and the corresponding surface normal vectors at these points, which determine the initial orientation of the probe, as shown in Fig.  $3(h)$  $3(h)$ . Based on this, we apply linear interpolation and spherical linear interpolation to interpolate the path points and the quaternions describing the probe orientation. The number of interpolation points is determined by the product of the scanning time and the robot's motion control frequency, ultimately resulting in a uniform initial path for robotic ultrasound scanning.

# <span id="page-5-0"></span>V. TRANSVERSE SCAN PATH PLANNING AND ONLINE PROBE POSTURE ADJUSTMENT

Due to the difference in body shape between the subject and the standard human body model, the intercostal centerlines estimated in the previous section inevitably deviate from their true positions. As a result, the planned scanning path cannot completely follow the target intercostal spaces, which has a more significant impact on transverse scanning. Moreover, during the scanning process, the probe posture needs to be adjusted according to the actual pleural lines to fully expose the A-lines and B-lines that are meaningful for the diagnosis of lung diseases. To solve these problems, a transverse scan path planning method based on reconstructed intercostal centerlines and an online probe posture adjustment method are proposed.

## *A. Pleural Line Segmentation*

Accurate pleural line segmentation is the key to reconstructing the intercostal centerline and online adjustment of the posture of the probe. In this work, segmentation of the intercostal pleural lines occurs in two scenarios: longitudinal scan and transverse scan. For the longitudinal scan, as shown in Fig. [4,](#page-6-0) scanning over the cartilages reveals the pleural lines beneath them due to their low acoustic impedance [\[28\]](#page-12-27). This results in a continuous high-echo line formed by the pleural line beneath the cartilage and the intercostal pleural line. In contrast, when scanning over the rib, the high acoustic impedance of the ribs causes posterior acoustic shadowing, making only the intercostal pleural line visible. This ribpleura-rib structure creates the "bat sign". In this case, only the pleural line in the target intercostal space needs to be precisely segmented while excluding interference from the pleural lines beneath the cartilages and the pleural lines of adjacent intercostal spaces. For the transverse scan, as shown in Fig. [1,](#page-1-0) due to the absence of obstruction of the rib, the entire intercostal pleural line is continuously visible and must be segmented.

<span id="page-5-3"></span>To segment specific pleural lines accurately in different scanning situations, we employ the nnU-Net framework [\[41\]](#page-12-40).

![](_page_6_Figure_2.jpeg)

<span id="page-6-0"></span>Fig. 4. Intercostal centerline reconstruction, (a) method diagram, where the yellow points represent the center points of the segmented pleural lines, and (b) reconstructed results.

Compared to other segmentation models that often require extensive manual parameter tuning, nnU-Net offers a robust and adaptable solution, automatically configuring itself to the specific dataset and task requirements, and has consistently achieved top performance in various medical image segmentation challenges. To train and test the network, we collected lung ultrasound images from 100 patients at a collaborating hospital. In addition, we used our own ultrasound machine to collect data from 10 healthy volunteers. All data were annotated by experienced clinicians. We then divided the data into training and test sets at a patient level in a 4:1 ratio. Ultimately, our training set contains 1,168 images from 80 patients and 8 volunteers, of which 508 are from transverse scans and 660 are from longitudinal scans. Our test set has 296 images, consisting of 128 transversely scanned images and 168 longitudinally scanned images.

All ultrasound images are with dimensions of 800  $\times$  600 pixels, and they were normalized using the Z-score method. Based on the nnU-Net framework, the U-Net structure with 6 downsampling layers and 6 upsampling layers was automatically configured. The training process utilized the Stochastic Gradient Descent (SGD) optimization algorithm with an initial learning rate of 0.01, a weight decay of  $3 \times 10^{-5}$ , and a momentum of 0.99 with Nesterov acceleration enabled. A polynomial learning rate scheduler was used, adjusting the learning rate to facilitate efficient convergence. The experiments were carried out using an NVIDIA 2080 Ti GPU with 11 GB of memory. The model was trained for 1000 epochs, with each epoch consisting of 250 iterations. During training, a batch size of 13 was employed to efficiently utilize GPU resources and ensure stable convergence. The loss function combines Cross Entropy Loss and Dice Loss with equal weighting and incorporates deep supervision to encourage better feature learning throughout the model, enhancing segmentation performance.

# *B. Transverse Scan Path Planning Based on Intercostal Centerline Reconstruction*

As shown in Fig. [4,](#page-6-0) during the longitudinal scan, we save the center points of the target intercostal pleural lines in Authorized licensed use limited to: Johns Hopkins University. Downloaded on July 26,2025 at 19:42:58 UTC from IEEE Xplore. Restrictions apply.

![](_page_6_Figure_8.jpeg)

<span id="page-6-1"></span>Fig. 5. General control scheme for autonomous robotic lung ultrasound scan. LS stands for longitudinal scan, and TS refers to transverse scan.

the ultrasound image coordinate system  $\{^{US}P_k|k=1,\ldots,K\}$ and their corresponding coordinate transformations  $\frac{RBT}{RE}I_k|k=$  $1, \ldots, K$ } to reconstruct the pleural centerline of the target intercostal space, which is considered the real target intercostal centerline. Considering the saving asynchrony between the pleural line centroids and the robot posture, we use timestamp alignment to get these paired data. Specifically, since the robot posture reading frequency is much higher than the ultrasound image processing, for each centroid of the pleural line, we search for the closest timestamp to its timestamp to obtain the corresponding robot posture. Then, the position of these centroids in the robot base coordinate system {*RB*} can be obtained by

$$
^{RB}\boldsymbol{P}_k = {}^{RB}_{RE}\boldsymbol{T}_k \cdot {}^{RE}_{US}\boldsymbol{T} \cdot {}^{US}\boldsymbol{P}_k,\tag{17}
$$

and their position in the camera coordinate system {*C*} can be further obtained by

$$
{}^{C}\boldsymbol{P}_{k} = \left(\begin{smallmatrix} R E & T \\ C & T \end{smallmatrix}\right)^{\mathrm{T}} \cdot \left(\begin{smallmatrix} R B & T \\ R E & T \end{smallmatrix}\right)^{\mathrm{T}} \cdot {}^{RB}\boldsymbol{P}_{k},\tag{18}
$$

where  $_{RE}^{RB}T_{\text{Cam}}$  is the posture of the robot when obtaining the point cloud of the body surface using the RGB-D camera. Thus, the target intercostal centerline and the body surface point cloud are unified under one coordinate system.

On this basis, the reconstructed intercostal centerline is smoothed using the cubic B-spline fitting method based on cumulative chord length parameterization to avoid the interference of outlier points. Using the method in Section [IV-C,](#page-5-2) a more accurate path along the intercostal space is obtained, when a transverse scan along this path is performed, the obtained ultrasound images will not have rib shadows.

## *C. Online Adaptive Adjustments of Probe Posture*

As shown in Fig. [5,](#page-6-1) there are two scenarios for online adaptive adjustment of the ultrasound probe posture, including probe translation along its long axis (X-axis) to prevent the target pleural line from exceeding the field of view of the ultrasound image during longitudinal scans, and rotation around its short axis (Y-axis) to visualize diagnostically meaningful

features, such as the A- and B-lines, during the whole scanning process, especially for transverse scan.

*1) Probe Translation Adjustment Along Its Long Axis:* During our experiments, we found that the estimated intercostal centerlines are more accurate at the middle of the body than near the side. As a result, when performing a longitudinal scan along the path planned based on an estimated intercostal centerline, moving the probe to the side of the body may cause the target pleural line to fall outside the imaging field of view. This situation hinders the accurate reconstruction of the true intercostal centerline and ultimately affects the effectiveness of the transverse scan.

To maintain the target pleural line in the center of the image, we introduce an adaptive probe translation compensation method. Let *d* represent the center deviation of the target pleural line from the vertical centerline of the image. Note that its value takes into account the pixel spacing of the ultrasound images, with units in mm. We introduce a threshold  $\tau$  to determine the necessity for probe adjustment, and probe translation compensation *t* in its X-direction is given by

$$
t = \begin{cases} d, \text{ if } |d| > \tau, \\ 0, \text{ if } |d| \le \tau. \end{cases}
$$
 (19)

*2) Probe Rotation Adjustment Around Its Short Axis:* During lung ultrasound scan, it is necessary to ensure that the ultrasound waves are emitted perpendicularly to the pleura so as to adequately visualize the A- and B-lines and other signs that are meaningful for the diagnosis of lung diseases. To this end, we propose to use the pleural line in the directly observed ultrasound image as a feedback, and adjust the probe posture in real time to maintain the alignment of the pleural line with the horizontal axis of the image.

Let  $\theta$  represent the angle deviation of the pleural line from the horizontal line, measured in degrees. We introduce an angular threshold  $\phi$  to determine the adjustment necessity, and the probe angle compensation  $r$  around its Y-axis is given by

$$
r = \begin{cases} \theta, & \text{if } |\theta| > \phi, \\ 0, & \text{if } |\theta| \le \phi. \end{cases}
$$
 (20)

The performance of our probe posture adjustment method relies on the accuracy of pleural line segmentation. To enhance stability in the presence of segmentation errors, we applied moving average filters to smooth the center and angle deviations of the pleural line during the actual scanning, and then we used these smoothed values for the probe adjustment control.

## *D. Compliant Robot Motion Control*

The ultrasound scanning procedure involves a significant amount of contact between the patient and the robot. Improper movement of the robot can result in excessive contact forces that could potentially injure the patient. To address this issue, we add an admittance controller to the robot's built-in motion controller. As shown in Fig. [5,](#page-6-1) after we obtain the desired motion from planned paths and online probe adjustments, we do not directly hand it over to the robot's motion controller for execution. Instead, we combine it with the contact forcebased admittance control to calculate a compliant motion as a reference for the inner motion control loop. Although this

![](_page_7_Figure_11.jpeg)

<span id="page-7-1"></span>Fig. 6. Experimental setups, including the compliant motion control experiment on (a) a chest phantom, and extensive experiments on (b) the anterior chest of volunteers to evaluate path planning methods, probe posture adjustment methods, and overall system performance. (c) Shows the preplanned paths for longitudinal scanning and the re-planned paths for transverse scanning during the scan of the first volunteer.

may lead to an imperfect execution of the desired motion, it allows the robot to automatically adjust its movement based on the contact force to avoid harming the patient.

In each compliant motion control cycle, given the desired 6-DoF posture of the ultrasound probe  $x_d \in \mathbb{R}^6$ , the desired 6-axis contact force/torque  $F_d \in \mathbb{R}^6$ , and the measured 6-axis force/torque  $\mathbf{F}_c \in \mathbb{R}^6$ , the compliant motion  $\mathbf{x}_c \in \mathbb{R}^6$  is determined by satisfying the condition

$$
M(\ddot{x}_c - \ddot{x}_d) + D(\dot{x}_c - \dot{x}_d) + K(x_c - x_d) = F_c - F_d, (21)
$$

where  $M, D, K \in \mathbb{R}^{6 \times 6}$  are the virtual inertia, damping, and stiffness matrices, respectively. They can be fine-tuned according to desired robot dynamics. The ideal contact force in the Z-direction is 8 N, while in other directions it is 0. After obtaining the Cartesian pose, we further use inverse kinematics to calculate the joint angles for the robot to execute.

## VI. EXPERIMENTS AND RESULTS

# <span id="page-7-2"></span><span id="page-7-0"></span>*A. Autonomous Robotic Lung Ultrasound Scan Workflow and Experimental Setups*

A robotic automatic lung ultrasound scanning workflow is designed to closely replicate clinician scanning practices. After planning scanning paths along the intercostal spaces based on the estimated subject's personalized intercostal centerlines, the robot scans each intercostal space one by one. For each intercostal space, the longitudinal scan is performed first, and the robot moves the probe from the middle of the body to the side along the planned path. During scanning, the target pleural

![](_page_8_Figure_2.jpeg)

<span id="page-8-0"></span>Fig. 7. Contact forces in Z-direction during scanning on a chest phantom (a) without compliant motion control, and (b) with compliant motion control. In (b), the red circle corresponds to the force when the probe remains stationary in other directions except the Z-direction.

lines are segmented from real-time ultrasound images, and probe translation adjustment along its long axis is performed based on the center deviation of the target pleural line. In addition, the center points of the target intercostal pleural lines in the ultrasound images along with the corresponding robot poses are saved in pairs. At the end of the longitudinal scan, the actual intercostal centerline is reconstructed and the scanning path is replanned for the next transverse scan. Then, the transverse scan is conducted from the side to the middle of the human body. During this process, the adjustment of the probe rotation around its short axis is made on the basis of the angle deviation of the pleural line. In each scan, when the probe moves to the initial position, compliant motion control is activated in the Z-direction, while other directions remain in position control to ensure positioning accuracy and safety. Upon locating the initial position or reaching the endpoint, compliant motion in the Z-direction remains enabled, while the other directions stay stationary for 4 s. During the 30-second movement along the planned path, compliant motion control is enabled in 6 DoFs.

As shown in Fig. [6,](#page-7-1) in order to ensure the safety of autonomous robotic lung ultrasound scanning, we first verified the compliant motion control method on a chest phantom. Based on this premise, we performed experimental evaluations of path planning methods, probe adjustment techniques, and overall system performance on the right anterior chest of two healthy male volunteers, aged 32 and 28. Both volunteers have body fat percentages within the normal range. For each volunteer, we scanned the representative second and third intercostal spaces of the anterior chest. The human experiment has been approved by the Institutional Review Board of the Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences (YSB-2020-Y0902).

## *B. Performance of Compliant Motion Control*

For the phantom experiment, after the surface point cloud of the phantom was obtained, a scanning path was planned on its right chest surface. Along this path, the longitudinal scan was

<span id="page-8-1"></span>TABLE I PLEURAL LINE SEGMENTATION RESULTS ON THE TEST SET IN TERMS OF MEAN VALUES OF FOUR QUANTITATIVE METRICS. ↑ / ↓ INDICATES THE HIGHER/LOWER THE SCORE, THE BETTER

| <b>Scenarios</b> | Dice $(\%)^{\uparrow}$ | HD95 (pixel) $\downarrow$ | ASD $(pixel)\downarrow$ | $JC(\%)^{\uparrow}$ |
|------------------|------------------------|---------------------------|-------------------------|---------------------|
| Transverse       | 85.08                  | 3.25                      | 1.37                    | 74.13               |
| Longitudinal     | 86.17                  | 3.24                      | 1.26                    | 76.18               |

first conducted from the middle to the side of the phantom, and then the transverse scan was performed from the side back to the starting position. We performed the experiments with and without compliant motion control, and the contact force in the Z-direction was recorded at 100 Hz.

As shown in Fig. [7,](#page-8-0) when compliant motion control was not enabled, during the longitudinal scan process, once the probe contacted the phantom,  $F_z$  increased sharply to nearly 60 N. As the probe moved, the contact force gradually decreased. In the latter part of the path, the contact force dropped to 0, indicating that the probe has lost contact with the phantom surface. In contrast, during the transverse scan process, the contact force started at 0 and gradually increased to more than 40 N. After enabling compliant motion control, during the longitudinal scan, when the probe contacted the phantom,  $F_z$ increased rapidly but did not exceed 13 N at its maximum. This is because force control was enabled only in the Z-direction of the probe, while other directions remained stationary at this time. When the probe began to move, the  $F<sub>z</sub>$  stabilized around the set value of 8 N. When the probe reached the end of the path and remained stationary except in the Z-direction,  $F<sub>z</sub>$  experienced a brief, slight increase. During the transverse scan process,  $F<sub>z</sub>$  behaved similarly. In the two scans, the mean value of  $F_z$  is 7.91 N, with a standard deviation of 2.23 N. The experimental results clearly demonstrate that using the compliant motion control ensures stable contact between the probe and the skin while avoiding excessive contact force that could harm the subject.

## *C. Pleural Line Segmentation Results*

To evaluate the performance of the pleural line segmentation model in the test set, four metrics are adopted. Among them, the dice coefficient indicates the overlap between the predicted and true segmentation, and the 95th percentile Hausdorff distance (HD95) measures the distance between two boundaries while reducing the sensitivity to outliers. The average surface distance (ASD) calculates the average distance between the surfaces of the segmented and groundtruth regions. Jaccard index (JC), also known as intersection over union (IoU), is similar to Dice, but penalizes disjoint sets more heavily. These metrics provide a comprehensive assessment of segmentation quality. The results of pleural line segmentation in two scanning scenarios, longitudinal and transverse scanning, are shown in Table [I.](#page-8-1) Overall, these results suggest that the model performs well in both scenarios, with good overlap and boundary accuracy. The segmentation results in the transverse scan are slightly worse than in the longitudinal scan because the visible pleural line tends to be thin and long in transverse scans, and its accurate segmentation

![](_page_9_Figure_1.jpeg)

<span id="page-9-0"></span>Fig. 8. Sample segmentation results in three cases. The first two rows show segmentation results from hospital patient data, and the last two rows show results from Volunteer 1's data. Segmented and ground truth region boundaries are shown by red and green contours, respectively.

is more likely to be disturbed by the surrounding tissue, as shown in Fig. [8.](#page-9-0)

# *D. Pre-Scan Path Planning & Intra-Scan Path Replanning*

In order to verify the validity of the pre-scan path planning method and the intra-scan path replanning method, we conducted robotic ultrasound scanning experiments on the second and third intercostal spaces of two volunteers. For each intercostal space, a longitudinal scan from the middle to the side of the body was performed along the paths planned by the two methods, respectively, as shown in Fig. [6\(](#page-7-1)c). During the scanning process, the center deviation *d* of the target intercostal pleural line, as well as the corresponding ultrasound images were recorded to characterize the precision of the paths along the intercostal spaces. Table  $II$  summarizes the mean and maximum absolute values of the center deviation of the target pleural line for each scan. It can be seen that when scanning along the paths from the pre-scan planning method, the mean values of  $|d|$  are high, even reaching the centimeter level. This indicates that the planned paths deviate significantly from the ideal paths along the centerlines of the intercostal spaces. The maximum |*d*| in all experiments is 17.38 mm, it is noteworthy that the C5-1 probe we used has a half field of view exceeding 3 mm at the pleural depth. Therefore, even with errors in the pre-scan path planning, the target intercostal pleura can still be scanned during longitudinal scanning. In contrast, when scanning along the paths planned by the intrascan path replanning method, the mean values of |*d*| are around 2 mm, and the maximum values do not exceed 5.5 mm. This clearly demonstrates that the replanned scanning path based on the reconstructed true intercostal centerline can ensure a more accurate transverse scanning along the intercostal space.

To gain a more detailed and intuitive understanding of the effects of the two path planning methods, we plot the center deviation variation curves during the scans of the second intercostal space for Volunteer 1, and also present several <span id="page-9-1"></span>EXPERIMENTAL RESULTS OF PRE-SCAN PATH PLANNING, INTRA-SCAN PATH REPLANNING, AND ONLINE PROBE TRANSLATION ADJUSTMENT METHODS ON TWO VOLUNTEERS, IN TERMS OF MEAN AND MAXIMUM VALUES OF |*d*|, UNIT: MM

| Volunteers     | ICS | Pre-planned path |       | Pre-planned path<br>with adjustment |       | Re-planned path |      |
|----------------|-----|------------------|-------|-------------------------------------|-------|-----------------|------|
|                |     | Mean             | Max   | Mean                                | Max   | Mean            | Max  |
|                | 2nd | 12.63            | 17.38 | 5.00                                | 10.16 | 1.93            | 4.86 |
|                | 3rd | 6.19             | 12.36 | 3.68                                | 7.17  | 2.07            | 5.39 |
| $\overline{c}$ | 2nd | 7.79             | 13.14 | 4.24                                | 8.77  | 2.02            | 5.11 |
|                | 3rd | 9.27             | 15.33 | 5.39                                | 9.42  | 2.18            | 5.26 |

![](_page_9_Figure_10.jpeg)

<span id="page-9-2"></span>Fig. 9. Experimental results of pre-scan path planning, intra-scan path replanning, and online probe translation adjustment methods, on the second intercostal space of Volunteer 1.

ultrasound images obtained at four evenly distributed time points, as shown in Fig. [9.](#page-9-2) During scanning following the pre-planned path, the center deviation of the target intercostal pleural line was within the desired range  $[-5, 5]$  (mm) when the probe was positioned near the middle of the body. As the probe was gradually moved to the lateral part of the body, the target intercostal pleural line gradually deviated from the center of the image, and finally the degree of deviation leveled off, with a maximum deviation magnitude of more than 17 mm throughout the scan. However, as shown in the ultrasound images Fig.  $9$  (b1)-(b4) at the four time nodes  $(0, 10, 20, 30)$ (s), corresponding to the center deviations of  $(-4.01, -12.78,$ −16.01, −16.02) (mm), the target intercostal pleural lines were completely visible in the ultrasound images throughout the scanning, confirming the effectiveness of the pre-scan path planning method for longitudinal scan. During scanning

<span id="page-10-0"></span>TABLE III EXPERIMENTAL RESULTS OF ONLINE PROBE ROTATION ADJUSTMENT METHOD ON TWO VOLUNTEERS, IN TERMS OF MEAN AND MAXIMUM VALUES OF  $|\theta|$ , UNIT:  $^{\circ}$ 

| Volunteers | ICS |       | Without rotation adjustment | With rotation adjustment |       |  |
|------------|-----|-------|-----------------------------|--------------------------|-------|--|
|            |     | Mean  | Max                         | Mean                     | Max   |  |
|            | 2nd | 12.02 | 22.15                       | 6.75                     | 21.50 |  |
|            | 3rd | 11.04 | 20.49                       | 4.98                     | 18.45 |  |
| 2          | 2nd | 14.38 | 26.16                       | 6.38                     | 26.49 |  |
|            | 3rd | 10.26 | 26.73                       | 5.29                     | 24.78 |  |

![](_page_10_Figure_3.jpeg)

<span id="page-10-1"></span>Fig. 10. Experimental results of online probe rotation adjustment method on the second intercostal space of Volunteer 1.

following the re-planned path, the center deviation values of the target intercostal pleural line remained within the desired range, and the target intercostal pleural lines were almost in the middle of the ultrasound images, just as in images Fig. [9](#page-9-2) (d1)-(d4) at the four time nodes, where the corresponding center deviations are 1.18 mm, 1.71 mm, 2.04 mm, and 2.10 mm, respectively. The result confirms that the path replanned based on the reconstructed intercostal pleural centerline can effectively ensure that the images obtained are not obstructed by ribs during the transverse scan.

## *E. Online Adaptive Adjustments of Probe Posture*

*1) Translation Adjustment of the Probe Along Its Long Axis:* To evaluate the effectiveness of the online probe translation adjustment method, the online probe translation adjustment function was enabled during robotic ultrasound scanning along the pre-planned path, and the comparative experimental results with and without the online probe translation adjustment are also shown in Table [II](#page-9-1) and Fig.  $9$ . As summarized in the table, along the same pre-planned paths, when probe translation adjustments are conducted based on pleural line feedback, the mean and maximum values of |*d*| significantly decrease, demonstrating the effectiveness of this method. As can be seen from the figure, starting from the same initial position, the center deviation of the target intercostal pleural line fluctuated around the −5 mm line as the probe traveled to the lateral part of the body with the probe translation adjustment, with a minimum value of  $-10.16$  mm, rather than decreasing all

the way below  $-17$  mm as it would have done without the probe translation adjustment. As shown in the ultrasound images at the four time points (0, 10, 20, 30) (s), the target intercostal pleural line always remained in the middle of the image in the presence of the probe translation adjustment, which not only facilitates the diagnosis of the disease based on the image obtained from the scan, but also prevents the target intercostal pleural line from exceeding the imaging range, thus guaranteeing the reconstruction accuracy of the target intercostal centerline to further replan the path for the following transverse scan.

*2) Rotation Adjustment of the Probe Around Its Short Axis:* To verify the effectiveness of the online probe rotation adjustment method, we compared the angle deviation  $\theta$  of the intercostal pleural line and the ultrasound images obtained during the transverse scans with and without the probe rotation adjustment. In these experiments, re-planned paths based on the reconstructed intercostal centerlines were followed, with the probe moving from the side to the middle part of the body.

As listed in Table  $III$ , for each scan, along the same scanning path, when the probe rotation adjustment is performed based on real-time pleural line feedback, the mean  $|\theta|$ decreases significantly. Furthermore, we observe that the maximum values of  $|\theta|$  remain largely unchanged with or without rotation adjustments in all experiments. This is because these maximum values typically occur at the starting position of the transverse scans, where the adjustment has not yet taken effect.

As shown in Fig. [10,](#page-10-1) starting from the same initial angle deviation, when with the probe rotation adjustment, the angle deviation was adjusted to below 5◦ at a significant rate and eventually fluctuated around  $-5^\circ$ . At four time points (0, 10, 20, 30) (s), (b1-b4) correspond to angle deviations of 21.34 $\degree$ , 2.58 $\degree$ ,  $-6.12\degree$  and  $-3.33\degree$ , respectively, and (c1-c4) correspond to angular deviations of 21.83◦, 11.66◦, −10.06◦, and −11.40◦, respectively. By comparing these ultrasound images, it can be seen that the pleural lines stabilize near the horizontal position of the images for a longer period of time in the presence of online probe rotation adjustment, and the A-lines are also more clearly visualized in this case. These results demonstrate the necessity of an online probe rotation adjustment based on the deviation of the pleural line angle and its effectiveness in obtaining diagnostically meaningful ultrasound images.

# *F. Performance of Robotic Lung Ultrasound Scan System*

Based on the verification of the aforementioned methods, we performed complete and coherent robotic autonomous lung ultrasound scanning on the second and third intercostal spaces of each volunteer according to the workflow described in Section [VI-A.](#page-7-2) At the same time, we invited two sonographers with more than ten years of ultrasound scanning experience to evaluate the performance of the robotic system. Assessment indicators included intercostal scanning ability, A-line imaging quality, scanning uniformity, and efficiency. Each indicator was quantified on a scale of 0 to 5, with higher scores indicating better performance.

The statistical results are shown in Table [IV,](#page-11-1) Overall, the sonographers recognized the ability of the robotic system to Authorized licensed use limited to: Johns Hopkins University. Downloaded on July 26,2025 at 19:42:58 UTC from IEEE Xplore. Restrictions apply.

![](_page_11_Figure_2.jpeg)

<span id="page-11-2"></span>Fig. 11. Visualization of (a) robotic lung ultrasound scan process along one intercostal space of the volunteer, as well as the (b) contact force in Z-direction. In (a), the red point in the ultrasound images is the center of the target intercostal pleural line during longitudinal scan, and the blue line connecting the two endpoints of the pleural line relative to the horizontal green line indicates the inclination of the probe relative to the pleura during transverse scan.

<span id="page-11-1"></span>TABLE IV EVALUATION RESULTS OF ROBOTIC SYSTEM PERFORMANCE BY SONOGRAPHERS BASED ON FOUR INDICATORS (SCORES FROM 0 TO 5, HIGHER SCORES INDICATE BETTER PERFORMANCE)

| <b>Indicators</b>       | Volunteers | Sonographer 1 | Sonographer 2 | Mean |
|-------------------------|------------|---------------|---------------|------|
| Intercostal<br>scanning |            |               |               | 3.75 |
| A line<br>imaging       |            |               |               | 2.75 |
| Uniformity              |            |               |               |      |
| Efficiency              |            |               |               | 2.5  |

scan along the intercostal spaces and its uniformity of the scan. However, they were not very satisfied with the quality of A-line imaging and scanning efficiency. The sonographers reported that they were mainly unhappy with the image quality of the A-lines during the initial short period of the transverse scans, when the angle deviation of the pleural line was significant. This is because our robot adjusts the probe posture while moving forward, and the adjustments only correct the probe angle for subsequent scans. To address this problem, on-site adjustments similar to those made by sonographers may be necessary to ensure the quality of A-line imaging at each location. Concerning the issue of low scanning efficiency, we intentionally set the scanning speed to a slower pace primarily to ensure safety during the experiments. Furthermore, volunteers reported no discomfort throughout the scanning process.

The key frames during the scanning of the second intercostal space for Volunteer 1 are shown in Fig. [11.](#page-11-2) It can be seen that the entire scanning process meets the expected results. We also

show the Z-axis contact force  $F_z$ , as in Fig. [11\(](#page-11-2)b). It can be seen that  $F<sub>z</sub>$  behaved in the same way on the volunteer as on the chest phantom during scanning, and the mean value was 8.13 N, with a standard deviation of 1.88 N. This confirms the safety of the robotic lung ultrasound scan on the human body.

## <span id="page-11-5"></span><span id="page-11-3"></span>VII. CONCLUSION

<span id="page-11-4"></span><span id="page-11-0"></span>In this paper, we addressed key challenges in the achievement of autonomous robotic lung ultrasound scans. We proposed a novel method for planning the longitudinal and transverse scan path along intercostal spaces, coupled with posture control of the ultrasound probe based on pleural line feedback from images. We developed a robotic lung ultrasound system and validated the effectiveness of our proposed methods in obtaining diagnostically significant ultrasound images on two volunteers. Although we achieved successful results, our experiments were conducted solely on healthy male volunteers with body fat percentages within the normal range, and were limited to the anterior chest area. In reality, factors such as the subject's body shape and health status can affect the accuracy of these methods, particularly the prescan path planning method. In the future, we will continue to refine these methods to enhance their applicability to patients with different body types, including women, and extend them to robotic full lung scanning. We will also improve the scanning workflow and efficiency of the robotic system, and incorporate metrics such as disease detection rates to systematically evaluate its performance. Furthermore, we will address challenges related to patient position changes [\[42\]](#page-12-41) and tissue deformation [\[43\]](#page-13-0) during the scanning process, and we will enhance the robot's capabilities with active learning [\[44\]](#page-13-1) for intercostal exploration. These advances aim to relieve doctors

from intensive and potentially infectious manual procedures, ultimately benefiting medical professionals and patients.

#### ACKNOWLEDGMENT

Long Lei is with the Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China, and also with the Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China (e-mail: longlei@cuhk.edu.hk).

Yingbai Hu is with the Multi-Scale Medical Robotics Centre Ltd., The Chinese University of Hong Kong, Hong Kong, China (e-mail: yingbaihu@mrc-cuhk.com).

Zixing Jiang and Xiao Luo are with the Department of Surgery, The Chinese University of Hong Kong, Hong Kong, China (e-mail: zxjiang@ surgery.cuhk.edu.hk; xluo@surgery.cuhk.edu.hk).

Juzheng Miao, Yu Zhang, and Pheng-Ann Heng are with the Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China (e-mail: jzmiao22@cse.cuhk.edu.hk; yzhang@cuhk.edu.hk; pheng@cse.cuhk.edu.hk).

Qiong Wang is with the Guangdong Provincial Key Laboratory of Computer Vision and Virtual Reality Technology, Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China (e-mail: wangqiong@siat.ac.cn).

Shujun Wang is with the Department of Biomedical Engineering, The Hong Kong Polytechnic University, Hong Kong, China (e-mail: shu-jun.wang@ polyu.edu.hk).

Zheng Li is with the Department of Surgery, Chow Yuk Ho Technology Centre for Innovative Medicine, Li Ka Shing Institute of Health Sciences, and Multiscale Medical Robotics Centre Ltd., The Chinese University of Hong Kong, Hong Kong, China (e-mail: lizheng@cuhk.edu.hk).

## **REFERENCES**

- <span id="page-12-0"></span>[\[1\]](#page-0-0) A. Saraogi, "Lung ultrasound: Present and future," *Lung India*, vol. 32, no. 3, pp. 250–257, 2015.
- <span id="page-12-1"></span>[\[2\]](#page-0-1) I. Blazic et al., "The use of lung ultrasound in COVID-19," *ERJ Open Res.*, vol. 9, no. 1, p. 196, 2023.
- <span id="page-12-2"></span>[\[3\]](#page-0-1) T. J. Marini et al., "Lung ultrasound volume sweep imaging for pneumonia detection in rural areas: Piloting training in rural Peru," *J. Clin. Imag. Sci.*, vol. 9, p. 35, Jul. 2019.
- <span id="page-12-3"></span>[\[4\]](#page-0-2) F. Mojoli, B. Bouhemad, S. Mongodi, and D. Lichtenstein, "Lung ultrasound for critically ill patients," *Amer. J. Respir. Crit. Care Med.*, vol. 199, no. 6, pp. 701–714, 2019.
- <span id="page-12-4"></span>[\[5\]](#page-0-3) T. J. Marini et al., "Lung ultrasound: The essentials," *Radiol., Cardiothoracic Imag.*, vol. 3, no. 2, 2021, Art. no. e200564.
- <span id="page-12-5"></span>[\[6\]](#page-0-4) G. Soldati, M. Demi, A. Smargiassi, R. Inchingolo, and L. Demi, "The role of ultrasound lung artifacts in the diagnosis of respiratory diseases," *Expert Rev. Respir. Med.*, vol. 13, no. 2, pp. 163–172, 2019.
- <span id="page-12-6"></span>[\[7\]](#page-0-5) S. Abbas and P. Peng, "Basic principles and physics of ultrasound," in *Ultrasound for Interventional Pain Management: An Illustrated Procedural Guide*. Cham, Switzerland: Springer, 2020, pp. 1–31.
- <span id="page-12-7"></span>[\[8\]](#page-0-5) J. Wang et al., "Review of machine learning in lung ultrasound in COVID-19 pandemic," *J. Imag.*, vol. 8, no. 3, p. 65, 2022.
- <span id="page-12-8"></span>[\[9\]](#page-0-6) L. Gargani and G. Volpicelli, "How I do it: Lung ultrasound," *Cardiovasc. Ultrasound*, vol. 12, pp. 1–10, Jul. 2014.
- <span id="page-12-9"></span>[\[10\]](#page-1-2) K. Li, Y. Xu, and M. Q.-H. Meng, "An overview of systems and techniques for autonomous robotic ultrasound acquisitions," *IEEE Trans. Med. Robot. Bionics*, vol. 3, no. 2, pp. 510–524, May 2021.
- <span id="page-12-10"></span>[\[11\]](#page-1-2) Z. Jiang, S. E. Salcudean, and N. Navab, "Robotic ultrasound imaging: State-of-the-art and future perspectives," *Med. Image Anal.*, vol. 89, Oct. 2023, Art. no. 102878.
- <span id="page-12-11"></span>[\[12\]](#page-1-3) X. Ma, Z. Zhang, and H. K. Zhang, "Autonomous scanning target localization for robotic lung ultrasound imaging," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2021, pp. 9467–9474.
- <span id="page-12-12"></span>[\[13\]](#page-1-3) L. Al-Zogbi et al., "Autonomous robotic point–of–care ultrasound imaging for monitoring of COVID-19–induced pulmonary diseases," *Front. Robot. AI*, vol. 8, May 2021, Art. no. 645756.
- <span id="page-12-13"></span>[\[14\]](#page-1-3) B. Zhang, K. Xu, H. Cong, and M. Sun, "Channel and spatial attention mechanism-based Yolo network for target detection of the lung ultrasound scanning robot," in *Proc. 9th Int. Conf. Control Sci. Syst. Eng. (ICCSSE)*, 2023, pp. 319–324.
- <span id="page-12-14"></span>[\[15\]](#page-1-3) B. Zhang, H. Cong, Y. Shen, and M. Sun, "Visual perception and convolutional neural network-based robotic autonomous lung ultrasound scanning Localization system," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 70, no. 9, pp. 961–974, Sep. 2023.
- <span id="page-12-15"></span>[\[16\]](#page-1-4) F. Suligoj, C. M. Heunis, J. Sikorski, and S. Misra, "RobUSt–an autonomous robotic ultrasound system for medical imaging," *IEEE Access*, vol. 9, pp. 67456–67465, 2021.

- <span id="page-12-16"></span>[\[17\]](#page-1-4) X. Ma, W.-Y. Kuo, K. Yang, A. Rahaman, and H. K. Zhang, "A-SEE: Active-sensing end-effector enabled probe self-normal-positioning for robotic ultrasound imaging applications," *IEEE Robot. Autom. Lett.*, vol. 7, no. 4, pp. 12475–12482, Oct. 2022.
- <span id="page-12-17"></span>[\[18\]](#page-1-4) J. Tan et al., "Fully automatic dual-probe lung ultrasound scanning robot for screening triage," *IEEE Trans. Ultrason., Ferroelectr., Freq. Control*, vol. 70, no. 9, pp. 975–988, Sep. 2023.
- <span id="page-12-18"></span>[\[19\]](#page-1-5) M. Blaivas, "Lung ultrasound in evaluation of pneumonia," *J. Ultrasound Med.*, vol. 31, no. 6, pp. 823–826, 2012.
- <span id="page-12-19"></span>[\[20\]](#page-1-6) D. A. Lichtenstein, "BLUE-protocol and FALLS-protocol: Two applications of lung ultrasound in the critically ill," *Chest*, vol. 147, no. 6, pp. 1659–1670, 2015.
- <span id="page-12-20"></span>[\[21\]](#page-1-7) X. Wang et al., "The value of bedside lung ultrasound in emergencyplus protocol for the assessment of lung consolidation and atelectasis in critical patients," *Zhonghua Nei Ke Za Zhi*, vol. 51, no. 12, pp. 948–951, 2012.
- <span id="page-12-21"></span>[\[22\]](#page-1-8) J. M. Smit et al., "Lung ultrasound in a tertiary intensive care unit population: A diagnostic accuracy study," *Crit. Care*, vol. 25, pp. 1–9, Sep. 2021.
- <span id="page-12-22"></span>[\[23\]](#page-1-9) Z. Wang et al., "Full-coverage path planning and stable interaction control for automated robotic breast ultrasound scanning," *IEEE Trans. Ind. Electron.*, vol. 70, no. 7, pp. 7051–7061, Jul. 2023.
- <span id="page-12-23"></span>[\[24\]](#page-1-10) Q. Huang, J. Lan, and X. Li, "Robotic arm based automatic ultrasound scanning for three-dimensional imaging," *IEEE Trans. Ind. Informat.*, vol. 15, no. 2, pp. 1173–1182, Feb. 2019.
- <span id="page-12-24"></span>[\[25\]](#page-1-11) K. Okuzaki, N. Koizumi, K. Yoshinaka, Y. Nishiyama, J. Zhou, and R. Tsumura, "Rib region detection for scanning path planning for fully automated robotic abdominal ultrasonography," *Int. J. Comput. Assist. Radiol. Surg.*, vol. 19, no. 3, pp. 449–457, 2024.
- <span id="page-12-25"></span>[\[26\]](#page-1-12) C. Graumann, B. Fuerst, C. Hennersperger, F. Bork, and N. Navab, "Robotic ultrasound trajectory planning for volume of interest coverage," in *Proc. IEEE Int. Conf. Robot. Autom. (ICRA)*, 2016, pp. 736–741.
- <span id="page-12-26"></span>[\[27\]](#page-2-2) Y. Bi, C. Qian, Z. Zhang, N. Navab, and Z. Jiang, "Autonomous path planning for intercostal robotic ultrasound imaging using reinforcement learning," 2024, *arXiv:2404.09927*.
- <span id="page-12-27"></span>[\[28\]](#page-2-3) Z. Jiang, C. Li, X. Lil, and N. Navab, "Thoracic cartilage ultrasound-ct registration using dense skeleton graph," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, 2023, pp. 6586–6592.
- <span id="page-12-28"></span>[\[29\]](#page-2-4) C. Yang, M. Jiang, M. Chen, M. Fu, J. Li, and Q. Huang, "Automatic 3-D imaging and measurement of human spines with a robotic ultrasound system," *IEEE Trans. Instrum. Meas.*, vol. 70, pp. 1–13, May 2021.
- <span id="page-12-29"></span>[\[30\]](#page-2-4) Z. Jiang et al., "Automatic normal positioning of robotic ultrasound probe based only on confidence map optimization and force measurement," *IEEE Robot. Autom. Lett.*, vol. 5, no. 2, pp. 1342–1349, Apr. 2020.
- <span id="page-12-30"></span>[\[31\]](#page-2-4) Z. Jiang, M. Grimm, M. Zhou, Y. Hu, J. Esteban, and N. Navab, "Automatic force-based probe positioning for precise robotic ultrasound acquisition," *IEEE Trans. Ind. Electron.*, vol. 68, no. 11, pp. 11200–11211, Nov. 2021.
- <span id="page-12-31"></span>[\[32\]](#page-2-4) G. Ning, H. Liang, X. Zhang, and H. Liao, "Autonomous robotic ultrasound vascular imaging system with decoupled control strategy for external-vision-free environments," *IEEE Trans. Biomed. Eng.*, vol. 70, no. 11, pp. 3166–3177, Nov. 2023.
- <span id="page-12-32"></span>[\[33\]](#page-2-5) B. Ihnatsenka and A. P. Boezaart, "Ultrasound: Basic understanding and learning the language," *Int. J. Shoulder Surg.*, vol. 4, no. 3, pp. 55–62, 2010.
- <span id="page-12-33"></span>[\[34\]](#page-2-6) Z. Jiang et al., "Autonomous robotic screening of tubular structures based only on real-time ultrasound imaging feedback," *IEEE Trans. Ind. Electron.*, vol. 69, no. 7, pp. 7064–7075, Jul. 2022.
- <span id="page-12-34"></span>[\[35\]](#page-2-6) G. Faoro, S. Maglio, S. Pane, V. Iacovacci, and A. Menciassi, "An artificial intelligence-aided robotic platform for ultrasound-guided transcarotid revascularization," *IEEE Robot. Autom. Lett.*, vol. 8, no. 4, pp. 2349–2356, Apr. 2023.
- <span id="page-12-35"></span>[\[36\]](#page-3-2) L. Lei et al., "Robotic needle insertion with 2D ultrasound–3D CT fusion guidance," *IEEE Trans. Autom. Sci. Eng.*, vol. 21, no. 4, pp. 6152–6164, Oct. 2024.
- <span id="page-12-36"></span>[\[37\]](#page-3-3) Y. Yu, R. Shi, and Y. Lou, "Bias estimation and gravity compensation for wrist-mounted force/torque sensor," *IEEE Sensors J.*, vol. 22, no. 18, pp. 17625–17634, Sep. 2022.
- <span id="page-12-37"></span>[\[38\]](#page-3-4) G. Farin, *Curves and Surfaces for Computer-Aided Geometric Design: A Practical Guide*. Amsterdam, The Netherlands Elsevier, 2014.
- <span id="page-12-38"></span>[\[39\]](#page-4-1) A. S. B. Mustafa et al., "Development of robotic system for autonomous liver screening using ultrasound scanning device," in *Proc. IEEE Int. Conf. Robot. Biomimetics (ROBIO)*, 2013, pp. 804–809.
- <span id="page-12-39"></span>[\[40\]](#page-4-1) R. Tsumura, Y. Koseki, N. Nitta, and K. Yoshinaka, "Towards fully automated robotic platform for remote auscultation," *Int. J. Med. Robot. Compu. Assist. Surg.*, vol. 19, no. 1, 2023, Art. no. e2461.
- <span id="page-12-40"></span>[\[41\]](#page-5-3) F. Isensee, P. F. Jaeger, S. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-net: A self-configuring method for deep learning-based biomedical image segmentation," *Nat. Methods*, vol. 18, no. 2, pp. 203–211, 2021.
- <span id="page-12-41"></span>[\[42\]](#page-11-3) Z. Jiang, Y. Gao, L. Xie, and N. Navab, "Towards autonomous atlasbased ultrasound acquisitions in presence of articulated motion," *IEEE Robot. Autom. Lett.*, vol. 7, no. 3, pp. 7423–7430, Jul. 2022.

- <span id="page-13-0"></span>[\[43\]](#page-11-4) Z. Jiang, Y. Zhou, D. Cao, and N. Navab, "DefCor-Net: Physicsaware ultrasound deformation correction," *Med. Image Anal.*, vol. 90, Dec. 2023, Art. no. 102923.
- <span id="page-13-1"></span>[\[44\]](#page-11-5) Z. Jiang, Y. Bi, M. Zhou, Y. Hu, M. Burke, and N. Navab, "Intelligent robotic sonographer: Mutual information-based disentangled reward learning from few demonstrations," *Int. J. Robot. Res.*, vol. 43, no. 7, pp. 981–1002, 2024.

![](_page_13_Picture_3.jpeg)

**Long Lei** received the Ph.D. degree in mechanical engineering from the Harbin Institute of Technology (Shenzhen), Shenzhen, China, in 2022. He is currently a Postdoctoral Fellow with the Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China. He is also an Assistant Research Fellow with the Shenzhen Institute of Advanced Technology, Chinese Academy of Science, Shenzhen. His research interests include medical robotics, surgical navigation, medical image analysis, and artificial intelligence.

![](_page_13_Picture_5.jpeg)

**Yingbai Hu** received the Ph.D. degree in computer science from the Technical University of Munich, Munich, Germany, in 2022, where he was a Research Postdoctoral Fellow from 2022 to 2023. He is currently a Research Postdoctoral Fellow with the Multi-Scale Medical Robotics Center Ltd., The Chinese University of Hong Kong, China. His main research interests include imitation learning, reinforcement learning, optimization control, and medical robotics.

**Zixing Jiang** received the B.Eng. degree in electroic information engineering from The Chinese University of Hong Kong (Shenzhen), Shenzhen, China, in 2023. He is currently pursuing the M.Phil. degree in surgery with the Chinese University of Hong Kong, Hong Kong, SAR, China. He has broad research interests in the field of medical robotics, particularly robot-assisted imaging, and

robotic image-guided interventions.

![](_page_13_Picture_7.jpeg)

**Yu Zhang** received the B.Eng. degree in electronic engineering from Chongqing University in 2013, and the M.S. degree in computer science from The Chinese University of Hong Kong in 2014. Since 2014, he has been holding a research position with the Chinese University of Hong Kong, while also serving as a product manager in the industry. His research focuses on translating engineering innovations into practical applications in medical education, clinical support, and health screening. He specializes in leveraging AI, VR, and XR tech-

nologies to drive industrialization, commercialization, and interdisciplinary collaboration in these fields.

![](_page_13_Picture_10.jpeg)

**Qiong Wang** received the Ph.D. degree from The Chinese University of Hong Kong, Hong Kong, China, in 2012. She is currently a Professor with the Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, Shenzhen, China. Her research interests include surgical robots, visualization, medical imaging, human–computer interaction, and computer graphics.

![](_page_13_Picture_12.jpeg)

**Shujun Wang** (Member, IEEE) is an Assistant Professor with PolyU BME. She is a member of the Research Institute for Smart Ageing and the Research Institute for Artificial Intelligence of Things. Her citation number is 3308 in Google Scholar dated January 2025. Her research is in the interdisciplinary field of artificial intelligence (AI) and healthcare. She is dedicated to designing AI-driven computational methods to enable reliable decision-making models for precision medicine, covering from disease diagnosis to prognosis. Her

team obtained the Best Paper Award of CMMCA Workshop at MICCAI 2022. She won the Champion of the REFUGE challenge in 2018 and the second place of PALM challenge in 2019.

![](_page_13_Picture_15.jpeg)

**Juzheng Miao** received the bachelor's and master's degrees in biomedical engineering from Southeast University, China. He is currently pursuing the Ph.D. degree with the Department of Computer Science and Engineering, The Chinese University of Hong Kong, under the supervision of Prof. P.-A. Heng. His research interests include label-efficient learning, reinforcement learning, medical image analysis, and causality inspired deep learning.

![](_page_13_Picture_17.jpeg)

**Zheng Li** (Senior Member, IEEE) received the Ph.D. degree in robotics from The Chinese University of Hong Kong, Hong Kong, in 2013, where he is currently an Associate Professor with the Department of Surgery, Chow Yuk Ho Technology Centre for Innovative Medicine, Li Ka Shing Institute of Health Science and Multi-Scale Medical Robotics Center. His research interests include design, kinematic modeling, sensing, control, and human—robot interaction of flexible/soft robots and magnetic actuated robots for medical applications.

![](_page_13_Picture_19.jpeg)

**Xiao Luo** received the B.S. degree from Chongqing University, Chongqing, China, in 2018, and the M.S. degree from Shanghai Jiao Tong University, Shanghai, China, in 2021. He is currently pursuing the Ph.D. degree with the Department of Surgery, The Chinese University of Hong Kong, Hong Kong, China. His research interests include the development of surgical robots, motion planning, and force control.

![](_page_13_Picture_21.jpeg)

**Pheng-Ann Heng** (Senior Member, IEEE) is a Choh-Ming Li Professor of Computer Science and Engineering with The Chinese University of Hong Kong. He is the Director of the Institute of Medical Intelligence and XR. According to Google Scholar, his publications have been cited over 60 000 times, with an H-index of 117. His research interests include AI/XR for medical and scientific applications, visualization, graphics, human—computer interaction, and computer vision. He is recognized as a Highly Cited Researcher by Clarivate in 2024 and

also recognized with the Research.com Computer Science in China Leader Award since 2023.