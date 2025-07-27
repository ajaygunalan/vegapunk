# Bias Estimation and Gravity Compensation for Wrist-Mounted Force/Torque Sensor

Yongqiang Y[u](https://orcid.org/0000-0001-8203-7795)<sup>®</sup>, Ran Sh[i](https://orcid.org/0000-0003-4057-4861)<sup>®</sup>, and Yunjiang Lou<sup>®</sup>, Senior Member, IEEE

**Abstract—Robot force-controlled task executions require an accurate perception of the contact force/torque. When a six-axis force/torque (F/T) sensor is attached to the robot wrist, the compensation precision is affected by many noncontact features, such as the end-effector gravity, inertial, centrifugal, Coriolis forces, and the mechanical errors of the sensor. However, it is complicated to identify all parameters at the same time. Some features will be ignored in practice applications, e.g., surface finishing, human-robot interaction, and haptic applications. In this article, a novel end-effector gravity compensation method for the F/T sensor is proposed, which identifies the end-effector gravity, its center, and the biases (the rotation between the sensor and robot, the robot**

![](_page_0_Figure_4.jpeg)

**installation declination, the bias of sensor). Our key is the rotation calibration while the transformation between the F/T sensor and robot is unknown. Simulation and experimental results show that the proposed method works effectively. In a wrist-mounted F/T sensor system, compared to the previous limited method, the compensation error in common force control channel** *Fz* **by our method diminishes 1.0% of the end-effector gravity (i.e., 21.7% relative improvement).**

<sup>18</sup> **Index Terms—Force sensor, robotics, calibration, gravity compensation.**

## **I. INTRODUCTION**

IN MANY robot applications, such as the physical humanrobot interaction [1] and milling [2], the end-effector needs to contact with the world. Especially there is a huge demand to control the interaction force between the robot and the environment  $[3]$  in assembly  $[4]$ , polishing  $[5]$ , and perception of the unknown surface  $[6]$ . Since the force/torque  $(F/T)$ sensors made the force control possible [7], a six-axis F/T sensor mounted on the wrist of the manipulator is typically used for its realization [8]. However, the force and torque measured by the F/T sensor always contain the external (actual contact) force and the internal (non-contact) force due to the robot motion [9], e.g., gravity, inertial force, Coriolis force, centrifugal force, and associated torques [10]. Furthermore, the error caused by the mechanical error of the F/T sensor is also non-ignorable.

In order to ensure the accuracy of the force control, identifying the non-contact force and the accurate relation

Manuscript received 13 October 2020; revised 18 December 2020; accepted 18 December 2020. Date of publication 3 February 2021; date of current version 14 September 2022. This work was supported in part by the NSFC-Shenzhen Robotics Basic Research Center Program under Grant U1713202 and in part by the Shenzhen Science and Technology Program under Grant JCYJ20180508152226630. The associate editor coordinating the review of this article and approving it for publication was Dr. Yin Zhang. (Corresponding author: Yunjiang Lou.)

The authors are with the School of Mechatronics Engineering and Automation, Harbin Institute of Technology Shenzhen, HIT Campus, University Town of Shenzhen, Xili, Nanshan, Shenzhen 518055, China (e-mail: louyj@hit.edu.cn).

Digital Object Identifier 10.1109/JSEN.2021.3056943

between the base frame and sensor frame is necessary [11]. In the low-speed application, e.g., the high-precision surface finishing processes [12], direct teaching and transcranial magnetic stimulation (TMS) [13], the inertial, Coriolis, centrifugal components are ignored but the gravity compensation <sup>41</sup> is considered inevitably. When the end-effector configuration and robot pose is changed to complete the trajectory-tracking based on the geometric surface models and the measurements from six-axis force sensors, the gravitational force and torque should be compensated in real-time  $[12]$ , in which case setting the simple bias before every execution isn't suitable.

If the end-effector gravity and center are known, the gravitational force and torque could be compensated [14]. However, in practice, the gravity and the center are all unknown quantities with complex shape, which need to be identified by experiment or any other way. To solve the problem, Vougioukas  $[15]$ used the least-squares method (LSM) to estimate the bias of the F/T sensor, the mass, and its center respectively from a set of properly chosen F/T sensor orientation when the robot is in free space and not moving. Wang *et al.* [16] proposed a method that verified the installation angle bias of robot to improve the precision of the F/T sensor measurements because the robot base frame is not parallel to the direction of gravity acceleration. Yang *et al.* [17] presented a multi-parameter coupled model that includes the gravity of the end-effector, the bias of the sensor, and the installation angle of the sixaxis F/T sensor. But it just considers the rotation in the yaw <sup>64</sup> of Tait-Bryan angles between the sensor and robot wrist.

1558-1748 © 2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

![](_page_1_Figure_1.jpeg)

Fig. 1. Different robot installation types for a manipulator ( $\alpha \in (0, 2\pi]$ ).

Several progresses have been made by the aforementioned approaches, but there are still some aspects can be improved. Firstly, the non-contact compensation accuracy is mostly dependent on the uncertainties in the system, e.g., the accuracy and installation error of the F/T sensor, and robot kinematics error. While there requires the expensive commercially available F/T sensors, it's potential that how to use a cheap F/T sensor with good calibration and compensation. An idea focuses on the neural network-based gravity compensation method. Lin *et al*. [18] developed a method for the end-effector gravity compensation by using back-propagation (BP) neural networks. It didn't identify any parameter, and need too many data sets and hidden layers. Secondly, all works above have a premise that the rotation matrix between the F/T sensor and robot is known or ensured to the yaw in Tait-Bryan angles, but unknown or difficult in some applications.

In this article, a novel method that makes the calibration of the rotation matrix between the F/T sensor and robot is proposed to achieve higher compensation performances and ignore any mechanical bias. Based on the proposed method, the end-effector gravity could be compensated effectively and the F/T sensor's measurements can be transformed into the robot coordinate when robot motions continuously. The contributions of this article are as the following bias estimation:

- <sup>90</sup> 1) The proposed method makes the rotation calibration between the F/T sensor and robot with the analytical solution since no iterative process is needed.
- <sup>93</sup> 2) The calibration considers the rotation between the gravity acceleration frame and robot base frame when different robot installation types shown in Fig. 1 [19].

The remainder of this article is organized as following. In Section II, the model of the contact and non-contact components on force/torque in the F/T sensor measurements is introduced, then the end-effector gravity compensation model and identification model are extracted based on the kinematicsbased method. In Section III, the method that identifies the parameters of bias and gravity is presented. In Section IV, the simulation and physical experiments by the proposed method for the wrist-mounted F/T sensor system are discussed. Finally, conclusions are concluded in Section V.

## **II. PROBLEM FORMULATION**

The force and torque measured by F/T sensor are represented by vector  ${}^sF$ ,  ${}^sT \in \mathbb{R}^3$  (The superscript 'T' donates the transpose of a matrix.)

$$
{}^{s}F = [{}^{s}F_{x}, {}^{s}F_{y}, {}^{s}F_{z}]^{T} {}^{s}T_{s} = [{}^{s}T_{x}, {}^{s}T_{y}, {}^{s}T_{z}]^{T} \qquad (1)
$$

![](_page_1_Figure_11.jpeg)

Fig. 2. The coordinate definition of the tool gravity compensation system. Note: the robot base frame isn't parallel with the gravity frame. ({B}: the robot base frame;  $\{E\}$ : the robot end frame;  $\{S\}$ : the sensor frame;  $\{G\}$ : the gravity frame, is directed vertically upward; {T}:the end-effector frame).

where  ${}^sF_x$ ,  ${}^sF_y$ ,  ${}^sF_z$  are the Cartesian force and  ${}^sT_x$ ,  ${}^sT_y$ ,  ${}^sT_z$ are the torque components.

In the kinematics-based force compensation principle, besides the actual contact force  ${}^sF_{contact}$  (torque  ${}^sT_{contact}$ ), the vector  ${}^sF$  ( ${}^sT$ ) includes the bias of the sensor  ${}^sF_0$  $(^sT_0)$  and non-contact force, i.e., the gravitational forces  $^sF_g$  $({}^{s}T_{g})$ , inertial force (linear and angular acceleration)  ${}^{s}F_{i}$  ( ${}^{s}T_{i}$ ), centrifugal and Coriolis forces  ${}^sF_c$  ( ${}^sT_c$ ). In order to get a pure contact force and torque, the non-contact portions have to be eliminated from the measurement.

When it's needed to identify the rotation matrix between the sensor and robot and other parameters, or robot system is in Low-speed applications, e.g., grinding, polishing, the inertial force, centrifugal and Coriolis forces could be ignored. Assuming  ${}^S F_i = 0$ ,  ${}^S F_c = 0$ ,  ${}^S T_i = 0$ ,  ${}^S T_c = 0$ , the force and torque compensation principle is

$$
{}^{s}F = {}^{s}F_{contact} + {}^{s}F_{g} + {}^{s}F_{0}
$$
 (2)

$$
{}^{s}T = {}^{s}T_{contact} + {}^{s}T_{g} + {}^{s}T_{0}
$$
 (3)

This article considers the scene that the robot arm is equipped with a tool mounted in its wrist and the F/T sensor is inserted between the wrist and the tool (see Fig. 2). Not limited to the chain robot, the parallel mechanism is also applicable.

According to Fig. 2, the values of components in  ${}^sF_g$ and  ${}^{s}T_{g}$  depend on the mass of tool *m*, the local gravity acceleration *g*, the gravity center in sensor frame  ${}_{g}^{s}P$ , the sensor frame orientation in the gravity frame  ${}_{g}^{s}R \in \mathbb{SO}(3)$ 

$$
\begin{bmatrix} {}^{s}F_{g} \\ {}^{s}T_{g} \end{bmatrix} = \begin{bmatrix} {}^{s}_{g}R & 0_{3\times3} \\ {}^{s}_{g}P^{\wedge} \cdot {}^{s}_{g}R & {}^{s}_{g}R \end{bmatrix} \begin{bmatrix} F_{mg} \\ 0_{3\times1} \end{bmatrix}
$$
 (4)

where  $F_{mg} = [0, 0, -mg]^T$ ;  $\mathbb{SO}(3) = \{R \in \mathbb{R}^{3 \times 3} | R \cdot R^T = 1\}$  $R^T \cdot R = I$ ,  $\det R = 1$ ;  $\frac{s}{g} P^{\wedge}$  is described as a 3 × 3 matrix as shown in  $(5)$ .

$$
{}_{g}^{s}P = \begin{bmatrix} p_{x} \\ p_{y} \\ p_{z} \end{bmatrix}, \quad {}_{g}^{s}P^{\wedge} = \begin{bmatrix} 0 & -p_{z} & p_{y} \\ p_{z} & 0 & -p_{x} \\ -p_{y} & p_{x} & 0 \end{bmatrix}
$$
 (5)

Since  $\frac{s}{g}R$  isn't known, it could be got indirectly from the rotation matrix of {S} with respect to {E}  ${}_{e}^{s}R \in \mathbb{SO}(3)$ , {E} with respect to {B}  ${}_{b}^{e}R \in \mathbb{SO}(3)$  (the robot forward kinematics) and  ${B}$  with respect to  ${G}$   $_g^bR \in \mathbb{SO}(3)$  as shown in Fig.  $2$ 

$$
{}_{g}^{s}R={}_{e}^{s}R\cdot {}_{b}^{e}R\cdot {}_{g}^{b}R
$$
 (6)

Substituting  $(4)$ ,  $(6)$  into  $(2)$ ,  $(3)$ , it changes to

$$
{}^{s}F = {}^{s}F_{contact} + {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot {}^{b}_{g}R \cdot F_{mg} + {}^{s}F_{0} \tag{7}
$$

$$
{}^{s}T = {}^{s}T_{contact} + {}^{s}_{g}P^{\wedge} \cdot {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot {}^{b}_{g}R \cdot F_{mg} + {}^{s}T_{0} \qquad (8)
$$

Actually, an unknown variable  $F_b$  can be defined to replace two unknown variables  ${}_{g}^{b}R$  and  $F_{mg}$ 

$$
F_b = \frac{b}{g} R \cdot F_{mg} \tag{9}
$$

With this variable, the gravitational compensation model of the wrist-mounted F/T sensor is

$$
{}^{s}F = {}^{s}F_{contact} + {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot F_{b} + {}^{s}F_{0}
$$
 (10)

$$
{}^{s}T = {}^{s}T_{contact} + {}^{s}_{g}P^{\wedge} \cdot {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot F_{b} + {}^{s}T_{0} \tag{11}
$$

To identify the unknown variables  ${}_{e}^{s}R$ ,  $F_{b}$ ,  ${}_{s}^{s}F_{0}$ ,  ${}_{g}^{s}P$ ,  ${}_{s}^{s}T_{0}$ , we assumes that  ${}^sF_{contact} = 0$ ,  ${}^sT_{contact} = 0$  without external contacted force and torque to get the identification model

$$
{}^{s}F = {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot F_{b} + {}^{s}F_{0} \tag{12}
$$

$$
{}^{s}T = {}^{s}_{g}P^{\wedge} \cdot ({}^{s}F - {}^{s}F_{0}) + {}^{s}T_{0}
$$
 (13)

Now, there are 4 unknown biases ( $\frac{s}{e}R$ ,  $\frac{b}{g}R$ ,  $\frac{s}{g}F_0$ ,  $\frac{s}{g}T_0$ ) and 2 other unknown variables ( $F_{mg}$ ,  ${}_{g}^{s}P$ ) of end-effector, which can be estimated by our method proposed in Section III.

## **III. METHOD DESCRIPTION**

In the proposed method, it estimates the gravity compensation parameters in three steps. The method first estimate the parameters of the force identified model (12) through sensor measurement and robot kinematics, and then estimate the parameters of the torque identified model  $(13)$ . At the end, the tool's gravity and the robot installation angle are estimated.

While there isn't a great closed-form solution by leastsquares method  $(LSM)$  for the force identified model  $(12)$ directly due to the rotation constraint  ${}_{e}^{s}R$ , our idea is estimating  $F_b$  firstly in Section III-A. With the preliminary work, in Section III-B, it solves the problem as the least-squares fitting of the two 3-D points sets, which exists in many computer vision. Next, In Section III-C, it takes the estimation of the torque identified model. Finally, the tool's gravity and robot bias are discussed in Section III-D.

#### A. The Gravitational Forces in Robot Base Frame

Due to the rotation constraint  ${}_{e}^{s}R$ , the force identified model (12) hasn't a great analytical solution by LSM directly. We first estimate the gravitational forces in the robot base frame  $F_b$  ignoring the condition for the rotation matrix, as other parameters by the least-squares fitting with the singular value decomposition (SVD). Considering the robot workspace limit, two methods are proposed to estimate  $F_b$ .

|  |  |  | ${}_{s}^{c}R_{01} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & +1 & 0 \\ 0 & 0 & +1 \end{bmatrix}$ , ${}_{s}^{c}R_{02} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$ , ${}_{s}^{c}R_{03} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & +1 & 0 \end{bmatrix}$ , ${}_{s}^{c}R_{04} = \begin{bmatrix} +1 & 0 & 0 \\ 0 & 0 & +1 \\ 0 & -1 & 0 \end{bmatrix}$                                                                                            |  |
|--|--|--|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  | ${}_{s}^{c}R_{0s} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & +1 & 0 \\ 0 & 0 & -1 \end{bmatrix}$ , ${}_{s}^{c}R_{0s} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & +1 \end{bmatrix}$ , ${}_{s}^{c}R_{07} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 0 & +1 \\ 0 & +1 & 0 \end{bmatrix}$ , ${}_{s}^{c}R_{0s} = \begin{bmatrix} -1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & -1 & 0 \end{bmatrix}$                                                                                            |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  | $\label{eq:R0p} \begin{aligned} \n\zeta R_{09} = & \begin{bmatrix} 0 & +1 & 0 \\ +1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \nonumber \\ \zeta R_{10} = & \begin{bmatrix} 0 & -1 & 0 \\ +1 & 0 & 0 \\ 0 & 0 & +1 \end{bmatrix}, \nonumber \\ \zeta R_{11} = & \begin{bmatrix} 0 & 0 & +1 \\ +1 & 0 & 0 \\ 0 & +1 & 0 \end{bmatrix}, \nonumber \\ \zeta R_{12} = & \begin{bmatrix} 0 & 0 & -1 \\ +1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$                                     |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  | ${}_{s}^{c}R_{13} = \begin{bmatrix} 0 & +1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & +1 \end{bmatrix}$ , ${}_{s}^{c}R_{14} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 0 & 0 \\ 0 & 0 & -1 \end{bmatrix}$ , ${}_{s}^{c}R_{15} = \begin{bmatrix} 0 & 0 & -1 \\ -1 & 0 & 0 \\ 0 & +1 & 0 \end{bmatrix}$ , ${}_{s}^{c}R_{16} = \begin{bmatrix} 0 & 0 & +1 \\ -1 & 0 & 0 \\ 0 & -1 & 0 \end{bmatrix}$                                                                                            |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  | $\label{eq:R17} \begin{aligned} \varepsilon R_{17} = \begin{bmatrix} 0 & +1 & 0 \\ 0 & 0 & +1 \\ +1 & 0 & 0 \end{bmatrix}, \end{aligned} \begin{aligned} \varepsilon R_{18} = \begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & -1 \\ +1 & 0 & 0 \end{bmatrix}, \end{aligned} \begin{aligned} \varepsilon R_{19} = \begin{bmatrix} 0 & 0 & -1 \\ 0 & +1 & 0 \\ +1 & 0 & 0 \end{bmatrix}, \end{aligned} \begin{aligned} \varepsilon R_{20} = \begin{bmatrix} 0 & 0 & +1 \\ 0 & -1 & $ |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  |                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |  |
|  |  |  | $\label{eq:R21} \begin{aligned} \ _{_{0}}^{^{c}}R_{21}=\begin{bmatrix} 0 & +1 & 0 \\ 0 & 0 & -1 \\ -1 & 0 & 0 \end{bmatrix}, \ _{s}\ ^{c}R_{22}=\begin{bmatrix} 0 & -1 & 0 \\ 0 & 0 & +1 \\ -1 & 0 & 0 \end{bmatrix}, \ _{_{0}}^{^{c}}R_{23}=\begin{bmatrix} 0 & 0 & +1 \\ 0 & +1 & 0 \\ -1 & 0 & 0 \end{bmatrix}, \ _{_{0}}^{^{c}}R_{24}=\begin{bmatrix} 0 & 0 & -1 \\ 0 & -1 & 0 \\ -1 & 0 & 0 \$                                                                      |  |

![](_page_2_Figure_21.jpeg)

1) Special Robot Orientation Method (SROM): If the robot can get to the full workspace, it is going to take the special rotation matrix  ${}^e_b R$  to separate the vector  $F_b = [f_{bx}, f_{by}, f_{bz}]^T$ to find an estimation. The method takes the 24 robot orientations  ${}^e_b R_1 \sim^e_b R_{24}$  as shown in Fig. 3 and get their corresponding force measurements  ${}^sF_1 \sim {}^sF_{24}$ . By (12), 24 equations got from every orientation and its corresponding force are brief in

$$
{}^{s}F_{i} = {}^{s}_{e}R \cdot {}^{e}_{b}R_{i} \cdot F_{b} + {}^{s}F_{0} \quad i = 1, 2, ..., 24 \quad (14)
$$

The estimation of  $f_{bx}$  that is the first variable of the vector  $F_b$  is taken as an example to show our method detail, which could be applied to the second and third. It takes  $i = 1, 2,$ 3, 4 and  $i = 5, 6, 7, 8$  into (14) to construct two following equations

$$
{}^{s}F_{1} + {}^{s}F_{2} + {}^{s}F_{3} + {}^{s}F_{4}
$$
\n
$$
= {}^{s}_{e}R({}^{e}_{b}R_{1} + {}^{e}_{b}R_{2} + {}^{e}_{b}R_{3} + {}^{e}_{b}R_{4})F_{b} + 4 \cdot {}^{s}F_{0}
$$
\n
$$
= {}^{s}_{e}R \cdot \begin{bmatrix} 4 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} F_{b} + 4 \cdot {}^{s}F_{0}
$$
\n
$$
= 4 \cdot r_{1} \cdot f_{bx} + 4 \cdot {}^{s}F_{0}
$$
\n
$$
{}^{s}F_{5} + {}^{s}F_{6} + {}^{s}F_{7} + {}^{s}F_{8}
$$
\n
$$
= {}^{s}_{e}R({}^{e}_{b}R_{5} + {}^{e}_{b}R_{6} + {}^{e}_{b}R_{7} + {}^{e}_{b}R_{8})F_{b} + 4 \cdot {}^{s}F_{0}
$$
\n
$$
= {}^{s}_{e}R \cdot \begin{bmatrix} -4 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} F_{b} + 4 \cdot {}^{s}F_{0}
$$
\n
$$
= -4 \cdot r_{1} \cdot f_{bx} + 4 \cdot {}^{s}F_{0}
$$
\n(16)

where  $r_1$  is the first column of the  ${}_{\rho}^{s}R$ .  $\frac{s}{e}R$ .

While  $(15)$  minus  $(16)$ , the definition of a new vector is

$$
p_{11} = \frac{1}{8}({}^{s}F_{1} + {}^{s}F_{2} + {}^{s}F_{3} + {}^{s}F_{4} - {}^{s}F_{5} - {}^{s}F_{6} - {}^{s}F_{7} - {}^{s}F_{8})
$$
  
=  $r_{1} \cdot f_{bx}$  (17)

Similarly, the method takes  $i = 9, 10, 11, 12, 13, 14, 15, 16$ and  $i = 17, 18, 19, 20, 21, 22, 23, 24$  into (14) to construct two equations as above

$$
p_{12} = \frac{1}{8} ({}^{s}F_{9} + {}^{s}F_{10} + {}^{s}F_{11} + {}^{s}F_{12} - {}^{s}F_{13}
$$
  
\n
$$
-{}^{s}F_{14} - {}^{s}F_{15} - {}^{s}F_{16})
$$
  
\n
$$
= r_{2} \cdot f_{bx}
$$
  
\n
$$
p_{13} = \frac{1}{8} ({}^{s}F_{17} + {}^{s}F_{18} + {}^{s}F_{19} + {}^{s}F_{20} - {}^{s}F_{21}
$$
  
\n
$$
-{}^{s}F_{22} - {}^{s}F_{23} - {}^{s}F_{24})
$$
  
\n
$$
= r_{3} \cdot f_{bx}
$$
  
\n(19)

where  $r_2$ ,  $r_3$  is the second, third column of  ${}_{\rho}^{s}R$ , respectively. Combining (17), (18) and (19) into a  $3 \times 3$  matrix

$$
f_{bx} \cdot [r_1, r_2, r_3] = f_{bx} \cdot {^s_e} R = [p_{11}, p_{12}, p_{13}] \tag{20}
$$

and then calculating the determinant of the matrix, the solution is obtained

$$
\hat{f}_{bx} = (det([p_{11}, p_{12}, p_{13}]))^{\frac{1}{3}} \in \mathbb{R}
$$
 (21)

where  $det({}^{s}_{e}R) = 1$ .

The second and third variable can be calculated as above

$$
\hat{f}_{by} = (det([p_{21}, p_{22}, p_{23}]))^{\frac{1}{3}} \in \mathbb{R}
$$
 (22)

$$
\hat{f}_{bz} = (det([p_{31}, p_{32}, p_{33}]))^{\frac{1}{3}} \in \mathbb{R}
$$
 (23)

where,

$$
p_{21} = \frac{1}{8} ({}^{s}F_{9} + {}^{s}F_{13} + {}^{s}F_{17} + {}^{s}F_{21}
$$
  
\n
$$
-{}^{s}F_{10} - {}^{s}F_{14} - {}^{s}F_{18} - {}^{s}F_{22})
$$
  
\n
$$
p_{22} = \frac{1}{8} ({}^{s}F_{1} + {}^{s}F_{5} + {}^{s}F_{19} + {}^{s}F_{23}
$$
  
\n
$$
-{}^{s}F_{2} - {}^{s}F_{6} - {}^{s}F_{20} - {}^{s}F_{24})
$$
  
\n
$$
p_{23} = \frac{1}{8} ({}^{s}F_{3} + {}^{s}F_{7} + {}^{s}F_{11} + {}^{s}F_{15}
$$
  
\n
$$
-{}^{s}F_{4} - {}^{s}F_{8} - {}^{s}F_{12} - {}^{s}F_{16})
$$
  
\n
$$
p_{31} = \frac{1}{8} ({}^{s}F_{11} + {}^{s}F_{16} + {}^{s}F_{20} + {}^{s}F_{23}
$$
  
\n
$$
-{}^{s}F_{12} - {}^{s}F_{15} - {}^{s}F_{19} - {}^{s}F_{24})
$$
  
\n
$$
p_{32} = \frac{1}{8} ({}^{s}F_{4} + {}^{s}F_{7} + {}^{s}F_{17} + {}^{s}F_{22}
$$
  
\n
$$
-{}^{s}F_{3} - {}^{s}F_{8} - {}^{s}F_{18} - {}^{s}F_{21})
$$
  
\n
$$
p_{33} = \frac{1}{8} ({}^{s}F_{1} + {}^{s}F_{6} + {}^{s}F_{10} + {}^{s}F_{13}
$$
  
\n
$$
-{}^{s}F_{2} - {}^{s}F_{5} - {}^{s}F_{9} - {}^{s}F_{14})
$$

2) Limited Robot Orientation Method (LROM): When in case that the robot can't reach the special orientations due to the hardware limit, the algorithm above doesn't work. To estimate the parameters of the gravitational forces in the robot base frame  $F_b$ , the least-squares method ignoring the orthogonality of  ${}_{e}^{s}R^{T}$  is taken. Pre-multiplied a  ${}_{e}^{s}R^{T}$ , the (12) is rewritten as

$$
{}_{e}^{s}R^{T} \cdot {}^{s}F - {}_{b}^{e}R \cdot F_{b} - {}_{e}^{s}R^{T} \cdot {}^{s}F_{0} = 0 \tag{24}
$$

Grouping the unknown parameters into the vector  $x \in \mathbb{R}^{15}$ 

$$
x = [r_1^T; r_2^T; r_3^T; F_b; ^s_e R^T \cdot ^s F_0]^T
$$
 (25)

where,

$$
{}_{e}^{s}R^{T} = [r_{1}; r_{2}; r_{3}] \text{ and } ||_{e}^{s}R^{T}||_{F}^{2} = 3
$$

a least square program is modeled as

$$
\min \ J = \|Ax\|_F^2 \quad \text{s.t. } \|Bx\|_F^2 = 1 \tag{26}
$$

where,  $n$  is the number of the measurements;

$$
B = \begin{bmatrix} \frac{\sqrt{3}}{3}I_9 & 0_{9\times6} \\ 0_{6\times9} & 0_{6\times6} \end{bmatrix}
$$
  
$$
A = \begin{bmatrix} \vdots & \vdots & \vdots \\ 0_{1\times3} & {}^{6}F_1^T & 0_{1\times3} \\ 0_{1\times3} & 0_{1\times3} & {}^{5}F_1^T \end{bmatrix}, \quad -{}^{e}_{b}R_i, \quad -I_{3\times3}
$$
  
$$
\vdots \quad \vdots \quad \vdots \quad \vdots \quad \vdots \quad \vdots
$$

Assuming  $x = \begin{bmatrix} x_9 \\ x_9 \end{bmatrix}$ *x*6  $\int$ , a new variable  $y =$  $\frac{\sqrt{3}}{3}I_9 \cdot x_9 = \sum_r x_9$ is defined with

$$
x_9 = \sum_{r}^{-1} \cdot y \tag{27}
$$

Substituting  $(27)$  into  $(26)$  to get

$$
Ax = [A_9, A_6] \begin{bmatrix} x_9 \\ x_6 \end{bmatrix} = A_9 x_9 + A_6 x_6 = A_9 \sum_r^{-1} \cdot y + A_6 x_6
$$
\n(28)

the problem  $(26)$  is equivalent to

$$
\min \ J = \|A_9 \sum_{r}^{-1} \cdot y + A_6 x_6\|_F^2 \quad \text{s.t. } \|y\|_F^2 = 1 \quad (29)
$$

While the solution of the optimal problem

$$
\underset{x_6}{\arg\min} \, \|A_9 \sum_{r}^{-1} \cdot y + A_6 x_6\|_F^2 \tag{30}
$$

is 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 2733 and 273

$$
x_6 = -(A_6^T A_6)^{-1} A_6^T A_9 \sum_r^{-1} \cdot y \tag{31}
$$

The minimization problem  $(29)$  is transformed into a general least-squares optimization

$$
\min J = ||(I - A_6(A_6^T A_6)^{-1} A_6^T) A_9 \sum_r^{-1} \cdot y||_F^2 = ||Hy||_F^2
$$
  
s.t.  $||y||_F^2 = 1$  (32)

Since the optimal solution of  $(32)$  is

$$
y^* = v \tag{33}
$$

where  $\nu$  is the eigenvector corresponding to the minimum eigenvalue of the  $H^T H$ , the solution  $x^*$  of (26) could be got from  $(27)$  and  $(31)$ . Based on the installation types in Fig. 1, the estimation of gravitational forces in robot base frame is

$$
\hat{F}_b
$$
\n
$$
= \begin{cases}\n-sign(x_{12}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in [100^\circ, 260^\circ] \\
sign(x_{12}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in (0, 80^\circ] \cup [280^\circ, 360^\circ] \\
-sign(x_{10}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in (80^\circ, 100^\circ), \text{x axis } \downarrow \\
-sign(x_{11}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in (80^\circ, 100^\circ), \text{else} \\
sign(x_{10}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in (260^\circ, 280^\circ), \text{x axis } \downarrow \\
sign(x_{11}^*) \cdot [x_{10}^*; x_{11}^*; x_{12}^*] & \text{if } \alpha \in (260^\circ, 280^\circ), \text{else}\n\end{cases}
$$
\n(34)

where  $\alpha$  is a angle in Fig.1;  $\downarrow$  represents the axis is downward along the direction of gravity.

Authorized licensed use limited to: Johns Hopkins University. Downloaded on July 26,2025 at 18:16:12 UTC from IEEE Xplore. Restrictions apply.

## **B.** Solution to the Force Compensation Problem

As the gravitational forces in the robot base frame has been estimated in Section III-A, the identical problem (12) can be solved as the Problem  $3$  in [20]. It changes into

$$
{}^{s}F = {}^{s}_{e}R \cdot {}^{e}_{b}R \cdot \hat{F}_{b} + {}^{s}F_{0}
$$
 (35)

Given the *n* measurements  ${}^sF_i$  with corresponding rotation matrix  ${}^e_b R_i$ , the cost function is

$$
J = \sum_{i=1}^{n} \| {}^{s}F_{i} - {}^{s}_{e}R \cdot {}^{e}_{b}R_{i} \cdot \hat{F}_{b} - {}^{s}F_{0} \|_{F}^{2}
$$
 (36)

It can be reduced to a minimization problem in  $[21]$  by considering the definition

$$
{}^{s}F_{0} = {}^{s}\bar{F} - {}^{s}_{e}R \cdot {}^{e}_{b}\bar{R} \cdot \hat{F}_{b}
$$
 (37)

where

$$
{}^{s}\bar{F} = \frac{1}{n}\sum_{i=1}^{n} {}^{s}F_{i} , {}^{e}_{b}\bar{R} = \frac{1}{n}\sum_{i=1}^{n} {}^{e}_{b}R_{i}
$$

Substituting  $(37)$  into  $(36)$ , the minimization problem is solved by minimizing

$$
J_1 = \sum_{i=1}^n \|({}^s F_i - {}^s \bar{F}) - {}^s e R({}^e_b R_i - {}^e b \bar{R}) \hat{F}_b\|_F^2 \tag{38}
$$

The solution of  ${}^sF_0$  in (37) is

$$
{}^{s}\hat{F}_{0} = {}^{s}\bar{F} - {}^{s}_{e}\hat{R} \cdot {}^{e}_{b}\bar{R} \cdot \hat{F}_{b}
$$
 (39)

where  $\frac{s}{e}\hat{R}$  is the solution of (38), which is equivalent to the maximization problem

$$
s_e^s \hat{R} = \arg \max \ tr \left[ s_R \cdot D^T \right] \quad s.t. \ s_e^s R \in \mathbb{SO}(3) \tag{40}
$$

where

$$
D^{T} = \sum_{i=1}^{n} (\frac{e}{b} R_{i} - \frac{e}{b} \bar{R}) \hat{F}_{b}({}^{s} F_{i} - {}^{s} \bar{F})^{T}
$$

The maximization solution of  $(40)$  will be given referencing <sup>314</sup> the **Theorem 1** in [20]. A singular value decomposition (SVD) of  $D$  in (40) is considerable and given by

$$
D = U \Sigma V^T \tag{41}
$$

where  $UU^T = I$ ,  $VV^T = I$ ,  $\Sigma = diag_{i=1,...,m} \{ \lambda_i I_{n_i \times n_i} \}$  and  $\lambda_1 > ... > \lambda_m \ge 0$  are the distinct singular values of *D*, with  $\sum_{m=1}^{m} n_m = 3$  Based the SVD shows the solution of (40) is  $\sum_{i=1}^{m} n_i = 3$ . Based the SVD above, the solution of (40) is

$$
{}_{e}^{s}\hat{R} = \begin{cases} U \cdot diag\left\{I_{2\times2}, detUdetV\right\} V^{T} & \text{if } rank D \ge 1\\ no \text{ solution} & \text{if } rank D = 0 \end{cases}
$$
(42)

When  $rankD = 1$ ,  $det D < 0$ , the minimum singular value of  $D$  is repeated and the solution is non-unique. But the solution is selected as other cases with no discussion on it.

## C. Solution of the Torque Identification Model

The closed-form solution of the torque identification model in  $(13)$  can be got by the least-squares method  $(LSM)$ . To identify the parameters,  $(13)$  could be rewritten as

$$
{}^{s}T = -({}^{s}F - {}^{s}F_{0})^{\wedge} \cdot {}^{s}_{g}P + {}^{s}T_{0}
$$
 (43)

Given *n* force measurements  ${}^{s}F_i$  ( $i = 1, ..., n$ ) with corresponding torque measurements  ${}^{s}T_i$  ( $i = 1, ..., n$ ), the parameters are the solution that minimizes the cost function

$$
J = \|Cy - b\|_F^2
$$
 (44)

where  $y = \begin{bmatrix} s & p \\ s & s \end{bmatrix}$  $3344$ 

$$
C = \begin{bmatrix} \vdots & \vdots & \vdots \\ -({}^{s}F_{i} - {}^{s}F_{0})^{\wedge}, & I_{3\times 3} & \vdots \\ \vdots & \vdots & \vdots \end{bmatrix}_{3n\times 6} b = \begin{bmatrix} \vdots \\ {}^{s}T_{i} \\ \vdots \end{bmatrix}_{3n\times 1}
$$

The minimizing solution is

$$
y^* = (C^T C)^{-1} C^T b \tag{45}
$$

## D. Tool's Gravity and Robot Installation Bias

Commonly, there is an angle bias between the robot base frame and the gravity acceleration frame because of the mechanical installation error. It can assume that the rotation matrix, by which the tool gravity can be transformed into the robot base frame, is defined in Tait-Bryan angles as

$$
\overset{b}{g}R = R_Z(0)R_Y(\beta)R_X(\alpha)
$$
\n
$$
= \begin{bmatrix} \cos\beta & \sin\alpha\sin\beta & \cos\alpha\sin\beta\\ 0 & \cos\alpha & -\sin\alpha\\ -\sin\beta & \sin\alpha\cos\beta & \cos\alpha\cos\beta \end{bmatrix}
$$
(46)

Substituting  $(46)$  into  $(9)$ , a new form of  $(9)$  is

$$
F_b = -mg \cdot [cos\alpha \cdot sin\beta, -sin\alpha, cos\alpha \cdot cos\beta]^T \quad (47)
$$

Once the parameter  $\hat{F}_b = [\hat{f}_{bx}, \hat{f}_{by}, \hat{f}_{bz}]^T$  has been estimated in Section III-A, the parameters are estimated as

$$
\beta = \arctan 2(\hat{f}_{bx}, \hat{f}_{bz})
$$
  
\n
$$
\alpha = \arctan 2(-\hat{f}_{by} \cdot \cos \beta, \hat{f}_{bz})
$$
  
\n
$$
mg = \|\hat{F}_b\|_F^2
$$
\n(48)

Finally, the approach is summarized in Algorithm 1.

#### IV. EXPERIMENTAL INVESTIGATION

In this section, a numerical example is firstly taken to demonstrate our rotation calibration method. It then gives a brief overview of our experimental system, estimates the biases and eliminates the tool's gravity component, compares our method with other approaches, and shows the errors in different end-effector setups.

For evaluating our approach, the following performance indexes are defined:

- 1) **estimating error**  $\epsilon_2$ : the 2-norm of the vector or matrix of the identified parameters.
- 2) **compensation error**  $\epsilon_{max}$ : the maximum of a sequence constructed with the 2-norm of the difference between the compensated and actual force/torque.

| <b>Algorithm 1 Proposed Calibration Approach</b>                                             |  |  |  |  |  |  |  |  |
|----------------------------------------------------------------------------------------------|--|--|--|--|--|--|--|--|
| <b>Ensure:</b> the Gravitational forces in robot base frame $F_b$                            |  |  |  |  |  |  |  |  |
| <b>Method 1</b> $F_b$ calibration, <b>SROM</b> in Section (III-A.1)                          |  |  |  |  |  |  |  |  |
| <b>Require:</b> Special $\{^s F_i\}$ and corresponding $\{^e_k R_i\}$                        |  |  |  |  |  |  |  |  |
| in $(14)$ , i = 1,, 24                                                                       |  |  |  |  |  |  |  |  |
| Estimate $F_b$ by (21), (22) and (23)                                                        |  |  |  |  |  |  |  |  |
| return $\hat{F}_h$                                                                           |  |  |  |  |  |  |  |  |
| <b>Method 2</b> $F_b$ calibration, <b>LROM</b> in Section (III-A.2)                          |  |  |  |  |  |  |  |  |
| <b>Require:</b> Any $\{^sF_i\}$ and corresponding $\{^e_kR_i\}$ , i =                        |  |  |  |  |  |  |  |  |
| $1,\ldots,n$                                                                                 |  |  |  |  |  |  |  |  |
| Estimate $F_b$ by (34)                                                                       |  |  |  |  |  |  |  |  |
| return $F_b$                                                                                 |  |  |  |  |  |  |  |  |
| <b>Ensure:</b> Rotation matrix ${}_{e}^{s}R$ and Force Bias ${}^{s}F_{0}$                    |  |  |  |  |  |  |  |  |
| <b>Require:</b> Any $\{^sF_i\}$ and corresponding $\{^e_kR_i\}$ , i =                        |  |  |  |  |  |  |  |  |
| $1, \ldots, n$                                                                               |  |  |  |  |  |  |  |  |
| Estimate ${}_{g}^{s}R$ by (42), ${}^{s}F_{0}$ by (39)                                        |  |  |  |  |  |  |  |  |
| return ${}_{\rho}^{s}R, {}^{s}F_{0}$                                                         |  |  |  |  |  |  |  |  |
| <b>Ensure:</b> Rotation matrix ${}_{g}^{b}R$ , Bias ${}^{s}T_0$ , Gravity <i>mg</i> and it's |  |  |  |  |  |  |  |  |
| center ${}_{g}^{s}P$                                                                         |  |  |  |  |  |  |  |  |
| <b>Require:</b> $\hat{F}_b$ , $^sF_0$ , Any $\{^sF_i\}$ and corresponding $\{^sT_i\}$ ,      |  |  |  |  |  |  |  |  |
| $i = 1,,n$                                                                                   |  |  |  |  |  |  |  |  |
| Estimate ${}^{s}T_{0}$ , ${}_{g}^{s}P$ by (44), (45)                                         |  |  |  |  |  |  |  |  |
| Estimate ${}_{g}^{b}R$ , mg by (46), (48)                                                    |  |  |  |  |  |  |  |  |
| <b>return</b> $_{g}^{b} \check{R}$ , $^{s} \hat{T}_{0}$ , $\hat{mg}$ , $_{g}^{s} \hat{P}$    |  |  |  |  |  |  |  |  |
| Worked: be used in the compensation model $(10)$ , $(11)$                                    |  |  |  |  |  |  |  |  |
|                                                                                              |  |  |  |  |  |  |  |  |

# A. Numerical Experiment

In the numerical example here, assuming no external contact  ${}^{s}F_{contact} = 0$ ,  ${}^{s}T_{contact} = 0$  and other pre-defined system parameters, it takes 24 special robot rotation matrices in Fig. 3 into the compensation model  $(10)$ ,  $(11)$  to generate the corresponding force/torque measurements for identification. As for calculating the compensation error, we generate <sup>375</sup> 200 random robot orientations and corresponding force/torque measurements with an external contact  ${}^sF_{contact} = 1N$ ,  ${}^sT_{contact} = 0.5N·m$  and other system parameters. What we use to make the calibration and calculate the error are numerical force/torque and manual robot rotation matrix.

To verify our method, with the six parameter setups in Table I, we generates manually six different numerical data as above and calculate the error of every setup. It defines  $R_R^s = R_Z(\alpha)R_Y(\beta)R_X(\gamma)$  in Tait-Bryan angles. In Fig. 4, the errors are less than  $10^{-12}$ , indicating that our method can estimate the parameters correctly and eliminate the gravity.

## <sup>386</sup> B. The Evaluation of F/T Sensor Mounted-On Robot

The F/T sensor wrist-mounted robot system shown in Fig. 5 is used to evaluate the performances of identification and compensation. The robot is ER3B-C20 from the EFFORT company. The force and torque data is sampled in  $7 kHz$ frequency based on the 6-axis F/T sensor of the ATI Mini40 for type SI-40-2.

In the experiment, the force/torque of the sensor, and the robot forward kinematics are sampled meanwhile. Under the condition that no external force acting on the end-effector,

![](_page_5_Figure_8.jpeg)

Fig. 4. Error in six parameter setups.  $\epsilon_2$ : sF0 (force's bias), sT0 (torque's bias),  $s$ P $g$  (tool's center),  $s$ Re (rotation for robot-sensor),  ${\it Fb}$  (tool's gravity n robot base);

TABLE I SEVERAL EXPERIMENTS SETUP OF PARAMETERS FOR VALIDATION

|   |          | ${}_{e}^{s}R$ (° |    | $F_b(N)$     | ${}_{\rho}^{s}P(m)$ | ${}^sF_0(N)$ | ${}^{s}T_{0}(N\cdot m)$ |  |
|---|----------|------------------|----|--------------|---------------------|--------------|-------------------------|--|
|   | α        |                  |    |              |                     |              |                         |  |
|   | 0        |                  | 3  | $[0;0;-10]$  | [0;0;2]             | [0;0;0]      | [0;0;0]                 |  |
| 2 | $\theta$ |                  | 3  | [0;0;10]     | [0;0;2]             | [0;0;0]      | [87;45;6]               |  |
| 3 | 0        |                  | 3  | $[0;0,-10]$  | [0,0,2]             | [63, 5; 21]  | [87, 45, 6]             |  |
| 4 | $\Omega$ |                  | 3  | $[0;0,-10]$  | [0.1; 0.1; 1]       | [63, 5; 21]  | [87, 45, 6]             |  |
| 5 | 45       |                  | 3  | $[1;3;-40]$  | [0.1;0.1;1]         | [63;5;21]    | [87;45;6]               |  |
| 6 | 70       | 15               | 13 | $[1;3; -40]$ | [0.1;0.1;1]         | [63;5;21]    | [87;45;6]               |  |

![](_page_5_Figure_12.jpeg)

Fig. 5. Experimental setup of the F/T sensor mounted-on robot.

150 measurements with the rotation set D shown in Fig. 6 are sampled to calculate the compensation error. The measurement in every orientation is actually the mean of a sequence sample.

Firstly, a physical dataset is sampled with the rotation sets A and E in Fig.  $6$  to validate the gravity compensation by our SROM method. Table II shows the identified parameters and that our method work. As the real tool's mass includes the tool and external placement of the sensor (about  $20.4$   $g$  here), the identified gravity and its center will have a small inaccuracy compared with the designed mechanism in the experiment.

From Fig. 7 and Fig. 8, we observe that the measured values of the sensor have changed during the experiment when the pose of the end-effector changes in a wide range of motion. More importantly, the gravity of the end-effector can be compensated obviously to estimate a more pure external contact <sup>410</sup> force. As a result, the error boundaries of F/T sensor in threeaxis force components measurement are  $\pm 1.259N$ ,  $\pm 1.122N$ ,

![](_page_6_Figure_2.jpeg)

Fig. 6. Robot rotations for sample. A: the 24 special rotations. B: the 12 random rotations. C: another 12 random rotations different with B. D: the 150 random rotations. E: the 24 random rotations in a limited workspace. F: the 24 random rotations on the opposite side of E. G: the 48 random rotations inside E. H: the half of E.

![](_page_6_Figure_4.jpeg)

Fig. 7. Compensation error on Force. The left is original data that sampled in sensor, and the right is force that after gravity compensation.

|                               | ${}_{e}^{s}R$                        |               | ${}^sF_0(N)$                  |
|-------------------------------|--------------------------------------|---------------|-------------------------------|
| $-0.4433$                     | $-0.8963$                            | $-0.0017$     | 2.992                         |
| 0.8963                        | $-0.4433$                            | $-0.0056$     | $-5.916$                      |
| 0.0043                        | $-0.0040$                            | 0.9999        | $-1.021$                      |
| mass $(kg)$                   | real $m$ (kg)                        | error $(\% )$ | $F_b(N)$                      |
| 2.044                         | 2.013                                | 1.5           | 0.072<br>0.092<br>$-20.047$   |
| ${}_{\alpha}^{s}P(m)$         | real ${}_{\sigma}^{s}P$ ( <i>m</i> ) | error(m)      | ${}^sT_0(N\cdot m)$           |
| 0.0000<br>$-0.0004$<br>0.0760 | 0.0<br>-0.0<br>0.08                  | 0.004         | $-0.071$<br>$-0.046$<br>0.017 |

TABLE II THE IDENTIFIED PARAMETERS

 $\pm 1.26N$ , while in torque are  $\pm 0.078N \cdot m$ ,  $\pm 0.084N \cdot m$ ,  $\pm 0.079N \cdot m$ .

Furthermore, to test how our method works effectively in the robot workspace, six different datasets by six rotation sets shown in Fig. 6 are sampled to estimate the parameters and calculate the error. A basic rule of data sample for calibration is obtained from Table III as following:

1) Case 8 has the best performance on compensation error, which even is better than Case 10 using all samples. It estimates  $F_b$  using 24 special rotations of SROM, while other parameters using 24 random local orientation data.

![](_page_6_Figure_11.jpeg)

Fig. 8. Compensation error on torque. The left is original data that sampled in sensor, and the right is torque that after gravity compensation.

![](_page_6_Figure_13.jpeg)

Fig. 9. Compensation error for three methods when yaw angle  $\alpha$  in Tait-Bryan  ${}_{\beta}^{s}$  =  $R_Z(\alpha)R_Y(0)R_X(0)$  varies. In Wang *et al.* 2018 [16] method, so lower than the set of 2010 [17] only estimates surface when the set of 2010 [17] only estimates surface when the set of 2010 [17] only es  $\frac{s}{\xi R}$  is known. Yang *et al.* 2019 [17] only estimates  $\alpha$ . Ours could identify eR.

- 2) If the robot can't reach 24 special rotations, it suggests that taking Case 6 will be better while Case 7 needs a double times sample but has a little improvement.
- 3) The error of Case 3, 4, 5, 6, 7 suggests that increasing the data number can get better accuracy, and a wider data feature could make an improvement obviously even though sampled in a local orientation workspace.
- 4) Comparing Case 6 and 11, we find that it works better and has obvious improvement in the inside of the data sample space than in the full orientation space.

In Fig. 9, it compares our method with Wang's  $[16]$  and Yang's [17] approaches. Six yaw angles  $\alpha$  in Tait-Bryan  $^s_eR =$  $R_Z(\alpha)R_Y(0)R_X(0)$  are set in the physical system. In every angle setup, It uses rotation set  $E$  to sample for estimation while D for calculating error. Wang's method knows the rotation between the sensor and robot roughly and works worst. Yang's method only estimates its yaw angle  $\alpha$ , while ours estimates any  ${}_{e}^{s}R$ . The result shows that Yang's and our method work close, but ours better on the force channel  $F_z$ .

## TABLE III MAX ERROR OF THE DIFFERENT SAMPLES AND METHODS. THE BOLD NUMBER IS THE MINIMUM OF CASE 1 TO 9. IN CASE 8, IT ESTIMATES $F_b$ BY SROM AND THEN OTHER PARAMETERS USING ANY ROTATION SAMPLES. THE TOOL'S GRAVITY AND CENTER IS 19.75 N, [0;0;8] cm

| Case | Method      | Data    | <b>Test Data</b> | $F_x(N)$ | $F_{\rm v}(N)$ | $F_{\tau}(N)$ | F(N)  | $T_r(N\cdot m)$ | $T_{\rm v}(N\cdot m)$ | $T_{\rm z}(N\cdot m)$ | $T(N \cdot m)$ | $mg(\%)$ | center $(\% )$ |
|------|-------------|---------|------------------|----------|----------------|---------------|-------|-----------------|-----------------------|-----------------------|----------------|----------|----------------|
|      | <b>SROM</b> |         | D                | . 547    | .130           | .547          | .780  | 0.084           | 0.104                 | 0.084                 | 0.108          | 1.5      | 5.1            |
|      | <b>LROM</b> | А       | D                | 1.591    | 1.235          | .591          | .854  | 0.084           | 0.105                 | 0.084                 | 0.109          | 1.6      | 5.1            |
|      | <b>LROM</b> | B       | D                | 2.531    | 5.041          | 2.531         | 5.167 | 0.442           | 0.267                 | 0.442                 | 0.444          | 13.6     | 1.8            |
| 4    | <b>LROM</b> | B+C     | D                | 1.698    | .625           | .698          | .882  | 0.103           | 0.111                 | 0.103                 | 0.126          | 2.0      | 4.7            |
|      | <b>LROM</b> | H       | D                | 1.878    | 2.149          | 1.878         | 2.263 | 0.156           | 0.132                 | 0.156                 | 0.172          | 1.3      | 5.0            |
|      | <b>LROM</b> | E       | D                | 1.259    | .403           | .259          | .612  | 0.098           | 0.076                 | 0.098                 | 0.108          | 1.3      | 5.0            |
|      | <b>LROM</b> | $E + F$ | D                | 1.184    | .394           | 1.184         | .566  | 0.095           | 0.080                 | 0.095                 | 0.106          | 1.2      | 5.0            |
| 8    | <b>SROM</b> | A+E     | D                | 1.259    | 1.122          | .259          | 1.507 | 0.079           | 0.084                 | 0.079                 | 0.090          | 1.5      | 5.0            |
|      | <b>LROM</b> | A+E     | D                | .387     | .308           | .387          | .640  | 0.093           | 0.092                 | 0.093                 | 0.099          | 1.5      | 5.0            |
| 10   | <b>LROM</b> | All     | D                | .201     | .369           | .201          | .682  | 0.090           | 0.076                 | 0.090                 | 0.099          | 1.4      | 5.0            |
|      | <b>LROM</b> | E       | G                | .053     | 0.990          | 1.053         | .242  | 0.063           | 0.070                 | 0.063                 | 0.079          | 1.3      | 5.0            |

![](_page_7_Figure_4.jpeg)

Fig. 10. Different tool's setups. (a) The center is about  $[0;0;0.08]$  m; (b) The mass is about 1  $kg$ , and the center  $[0;0;x]$  m varies.

## C. The Error Analysis on Different Tool Setups

When the robot inevitably mounts different tools in different applications, there is a relation between the error and the endeffector. Different setups of end-effector shown in Fig. 10 is taken for calibration. By LROM, the robot rotation sets A and E are used to sample for estimation as rotation set D for error.

From Fig. 11, there is a linear relationship between the com-<sup>451</sup> pensation error of force/torque and the tool's mass. We find that the error has a strong linear correlation with the range transformed from the tool's mass, and the linear coefficients of force and torque are very close at 0.03. The tool's mass affects the estimation error of the tool's mass positively but has no correlation with the estimation error of its center.

When the tool's center increases, the accuracy of compensated torque, estimated mass, and estimated center will get worse, but it does not have much effect on the compensation error of the force. From Fig. 12, only the torque variation range of the sensor is affected by the tool's center. The compensation error of the torque is strongly linearly related to this variation range, and the coefficient is very close to the coefficient when the tool's mass varies (about 0.03).

In Fig. 9, our method has a little improvement in z-axis force  $F_z$ . Since it is more common that force control task stresses on the control of z-axis force  $F_z$ , an additional experiment is taken to address the gap improvement. The experiment takes our method (Case  $8$  in Table III) and Yang's method [6] to calculate the linear coefficient between the endeffector gravity and the compensation error in  $F<sub>z</sub>$  as Fig. 11. In Fig. 13, compared to the Yang's method, the compensation error in  $F<sub>z</sub>$  by the proposed method diminishes 1.0% of the end-effector gravity (i.e., 21.7% relative improvement).

![](_page_7_Figure_11.jpeg)

![](_page_7_Figure_12.jpeg)

![](_page_7_Figure_13.jpeg)

Fig. 12. Error analysis when the tool's center  $[0;0;x]$  m varies.

To probe into the nexus between the system and linear coefficient, two numerical samples are taken. With no external contact, different tool's gravity in  $[0, 40]$  *N* are assigned to change the sensor range of the force/torque, while its center just only changes the range on torque. The linear coefficients

![](_page_8_Figure_1.jpeg)

Fig. 13. The improvement of linear coefficient on  $F_z$  compensation error.

![](_page_8_Figure_3.jpeg)

Fig. 14. Linear coefficient analysis when the noise varies.  ${}^{s}F =$  $\overset{s}{e}R\cdot\overset{e}{b}R\cdot F_{b}+{}^{S}\digamma_{0}+n_{1},\ ^{S}\digamma=\overset{s}{g}\overset{\rightarrow}{P^{\wedge}}\cdot\overset{s}{e}R\cdot\overset{e}{b}R\cdot F_{b}+{}^{S}\digamma_{0}+n_{1},\ \overset{s}{e}R=$  $\tilde{R}_{Z}$ (45°) $R_{Y}$ (7°) $R_{X}$ (3°) in Tait-Bryan angles,  ${}^{S}_{S}P=$  [0.1;0.1;0.1] $m,$   ${}^{S}F_{0}=$ [63;5;21]N,  ${}^5T_0 = [87;45;6]N \cdot m$ ,  $F_b = \alpha$  [1;3;–40]N with  $\alpha \in [0,1]$ ensuring tool's gravity in [0,40]N. It generates 24 special  ${}^e_B R = R_Z(\alpha +$  $n_2$ ) $R_Y(\beta + n_2)R_X(\gamma + n_2)$  to get the data for estimation and 500 random  ${}_{b}^{e}$ R for calculating the error.  $n_1$  is the F/T random noise and  $n_2$  is the rotation random noise, when the axis are their amplitude.

![](_page_8_Figure_5.jpeg)

Fig. 15. The coupling between the Linear Coefficient and tool's center. The white noise on F/T sensor and robot rotation are 0.2 and 0.04 when other parameter set as Fig. 14.

on force/torque are calculated as the Fig. 11. Fig. 14 depicts how the linear coefficient varies with different noise in robot forward kinematics and F/T sensor. There is a same trend in both force and torque. If the noise on robot kinematics increases, the linear coefficient gets bigger. However, a bigger. noise on force/torque components makes a smaller coefficient.

In Fig. 15, it sets different tool's center and there are couplings between the tool's center and the linear coefficient on torque. For instance, when set the tool's center as  $[0;0;0.1]$   $m$ , the linear coefficient on the  $x$  or  $y$  axis is overt but on the  $\zeta$  axis is less impact. The phenomenon is same as that shown in the physical experiment Fig.  $11$ . It suggests that if the task needs to control the *z*-axis torque, set the tool's center on  $x$ ,  $y$  axis to 0 will be better. Also, the linear relations of every axis are stacked together with different tool's centers.

In summary, the linear coefficient of error is decided by the system. The compensation error has an approximately linear relation with the sensor range of force/torque, which range is caused by the end-effector (includes its gravity and center). The linear coefficient is constant and decided by the system noise (the inaccuracy of the robot kinematics and F/T sensor), while ours is about  $3\%$ .

## **V. CONCLUSION**

This work has demonstrated the rotation calibration between the robot and the six-axis F/T sensor. Based on the bias calibration, the end-effector gravity is compensated effectively on the measurement. In our robot deburring system, it is convenient with the F/T sensor's mechanical assemble and robust for different robot system setups. Then the measurements could be transformed from the sensor space into robot space for force control. The key is using our two ideas to find an analytical rotation solution when the F/T sensor is assembled arbitrarily. The proposed framework is potential in the surface finishing and human-robot interaction. Future work will concern on the uncertainty of the robot kinematics and the F/T sensor.

#### **REFERENCES**

- [1] S. Haddadin and E. Croft, "Erratum to: Physical human-robot interaction," in Springer Handbook of Robotics, 2nd ed. Cham, Switzerland: Springer, 2016, p. E1.
- [2] R. Johansson, K. Nilsson, and A. Robertsson, "Force control," in *Hand*book of Manufacturing Engineering and Technology, 1st ed. London, U.K.: Springer, 2015, pp. 1933-1965.
- [3] L. Villani and J. D. Schutter, "Force control," in *Springer Handbook of Robotics*, 2nd ed. Cham, Switzerland: Springer, 2016, pp. 195-220.
- [4] A. Salem and Y. Karayiannidis, "Robotic assembly of rounded parts with and without threads," IEEE Robot. Autom. Lett., vol. 5, no. 2, pp. 2467–2474, Apr. 2020.
- [5] Y. Zhou et al., "Global vision-based impedance control for robotic wall polishing," in *Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS)*, Macao, China, Nov. 2019, pp. 6022-6027.
- $[6]$  Y. H. Yin, Y. Xu, Z. H. Jiang, and O. R. Wang, "Tracking and understanding unknown surface with high speed by force sensing and control for robot," *IEEE Sensors J.*, vol. 12, no. 9, pp. 2910-2916, Sep. 2012.
- [7] G. Zeng and A. Hemami, "An overview of robot force control," *Robotica*, vol. 15, pp. 473-482, Sep. 1997.
- J. L. Nevins and D. E. Whitney, "The force vector assembler concept," in *On Theory and Practice of Robots and Manipulators*, 1st ed. Berlin, Germany: Springer, 1972, pp. 273-288.
- [9] D. Kubus and F. M. Wahl, "Scaling and eliminating non-contact forces and torques to improve bilateral teleoperation," in *Proc. IEEE/RSJ* Int. Conf. Intell. Robots Syst., St. Louis, MO, USA, Oct. 2009, pp. 5133-5139.
- [10] A. Winkler and J. Suchy, "Dynamic force/torque measurement using a 12DOF sensor," in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst., San Diego, CA, USA, Oct. 2007, pp. 1870-1875.
- [11] D. Kubus, T. Kroger, and F. M. Wahl, "Improving force control performance by computational elimination of non-contact forces/torques," in Proc. IEEE Int. Conf. Robot. Autom., Pasadena, CA, USA, May 2008, pp. 2617-2622.
- [12] F. Tian, C. Lv, Z. Li, and G. Liu, "Modeling and control of robotic automatic polishing for curved surfaces," CIRP J. Manuf. Sci. Technol., vol. 14, pp. 55–64, Aug. 2016.

- [13] L. Richter, R. Bruder, and A. Schweikard, "Hand-assisted positioning and contact pressure control for motion compensated robotized transcra-<sup>555</sup> nial magnetic stimulation," *Int. J. Comput. Assist. Radiol. Surg.*, vol. 13, pp. 845–852, Mar. 2012.
- [14] B. R. Shetty and M. H. Ang, "Active compliance control of a PUMA <sup>558</sup> 560 robot," in *Proc. IEEE Int. Conf. Robot. Autom.*, Minneapolis, MN, USA, Apr. 1996, pp. 3720-3725.
- [15] S. Vougioukas, "Bias estimation and gravity compensation for force-<sup>561</sup> torque sensors," in *Proc. WSEAS Symp. Math. Methods Comput. Techn.* <sup>562</sup> *Electr. Eng.*, Athens, Greece, 2001, pp. 82–85.
  - [16] N. Wang, J. Zhou, and X. Zhang, "Research on the estimation of sensor bias and parameters of load based on force-feedback," in *Proc. Int. Conf.* <sup>565</sup> *Intell. Robot. Appl.*, Newcastle, NSW, Australia, 2018, pp. 404–413.
  - <sup>566</sup> [17] X. Yang, F. Li, R. Gao, R. Song, and Y. Li, "Force perception of <sup>567</sup> industrial robot based on multi-parameter coupled model," in *Proc.* <sup>568</sup> *IEEE Int. Conf. Robot. Biomimetics (ROBIO)*, Dali, China, Dec. 2019, pp. 1676-1681.
  - [18] Z. Lin, W. Xin, J. Yang, Z. QingPei, and L. ZongJie, "Dynamic trajectory-tracking control method of robotic transcranial magnetic stimulation with end-effector gravity compensation based on force sensors," <sup>573</sup> *Ind. Robot.*, vol. 45, no. 6, pp. 722–731, 2018.
  - [19] Y. Gan, X. Dai, and D. Dong, "Robot calibration for cooperative process under typical installation," *J. Appl. Math.*, vol. 2014, pp. 1-12, Apr. 2014.
  - [20] D. Ruiter, H. J. Anton, and J. R. Forbes, "On the solution of Wahba's problem on SO (n)," *J. Astronaut. Sci.*, vol. 60, no. 1, pp. 1–31, 2013.
  - [21] S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, pp. 376-380, Apr. 1991.

![](_page_9_Picture_10.jpeg)

Yonggiang Yu received the B.S. degree in automation from the Harbin Institute of Technology, Harbin, China, in 2018. He is currently pursuing the M.S. degree in control science and engineering with the Harbin Institute of Technology, Shenzhen, China. His research interests include robotics, compliance control, and robot learning.

![](_page_9_Picture_12.jpeg)

**Ran Shi** received the M.E. degree in control science and engineering from Xi'an Jiaotong University, Xi'an, China, in 2013, and the Ph.D. degree in control science and engineering from the Harbin Institute of Technology, Shenzhen, China, in 2019. His research interests include control theory, motion control, and robotics.

![](_page_9_Picture_14.jpeg)

**Yunjiang Lou** (Senior Member, IEEE) received the B.S. and M.E. degrees in automation from the University of Science and Technology of China, Hefei, China, in 1997 and 2000, respectively, and the Ph.D. degree in electrical and electronic engineering from the Hong Kong University of Science and Technology, Hong Kong, in 2006. He is currently with the State Key Laboratory of Robotics and Systems, School of Mechatronics Engineering and Automation, Harbin Institute of Technology, Shenzhen, China. His research

interests include motion control, robot mobile manipulation, and industrial robots. robots.