# Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering

Yutao Feng<sup>1,2∗</sup> Xiang Feng<sup>1,2∗</sup> Yintong Shang<sup>1</sup> Ying Jiang<sup>3</sup> Chang Yu<sup>3</sup> Zeshun Zong<sup>3</sup> Tianjia Shao<sup>2</sup> Hongzhi Wu<sup>2</sup> Kun Zhou<sup>2</sup> Chenfanfu Jiang<sup>3</sup> Yin Yang<sup>1</sup> <sup>1</sup>University of Utah, USA  $\frac{2\pi}{2}$ Zhejiang University, China  $\frac{3}{2}$ UCLA, USA

![](_page_0_Picture_2.jpeg)

Figure 1. Versatile Motion Synthesis and Rendering. Gaussian Splashing (GSP) is a unified framework combining 3D Gaussian Splatting (3DGS) and position-based dynamics. By leveraging their coherent point-based representations, GSP delivers high-quality rendering for novel dynamic views involving interacting deformable bodies, rigid objects, and fluids. GSP enables a variety of compelling effects and new human-computer interaction modalities not available with existing NeRF/3DGS-based systems. The teaser figure showcases a cliff battered by waves (a), a deformable ficus plant (b), flooding garden (c) and piled and scattered rigid lego bulldozers in a box (d). The reconstructed Gaussians not only capture the nonlinear dynamics of fluids and solids but can also be rasterized to realistically render with both diffuse and specular shadings. GSP re-engineers several state-of-the-art techniques from neural surface reconstruction, specular-aware Gaussian shading, position-based surface tension, and AI inpainting to ensure the quality of both simulation and rendering with 3DGS.

## Abstract

*We demonstrate the feasibility of integrating physicsbased animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes reconstructed using 3DGS. Leveraging the coherence of the Gaussian Splatting and Position-Based Dynamics (PBD) in the underlying representation, we manage rendering, view synthesis, and the dynamics of solids and fluids in a cohesive manner. Similar to GaussianShader, we enhance each Gaussian kernel with an added normal, aligning the kernel's orientation with the surface normal to refine the PBD simulation. This approach effectively eliminates spiky noises that arise from rotational deformation in solids. It also allows us to integrate physically based rendering to augment the dynamic surface reflections on fluids. Consequently, our framework is capable of realistically reproducing surface highlights on dynamic fluids and facilitating interactions between scene objects and fluids from new views. For more information, please visit our project page at* <https://gaussiansplashing.github.io/>*.*

# <span id="page-0-0"></span>1. Introduction

Visualization and reconstruction of 3D scenes have been the core of 3D graphics and vision. Recent development of learning-based techniques such as the neural radiance fields (NeRFs) [\[44\]](#page-9-0) sheds new light on this topic. NeRF casts the reconstruction pipeline as a training procedure and delivers state-of-the-art results by encapsulating the color, texture, and geometry of the 3D scene into an implicit MLP net. Its superior convenience and efficacy inspire many follow-ups, e.g., with improved visual quality [\[38\]](#page-9-1), faster performance [\[19,](#page-8-0) [76\]](#page-10-0), and sparser inputs [\[25,](#page-8-1) [77\]](#page-10-1). NeRF is computationally expensive. Image synthesis with NeRF has to follow the path integral, which is less suitable for realtime or interactive applications unless dedicated compression or acceleration methods are employed e.g., with NGP encoding [\[50\]](#page-9-2). 3D Gaussian Splatting (3DGS) [\[29\]](#page-8-2) provides an elegant alternative. As the name suggests, 3DGS learns a collection of Gaussian kernels from the input images.Apart from NeRF, a novel view of the scene from an unseen camera pose is generated using rasterization with the tile-splatting technique. Therefore, fast rendering with 3DGS is feasible.

It is noted that Gaussian kernels not only serve as a

<sup>∗</sup> indicates equal contributions.

good rendering agent but also explicitly encode rich information of the scene geometry. This feature suggests 3DGS a good candidate for dynamic scenes [\[16,](#page-8-3) [67,](#page-10-2) [75\]](#page-10-3), animated avatars [\[46,](#page-9-3) [81\]](#page-10-4), or simulated physics [\[69\]](#page-10-5). We expand on this intuition, enhancing the current 3DGS ecosystem by injecting physics-based fluid and solid interactions into a 3DGS scene. This appears straightforward at first sight. Since 3DGS kernels are essentially a collection of ellipsoids, they can be used for the discretization of the fluid and solid dynamics just as position-based dynamics [\[42\]](#page-9-4), oriented particles [\[47\]](#page-9-5) or other particle-based simulation techniques. Unfortunately, a simple combination of those techniques does not yield the results expected. Large rotational deformation of the solid objects affects the splatting results with sharp and spiky noises. During fluid motion, fluid particles undergo substantial positional shifts, moving from the inside to the outside or vice versa. Fluids are both translucent and specular. The vanilla 3DGS simplifies the composition of the light field without well-defined appearance properties. This limitation makes fluid rendering cumbersome with 3DGS.

This paper presents a system namely Gaussian Splashing (GSP), a 3DGS-based framework that enables realistic interactions between solid objects and fluids in a physically meaningful way, and thus generates two-way coupled fluids-solids dynamics in novel views. GSP integrates Lagrangian fluid and 3DGS scenes through a unified framework of position-based dynamics (PBD) [\[42,](#page-9-4) [48\]](#page-9-6). We follow a recent contribution of GaussianShader [\[27\]](#page-8-4) to augment Gaussian kernels with additional environmental information so that specular shading can be dynamically synthesized along with the fluid's movement. For solid objects, GSP uses an anisotropy loss to cap the stretching ratio during 3DGS training and mitigate the rendering artifacts induced by rotational deformation. We approximate the normal of a fluid kernel based on the surface tension if it is near the fluid surface. For scattered fluid droplets, we resort to a depth volume rendered via the current camera pose to estimate the normal information [\[64\]](#page-10-6). GSP is versatile, due to the flexibility of PBD. It handles deformable bodies, rigid objects, and fluid dynamics in a unified way. While it is possible to incorporate more complicated constitutional models as in [\[17\]](#page-8-5) and [\[31\]](#page-9-7). We found that PBD-based simulation suffices in many situations. We further augment GSP with an image-space segmentation module to select objects of interest from the 3DGS scene. We exploit the latest generative AI to fill the missing pixels to enable interesting physicsbased scene editing.

In a nutshell, GSP leverages a unified, volumetric, particle-based representation for rendering, 3D reconstruction, view synthesis, and dynamic simulation. It contributes a novel 3D graphics/vision system that allows natural and realistic solid-fluid interactions in real-world 3DGS scenes. This is achieved by carefully engineering the pipeline to overcome the limitations of the vanilla 3DGS. GSP could enable a variety of intriguing effects and new humancomputer interaction modalities in a diverse range of applications. For instance, one can pour water to flood the scene, floating the objects within, or directly liquefy an object, just as in science fiction. Fig. [1](#page-0-0) showcases the dynamic interaction between a LEGO excavator and the splashing waves. There are 334,815 solid Gaussian kernels and 280,000 fluid Gaussian kernels. Through the two-way coupling dynamics, the excavator is animated to surf on the splashing waves.

# 2. related work

Dynamic neural radiance field Dynamic neural radiance fields generalize the original NeRF system to capture timevarying scenes e.g., by decomposing time-dependent neural fields into an inverse displacement field and canonical timeinvariant neural fields [\[52,](#page-9-8) [53,](#page-9-9) [63,](#page-10-7) [66\]](#page-10-8), or estimating the time-continuous 3D motion field [\[15,](#page-8-6) [18,](#page-8-7) [22,](#page-8-8) [33,](#page-9-10) [37,](#page-9-11) [56,](#page-9-12) [68\]](#page-10-9) with an added temporal dimension. Existing arts enable direct edits of NeRF reconstructions [\[3,](#page-8-9) [13,](#page-8-10) [26,](#page-8-11) [32\]](#page-9-13). In dynamic scenes, the rendering process needs to trace deformed sample points back to rest-shape space to correctly retrieve the color/texture information [\[54,](#page-9-14) [58,](#page-10-10) [72,](#page-10-11) [78\]](#page-10-12). They often extract a mesh/grid from the NeRF volume. It is also possible to integrate physical simulation with NeRF using meshless methods [\[17,](#page-8-5) [31\]](#page-9-7). Point NeRF [\[71\]](#page-10-13) and 3DGS [\[29\]](#page-8-2) offer a different perspective to scene representation explicitly using points/Gaussian kernels to encode the scene. The success of 3DGS has inspired many studies to transplant techniques for dynamic NeRF to 3DGS [\[36,](#page-9-15) [67,](#page-10-2) [73\]](#page-10-14). They incorporate learning-based deformation and editing techniques to reconstruct or generate dynamics of NeRF scenes. It is noteworthy that a recent work from Xie et al. [\[69\]](#page-10-5) integrates physical simulation with 3DGS, leveraging the unified proxy for both simulating and rendering.

Lagrangian fluid simulation Lagrangian fluid simulation tracks fluid motion using individual particles as they traverse the simulation domain. A seminal approach within this domain is smoothed particle hydrodynamics (SPH) [\[45\]](#page-9-16), which solves fluid dynamics equations by assessing the influence of neighboring particles. Despite its efficacy, SPH, particularly in its standard and weakly compressible forms (WCSPH) [\[6\]](#page-8-12), suffers from parameter sensitivity, e.g., kernel radius and time-step size for stiff equations. To relax the time-step restriction, predictivecorrective incompressible SPH (PCISPH) [\[59\]](#page-10-15) iteratively corrects pressure based on the density error. Similarly, position-based dynamics [\[48\]](#page-9-6) provides a robust method of solving a system of non-linear constraints using Gauss-Seidel iterations by updating particle positions directly, which can also be employed in fluid simulation [\[40\]](#page-9-17) with

improved stability. Furthermore, surface tension can also be generated [\[70\]](#page-10-16) using the position-based iterative solver by tracking surface particles and solving constraints on them to minimize the surface area.

Reflective object rendering Achieving precise rendering of reflective surfaces relies on accurately estimating scene illumination, such as environmental light, and material properties like bidirectional reflectance distribution function (BRDF). This task falls under the domain of inverse rendering [\[4,](#page-8-13) [51\]](#page-9-18). Some NeRF-related methodologies disentangle the visual appearance into lighting and material properties, which can jointly optimize environmental illumination, surface geometry, and material  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$  $[7-9, 60, 79, 80]$ . Other NeRF studies [\[28,](#page-8-16) [34,](#page-9-19) [35,](#page-9-20) [39\]](#page-9-21) aim to enhance the accuracy of the normal estimation in physically based rendering (PBR). Nevertheless, these efforts face challenges such as time-consuming training and slow rendering speed. On the contrary, 3DGS naturally offers a good normal estimation as the shortest axis of the Gaussian kernel [\[21,](#page-8-17) [27\]](#page-8-4). Following this idea, it is possible to achieve high-quality rendering of reflective objects and training efficiency simultaneously [\[27\]](#page-8-4).

Point-based rendering Point-based rendering has been an active topic in computer graphics since the 1980s [\[30\]](#page-9-22). The simplest method [\[20\]](#page-8-18) renders a set of points with a fixed size. It suffers from holes and relies on post-processing steps such as hole-filling [\[20,](#page-8-18) [55\]](#page-9-23) to correct the resulting rendering. An improvement is to use ellipsoids instead of volume-less points. This strategy is usually referred to as *point splatting* [\[82\]](#page-10-20). The ellipsoids are rendered with a Gaussian alpha-mask to eliminate visible artifacts between neighboring splats and then combined by a normalizing blend function [\[2,](#page-8-19) [82\]](#page-10-20). Point-based rendering well synergizes with Lagrangian fluid rendering, enabling the calculation of fluid thickness and depth through splatting. This approach [\[64\]](#page-10-6) achieves fluid rendering at an impressive realtime speed. Further extensions to splatting aim to automatically compute the shape and color of ellipsoids, for example, auto splats [\[12\]](#page-8-20). With the development of deep learning in recent years, learning-based approaches improve the image quality of splatting [\[11,](#page-8-21) [74\]](#page-10-21). 3DGS [\[29\]](#page-8-2) has introduced this technique into 3D reconstruction, enabling high-quality real-time novel view synthesis for reconstructed scenes. A natural idea is to combine 3DGS with fluid rendering, enabling interaction between reconstructed scenes and dynamic fluids.

# 3. Background

To make our paper self-contained, we start with a brief review of PBD and 3DGS on which our pipeline is built. More detailed discussions are available in the supplementary material and relevant literature e.g., [\[27,](#page-8-4) [29,](#page-8-2) [42,](#page-9-4) [48\]](#page-9-6).

#### 3.1. Position-Based Dynamics

PBD/XPBD treats a dynamic system as a set of  $N$  vertices and M constraints. This perspective offers an easy and efficient simulation modality, converting the variational optimization to the so-called constraint projections. Specifically, XBPD considers the total system potential  $U$  as a quadratic form of all the constraints  $C(x) = [C_1(x), C_2(x), ..., C_M(x)]^{\top}$  such that  $U =$  $\frac{1}{2}C^\top(x)\alpha^{-1}C(x)$ . Here,  $x$  represents the position of vertices and  $\alpha$  is the compliance matrix, i.e., the inverse of the constraint stiffness. XBPD estimates an update of constraint force (i.e., the multiplier)  $\Delta\lambda$  by solving:

$$
\left[\Delta t^2 \nabla C(\mathbf{x}) \mathbf{M}^{-1} \nabla C^{\top}(\mathbf{x}) + \alpha\right] \Delta \lambda = -\Delta t^2 C(\mathbf{x}) - \alpha \lambda,
$$
\n(1)

where  $\Delta t$  is the time step size, and M is the lumped mass matrix. The update of the primal variable  $\Delta x$  can then be computed as:

$$
\Delta x = M^{-1} \nabla C^{\top}(x) \Delta \lambda. \tag{2}
$$

We note that such constraint-projection-based simulation naturally synergizes with 3DGS. It is also versatile and can deal with a wide range of physical problems such as fluid [\[40,](#page-9-17) [70\]](#page-10-16), rigid bodies [\[49\]](#page-9-24), or hyperelastic objects [\[41\]](#page-9-25).

#### 3.2. 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) is a learning-based rasterization technique for 3D scene reconstruction and novel view synthesis. 3DGS encodes a radiance field using a set of Gaussian kernels  $P$  with trainable parameters  $\mathbf{x}_p, \sigma_p, \mathbf{A}_p, \mathbf{c}_p$  for  $p \in \mathcal{P}$ , where  $\mathbf{x}_p, \sigma_p, \mathbf{A}_p$  and  $\mathbf{c}_p$  represent the center, opacity, covariance matrix, and color function of each kernel. To generate a scene render, 3DGS projects these kernels onto the imaging plane according to the viewing matrix and blends their colors based on the opacity and depth. The final color of the  $i$ -th pixel is computed as:

$$
\boldsymbol{c}_i = \sum_k G_k(i) \sigma_k \boldsymbol{c}_k(\boldsymbol{r}_i) \prod_{j=1}^{k-1} (1 - G_j(i) \sigma_j).
$$
 (3)

Here, all the kernels are re-ordered based on the z-values at kernel centers under the current view.  $G_k(i)$  denotes the 2D Gaussian weight of the k-th kernel at pixel i, and  $r_i$  is the view direction from camera to pixel  $i$ . The color functions only depend on the viewing direction.

GaussianShader [\[27\]](#page-8-4) further enhances 3DGS by incorporating additional trainable material parameters for kernel p such as diffuse  $d_p$ , specular  $s_p$ , roughness  $\rho_p$ , and normal

<span id="page-3-0"></span>![](_page_3_Figure_0.jpeg)

Figure 2. An overview of GSP pipeline. The input to our system comprises multi-view images that capture a 3D scene. During the preprocessing stage, foreground objects are isolated and reconstructed. This is followed by point sampling to facilitate scene discretization for PBD simulation and Gaussian rendering. We train the Gaussian kernels using differentiable 3DGS, which takes into account appearance materials and lighting conditions. These kernels are animated using PBD, in conjunction with fluid particles, to tackle the dynamics of both solids and fluids within the scene. Finally, the dynamic scene is rendered into images. This rendering process includes detailed modeling of specular reflections, thereby providing visually accurate representations of the simulated interactions between solids and fluids.

 $n_p$ , along with a global environment map. It fuses more information into the kernel's color:

<span id="page-3-1"></span>
$$
\boldsymbol{c}_p(\boldsymbol{r}_i) = \boldsymbol{d}_p + \boldsymbol{s}_p \odot L_s(\boldsymbol{r}_i, \boldsymbol{n}_p, \rho_p), \tag{4}
$$

where  $L_s(\mathbf{r}_i, \mathbf{n}_p, \rho_p)$  is the specular light for the kernel along  $r_i$  given the normal and roughness of the kernel. It can be pre-filtered into multiple mip maps. The symbol ⊙ denotes the element-wise multiplication.

# 4. Method

As shown in Fig. [2,](#page-3-0) the input of our system is a collection of images of a given 3D scene taken from different viewpoints. We separate foreground objects and the image background for all the inputs and extract the surface of masked objects. We apply an anisotropy loss to mitigate undesired splatting render to prevent over-stretching Gaussian kernels when training 3DGS for the solid object. Doing so mitigates the rendering artifacts near the surface of solid models. We decouple solid simulation and rendering by utilizing a separate set of sampled particles for simulation, and interpolating deformations of these particles onto trained Gaussian kernels during rendering. This approach ensures high-quality and robust results in both simulation and rendering. On the other hand, fluids use a unified set of Gaussian kernels (for rendering) or particles (for simulation). We track the fluids surface during simulation and use surface normal to properly synthesize specular effects by augmenting them with a decomposed environment map. Under the PBD framework, both fluids and solids are made of particles, and can be animated in a unified way based on local constraint projection.

## 4.1. Training

Our training process is generally similar to traditional Gaussian Splatting. However, due to our use of Physically-Based Rendering (PBR) modeling, the visual attributes or material parameters (e.g., diffuse, specular, roughness) of Gaussian kernels need to be determined. Without them, high-quality rendering results are not possible. Similar to [\[27\]](#page-8-4), we leverage the differentiable 3DGS pipeline to optimize the appearance of every Gaussian kernel. All Gaussian kernels are first shaded with their corresponding material parameters (Eq. [\(4\)](#page-3-1)) and are then splatted to a rendered image. Comparing it with the training view gives a loss back-propagated to update the corresponding parameters of Gaussian kernels. Specifically, the trainable parameters for each kernel p are position  $x_p$ , opacity  $\sigma_p$ , covariance  $A_p$  and material  ${d_p, s_p, \rho_p, n_p}$ . The loss is defined as:

$$
\mathcal{L} = \mathcal{L}_{\text{color}} + \lambda_n \mathcal{L}_{\text{normal}} + \lambda_a \mathcal{L}_{\text{aniso}},
$$
 (5)

where  $\mathcal{L}_{color}$  is the color loss between render and training image;  $\mathcal{L}_{\text{normal}}$  is normal consistency regularization adopted from GaussianShader [\[27\]](#page-8-4). The anisotropy loss  $\mathcal{L}_{aniso}$  is designed to prevent Gaussian kernels from becoming excessively elongated or compressed and potentially producing artifacts under large deformations. It is defined as:

$$
\mathcal{L}_{\text{aniso}} = \frac{1}{|\mathcal{P}|} \sum_{p \in \mathcal{P}} \max \left\{ \frac{S_p^1}{S_p^2} - a, 0 \right\},\tag{6}
$$

where  $a$  is a ratio threshold and  $\mathcal{\mathcal{S}}_p = \left\{\mathcal{S}^1_p, \mathcal{S}^2_p, \mathcal{S}^3_p\right\}$  are the scalings of Gaussian kernels with  $S_p^1$  being the largest scaling and  $S_p^3$  being the smallest scaling. Note that as the shading normal is based on the minimal axis of Gaussian kernels, we do not constrain the minimal axis in the anisotropy loss. Otherwise, a spherical Gaussian kernel will result in normal ambiguity. We set  $a = 1.1$ ,  $\lambda_n = 0.2$ , and  $\lambda_a = 10$ in our experiments.

#### 4.2. Solid Simulation and Interpolation

In many dynamic Gaussian Splatting applications [\[57,](#page-9-26) [69\]](#page-10-5), simulation and rendering are often coupled, utilizing the Gaussian kernels for both processes. This approach presents several issues. First, Gaussian kernels tend to distribute primarily on the surface of objects, necessitating the addition of internal kernels that, unless they contribute to effects like fracturing or breakage, are predominantly unrendered. Additionally, an uneven distribution of surface kernels can compromise the quality of the simulation, while a distribution that is too uniform can detrimentally affect rendering quality. Therefore, decoupling simulation and rendering is necessary to maintain quality in both aspects. In our pipeline, we leverage reconstructed 3DGS kernels for solid rendering and utilize a distinct set of particles for simulation.

To properly handle the object-object interaction or fluidsolid coupling, the particles are required to have an accurate description of object boundaries and interiors. To this end, we use the Poisson disk sampling [\[10\]](#page-8-22) to place simulation particles within the surface mesh, which is explicitly reconstructed for segmented foreground model using NeuS [\[65\]](#page-10-22). NeuS extracts the zero-level set of a signed distance function corresponding to the foreground object. It is important to note that existing frameworks that incorporate physics with 3DGS [\[69\]](#page-10-5) also require spatial discretization over the model to numerically solve the dynamics equation. Those prior approaches sample the model based on the trained 3DGS kernels. This method can result in sparsely sampled regions, especially for objects with thin parts, potentially affecting simulation quality (see Fig. [11](#page-12-0) in supplementary material).

Once the simulation particles are sampled and in place, we perform PBD on them to animate frames of dynamic motion. Each frame, deformation gradients and displacements are interpolated from the simulation particles to the trained Gaussian kernels using generalized moving least squares (GMLS) [\[43\]](#page-9-27), which produces smooth and robust results.

### 4.3. Position-Based Fluids

We employ the Position-Based Fluids (PBF) [\[40\]](#page-9-17) as our Lagrangian fluid synthesizer. To enforce the fluid incompressibility, PBF imposes a density constraint  $C_i^{\rho}$  on each particle, maintaining the integrated density  $\rho_i$  computed by the SPH kernel as:

$$
C_i^{\rho} = \frac{\rho_i}{\rho_0} - 1 = \sum_j \frac{m_j}{\rho_0} W(\mathbf{p}_i - \mathbf{p}_j) - 1, \qquad (7)
$$

where  $m_j$  is the mass of particle j.  $p_i$  is the position of particle  $i$ , and  $W$  is the SPH kernel function. Each constraint can be straightforwardly parallelized using GPUs.

GSP incorporates a position-based tension model [\[70\]](#page-10-16) to more accurately simulate fluid surface dynamics. Surface detection for a particle is performed through occlusion estimation, where each particle is enclosed by a spherical screen. Neighboring particles project onto this screen, and a particle is classified as on the surface if the cumulative projection area is below a specified threshold, reflecting its partial exposure. Given the natural tendency of tensions to reduce surface area, PBF enforces an area constraint on each surface particle to minimize local surface area. This process begins with the calculation of the normal  $n_i$  for surface particle *i* as:

$$
n_i = \text{normalize}(-\nabla_{p_i} C_i^{\rho}), \tag{8}
$$

where  $C_i^{\rho} = 0$  indicates the particle is inside the fluid, and  $C_i^{\rho} = -1$  indicates it is outside. After that, we project the neighboring surface particles onto a plane perpendicular to  $n<sub>i</sub>$  and triangularize the plane. The area constraint can then be built as:

$$
C_i^A = \sum_{t \in T(i)} \frac{1}{2} ||(\boldsymbol{p}_{t^2} - \boldsymbol{p}_{t^1}) \times (\boldsymbol{p}_{t^3} - \boldsymbol{p}_{t^1})|| \qquad (9)
$$

where  $T(i)$  is the set of neighboring triangles for particle i. To promote a more uniform particle distribution, additional distance constraints are introduced to push apart particles that are too close to each other:

$$
C_{ij}^D = \min\{0, ||\mathbf{p}_i - \mathbf{p}_j|| - d_0\},\tag{10}
$$

where  $d_0$  is the distance threshold. The original version was parallelized on CPUs; we have enhanced it for GPU parallelization. Calculations for  $n_i$ ,  $C_i^A$  and  $C_{ij}^D$  are efficiently parallelizable on GPUs. Considering that the number of surface particles surrounding each particle is typically low, local triangulation for each particle is conducted independently on separate threads.

## 4.4. Rendering

The rendering of the dynamic scene reuses the existing 3DGS pipeline. For dynamic solids, we first transform each solid Gaussian kernel from rest positions  $x_p$  to deformed positions  $x_p^t$  where t indicates the time step index. We directly place these kernels at deformed positions. For a kernel with deformation gradient  $F_p$ , its covariance  $A_p^t$  and normal  $n_p^t$  after deformation is updated by:

$$
\boldsymbol{A}_p^t = \boldsymbol{F}_p \boldsymbol{A}_p \boldsymbol{F}_p^\top, \text{ and } \boldsymbol{n}_p^t = \frac{\boldsymbol{F}_p^{-\top} \boldsymbol{n}_p}{\left\| \boldsymbol{F}_p^{-\top} \boldsymbol{n}_p \right\|}. \qquad (11)
$$

We then shade the deformed Gaussian kernels  $\{\boldsymbol{x}_p^t, \sigma_p, \boldsymbol{A}_p^t, \boldsymbol{n}_p^t\}$  with material  $\{\boldsymbol{d}_p, s_p, \rho_p\}$ , i.e., Eq. [\(4\)](#page-3-1) and splat them into an image  $c^{\text{bg}}$ . As shadows are important to visual outcomes in dynamic scenes, we further

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

Figure 3. GSP rendering. GSP synthesizes high-quality images corresponding to dynamically interacting fluids and solids. (a) The final rendered image combining rendered solids, fluids, and foams. (b) The rendering result of solids. (c) The rendering result of fluids. (d) The fluid thickness by additive splatting, where the darker color indicates the higher thickness. (e) The normal of fluids. (f) The intensity of foam particles. The insets in (b) and (e) represent solid and fluid Gaussian kernels, respectively.

re-engineer nearly-soft shadows [\[14\]](#page-8-23) into our system to enhance the realism.

The dynamic particle-based fluids are rendered with ellipsoids splatting [\[40\]](#page-9-17), which is inherently compatible with the existing 3DGS pipeline. We begin by generating spherical fluid Gaussian kernels at each fluid particle. The initial covariance  $A_p$  of each kernel is determined by the particle radius. The normal  $n_p$  adopts the surface normal of the nearest surface fluid particle from PBF simulation. We proceed to set up the appearance material for each fluid Gaussian kernel, which requires careful engineering due to the fluid's strong reflection and refraction effects. We employ the current PBR workflow to model the reflection, which adopts the same formula of Eq. [\(4\)](#page-3-1). We set a specular material ( $s_p = 1$ ,  $\rho_p = 0.05$  in our experiments) for all fluid kernels to imitate reflective behavior. The refraction needs careful treatment, which we model into a thicknessdependent diffuse color  $d_p$ . As the fluid thickness increases, the absorption of light within the fluid intensifies, resulting in reduced visibility for objects behind. Conversely, when there is less fluid present, it exhibits a more transparent appearance. The fluid thickness  $\tau$  comes from the modified splatting pipeline, with the alpha blending replaced by additive blending. The refraction color  $d_p$  is then represented by Beer's Law [\[62\]](#page-10-23):

$$
\boldsymbol{d}_p = e^{-k\tau_p} \boldsymbol{c}_p^{\text{bg}}.\tag{12}
$$

Here, the absorption coefficient  $k$  is defined differently for each color channel,  $\tau_p, c_p^{\text{bg}}$  is the fluid thickness and background back-projected to each Gaussian kernel respectively. Note that for background back-projection, a distortion  $\beta n_p$ is added to mimic the change of light path due to refraction. Opacity  $\sigma_p$  is set to 1 as most of transmission and refraction has already been modeled into  $d_p$ . We finally shade all fluid Gaussian kernels  $\{\boldsymbol{x}^t, \sigma_p, \boldsymbol{A}^t_p, \boldsymbol{n}^t_p~\boldsymbol{d}_p, s_p, \rho_p\}$  and splat them to the fluid rendering result  $c^{\text{fluid}}$  with Gaussian Splatting. To enhance the realism of fluids, foam particles are synthesized [\[23\]](#page-8-24) and rendered [\[1\]](#page-8-25). The 3DGS pipeline is re-engineered to incorporate additive splatting with distinct kernels for foam, bubble, and spray particles. The final ren-

<span id="page-5-1"></span>![](_page_5_Figure_6.jpeg)

Figure 4. Anistropy regularization. Anistropy regularization effectively maintains rendering quality under large deformations. Without the regularization term, 3DGS tends to generate fuzzy and spiky artifacts, especially near the surface of the model (left). When the regularization is applied, image quality is greatly improved with correct specular effects.

dering result is achieved by combining the  $c^{\rm bg}$  and  $c^{\rm fluid}$ , as shown in Fig. [3.](#page-5-0) Please refer to the supplementary material for more details on the rendering part.

## 4.5. Inpainting

Displaced object exposes unseen areas that were originally covered to the camera. Since they are not present in the input image, 3DGS is unable to recover the color and texture information of these areas, leading to black smudges and dirty textures in the result. GSP remedies this issue using an inpainting trick. First, we remove all the Gaussian kernels of the object that may be displaced. We then use LaMa [\[61\]](#page-10-24) to inpaint the rendered results, noting the new colors of pixels originally in spots and assigning them to the first Gaussian kernel encountered by the rays emitted from these pixels. We average the recorded colors on all noted Gaussian kernels for their diffuse color, set their opacity to 1, and their specular color to 0 to prevent highlights.

# 5. Experiments

We implemented Gaussian Splashing pipeline using Python, C++ and CUDA on a desktop PC equipped with a 12-core Intel i7-12700F CPU and an NVIDIA RTX 3090 GPU. Specifically, for the rendering part, we ported the published implementation of GaussianShader [\[27\]](#page-8-4) and integrated our fluid rendering using PyTorch [\[24\]](#page-8-26). PBD/PBF engine was implemented with CUDA, and we also group independent constraints to efficiently parallelize the constraint projections on the GPU. Please refer to the supplementary material for more details of the implementation.

#### 5.1. Ablation

Anisotropy Regularization 3DGS is originally designed for view synthesis. 3DGS obtained from a static scene produces low-quality renders when Gaussian kernels undergo large rotational deformations. The anisotropy regularization is effective against this limitation as shown Fig. [4.](#page-5-1) Detailed statistics regarding the experiment settings and timings are

<span id="page-6-0"></span>![](_page_6_Picture_0.jpeg)

Figure 5. Ablation study of specular. We demonstrate the impact of specular highlights on the quality of rendering. On the left is a fluid rendered with diffuse color only. On the right, surface reflective specular are added, which exhibits a more realistic and dynamic fluid.

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Figure 6. Shadow map. GSP incorporates dynamic shadows into the rendering pipeline to enhance visual realism. We re-engineer variance shadow mapping [\[14\]](#page-8-23) within the existing 3DGS pipeline to produce nearly-soft shadows.

reported in Tab. [1.](#page-7-0) Most experiments can also be found in the supplementary video.

**PBR Material** To show the importance of specular highlights in fluid rendering, we show a side-by-side comparison in a 3DGS scene. As shown in Fig. [5,](#page-6-0) we compare fluids rendered using purely diffuse materials with those that incorporated specular reflections. The fluids without specular reflection appear almost smoke-like, while the inclusion of specular term significantly enhances the realism of the fluids. It should be noted that incorporating PBR in 3DGS is key to improving the render quality of largely deformed fluids. This is in contrast to previous work [\[69\]](#page-10-5), which baked lighting into spherical harmonics and failed to produce realistic rendering when fluids underwent significant motion.

Shadow Map In Fig. [6,](#page-6-1) we demonstrate that GSP, with the addition of nearly soft shadows, significantly improves visual realism compared to vanilla 3DGS. Without these shadows, objects appear like flat layers pasted onto the background, lacking a sense of depth.

Inpainting We conduct an ablation experiment on inpainting as shown in Fig. [7.](#page-6-2) In this indoor scene, displacing cup and dog with vanilla 3DGS results in the occurance of black smudges and dirty textures, as the hidden area by the object could not be reconstructed properly due to missing information in the input images. GSP addresses this by using LaMa [\[61\]](#page-10-24) for inpainting.

<span id="page-6-2"></span>![](_page_6_Figure_8.jpeg)

Figure 7. 3DGS inpainting. In this indoor scene, both the paper cup and the stuffed toy dog are segmented from the input image (left). 3DGS leaves empty spots and dirty textures blended from irrelevant kernels, as highlighted in the middle figure. Applying the inpainting with generative AI [\[61\]](#page-10-24) ameliorates this issue (right).

## 5.2. Evaluation

We evaluate GSP through a diverse set of experiments, covering the dynamics of deformable bodies, rigid bodies, and fluids, as well as two-way coupling between solids and fluids. For additional frames and more detailed results, please refer to the supplementary material. All experiments are also available in the supplementary video. A preliminary test is depicted in Fig. [12,](#page-14-0) where a soft chair from the NeRF synthetic datasets [\[44\]](#page-9-0) falls into a pool, demonstrating the two-way coupling between deformable solids and fluids. The chair deforms, floats due to buoyancy, and generates fluid ripples. Fig. [13](#page-14-1) illustrates another dynamic fluid scene featuring a coastal cliff and waves. The waves continuously push towards the shore from a distance, and upon colliding with the cliff, they splash out foam and spray. This accurately models the complex interactions between the fluid and the solid cliff face. Fig. [14](#page-14-2) showcases another fluidsolid interaction test. The garden scene is sourced from the Mip-NeRF 360 [\[5\]](#page-8-27) dataset. The foreground objects consist of a fixed table and a potted plant. We pour water into the garden slowly. The water rises up and eventually sinks the table, and sweeps the plant. In Fig. [8](#page-7-1) and Fig. [15,](#page-14-3) a lego bulldozer is surfing on the splashing waves. Through the two-way coupling, the baseplate and the bucket deform and vibrate under the impact of the waves.

GSP has a semantic segmentation module. Therefore, the user is able to freely manipulate the models in the scene. Furthermore, since everything is represented as Gaussian particles, GSP allows the user to transform the state of the model. An example is shown in Fig. [16,](#page-14-4) the scene includes a round white table, on which a paper cup and a stuffed toy dog are placed. Water is poured into the cup. The state of the toy dog and the cup are changed to water, and splashes on the desk. As mentioned, we use LaMA to inpaint the table texture so that the user does not observe rendering artifacts when the liquefied cup and toy dog splash away.

Our fluid simulator can also capture the surface tension within the PBD framework. This enables realistic low-

<span id="page-7-0"></span>Table 1. Time performance. We present detailed time statistics for the experiments reported in the paper. All time-related evaluations are expressed in seconds. From left to right, (1) # Kernels indicates the number of Gaussian kernels, while # Solids, # Fluids, and # BG denote the foreground solid, the fluid, and the background, respectively. In some experiments (e.g., Cup & Dog and Headset), the number of fluid particles varies over time. We report the maximum number of fluid particles. (2) Sim. Time provides time statistics for simulations, with Overall representing the total simulation time per time step and Tension indicating the time taken for each surface tension constraint projection. (3) Render Time details the time statistics for rendering, with Solids, Fluids, and BG denoting the time spent on rendering the solids, fluids, and background, respectively.

| Scene (Fig.)                | # Kernels |                   |          | Sim. Time (s) |                                      | <b>Render Time</b> $(\times 10^{-2} s)$ |          |          |
|-----------------------------|-----------|-------------------|----------|---------------|--------------------------------------|-----------------------------------------|----------|----------|
|                             |           | # Solids # Fluids | # BG     |               | <b>Overall Tension Solids Fluids</b> |                                         |          | ВG       |
| Chair (Fig. 12)             | 315K      | 300K              | $\Omega$ | 5.4           | 1.1                                  | 3.9                                     | 5.9      | $\Omega$ |
| Waves $(Fig. 13)$           | 420K      | 817K              | $\Omega$ | 8.1           | 3.1                                  | 8.2                                     | 11.6     | $\Omega$ |
| Garden (Fig. 14)            | 450K      | 614K              | 2.27M    | 7.3           | 1.6                                  | 6.9                                     | 10.3     | 2.3      |
| Lego (Fig. 8&15)            | 330K      | 280K              | 290K     | 3.8           | 1.0                                  | 2.9                                     | 4.6      | 1.9      |
| Cup & dog (Fig. $16$ )      | 156K      | 160K              | 310K     | 2.1           | 0.6                                  | 2.4                                     | 4.1      | 1.9      |
| Headset (Fig. $9&17$ )      | 357K      | 64K               | 2.22M    | 1.9           | 1.8                                  | 1.5                                     | 3.8      | 2.7      |
| Can (Fig. 18)               | 390K      | 254K              | 1.19M    | 1.6           | 0.8                                  | 2.5                                     | 4.7      | 2.0      |
| <b>Astronaut</b> (Fig. 19)  | 0         | 145K              | $\Omega$ | 0.8           | 0.6                                  | $\Omega$                                | 4.0      | $\Omega$ |
| Ficus $(Fig. 20)$           | 204K      | $\Omega$          | $\Omega$ | 2.3           | $\Omega$                             | 2.6                                     | $\Omega$ | $\Omega$ |
| <b>Bulldozers</b> (Fig. 21) | 6.67M     | $\Omega$          | 350K     | 4.5           | $\Omega$                             | 24.6                                    | $\Omega$ | 2.1      |

<span id="page-7-1"></span>![](_page_7_Figure_2.jpeg)

Figure 8. Splashing LEGO. Through the two-way coupling dynamics, the LEGO bulldozer is animated to surf on the splashing waves.

<span id="page-7-2"></span>![](_page_7_Picture_4.jpeg)

Figure 9. Headset droplets. Water flows from a headset hanging above an office desk, resembling a faucet. Due to surface tension, the water forms droplets, sliding down the computer screen and splashing onto the desk, creating a puddle.

volume fluid-solid interaction. As shown in Fig. [9](#page-7-2) and Fig. [17,](#page-15-0) water flows from a headset hanging above an office desk, resembling a faucet. Due to surface tension, the water forms droplets as it falls, sliding down the computer screen and splashing onto the desk, creating a puddle. In Fig. [18,](#page-15-1) droplets of water continuously drip onto the top of the can until they reaches the capacity and overflow. The surface tension of the liquid causes the droplets to gradually aggregate at the surface. As more droplets fall onto the liquid surface, they rise above the edge of the can. Eventually, the accumulated water exceeds the limit of surface tension and spills over. Fig. [19](#page-15-2) showcases another interesting use of GSP, where the user applies Trisolarans' black magic on an astronaut. The astronaut is strucked by the magic and is transformed to water. It finally collapses into a water ball in a zero-gravity space due to the presence of surface tension.

GSP is a versatile system and supports the manipulation of both rigid objects and deformable bodies. As shown in Fig. [20,](#page-15-3) a deformable ficus is waving at you. Due to the external force applied to it, the plant undergoes continuous deformation. In Fig. [21,](#page-15-4) bulldozers are piled and dropped in a bowl. They collide and contact with each other and eventually scattered within the box.

# 6. Conclusion

GSP is a novel pipeline combining versatile position-based dynamics with 3DGS. The principle design philosophy of Gaussian Splashing is to harness the consistency of volume particle-based discretization to enable integrated processing of various 3D graphics and vision tasks, such as 3D reconstruction, deformable simulation, fluid dynamics, rigid contacts, and rendering from novel viewpoints. While the concept is straightforward, building Gaussian Splashing involves significant research and engineering efforts. The presence of fluid complicates the 3DGS processing due to the specular highlights at the fluid surface. Fluid-solid coupling resorts to accurate surface information; Large deformation on the solid object generates defective rendering; Displaced models also leave empty regions that the input images fail to capture. We overcome those difficulties by systematically integrating and adapting a collection of state-of-the-art technologies into the framework. As a result, Gaussian Splashing enables realistic view synthesis not only under novel camera poses but also with novel physically-based dynamics for various deformable, rigid, and fluid objects, or even novel object state transform. It should be noted that incorporating physically-based fluid dynamics in NeRF/3DGS has not been explored previously. The primary contribution of this work is to showcase the feasibility of building a unified framework for integrated physics and learning-based 3D reconstruction. Gaussian Splashing still has many limitations. For instance, PBD is known to be less physically accurate. It may be worth generalizing PBD with other meshless simulation methods. The fluid rendering in Gaussian Splashing in its current form is far from perfect – ellipsoid splatting is an ideal candidate for position-based fluid but does not physically handle real world light transport, e.g. refraction.

## References

- <span id="page-8-25"></span>[1] Nadir Akinci, Alexander Dippel, Gizem Akinci, and Matthias Teschner. Screen space foam rendering. 2013.
- <span id="page-8-19"></span>[2] Marc Alexa, Markus Gross, Mark Pauly, Hanspeter Pfister, Marc Stamminger, and Matthias Zwicker. Point-based computer graphics. In *ACM SIGGRAPH 2004 Course Notes*, pages 7–es. 2004.
- <span id="page-8-9"></span>[3] Chong Bao, Yinda Zhang, Bangbang Yang, Tianxing Fan, Zesong Yang, Hujun Bao, Guofeng Zhang, and Zhaopeng Cui. Sine: Semantic-driven image-based nerf editing with prior-guided editing field. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 20919–20929, 2023.
- <span id="page-8-13"></span>[4] Jonathan T Barron and Jitendra Malik. Shape, illumination, and reflectance from shading. *IEEE transactions on pattern analysis and machine intelligence*, 37(8):1670–1687, 2014.
- <span id="page-8-27"></span>[5] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. *CVPR*, 2022.
- <span id="page-8-12"></span>[6] Markus Becker and Matthias Teschner. Weakly compressible sph for free surface flows. In *Proceedings of the 2007 ACM SIGGRAPH/Eurographics Symposium on Computer Animation*, page 209–217, Goslar, DEU, 2007. Eurographics Association.
- <span id="page-8-14"></span>[7] Sai Bi, Zexiang Xu, Pratul Srinivasan, Ben Mildenhall, Kalyan Sunkavalli, Miloš Hašan, Yannick Hold-Geoffroy, David Kriegman, and Ravi Ramamoorthi. Neural reflectance fields for appearance acquisition. *arXiv preprint arXiv:2008.03824*, 2020.
- [8] Mark Boss, Raphael Braun, Varun Jampani, Jonathan T Barron, Ce Liu, and Hendrik Lensch. Nerd: Neural reflectance decomposition from image collections. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 12684–12694, 2021.
- <span id="page-8-15"></span>[9] Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan Barron, and Hendrik Lensch. Neural-pil: Neural pre-integrated lighting for reflectance decomposition. *Advances in Neural Information Processing Systems*, 34: 10691–10704, 2021.
- <span id="page-8-22"></span>[10] Robert Bridson. Fast poisson disk sampling in arbitrary dimensions. *SIGGRAPH sketches*, 10(1):1, 2007.
- <span id="page-8-21"></span>[11] Giang Bui, Truc Le, Brittany Morago, and Ye Duan. Pointbased rendering enhancement via deep learning. *The Visual Computer*, 34:829–841, 2018.
- <span id="page-8-20"></span>[12] H Childs, T Kuhlen, and F Marton. Auto splats: Dynamic point cloud visualization on the gpu. In *Proc. Eurographics Symp. Parallel Graph. Vis.*, pages 1–10, 2012.
- <span id="page-8-10"></span>[13] Jiahua Dong and Yu-Xiong Wang. Vica-nerf: Viewconsistency-aware 3d editing of neural radiance fields. In *Thirty-seventh Conference on Neural Information Processing Systems*, 2023.
- <span id="page-8-23"></span>[14] William Donnelly and Andrew Lauritzen. Variance shadow maps. In *Proceedings of the 2006 symposium on Interactive 3D graphics and games*, pages 161–165, 2006.
- <span id="page-8-6"></span>[15] Yilun Du, Yinan Zhang, Hong-Xing Yu, Joshua B Tenenbaum, and Jiajun Wu. Neural radiance flow for 4d view

synthesis and video processing. In *2021 IEEE/CVF International Conference on Computer Vision (ICCV)*, pages 14304–14314. IEEE Computer Society, 2021.

- <span id="page-8-3"></span>[16] Bardienus P Duisterhof, Zhao Mandi, Yunchao Yao, Jia-Wei Liu, Mike Zheng Shou, Shuran Song, and Jeffrey Ichnowski. Md-splatting: Learning metric deformation from 4d gaussians in highly deformable scenes. *arXiv preprint arXiv:2312.00583*, 2023.
- <span id="page-8-5"></span>[17] Yutao Feng, Yintong Shang, Xuan Li, Tianjia Shao, Chenfanfu Jiang, and Yin Yang. Pie-nerf: Physics-based interactive elastodynamics with nerf, 2023.
- <span id="page-8-7"></span>[18] Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic view synthesis from dynamic monocular video. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5712–5721, 2021.
- <span id="page-8-0"></span>[19] Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf: High-fidelity neural rendering at 200fps. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 14346– 14355, 2021.
- <span id="page-8-18"></span>[20] Jeffrey P Grossman and William J Dally. Point sample rendering. In *Rendering Techniques' 98: Proceedings of the Eurographics Workshop in Vienna, Austria, June 29—July 1, 1998 9*, pages 181–192. Springer, 1998.
- <span id="page-8-17"></span>[21] Antoine Guédon and Vincent Lepetit. Sugar: Surfacealigned gaussian splatting for efficient 3d mesh reconstruction and high-quality mesh rendering. *arXiv preprint arXiv:2311.12775*, 2023.
- <span id="page-8-8"></span>[22] Xiang Guo, Jiadai Sun, Yuchao Dai, Guanying Chen, Xiaoqing Ye, Xiao Tan, Errui Ding, Yumeng Zhang, and Jingdong Wang. Forward flow for novel view synthesis of dynamic scenes. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 16022–16033, 2023.
- <span id="page-8-24"></span>[23] Markus Ihmsen, Nadir Akinci, Gizem Akinci, and Matthias Teschner. Unified spray, foam and air bubbles for particlebased fluids. *The Visual Computer*, 28:669–677, 2012.
- <span id="page-8-26"></span>[24] Sagar Imambi, Kolla Bhanu Prakash, and GR Kanagachidambaresan. Pytorch. *Programming with TensorFlow: Solution for Edge Computing Applications*, pages 87–104, 2021.
- <span id="page-8-1"></span>[25] Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5885–5894, 2021.
- <span id="page-8-11"></span>[26] Kaiwen Jiang, Shu-Yu Chen, Feng-Lin Liu, Hongbo Fu, and Lin Gao. Nerffaceediting: Disentangled face editing in neural radiance fields. In *SIGGRAPH Asia 2022 Conference Papers*, pages 1–9, 2022.
- <span id="page-8-4"></span>[27] Yingwenqi Jiang, Jiadong Tu, Yuan Liu, Xifeng Gao, Xiaoxiao Long, Wenping Wang, and Yuexin Ma. Gaussianshader: 3d gaussian splatting with shading functions for reflective surfaces, 2023.
- <span id="page-8-16"></span>[28] Dahyun Kang, Daniel S Jeon, Hakyeong Kim, Hyeonjoong Jang, and Min H Kim. View-dependent scene appearance synthesis using inverse rendering from light fields. In *2021 IEEE International Conference on Computational Photography (ICCP)*, pages 1–12. IEEE, 2021.
- <span id="page-8-2"></span>[29] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time

radiance field rendering. *ACM Transactions on Graphics*, 42 (4), 2023.

- <span id="page-9-22"></span>[30] Marc Levoy and Turner Whitted. The use of points as a display primitive. 1985.
- <span id="page-9-7"></span>[31] Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. PAC-neRF: Physics augmented continuum neural radiance fields for geometry-agnostic system identification. In *The Eleventh International Conference on Learning Representations*, 2023.
- <span id="page-9-13"></span>[32] Yuan Li, Zhi-Hao Lin, David Forsyth, Jia-Bin Huang, and Shenlong Wang. Climatenerf: Extreme weather synthesis in neural radiance field. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023.
- <span id="page-9-10"></span>[33] Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural scene flow fields for space-time view synthesis of dynamic scenes. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 6498– 6508, 2021.
- <span id="page-9-19"></span>[34] Ruofan Liang, Jiahao Zhang, Haoda Li, Chen Yang, and Nandita Vijaykumar. Spidr: Sdf-based neural point fields for illumination and deformation. *arXiv preprint arXiv:2210.08398*, 2022.
- <span id="page-9-20"></span>[35] Ruofan Liang, Huiting Chen, Chunlin Li, Fan Chen, Selvakumar Panneer, and Nandita Vijaykumar. Envidr: Implicit differentiable renderer with neural environment lighting. *arXiv preprint arXiv:2303.13022*, 2023.
- <span id="page-9-15"></span>[36] Yiqing Liang, Numair Khan, Zhengqin Li, Thu Nguyen-Phuoc, Douglas Lanman, James Tompkin, and Lei Xiao. Gaufre: Gaussian deformation fields for real-time dynamic novel view synthesis, 2023.
- <span id="page-9-11"></span>[37] Jia-Wei Liu, Yan-Pei Cao, Weijia Mao, Wenqiao Zhang, David Junhao Zhang, Jussi Keppo, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Devrf: Fast deformable voxel radiance fields for dynamic scenes. *Advances in Neural Information Processing Systems*, 35:36762–36775, 2022.
- <span id="page-9-1"></span>[38] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. *Advances in Neural Information Processing Systems*, 33:15651–15663, 2020.
- <span id="page-9-21"></span>[39] Yuan Liu, Peng Wang, Cheng Lin, Xiaoxiao Long, Jiepeng Wang, Lingjie Liu, Taku Komura, and Wenping Wang. Nero: Neural geometry and brdf reconstruction of reflective objects from multiview images. *arXiv preprint arXiv:2305.17398*, 2023.
- <span id="page-9-17"></span>[40] Miles Macklin and Matthias Müller. Position based fluids. *ACM Transactions on Graphics (TOG)*, 32(4):1–12, 2013.
- <span id="page-9-25"></span>[41] Miles Macklin and Matthias Muller. A constraint-based formulation of stable neo-hookean materials. In *Proceedings of the 14th ACM SIGGRAPH conference on motion, interaction and games*, pages 1–7, 2021.
- <span id="page-9-4"></span>[42] Miles Macklin, Matthias Müller, and Nuttapong Chentanez. Xpbd: position-based simulation of compliant constrained dynamics. In *Proceedings of the 9th International Conference on Motion in Games*, pages 49–54, 2016.
- <span id="page-9-27"></span>[43] Sebastian Martin, Peter Kaufmann, Mario Botsch, Eitan Grinspun, and Markus Gross. Unified simulation of elastic rods, shells, and solids. *ACM Trans. Graph.*, 29(4), 2010.

- <span id="page-9-0"></span>[44] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In *The European Conference on Computer Vision (ECCV)*, 2020.
- <span id="page-9-16"></span>[45] Joe J Monaghan. Smoothed particle hydrodynamics. *Annual review of astronomy and astrophysics*, 30(1):543–574, 1992.
- <span id="page-9-3"></span>[46] Arthur Moreau, Jifei Song, Helisa Dhamo, Richard Shaw, Yiren Zhou, and Eduardo Pérez-Pellitero. Human gaussian splatting: Real-time rendering of animatable avatars. *arXiv preprint arXiv:2311.17113*, 2023.
- <span id="page-9-5"></span>[47] Matthias Müller and Nuttapong Chentanez. Solid simulation with oriented particles. In *ACM SIGGRAPH 2011 papers*, pages 1–10. 2011.
- <span id="page-9-6"></span>[48] Matthias Müller, Bruno Heidelberger, Marcus Hennix, and John Ratcliff. Position based dynamics. *Journal of Visual Communication and Image Representation*, 18(2):109–118, 2007.
- <span id="page-9-24"></span>[49] Matthias Müller, Miles Macklin, Nuttapong Chentanez, Stefan Jeschke, and Tae-Yong Kim. Detailed rigid body simulation with extended position based dynamics. In *Computer graphics forum*, pages 101–112. Wiley Online Library, 2020.
- <span id="page-9-2"></span>[50] Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. *ACM Trans. Graph.*, 41(4), 2022.
- <span id="page-9-18"></span>[51] Merlin Nimier-David, Delio Vicini, Tizian Zeltner, and Wenzel Jakob. Mitsuba 2: A retargetable forward and inverse renderer. *ACM Transactions on Graphics (TOG)*, 38(6):1– 17, 2019.
- <span id="page-9-8"></span>[52] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5865–5874, 2021.
- <span id="page-9-9"></span>[53] Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Ricardo Martin-Brualla, and Steven M Seitz. Hypernerf: A higherdimensional representation for topologically varying neural radiance fields. *arXiv preprint arXiv:2106.13228*, 2021.
- <span id="page-9-14"></span>[54] Yicong Peng, Yichao Yan, Shengqi Liu, Yuhao Cheng, Shanyan Guan, Bowen Pan, Guangtao Zhai, and Xiaokang Yang. Cagenerf: Cage-based neural radiance field for generalized 3d deformation and animation. In *Advances in Neural Information Processing Systems*, pages 31402–31415. Curran Associates, Inc., 2022.
- <span id="page-9-23"></span>[55] Ruggero Pintus, Enrico Gobbetti, and Marco Agus. Realtime rendering of massive unstructured raw point clouds using screen-space operators. In *Proceedings of the 12th International conference on Virtual Reality, Archaeology and Cultural Heritage*, pages 105–112, 2011.
- <span id="page-9-12"></span>[56] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 10318–10327, 2021.
- <span id="page-9-26"></span>[57] Shenhan Qian, Tobias Kirschstein, Liam Schoneveld, Davide Davoli, Simon Giebenhain, and Matthias Nießner. Gaus-

sianavatars: Photorealistic head avatars with rigged 3d gaussians. *arXiv preprint arXiv:2312.02069*, 2023.

- <span id="page-10-10"></span>[58] Yi-Ling Qiao, Alexander Gao, Yiran Xu, Yue Feng, Jia-Bin Huang, and Ming C Lin. Dynamic mesh-aware radiance fields. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 385–396, 2023.
- <span id="page-10-15"></span>[59] Barbara Solenthaler and Renato Pajarola. Predictivecorrective incompressible sph. In *ACM SIGGRAPH 2009 papers*, pages 1–6. 2009.
- <span id="page-10-17"></span>[60] Pratul P Srinivasan, Boyang Deng, Xiuming Zhang, Matthew Tancik, Ben Mildenhall, and Jonathan T Barron. Nerv: Neural reflectance and visibility fields for relighting and view synthesis. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 7495–7504, 2021.
- <span id="page-10-24"></span>[61] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In *Proceedings of the IEEE/CVF winter conference on applications of computer vision*, pages 2149– 2159, 2022.
- <span id="page-10-23"></span>[62] Donald F Swinehart. The beer-lambert law. *Journal of chemical education*, 39(7):333, 1962.
- <span id="page-10-7"></span>[63] Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, and Christian Theobalt. Nonrigid neural radiance fields: Reconstruction and novel view synthesis of a dynamic scene from monocular video. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 12959–12970, 2021.
- <span id="page-10-6"></span>[64] Wladimir J. van der Laan, Simon Green, and Miguel Sainz. Screen space fluid rendering with curvature flow. In *Proceedings of the 2009 Symposium on Interactive 3D Graphics and Games*, page 91–98, New York, NY, USA, 2009. Association for Computing Machinery.
- <span id="page-10-22"></span>[65] Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus: Learning neural implicit surfaces by volume rendering for multi-view reconstruction. *NeurIPS*, 2021.
- <span id="page-10-8"></span>[66] Chung-Yi Weng, Brian Curless, Pratul P Srinivasan, Jonathan T Barron, and Ira Kemelmacher-Shlizerman. Humannerf: Free-viewpoint rendering of moving people from monocular video. In *Proceedings of the IEEE/CVF conference on computer vision and pattern Recognition*, pages 16210–16220, 2022.
- <span id="page-10-2"></span>[67] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. *arXiv preprint arXiv:2310.08528*, 2023.
- <span id="page-10-9"></span>[68] Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time neural irradiance fields for free-viewpoint video. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 9421–9431, 2021.
- <span id="page-10-5"></span>[69] Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang. Physgaussian: Physicsintegrated 3d gaussians for generative dynamics. *arXiv preprint arXiv:2311.12198*, 2023.

- <span id="page-10-16"></span>[70] Jingrui Xing, Liangwang Ruan, Bin Wang, Bo Zhu, and Baoquan Chen. Position-based surface tension flow. *ACM Trans. Graph.*, 41(6), 2022.
- <span id="page-10-13"></span>[71] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Point-nerf: Point-based neural radiance fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5438–5448, 2022.
- <span id="page-10-11"></span>[72] Tianhan Xu and Tatsuya Harada. Deforming radiance fields with cages. In *ECCV*, 2022.
- <span id="page-10-14"></span>[73] Yuelang Xu, Benwang Chen, Zhe Li, Hongwen Zhang, Lizhen Wang, Zerong Zheng, and Yebin Liu. Gaussian head avatar: Ultra high-fidelity head avatar via dynamic gaussians, 2023.
- <span id="page-10-21"></span>[74] Zhenpei Yang, Yuning Chai, Dragomir Anguelov, Yin Zhou, Pei Sun, D Erhan, Sean Rafferty, and Henrik Kretzschmar. Surfelgan: Synthesizing realistic sensor data for autonomous driving. 2020 ieee. In *CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pages 11115–11124, 2020.
- <span id="page-10-3"></span>[75] Zeyu Yang, Hongye Yang, Zijie Pan, Xiatian Zhu, and Li Zhang. Real-time photorealistic dynamic scene representation and rendering with 4d gaussian splatting. *arXiv preprint arXiv:2310.10642*, 2023.
- <span id="page-10-0"></span>[76] Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 5752– 5761, 2021.
- <span id="page-10-1"></span>[77] Yu-Jie Yuan, Yu-Kun Lai, Yi-Hua Huang, Leif Kobbelt, and Lin Gao. Neural radiance fields from sparse rgb-d images for high-quality view synthesis. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2022.
- <span id="page-10-12"></span>[78] Yu-Jie Yuan, Yang-Tian Sun, Yu-Kun Lai, Yuewen Ma, Rongfei Jia, and Lin Gao. Nerf-editing: geometry editing of neural radiance fields. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 18353–18364, 2022.
- <span id="page-10-18"></span>[79] Kai Zhang, Fujun Luan, Qianqian Wang, Kavita Bala, and Noah Snavely. Physg: Inverse rendering with spherical gaussians for physics-based material editing and relighting. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 5453–5462, 2021.
- <span id="page-10-19"></span>[80] Xiuming Zhang, Pratul P Srinivasan, Boyang Deng, Paul Debevec, William T Freeman, and Jonathan T Barron. Nerfactor: Neural factorization of shape and reflectance under an unknown illumination. *ACM Transactions on Graphics (ToG)*, 40(6):1–18, 2021.
- <span id="page-10-4"></span>[81] Wojciech Zielonka, Timur Bagautdinov, Shunsuke Saito, Michael Zollhöfer, Justus Thies, and Javier Romero. Drivable 3d gaussian avatars. *arXiv preprint arXiv:2311.08581*, 2023.
- <span id="page-10-20"></span>[82] Matthias Zwicker, Hanspeter Pfister, Jeroen Van Baar, and Markus Gross. Surface splatting. In *Proceedings of the 28th annual conference on Computer graphics and interactive techniques*, pages 371–378, 2001.

# Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering

Supplementary Material

# 7. Simulation Details

### 7.1. Position-Based Dynamics

PBD/XPBD treats a dynamic system as a set of  $N$  vertices, i.e.,  $x = [x_0, x_1, ..., x_N]^\top$  and M constraints, i.e.,  $\mathbf{C}(\mathbf{x}) = [C_1(\mathbf{x}), C_2(\mathbf{x}), ..., C_M(\mathbf{x})]^\top$ . Here, x represents the position of vertices and  $C(x)$  represents the set of constraints. Specifically, the total system potential  $U$  is defined as a quadratic form of all the constraints such that  $U = \frac{1}{2} C^{\top}(x) \alpha^{-1} C(x)$ . Here,  $\alpha$  is the compliance matrix, i.e., the inverse of the constraint stiffness. For example, if there are only two vertices and they form a mass-spring system, constraint and compliance matrix could be written as  $\boldsymbol{C}(\boldsymbol{x}) = [\|\boldsymbol{x_0}-\boldsymbol{x_1}\|-d_0]^\top$  and  $\boldsymbol{\alpha} = [k]$ , where  $d_0$  is the rest length of the spring and  $k$  is the stiffness of the spring.

Motion at each time step can be solved by minimizing the system energy. However, PBD/XPBD offers an easy and efficient simulation modality, converting the variational optimization to the so-called constraint projections.

XBPD estimates an update of constraint force (i.e., the multiplier)  $\Delta \lambda$  by solving:

$$
\left[\Delta t^2 \nabla C(\mathbf{x}) M^{-1} \nabla C^{\top}(\mathbf{x}) + \alpha \right] \Delta \lambda = -\Delta t^2 C(\mathbf{x}) - \alpha \lambda,
$$
\n(13)

where  $\Delta t$  is the time step size, and M is the lumped mass matrix. The update of the primal variable  $\Delta x$  can then be computed as:

<span id="page-11-2"></span>
$$
\Delta x = M^{-1} \nabla C^{\top}(x) \Delta \lambda. \tag{14}
$$

The parallelization of XPBD is enabled with a Gauss-Seidel-like scheme, which computes  $\Delta\lambda_j$  at each constraint  $C_i$  independently:

<span id="page-11-1"></span>
$$
\Delta \lambda_j \leftarrow \frac{-\Delta t^2 C_j(\boldsymbol{x}) - \boldsymbol{\alpha}_j}{\Delta t^2 \nabla C_j \boldsymbol{M}^{-1} \nabla C_j^\top + \boldsymbol{\alpha}_j}.
$$
 (15)

A typical XPBD simulation loop is shown in Algorithm [1.](#page-11-0)

## 7.2. Position-Based Fluids

We employ the Position-Based Fluids (PBF) [\[40\]](#page-9-17) as our Lagrangian fluid synthesizer. PBF is based on PBD, which means it also use constraint projections to simulate fluid behaviour. In PBF, fluid is composed of a large amount of particles. To enforce the fluid incompressibility, PBF imposes a density constraint  $C_i^{\rho}$  on each particle, maintaining the integrated density  $\rho_i$  computed by the SPH kernel as:

<span id="page-11-0"></span>

| <b>Algorithm 1</b> XPBD simulation loop for time step $n + 1$                                  |  |  |  |  |  |
|------------------------------------------------------------------------------------------------|--|--|--|--|--|
| 1: predict position $\tilde{x} \Leftarrow x^n + \Delta t v^n + \Delta t^2 M^{-1} f_{ext}(x)^n$ |  |  |  |  |  |
| 2:                                                                                             |  |  |  |  |  |
| 3: initialize solve $x_0 \leftarrow x$                                                         |  |  |  |  |  |
| 4: initialize multipliers $\lambda_0 \Leftarrow 0$                                             |  |  |  |  |  |
| 5: while $i <$ solver Iterations do                                                            |  |  |  |  |  |
| for all constraints do<br>6:                                                                   |  |  |  |  |  |
| compute $\Delta\lambda$ using Eq. 15<br>7:                                                     |  |  |  |  |  |
| compute $\Delta x$ using Eq. 14<br>8:                                                          |  |  |  |  |  |
| update $\lambda_{i+1} \Leftarrow \lambda_i + \Delta \lambda$<br>9:                             |  |  |  |  |  |
| update $x_{i+1} \Leftarrow x_i + \Delta x$<br>10:                                              |  |  |  |  |  |
| end for<br>11:                                                                                 |  |  |  |  |  |
| $12:$ end while                                                                                |  |  |  |  |  |
| 13: update positions $x^{n+1} \leftarrow x_i$                                                  |  |  |  |  |  |
| 14: update velocities $v^{n+1} \leftarrow \frac{1}{\Delta t}(x^{n+1} - x^n)$                   |  |  |  |  |  |
|                                                                                                |  |  |  |  |  |

$$
C_i^{\rho} = \frac{\rho_i}{\rho_0} - 1 = \sum_j \frac{m_j}{\rho_0} W(\mathbf{p}_i - \mathbf{p}_j, r) - 1, \qquad (16)
$$

where  $m_j$  is the mass of particle j.  $p_i$  is the position of particle  $i$ ,  $W$  is the SPH kernel function and  $r$  is the kernel radius. Intuitively, projecting this constraint to 0 ensures that the density at the current time remains consistent with the initial state. We use the following cubic SPH kernel:

$$
W(\mathbf{p}, r) = \begin{cases} \frac{8}{\pi r^3} (6q^2(q-1) + 1), & 0 \le q \le 0.5\\ \frac{16}{\pi r^3} (1-q)^3, & 0.5 < q \le 1\\ 0, & \text{otherwise} \end{cases}
$$
(17)

where  $q = \frac{\Vert \mathbf{p} \Vert}{r}$  $\frac{p_{\parallel}}{r}$ . The Jacobian of constraint is computed as:

$$
\nabla_{\mathbf{p}_k} C_i^{\rho} = \begin{cases} \sum_j \frac{m_j}{\rho_0} \nabla_{\mathbf{p}_i} W(\mathbf{p}_i - \mathbf{p}_j, r), & k = i \\ \frac{m_j}{\rho_0} \nabla_{\mathbf{p}_j} W(\mathbf{p}_i - \mathbf{p}_j, r), & k = j. \end{cases}
$$
(18)

The gradient of the kernel function is:

$$
\nabla_{\boldsymbol{p}} W(\boldsymbol{p}, r) = \begin{cases} \frac{48}{\pi r^5} (3q - 2) \boldsymbol{p}, & 0 \le \frac{\|\boldsymbol{p}\|}{r} \le 0.5\\ -\frac{48}{\pi r^5} \frac{(1-q)^2}{q} \boldsymbol{p}, & 0.5 < \frac{\|\boldsymbol{p}\|}{r} \le 1\\ 0, & \text{otherwise} \end{cases}
$$
(19)

GSP also includes a position-based surface tension model [\[70\]](#page-10-16) to better capture the dynamics of the fluid surface. We first detect whether a particle (i.e., a Gaussian kernel) is on the fluid surface based on occlusion estimation. Specifically, we encapsulate a particle with a spherical cover or screen. Each of its neighboring particles generates a projection on the screen (because a particle has a finite volume). The particle is considered on the fluid surface if the total projection area from its neighbors is below a given threshold.

In the original paper  $[70]$ , surface detection is implemented on the CPU. It is noteworthy that surface detection can be parallelized on the GPU to expedite the simulation, as the calculation of each particle's occluding ratio on the screen is independent of the others.

For each neighboring particle, its occluding area on the spherical screen is calculated as follows:

$$
\theta = \tan^{-1}\left(\frac{\Delta p_y}{\Delta p_x + \Delta p_z^2}\right)
$$
$$
\phi = \tan^{-1}\left(\frac{\Delta p_x}{\Delta p_z}\right) \qquad (20)
$$
$$
\Delta \theta = \tan^{-1}\frac{R}{\|\Delta p\|^2 - R^2}
$$
$$
\Delta \phi = \Delta \theta
$$

where  $\Delta p$  is the vector from the detection particle to the neighboring particle and R is the particle radius. The shad-

![](_page_12_Picture_4.jpeg)

Figure 10. Detection of surface particles. An interior particle is detected if its screen is widely shadowed by its neighbors. A boundary particle is detected if at least one part of the particle's screen is not shadowed.

owed area on the spherical screen is then  $[\theta - \Delta \theta, \theta + \Delta \theta] \times$  $[\phi-\Delta\phi, \phi+\Delta\phi]$ . We parameterize the screen as an 18 × 36 environment map, with each column of the environment map corresponding to 18 bits of an integer. We then mask 36 integers and count the mask ratio.

After detecting surface particles on the fluid, we apply tension on the surface. Tensions tends to minimize surface area. Therefore, PBF applies an area constraint to each surface particle to minimize the local surface area nearby. We start by calculating the normal  $n_i$  of surface particles i as:

$$
n_i = \text{normalize}(-\nabla_{p_i} C_i^{\rho}), \qquad (21)
$$

where  $C_i^{\rho} = 0$  indicates the particle is inside the fluid, and  $C_i^{\rho} = -1$  indicates it is outside. After that, we project the neighboring surface particles onto a plane perpendicular to  $n<sub>i</sub>$  and triangularize the plane. The area constraint can then be built as:

$$
C_i^A = \sum_{t \in T(i)} \frac{1}{2} ||(\boldsymbol{p}_{t^2} - \boldsymbol{p}_{t^1}) \times (\boldsymbol{p}_{t^3} - \boldsymbol{p}_{t^1})|| \qquad (22)
$$

where  $T(i)$  is the set of neighboring triangles for particle i. We use the 2D Delaunay triangulation to construct the triangles on the local surface. This process is sequential and cannot be parallelized on the GPU. However, it is sufficiently fast and we translate it from CPU to GPU.

<span id="page-12-0"></span>![](_page_12_Picture_13.jpeg)

Figure 11. Different sampling strategies. We compare the results of different sampling strategies: (left) fill the particle based on the density grid calculated using Gaussian kernels [\[69\]](#page-10-5), and (right) uniformly sample within NeuS reconstruction. The point distribution generated by vanilla 3DGS is uneven, which hardly samples the legs or seat of the chair.

To promote a more uniform particle distribution, additional distance constraints are introduced to push apart particles that are too close to each other:

$$
C_{ij}^D = \min\{0, ||\mathbf{p}_i - \mathbf{p}_j|| - d_0\},\tag{23}
$$

where  $d_0$  is the distance threshold. The Jacobian of aboved constraints are:

$$
\nabla_{t} C_{t}^{A}(\mathbf{p}) = \frac{(\mathbf{p}_{t^{2}} - \mathbf{p}_{t^{1}}) \times (\mathbf{p}_{t^{3}} - \mathbf{p}_{t^{1}}) \times (\mathbf{p}_{t^{3}} - \mathbf{p}_{t^{2}})}{2\|(\mathbf{p}_{t^{2}} - \mathbf{p}_{t^{1}}) \times (\mathbf{p}_{t^{3}} - \mathbf{p}_{t^{1}})\|},
$$
\n
$$
\nabla_{t^{2}} C_{t}^{A}(\mathbf{p}) = \frac{(\mathbf{p}_{t^{3}} - \mathbf{p}_{t^{2}}) \times (\mathbf{p}_{t^{1}} - \mathbf{p}_{t^{2}}) \times (\mathbf{p}_{t^{1}} - \mathbf{p}_{t^{3}})}{2\|(\mathbf{p}_{t^{3}} - \mathbf{p}_{t^{2}}) \times (\mathbf{p}_{t^{1}} - \mathbf{p}_{t^{2}})\|},
$$
\n
$$
\nabla_{t^{3}} C_{t}^{A}(\mathbf{p}) = \frac{(\mathbf{p}_{t^{1}} - \mathbf{p}_{t^{3}}) \times (\mathbf{p}_{t^{2}} - \mathbf{p}_{t^{3}}) \times (\mathbf{p}_{t^{2}} - \mathbf{p}_{t^{1}})}{2\|(\mathbf{p}_{t^{1}} - \mathbf{p}_{t^{3}}) \times (\mathbf{p}_{t^{2}} - \mathbf{p}_{t^{3}})\|},
$$
\n
$$
\nabla_{i} C_{ij}^{D}(\mathbf{p}) = \begin{cases}\n\mathbf{0}, & \|\mathbf{p}_{i} - \mathbf{p}_{j}\| > d_{0}, \\
\frac{\mathbf{p}_{i} - \mathbf{p}_{j}}{\|\mathbf{p}_{i} - \mathbf{p}_{j}\|}, & \text{others.} \\
\frac{\mathbf{p}_{j} - \mathbf{p}_{i}}{\|\mathbf{p}_{i} - \mathbf{p}_{j}\|}, & \text{others.} \\
\frac{\mathbf{p}_{j} - \mathbf{p}_{i}}{\|\mathbf{p}_{i} - \mathbf{p}_{j}\|}, & \text{others.} \\
\end{cases}
$$
\n
$$
(24)
$$

# 8. Sampling and Interpolation Details

In practice, we found that an uneven distribution of Gaussian kernels results in unstable and inaccurate motion synthesis, while a distribution that is too uniform can detrimentally affect rendering quality. Gaussian kernels tend to distribute primarily unevenly around the surfaces and edges of objects, leading to inaccurate boundary descriptions, which are crucial for interactions between objects in simulation. Conversely, adaptively distributed anisotropic

Gaussian kernels are key to representing the spatially varying texture on the object. To address this issue, we maintain two separate sets of points: one sampled from the NeuS mesh surface for simulation, and the other consisting of trained Gaussian kernels for rendering. We compare the results of directly sampling from trained Gaussian kernels to those of sampling from NeuS in Fig. [11.](#page-12-0) The former method can result in sparsely sampled regions, especially for objects with thin parts, potentially affecting simulation quality.

We then discuss how to animate trained Gaussian kernels for rendering dynamics. Denote the set of trained Gaussian kernels for rendering as  $S_r$ , and the set of sampled points from NeuS for simulation as  $S_s$ . At time 0, we initialize GMLS kernels on  $S<sub>s</sub>$  and find the k nearest neighbors  $\{p^0_{s,j} : j \in \mathcal{N}(i)\}$  from  $S_s$  for each point  $p_{r,i}$  in  $S_r$ . Here,  $\mathcal{N}(i)$  denotes the set of k nearest neighboring particles' indices of  $p_{r,i}$  in  $S_s$  at time 0. As the simulation or motion synthesis proceeds, the position  $p_{s,j}^n$  evolves with time step n. We then update  $p_{r,j}$  by interpolating from  ${p_{s,j}^n : j \in \mathcal{N}(i)}$  with the pre-built GMLS kernel. The interpolation of deformation is achieved in the same way, replacing the physical quantity position  $p$  to deformation gradient F.

## 9. Rendering Details

Shadow As shadows are crucial to the visual outcomes in dynamic scenes, we re-engineer nearly-soft shadows [\[14\]](#page-8-23) into our system to enhance realism. Following shadow mapping, we splat all Gaussian kernels to a camera positioned at the point light's location, which we denote as light-view. The point light is aligned with the direction of the significantly bright light in the environment map. The resolution of the light-view image is three times that of the original image resolution to address visual discrepancies caused by under-sampling.

We then reproject the points seen in the camera view to the light-view and compare their depths to the previously splatted light-view depth image. A larger depth indicates that the point is occluded by a nearer point and will therefore cast a shadow. A more robust variance method is discussed in [\[14\]](#page-8-23). Softer shadows can be achieved by blurring and averaging the light-view depth image. We compute the shadowing probability using Chebyshev's inequality [\[14\]](#page-8-23) and store the results in a shadow map. Finally, we composite the rendered image with the shadow map to achieve nearly-soft dynamic shadows.

Spray, foam, and bubble To enhance the realism of fluids, foam, spray and bubble particles are synthesized with [\[23\]](#page-8-24) as a post-processing step. Fluid-air mixtures are generated at the crest of the wave and in the impact zone of the wave. Spray, foam, and bubble particles are advected by the fluid and dissipate within their predefined lifetimes.

We splat these particles into a foam intensity image using modified additive splatting. Different types of particles use different kernels during splatting [\[1\]](#page-8-25). We preferred a larger overall intensity for surface foam particles to increase their visibility, while we used a comparatively smaller intensity value for spray particles to make them less prominent. Furthermore, we used hollow circle structures for the bubble particles to make their appearance more convincing underwater. The kernel typically has a radius of 2 pixels. However, in practice, we found that the kernel radius should be scaled based on the particle's depth in the view, as particles near the camera occupy more of the view compared to those farther away. Finally, we apply a curve [\[1\]](#page-8-25) on the foam intensity image to scale it into  $[0, 1]$  and compose it with rendering.

## 10. Implementation Details

We set the simulation time step as 0.005 seconds throughout the simulations. In our PBD solver, we used 10 iterations for fluids and 50 iterations for solids for our experiments, since mass particles on the solid models are more strongly coupled than the ones in the fluid. During the PBF simulation, the surface particles of fluids are updated every two time steps.

# 11. More Results

We show the results of **Chair** (Fig. [12\)](#page-14-0), **Waves** (Fig. [13\)](#page-14-1), Garden (Fig. [14\)](#page-14-2), Lego (Fig. [15\)](#page-14-3), Cup & dog (Fig. [16\)](#page-14-4), Headset (Fig. [17\)](#page-15-0), Can (Fig. [18\)](#page-15-1), Astronaut (Fig. [19\)](#page-15-2), Ficus (Fig. [20\)](#page-15-3), and Bulldozers (Fig. [21\)](#page-15-4). For better visualization, please refer to the supplementary video.

<span id="page-14-0"></span>![](_page_14_Picture_0.jpeg)

Figure 12. Chair. A soft chair falls into the pool, causing deformation and ripples.

<span id="page-14-1"></span>![](_page_14_Picture_2.jpeg)

Figure 13. Waves crashing on cliff. A coastal cliff rises from the sandy beach, while the sea waves continuously crash against the rocky surface, generating splashes and foam upon collision.

<span id="page-14-2"></span>![](_page_14_Picture_4.jpeg)

Figure 14. Flooding garden. Water leaks into the garden and submerge the table. As the water level goes up, the surface gets more vibrant and washes the potted plant away.

<span id="page-14-3"></span>![](_page_14_Picture_6.jpeg)

Figure 15. Splashing LEGO. Through the two-way coupling dynamics, the LEGO bulldozer is animated to surf on the splashing waves.

<span id="page-14-4"></span>![](_page_14_Picture_8.jpeg)

Figure 16. "Everything is water". Pouring water into the paper cup on the table and transforming the cup and the dog toy into water. The water spills out.

<span id="page-15-0"></span>![](_page_15_Picture_0.jpeg)

Figure 17. Headset waterfall. Water flows from a headset hanging above an office desk, resembling a faucet. Due to surface tension, the water forms droplets as it falls, sliding down the computer screen and splashing onto the desk, creating a puddle.

<span id="page-15-1"></span>![](_page_15_Picture_2.jpeg)

Figure 18. Water droplets on can. Droplets of water fall onto the surface of a soda can, coalesce due to surface tension and gradually overflow.

<span id="page-15-2"></span>![](_page_15_Picture_4.jpeg)

Figure 19. Black magic. An astronaut in space strucked by the black magic of the Trisolarans, and get transformed into a water sphere.

<span id="page-15-3"></span>![](_page_15_Picture_6.jpeg)

Figure 20. Deformable ficus. A deformable ficus plant undergoes continuous shape changes as it is dragged and manipulated by external forces.

<span id="page-15-4"></span>![](_page_15_Picture_8.jpeg)

Figure 21. LEGO bulldozers in glass bowl. A collection of LEGO bulldozer rigid bodies fall into a round glass box, colliding with each other. They cast shadow on the ground and eventually stack and scatter throughout the box.