# Your ViT is Secretly a Hybrid Discriminative-Generative Diffusion Model

Xiulong Yang<sup>∗</sup> Georgia State University xyang22@gsu.edu

Sheng-Min Shih, Yinlin Fu, Xiaoting Zhao Etsy Inc. {sshih,yfu,xzhao}@etsy.com

Shihao Ji Georgia State University sji@gsu.edu

## Abstract

Diffusion Denoising Probability Models (DDPM) [\[31\]](#page-12-0) and Vision Transformer (ViT) [\[14\]](#page-12-1) have demonstrated significant progress in generative tasks and discriminative tasks, respectively, and thus far these models have largely been developed in their own domains. In this paper, we establish a direct connection between DDPM and ViT by integrating the ViT architecture into DDPM, and introduce a new generative model called Generative ViT (GenViT). The modeling flexibility of ViT enables us to further extend GenViT to hybrid discriminativegenerative modeling, and introduce a Hybrid ViT (HybViT). Our work is among the first to explore a single ViT for image generation and classification jointly. We conduct a series of experiments to analyze the performance of proposed models and demonstrate their superiority over prior state-of-the-arts in both generative and discriminative tasks. Our code and pre-trained models can be found in [https://github.com/sndnyang/Diffusion\\_ViT](https://github.com/sndnyang/Diffusion_ViT).

# 1 Introduction

Discriminative models and generative models based on the Convolutional Neural Network (CNN) [\[40\]](#page-13-0) architectures, such as GAN [\[20\]](#page-12-2) and ResNet [\[27\]](#page-12-3), have achieved state-of-the-art performance in a wide range of learning tasks. Thus far, they have largely been developed in two separate domains. In recent years, ViTs have started to rival CNNs in many vision tasks. Unlike CNNs, ViTs can capture the features from an entire image by self-attention, and they have demonstrated superiority in modeling non-local contextual dependencies as well as their efficiency and scalability to achieve comparable classification accuracy with smaller computational budgets (measured in FLOPs). Since the inception, ViTs have been exploited in various tasks such as object detection [\[6\]](#page-11-0), video recognition [\[4\]](#page-11-1), multimodal pre-training [\[37\]](#page-13-1), and image generation [\[33,](#page-12-4) [41\]](#page-13-2). Especially, VQ-GAN [\[18\]](#page-12-5), TransGAN [\[33\]](#page-12-4) and ViTGAN [\[41\]](#page-13-2) investigate the application of ViT in image generation. However, VQ-GAN is built upon an extra CNN-based VQ-VAE, and the latter two require two ViTs to construct a GAN for generation tasks. Therefore we ask the following question: is it possible to train a generative model using a single ViT?

DDPM is a class of generative models that matches a data distribution by learning to reverse a multistep diffusion process. It has recently been shown that DDPMs can even outperform prior SOTA GAN-based generative models [\[12,](#page-11-2) [5,](#page-11-3) [36\]](#page-13-3). Unlike GAN which needs to train with two competing networks, DDPM utilizes a UNet [\[53\]](#page-13-4) as a backbone for image generation and is trained to optimize maximum likelihood to avoid the notorious instability issue in GAN [\[46,](#page-13-5) [5\]](#page-11-3) and EBM [\[17,](#page-12-6) [23\]](#page-12-7).

In this paper, we establish a direct connection between DDPM and ViT for the task of image generation and classification. Specifically, we answer the question whether a single ViT can be

<sup>∗</sup>This work was done during an internship at Etsy

trained as a generative model. We design Generative ViT (GenViT) for pure generation tasks, as well as Hybrid ViT (HybViT) that extends GenViT to a hybrid model for both image classification and generation. As shown in Fig [2](#page-3-0) and [3,](#page-5-0) the reconstruction of image patches and the classification are two routines independent to each other and train a shared set of features together.

Our experiments show that HybViT outperforms previous state-of-the-art hybrid models. In particular, the Joint Energy-based Model (JEM), the previous state-of-the-art proposed by [\[23,](#page-12-7) [67\]](#page-14-0), requires extremely expensive MCMC sampling, which introduce instability and causes the training processes to fail for large-scale datasets due to the long training procedures required. To the best of our knowledge, GenViT is the first model that utilizes a single ViT as a generative model, and HybViT is a new type of hybrid model without the expensive MCMC sampling during training. Compared to existing methods, our new models demonstrate a number of conceptual advantages [\[17\]](#page-12-6): 1) Our methods provide simplicity and stability similar to DDPM, and are less prone to collapse compared to GANs and EBMs. 2) The generative and discriminative paths of our model are trained with a single objective which enables sharing of statistical strengths. 3) Advantageous computational efficiency and scalability to growing model and data sizes inherited from the ViT backbone.

Our contributions can be summarized as following:

- 1. We propose GenViT, which to the best of our knowledge, is the first approach to utilize a single ViT as an alternative to the UNet in DDPM.
- 2. We introduce HybViT, a new hybrid approach for image classification and generation leveraging ViT, and show that HybViT considerably outperforms the previous state-ofthe-art hybrid models on both classification and generation tasks while at the same time optimizes more effectively than MCMC-based models such as JEM/JEM++.
- 3. We perform comprehensive analysis on model characteristics including adversarial robustness, uncertainty calibration, likelihood and OOD detection, comparing GenViT and HybViT with existing benchmarks.

## 2 Related Work

#### 2.1 Denoising Diffusion Probabilistic Models

We first review the derivation of DDPM [\[31\]](#page-12-0). DDPM is built upon the theory of Nonequilibrium Thermodynamics [\[56\]](#page-13-6) with a few simple yet effective assumptions. It assumes diffusion is a noising process  $q$  that accumulates isotropic Gaussian noises over timesteps (Figure [1\)](#page-1-0).

![](_page_1_Figure_9.jpeg)

<span id="page-1-0"></span>Figure 1: A graphical model of diffusion process.

Starting from the data distribution  $\vec{x}_0 \sim q(\vec{x}_0)$ , the diffusion process q produces a sequence of latents  $\vec{x}_1$  through  $\vec{x}_T$  by adding Gaussian noise at each time  $t \in [0, \dots, T-1]$  with variance  $\beta_t \in (0, 1)$ as follows:

$$
q(\vec{x}_1, ..., \vec{x}_T | \vec{x}_0) := \prod_{t=1}^T q(\vec{x}_t | \vec{x}_{t-1})
$$
\n(1)

$$
q(\vec{x}_t|\vec{x}_{t-1}) := \mathcal{N}(\vec{x}_t; \sqrt{1-\beta_t}\vec{x}_{t-1}, \beta_t \mathbf{I})
$$
\n(2)

Then, the process in reverse aims to get a sample in  $q(\vec{x}_0)$  from sampling  $\vec{x}_T \sim \mathcal{N}(0, \mathbf{I})$  by using a neural network:

$$
p_{\theta}(\vec{x}_{t-1}|\vec{x}_t) \coloneqq \mathcal{N}(\vec{x}_{t-1}; \mu_{\theta}(\vec{x}_t, t), \Sigma_{\theta}(\vec{x}_t, t)) \tag{3}
$$

With the approximation of q and p, DDPM gets a variational lower bound (VLB) as follows:

$$
\log p_{\theta(\boldsymbol{x}_0)} \ge \log p_{\theta(\boldsymbol{x}_0)} - D_{KL}(q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \mid p_{\theta}(\boldsymbol{x}_{0:T}))
$$
  
= 
$$
-\mathbb{E}_q \left[ \frac{q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)}{p_{\theta}(\boldsymbol{x}_{0:T})} \right]
$$
 (4)

Then they derive a loss for VLB as:

$$
L_{\rm vib} = L_0 + L_1 + \dots + L_{T-1} + L_T \tag{5}
$$

<span id="page-2-3"></span><span id="page-2-0"></span>
$$
L_0 = -\log p_\theta(\vec{x}_0|\vec{x}_1) \tag{6}
$$

$$
L_{t-1} = D_{KL}(q(\vec{x}_{t-1}|\vec{x}_t, \vec{x}_0) \mid p_{\theta}(\vec{x}_{t-1}|\vec{x}_t))
$$
\n(7)

$$
L_T = D_{KL}(q(\vec{x}_T|\vec{x}_0) \parallel p(\vec{x}_T))
$$
\n(8)

where  $L_0$  is modeled by an independent discrete decoder from the Gaussian  $\mathcal{N}(\vec{x}_0; \mu_\theta(\vec{x}_1, 1), \sigma_1^2 \vec{I})$ , and  $L_T$  is constant and can be ignored.

As noted in [\[31\]](#page-12-0), the forward process can sample an arbitrary timestep  $x_t$  directly conditioned on the input  $x_0$  in a closed form. With the nice property, we define  $\alpha_t := 1 - \beta_t$  and  $\bar{\alpha}_t := \prod_{s=0}^t \alpha_s$ . Then we have

$$
q(\vec{x}_t|\vec{x}_0) = \mathcal{N}(\vec{x}_t; \sqrt{\bar{\alpha}_t}\vec{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$
\n(9)

<span id="page-2-1"></span>
$$
\vec{x}_t = \sqrt{\bar{\alpha}_t} \vec{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon \tag{10}
$$

where  $\epsilon \sim \mathcal{N}(0, \mathbf{I})$  using the reparameterization. Then using Bayes theorem, we can calculate the posterior  $q(\vec{x}_{t-1}|\vec{x}_t, \vec{x}_0)$  in terms of  $\tilde{\beta}_t$  and  $\tilde{\mu}_t(\vec{x}_t, \vec{x}_0)$  as follows:

$$
q(\vec{x}_{t-1}|\vec{x}_t, \vec{x}_0) = \mathcal{N}(\vec{x}_{t-1}; \tilde{\mu}(\vec{x}_t, \vec{x}_0), \tilde{\beta}_t \mathbf{I})
$$
\n
$$
(11)
$$

$$
\tilde{\mu}_t(\vec{x}_t, \vec{x}_0) \coloneqq \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \vec{x}_0 + \frac{\sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \vec{x}_t \tag{12}
$$

<span id="page-2-2"></span>
$$
\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t \tag{13}
$$

As we can observe, the objective in Eq. [5](#page-2-0) is a sum of independent terms  $L_{t-1}$ . Using Eq. [10,](#page-2-1) we can sample from an arbitrary step of the forward diffusion process and estimate  $L_{t-1}$  efficiently. Hence, DDPM uniformly samples  $t$  for each sample in each mini-batch to approximate the expectation  $E_{\mathbf{x}_0,t,\epsilon}[L_{t-1}]$  to estimate  $L_{\text{vlb}}$ .

To parameterize  $\mu_{\theta}(\vec{x}_t, t)$  for Eq. [12,](#page-2-2) we can predict  $\mu_{\theta}(\vec{x}_t, t)$  directly with a neural network. Alter-natively, we can first use Eq. [10](#page-2-1) to replace  $x_0$  in Eq. [12](#page-2-2) to predict the noise  $\epsilon$  as

$$
\mu_{\theta}(\vec{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( \vec{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(\vec{x}_t, t) \right), \tag{14}
$$

[\[31\]](#page-12-0) finds that predicting the noise  $\epsilon$  worked best with a reweighted loss function:

<span id="page-2-4"></span>
$$
L_{\text{simple}} = E_{t, \vec{x}_0, \epsilon} \left[ ||\epsilon - \epsilon_{\theta}(\vec{x}_t, t)||^2 \right]. \tag{15}
$$

This objective can be seen as a reweighted form of  $L_{\text{vlb}}$  (without the terms affecting  $\Sigma_{\theta}$ ). For more details of the training and inference, we refer the readers to [\[31\]](#page-12-0). A closely related branch is called score matching [\[57,](#page-13-7) [58\]](#page-13-8), which builds a connection bridging DDPMs and EBMs. Our work is mainly built upon DDPM, but it's straightforward to substitute DDPM with a score matching method.

# 3 Vision Transformers

Transformers [\[63\]](#page-14-1) have made huge impacts across many deep learning fields [\[26\]](#page-12-8) due to their prediction power and flexibility. They are based on the concept of self-attention, a function that allows interactions with strong gradients between all inputs, irrespective of their spatial relationships. The self-attention layer (Eq. [16\)](#page-3-1) encodes inputs as key-value pairs, where values  $\vec{V}$  represent embedded

![](_page_3_Figure_0.jpeg)

<span id="page-3-0"></span>Figure 2: The backbone architecture for GenViT and HybViT. For generative modeling,  $x_t$  with a time embedding of t is fed into the model. For the classification task in HybViT, we compute logits from CLS with the input  $x_0$ .

inputs and keys  $\vec{K}$  act as an indexing method, and subsequently, a set of queries  $\vec{Q}$  are used to select which values to observe. Hence, a single self-attention head is computed as:

<span id="page-3-1"></span>
$$
Attn(\vec{Q}, \vec{K}, \vec{V}) = softmax\left(\frac{\vec{Q}\vec{K}^T}{\sqrt{d_k}}\right)\vec{V}.
$$
\n(16)

where  $d_k$  is the dimension of K.

Vision transformers (ViT) ViT2021 has emerged as a famous architecture that outperforms CNNs in various vision domains. The transformer encoder is constructed by alternating layers of multi-headed self-attention (MSA) and MLP blocks (Eq. [18,](#page-3-2) [19\)](#page-3-3), and layernorm (LN) is applied before every block, followed by residual connections after every block [\[64,](#page-14-2) [2\]](#page-11-4). The MLP contains two layers with a GELU non-linearity. The 2D image  $x \in \mathbb{R}^{H \times W \times C}$  is flattened into a sequence of image patches, denoted by  $x_p \in \mathbb{R}^{L \times (P^2 \cdot C)}$ , where  $L = \frac{H \times W}{P^2}$  is the effective sequence length and  $P \times P \times C$  is the dimension of each image patch.

Following BERT [\[11\]](#page-11-5), we prepend a learnable classification embedding  $x_{\text{class}}$  to the image patch sequence, then the 1D positional embeddings  $E_{pos}$  are added to formulate the patch embedding  $z_0$ . The overall pipeline of ViT is shown as follows:

$$
\boldsymbol{z}_{0} = [\boldsymbol{x}_{\text{class}}; \boldsymbol{x}_{p}^1 \boldsymbol{E}; \boldsymbol{x}_{p}^2 \boldsymbol{E}; \cdots; \boldsymbol{x}_{p}^N \boldsymbol{E}] + \boldsymbol{E}_{pos},
$$
  
$$
\boldsymbol{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \boldsymbol{E}_{pos} \in \mathbb{R}^{(N+1) \times D}
$$
 (17)

<span id="page-3-2"></span>
$$
\mathbf{z}'_{\ell} = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \ \ \ell = 1 \dots L \tag{18}
$$

$$
z_{\ell} = \text{MLP}(\text{LN}(z'_{\ell})) + z'_{\ell}, \qquad \ell = 1 \dots L \tag{19}
$$

<span id="page-3-3"></span>
$$
y = LN(z_L^0) \tag{20}
$$

ViT have made significant breakthroughs in various discriminative tasks and generative tasks, including image classification, multi-modal, and high-quality image and text generation [\[14,](#page-12-1) [15,](#page-12-9) [41\]](#page-13-2). Inspired by the parallelism between patches/embeddings of ViT, we experiment with applying a standard ViT directly to generative modeling with minimal possible modifications.

## 3.1 Hybrid models

Hybrid models [\[52\]](#page-13-9) commonly model the density function  $p(x)$  and perform discriminative classification jointly using shared features. Notable examples are [\[13,](#page-11-6) [7,](#page-11-7) [48,](#page-13-10) [23,](#page-12-7) [24,](#page-12-10) [1\]](#page-11-8).

Hybrid models can utilize two or more classes of generative model to balance the trade-off such as slow sampling and poor scalability with dimension. For example, VAE can be increased by applying a second generative model such as a Normalizing Flow [\[38,](#page-13-11) [22,](#page-12-11) [62\]](#page-14-3) or EBM [\[51\]](#page-13-12) in latent space. Alternatively, a second model can be used to correct samples [\[66\]](#page-14-4). In our work, we focus on training a single ViT as a hybrid model without the auxiliary model.

#### 3.2 Energy-Based Models

Energy-based models (EBMs) are an appealing family of models to represent data as they permit unconstrained architectures. Implicit EBMs define an unnormalized distribution over data typically learned through contrastive divergence [\[17,](#page-12-6) [30\]](#page-12-12).

Joint Energy-based Model (JEM) [\[23\]](#page-12-7) reinterprets the standard softmax classifier as an EBM and trains a single network to achieve impressive hybrid discriminative-generative performance. Beyond that, JEM++ [\[67\]](#page-14-0) proposes several training techniques to improve JEM's accuracy, training stability, and speed, including proximal gradient clipping, YOPO-based SGLD sampling, and informative initialization. Unfortunately, training EBMs using SGLD sampling is still impractical for highdimensional data.

# 4 Method

#### 4.1 A Single ViT is a Generative Model

We propose GenViT by substituting UNet, the backbone of DDPM, with a single ViT. In our model design, we follow the standard ViT [\[14\]](#page-12-1) as close as possible. An overview of the architecture of the proposed GenViT is depicted in Fig [2.](#page-3-0)

Given the input  $x_t$  from DDPM, we follow the raster scan to get a sequence of image patches  $x_n$ , which is fed into GenViT as:

$$
\mathbf{h}_0 = [\boldsymbol{x}_{\text{class}}; \boldsymbol{x}_p^1 \mathbf{E}; \boldsymbol{x}_p^2 \mathbf{E}; \cdots; \boldsymbol{x}_p^N \mathbf{E}] + \mathbf{E}_{pos},
$$
  
\n
$$
\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}
$$
  
\n
$$
\mathbf{h}'_{\ell} = \text{MSA}(\text{LN}(M(\mathbf{h}_{\ell-1}, \mathbf{A}))) + \mathbf{h}_{\ell-1}, \ell = 1, ..., L
$$
  
\n
$$
\mathbf{h}_{\ell} = \text{MLP}(\text{LN}(M(\mathbf{h}'_{\ell}, \mathbf{A}))) + \mathbf{h}'_{\ell}, \ell = 1, ..., L
$$
  
\n
$$
\mathbf{y} = \mathbf{h}_L = [\mathbf{y}^1, \cdots, \mathbf{y}^N]
$$
  
\n
$$
\mathbf{x}' = [\mathbf{x}_p^1, \cdots, \mathbf{x}_p^N] = [f_r(\mathbf{y}^1), \dots, f_r(\mathbf{y}^N)],
$$
  
\n
$$
\mathbf{x}_p^i \in \mathbb{R}^{P^2 \times C}, \mathbf{x}' \in \mathbb{R}^{H \times W \times C}.
$$
  
\n(21)

Different from ViT, GenViT takes the embedding of t as input to control the hidden features  $h_\ell$ every layer, and finally reconstruct L-th layer output  $h_L \in \mathbb{R}^{(\bar{N}+1)\times D}$  to an image  $x'$ . Following the design of UNet in DDPM, we first compute the embedding of t using an MLP  $\overline{A} = \text{MLP}_t(t)$ . Then we compute  $M(h_\ell, A) = h_\ell * (\mu_\ell(A) + 1) + \sigma_\ell(A)$  for each layer, where  $\mu_\ell(A) = MLP_\ell(A)$ .

#### 4.2 ViT as a Hybrid Model

JEM reinterprets the standard softmax classifier as an EBM and trains a single network for hybrid discriminative-generative modeling. Specifically, JEM maximizes the logarithm of joint density function  $p_{\theta}(x, y)$ :

<span id="page-4-0"></span>
$$
\log p_{\theta}(\boldsymbol{x}, y) = \log p_{\theta}(y|\boldsymbol{x}) + \log p_{\theta}(\boldsymbol{x}), \qquad (22)
$$

where the first term is the cross-entropy classification objective, and the second term can be optimized by the maximum likelihood learning of EBM using contrastive divergence and MCMC sampling. However, MCMC-based EBM is notorious due to the expensive K-step MCMC sampling that requires K full forward and backward propagations at every iteration. Hence, removing the MCMC sampling in training is a promising direction [\[24\]](#page-12-10).

We propose Hybrid ViT (HybViT), a simple framework to extend GenViT for hybrid modeling. We substitute the optimization of  $\log p_{\theta}(x)$  in Eq. [22](#page-4-0) with the VLB of GenViT as Eq. [4.](#page-2-3) Hence, we can train  $p(y|x)$  using standard cross-entropy loss and optimize  $p(x)$  using  $L_{simple}$  loss in Eq [15.](#page-2-4) The final loss of our HybViT is

$$
L = L_{\rm CE} + \alpha L_{\rm simple} \tag{23}
$$

$$
=E_{\boldsymbol{x}_0,y}\left[H(\boldsymbol{x}_0,y)\right]+\alpha E_{t,\vec{x}_0,\epsilon}\left[||\epsilon-\epsilon_\theta(\vec{x}_t,t)||^2\right]
$$
\n(24)

![](_page_5_Figure_0.jpeg)

<span id="page-5-0"></span>Figure 3: The pipeline of HybViT.

We empirically find that a larger  $\alpha = 100$  improves the generation quality while retaining comparable classification accuracy. The training pipeline can be viewed in Fig [3.](#page-5-0)

# 5 Experiments

This section evaluates the discriminative and generative performance on multiple benchmark datasets, including CIFAR10, CIFAR100, STL10, CelebA-HQ-128, Tiny-ImageNet, and ImageNet 32x32.

Our code is largely built on top of ViT  $[42]^2$  $[42]^2$  $[42]^2$  and DDPM<sup>[3](#page-5-2)</sup>. Note that we set the batch size as 128, and we update all ViT-based models with 1170 iterations in one epoch, while 390 iterations for CNN-based methods<sup>[4](#page-5-3)</sup>. Most experiments of ViTs run for 500 epochs, but 2500 epochs for STL10 and 100 epochs for ImageNet 32x32. Thanks to the memory efficiency of ViT, all our experiments can be performed with PyTorch on a single Nvidia GPU. For reproducibility, our source code is provided in the supplementary material.

![](_page_5_Picture_6.jpeg)

(a) CIFAR10 (b) CelebA 128 Figure 4: GenViT Generated samples of CIFAR10 and CelebA 128.

# <span id="page-5-4"></span>5.1 Hybrid Modeling

We first compare the performance with the state-of-the-art hybrid models, stand-alone discriminative and generative models on CIFAR10. We use accuracy, Inception Score (IS) [\[54\]](#page-13-14) and Fréchet Inception Distance (FID) [\[29\]](#page-12-13) as evaluation metrics. IS and FID are employed to evaluate the quality of generated images. The results on CIFAR10 are shown in Tables [1.](#page-6-0) HybViT outperforms

<span id="page-5-1"></span> $^{2}$ [https://github.com/aanna0701/SPT\\_LSA\\_ViT](https://github.com/aanna0701/SPT_LSA_ViT)

<span id="page-5-2"></span> $^3$ <https://github.com/lucidrains/denoising-diffusion-pytorch>

<span id="page-5-3"></span><sup>&</sup>lt;sup>4</sup>ViT-based models use  $3\times$  repeated augmentations [\[61\]](#page-14-5)

other hybrid models including JEM ( $K = 20$ ) and JEM++ ( $M = 20$ ) on accuracy (95.9%) and FID score (26.4), when the original ViT achieves comparable accuracy to WideResNet(WRN) 28-10. Moreover, GenViT and HybViT are superior in training stability. HybViT matches or outperforms the classification accuracy of JEM++  $(M = 20)$ , and in the meantime, it exhibits high stability during training while JEM ( $K = 20$ ) and JEM++ ( $M = 5$ ) would easily diverge at early epochs. The comparison results on more benchmark datasets, including CIFAR100, STL10, CelebA-128, Tiny-ImageNet, ImageNet 32x32 are shown in Table [2.](#page-7-0) Example images generated by GenViT and HybViT are provided in Fig [4](#page-5-4) and [5,](#page-6-1) respectively. More generated images can be found in the appendix.

| Model              | Acc $\% \uparrow$        | IS $\uparrow$ | $FID \downarrow$ |  |  |  |
|--------------------|--------------------------|---------------|------------------|--|--|--|
| <b>ViT</b>         | 96.5                     |               |                  |  |  |  |
| GenViT             |                          | 8.17          | 20.2             |  |  |  |
| HybViT             | 95.9                     | 7.68          | 26.4             |  |  |  |
|                    | Single Hybrid Model      |               |                  |  |  |  |
| <b>IGEBM</b>       | 49.1                     | 8.30          | 37.9             |  |  |  |
| JEM                | 92.9                     | 8.76          | 38.4             |  |  |  |
| $JEM++ (M=20)$     | 94.1                     | 8.11          | 38.0             |  |  |  |
| <b>JEAT</b>        | 85.2                     | 8.80          | 38.2             |  |  |  |
|                    | <b>Generative Models</b> |               |                  |  |  |  |
| SNGAN              |                          | 8.59          | 21.7             |  |  |  |
| StyleGAN2-ADA      |                          | 9.74          | 2.92             |  |  |  |
| <b>DDPM</b>        |                          | 9.46          | 3.17             |  |  |  |
| <b>DiffuEBM</b>    |                          | 8.31          | 9.58             |  |  |  |
| <b>VAEBM</b>       |                          | 8.43          | 12.2             |  |  |  |
| FlowEBM            |                          |               | 78.1             |  |  |  |
| Other Models       |                          |               |                  |  |  |  |
| WRN-28-10          | 96.2                     |               |                  |  |  |  |
| VERA(w/ generator) | 93.2                     | 8.11          | 30.5             |  |  |  |

<span id="page-6-0"></span>Table 1: Results on CIFAR10.

We compare with results reported by SNGAN [\[46\]](#page-13-5), StyleGAN2- ADA [\[36\]](#page-13-3), DDPM [\[31\]](#page-12-0), DiffuEBM [\[19\]](#page-12-14), VAEBM [\[66\]](#page-14-4), VERA [\[24\]](#page-12-10), FlowEBM [\[50\]](#page-13-15).

![](_page_6_Picture_4.jpeg)

<span id="page-6-1"></span>Figure 5: HybViT Generated samples of CIFAR10 and STL10.

It's worth mentioning that the overall quality of synthesis is worse than UNet-based DDPM. In particular, our methods don't generate realistic images for complex and high-resolution data. ViT is known to model global relations between patches and lack of local inductive bias. We hope advances in ViT architectures and DDPM may address these issues in future work, such as Performer [\[8\]](#page-11-9), Swin Transformer [\[44\]](#page-13-16), CvT [\[65\]](#page-14-6) and Analytic-DPM [\[3\]](#page-11-10).

| Model            | Acc % $\uparrow$ | IS $\uparrow$ | $FID \downarrow$ |
|------------------|------------------|---------------|------------------|
|                  | CIFAR100         |               |                  |
| ViT              | 77.8             |               |                  |
| GenViT           |                  | 8.19          | 26.0             |
| HybViT           | 77.4             | 7.45          | 33.6             |
| <b>WRN-28-10</b> | 79.9             |               |                  |
| <b>SNGAN</b>     |                  | 9.30          | 15.6             |
| BigGAN           |                  | 11.0          | 11.7             |
|                  | Tiny-ImageNet    |               |                  |
| ViT              | 57.6             |               |                  |
| GenViT           |                  | 7.81          | 66.7             |
| HybViT           | 56.7             | 6.79          | 74.8             |
| PreactResNet18   | 55.5             |               |                  |
| <b>ADC-GAN</b>   |                  |               | 19.2             |
|                  | STL10            |               |                  |
| ViT              | 84.2             |               |                  |
| GenViT           |                  | 7.92          | 110              |
| HybViT           | 80.8             | 7.87          | 109              |
| <b>WRN-16-8</b>  | 76.6             |               |                  |
| <b>SNGAN</b>     |                  | 9.10          | 40.1             |
|                  | ImageNet 32x32   |               |                  |
| ViT              | 57.5             |               |                  |
| GenViT           |                  | 7.37          | 41.3             |
| HybViT           | 53.5             | 6.66          | 46.4             |
| <b>WRN-28-10</b> | 59.1             |               |                  |
| <b>IGEBM</b>     |                  | 5.85          | 62.2             |
| <b>KL-EBM</b>    |                  | 8.73          | 32.4             |
|                  | CelebA 128       |               |                  |
| GenViT           |                  |               | 22.07            |
| <b>KL-EBM</b>    |                  |               | 28.78            |
| <b>SNGAN</b>     |                  |               | 24.36            |
| <b>UNet GAN</b>  |                  |               | 2.95             |

<span id="page-7-0"></span>Table 2: Results on STL10, CelebA 128, Tiny-ImageNet, and ImageNet 32x32. Baseline models are selected based on availability in the literature.

IGEBM [\[17\]](#page-12-6), KL-EBM [\[16\]](#page-12-15), SNGAN [\[46\]](#page-13-5), BigGAN [\[5\]](#page-11-3), ADC-GAN [\[32\]](#page-12-16), UNet GAN [\[55\]](#page-13-17), .

# 5.2 Model Evaluation

In this section, we conduct a thorough evaluation of proposed methods beyond the accuracy and generation quality. Note that it is not our intention to propose approaches to match or outperform the best models in all metrics.

# 5.2.1 Calibration

Recent works show that the predictions of modern convolutional neural networks could be overconfident due to increased model capacity [\[25\]](#page-12-17). Incorrect but confident predictions can be catastrophic for safety-critical applications. Hence, we investigate ViT and HybViT in terms of calibration using the metric Expected Calibration Error (ECE). Interestingly, Fig [6](#page-8-0) shows that predictions of both HybViT and ViT look like well-calibrated when trained with strong augmentations, however they are less confident and have worse ECE compared to WRN. More comparison results can be found in the appendix.

# 5.2.2 Out-of-Distribution Detection

Determining whether inputs are out-of-distribution (OOD) is an essential building block for safely deploying machine learning models in the open world. The model should be able to assign lower scores to OOD examples than to in-distribution examples such that it can be used to distinguish

![](_page_8_Figure_0.jpeg)

<span id="page-8-0"></span>Figure 6: Calibration results on CIFAR10. The smaller ECE is, the better. However, ViTs are better calibrated.

OOD examples from in-distribution ones. For evaluating the performance of OOD detection, we use a threshold-free metric, called Area Under the Receiver-Operating Curve (AUROC) [\[28\]](#page-12-18). Using the input density  $p_{\theta}(x)$  [\[47\]](#page-13-18) as the score, ViTs performs better in distinguishing the in-distribution samples from out-of-distribution samples as shown in Table [3,](#page-8-1).

<span id="page-8-1"></span>

| $s_{\boldsymbol{\theta}}(\boldsymbol{x})$ | Model         | <b>SVHN</b> | Interp | C <sub>100</sub> | CelebA |
|-------------------------------------------|---------------|-------------|--------|------------------|--------|
|                                           | WRN*          | .91         |        | .87              | .78    |
|                                           | <b>IGEBM</b>  | .63         | .70    | .50              | .70    |
| $\log p_{\theta}(\boldsymbol{x})$         | <b>JEM</b>    | .67         | .65    | .67              | .75    |
|                                           | $JEM++$       | .85         | .57    | .68              | .80    |
|                                           | <b>VERA</b>   | .83         | .86    | .73              | .33    |
|                                           | <b>KL-EBM</b> | .91         | .65    | .83              |        |
|                                           | ViT           | .93         | .93    | .82              | .81    |
|                                           | HybViT        | .93         | .92    | .84              | .76    |
|                                           |               |             |        |                  |        |

Table 3: OOD detection results. Models are trained on CIFAR10 and values are AUROC.

\* The result is from [\[43\]](#page-13-19).

# 5.2.3 Robustness

Adversarial examples [\[60,](#page-14-7) [21\]](#page-12-19) tricks the neural networks into giving incorrect predictions by applying minimal perturbations to the inputs, and hence, adversarial robustness is a critical characteristics of the model, which has received an influx of research interest. In this paper, we investigate the robustness of models trained on CIFAR10 using the white-box PGD attack [\[45\]](#page-13-20) under an  $L_{\infty}$  or  $L_2$ constraint. Fig [7](#page-8-2) compares ViT and HybViT with the baseline WRN-based classifier. We can see that ViT and HybViT have similar performance and both outperform WRN-based classifiers.

![](_page_8_Figure_8.jpeg)

<span id="page-8-2"></span>![](_page_8_Figure_9.jpeg)

## 5.2.4 Likelihood

An advantage of DDPM is that it can use the VLB as the approximated likelihood while most EBMs can't compute the intractable partition function w.r.t  $x$ . Table [4](#page-9-0) reports the test negative log-likelihood(NLL) in bits per dimension on CIFAR10. As we can observe, HybViT achieves comparable result to GenViT, and both are worse than other methods.

<span id="page-9-0"></span>

| Model            | $BPD\downarrow$ |
|------------------|-----------------|
| GenViT           | 3.78            |
| HybViT           | 3.84            |
| <b>DDPM</b> [31] | 3.70            |
| iDDPM [49]       | 2.94            |
| DiffuEBM[19]     | 3.18            |
| DistAug [34]     | 2.56            |

#### 5.3 Ablation Study

In this section, we study the effect of different training configurations on the performance of image classification and generation by conducting an exhaustive ablation study on CIFAR10. We investigate the impact of 1) training epochs, 2) the coefficient  $\alpha$ , and 3) configurations of ViT/HybViT architecture in the main content. Due to page limitations, more results can be found in the appendix.

<span id="page-9-1"></span>

| Model                 | Acc $\% \uparrow$ | IS $\uparrow$ | $FID \downarrow$ |
|-----------------------|-------------------|---------------|------------------|
| $ViT$ (epoch= $100$ ) | 94.2              |               |                  |
| $ViT$ (epoch=300)     | 96.2              |               |                  |
| $ViT$ (epoch=500)     | 96.5              |               |                  |
| GenViT(epoch=100)     |                   | 7.25          | 33.3             |
| GenViT(epoch=300)     |                   | 7.67          | 26.2             |
| GenViT(epoch=500)     |                   | 8.17          | 20.2             |
| HybViT(epoch=100)     | 93.1              | 7.15          | 35.0             |
| HybViT(epoch=300)     | 95.9              | 7.59          | 29.5             |
| $HybViT(epoch=500)$   | 95.9              | 7.68          | 26.4             |

The results are reported in Table [5](#page-9-1) and [6.](#page-10-0) First, Table [5](#page-9-1) shows a trade-off between the overall performance and computation time. The gain of classification and generation is relatively large when we prolong the training from 100 epochs to 300. With more training epochs, the accuracy gap between ViT and HybViT decreases. Furthermore, The generation quality can slightly improve after 300 epochs. Then we thoroughly explore the settings of the backbone ViT for GenViT and HybViT in Table [6.](#page-10-0) It can be observed that larger  $\alpha$  is preferred with high-quality generation and only small drop in accuracy. The number of heads also has a small effect on the trade-off between classification accuracy and generation quality. Enlarging the model capacity, depth, or hidden dimensions can improve the accuracy and generation quality.

While it is challenging for our methods to generate realistic images for complex and high-resolution data, it is beyond the scope of this work to further improve the generation quality for high-resolution data. Thus, it warrants an exciting direction of future work. We suppose the large patch size of the ViT's architecture is the critical causing factor. Hence, we investigate the impact of different patch sizes on STL10 in Table [7.](#page-10-1) However, even though a smaller patch size can improve the accuracy by a notably margin at the cost of increasing model sizes, but the generation quality for high-resolution images plateaued around  $p = 6$ . These results indicate that the bottleneck of image generation comes from other components, such as the linear projections and reconstruction projections, other than the multi-head self-attention. Note that a larger patch size (ps=12) do further deteriorate the generation quality and would lead to critical issues for high-resolution data like ImageNet, since the corresponding patch size is typically set to 14 or larger.

## 5.3.1 Training Speed

We report the empirical training speeds of our models and baseline methods on a single GPU for CIFAR10 in Table [8](#page-11-11) and those for ImageNet 32x32 is in the appendix. As discussed previously, two mini-batches are utilized in HybViT: one for training of  $L_{simple}$  and the other for training of the cross entropy loss. Hence, HybViT requires about  $2\times$  training time compared to GenViT. One of the advantages of GenViT and HybViT is that even with much more  $(7.5\times)$  iterations, they still reduce training time significantly compared to EBMs. The results demonstrate that our new methods are much faster and affordable for academia research settings.

| Model                 | Acc % $\uparrow$ | IS $\uparrow$ | $FID \downarrow$ |
|-----------------------|------------------|---------------|------------------|
| HybViT                | 95.9             | 7.68          | 26.4             |
| $HybViT(\alpha=1)$    | 96.6             | 4.74          | 68.9             |
| HybViT( $\alpha$ =10) | 97.0             | 6.40          | 38.2             |
| $HybViT(head=6)$      | 96.0             | 7.51          | 30.0             |
| HybViT(head=8)        | 95.9             | 7.74          | 28.0             |
| $HybViT(head=16)$     | 95.4             | 7.79          | 27.1             |
| $HybViT(depth=6)$     | 94.7             | 7.39          | 30.6             |
| HybViT(depth=12)      | 96.6             | 7.78          | 24.3             |
| $HybViT(dim=192)$     | 94.1             | 7.06          | 35.0             |
| HybViT(dim=768)       | 96.4             | 8.04          | 19.9             |
| $GenViT(dim=192)$     |                  | 7.26          | 32.5             |
| GenViT(dim=384)       |                  | 8.17          | 20.2             |
| GenViT(dim=768)       |                  | 8.32          | 18.7             |

<span id="page-10-0"></span>Table 6: Ablation Study on CIFAR10. The configurations of baseline of HybViT is  $\alpha$  = 100,head=12,depth=9,dim=384.

<span id="page-10-1"></span>Table 7: Ablation Study on STL10. All models are trained for 500 epochs. NoP means Number of Parameters. ps means Patch Size.

| Model                                                                                | N <sub>o</sub> P                          | Acc $\% \uparrow$                    | IS $\uparrow$                | $FID \downarrow$                 |
|--------------------------------------------------------------------------------------|-------------------------------------------|--------------------------------------|------------------------------|----------------------------------|
| $ViT(ps=8)$<br>$HybViT(ps=4)$<br>$HybViT(ps=6)$<br>$HybViT(ps=8)$<br>$HybViT(ps=12)$ | 12.9M<br>41.1M<br>17.0M<br>12.9M<br>11.4M | 78.7<br>87.1<br>81.7<br>77.5<br>69.1 | 6.90<br>7.30<br>6.95<br>2.55 | 125.5<br>123.6<br>125.2<br>240.2 |
| $GenViT(dim=384)$<br>GenViT $(dim=576)$<br>GenViT $(dim=768)$                        | 12.9M<br>26.4M<br>45.2M                   |                                      | 6.95<br>7.02<br>7.01         | 125.2<br>124.1<br>126.6          |

# 6 Limitations

As shown in previous sections, our models GenViT and HybViT exhibit promising results. However, compared to CNN-based methods, the main limitations are: 1) The generation quality is relatively low compared with pure generation (non-hybrid) SOTA models. 2) They require more training iterations to achieve high classification performance compared with pure classification models. 3) The sampling speed during inference is slow (typically  $T \geq 1000$ ) while GAN only needs one-time forward.

We believe the results presented in this work are sufficient to motivate the community to solve these limitations and improve speed and generative quality.

# 7 Conclusion

In this work, we integrate a single ViT into DDPM to propose a new type of generative model, GenViT. Furthermore, we present HybViT, a simple approach for training hybrid discriminativegenerative models. We conduct a series of thorough experiments to demonstrate the effectiveness of these models on multiple benchmark datasets with state-of-the-art results in most of the tasks of image classification, and image generation. We also investigate the intriguing properties, including likelihood, adversarial robustness, uncertainty calibration, and OOD detection. Most importantly, the proposed approach HybViT provides stable training, and outperforms the previous state-of-theart hybrid models on both discriminative and generation tasks. While there are still challenges in training the models for high-resolution images, we hope the results presented here will encourage the community to improve upon current approaches.

| Model            | NoP(M) | Min/Epoch<br>Runtime(Hours) |                                    |  |  |
|------------------|--------|-----------------------------|------------------------------------|--|--|
|                  |        | ViT-based Models            |                                    |  |  |
| $ViT(d=384)$     | 11.2   | 1.72                        | 14.4                               |  |  |
| GenViT $(d=384)$ | 11.2   | 2.11                        | 17.6                               |  |  |
| $HybViT(d=192)$  | 3.2    | 2.14                        | 17.9                               |  |  |
| $HybViT(d=384)$  | 11.2   | 3.71                        | 31.2                               |  |  |
| $HybViT(d=768)$  | 43.2   | 9.34                        | 77.8                               |  |  |
| WRN-based Models |        |                             |                                    |  |  |
| <b>WRN 28-10</b> | 36.5   | 1.53                        | 5.2                                |  |  |
| $JEM(K=20)$      | 36.5   | 30.2                        | 101.3                              |  |  |
| $JEM++(K=10)$    | 36.5   | 20.4                        | 67.4                               |  |  |
| <b>VERA</b>      | 40     | 19.3                        | 64.3                               |  |  |
| <b>IGEBM</b>     |        |                             | 1 GPU for 2 days                   |  |  |
| KL-EBM           | 6.64   |                             | 1 GPU for 1 day                    |  |  |
| VAEBM*           | 135    |                             | 400 epochs, 8 GPUs, 55 hours       |  |  |
| <b>DDPM</b>      | 35.7   |                             | 800k iter, 8 TPUs, 10.6 hours      |  |  |
| DiffuEBM         | 34.8   |                             | 240 $k$ iter, 8 TPUs, 40 $+$ hours |  |  |

<span id="page-11-11"></span>Table 8: Run-time comparison on CIFAR10. We set 1170 iterations as one epoch for all ViTs, and 390 for WRN-based models. All ViTs are trained for 500 epochs and WRN-based models are trained for 200 epochs.

\* The runtime is for pretraining NVAE only. It further needs 25,000 iterations (or 16 epochs) on CIFAR-10 using one GPU for VAEBM.

# References

- <span id="page-11-8"></span>[1] Michael Arbel and Arthur Zhou, Liang andGretton. Generalized energy based models. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-11-4"></span>[2] Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. In *ICLR*, 2019.
- <span id="page-11-10"></span>[3] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analytic-DPM: an analytic estimate of the optimal reverse variance in diffusion probabilistic models. In *International Conference on Learning Representations*, 2022.
- <span id="page-11-1"></span>[4] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In *ICML*, 2021.
- <span id="page-11-3"></span>[5] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In *International Conference on Learning Representations (ICLR)*, 2019.
- <span id="page-11-0"></span>[6] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In *European conference on computer vision (ECCV)*, 2020.
- <span id="page-11-7"></span>[7] LI Chongxuan, Taufik Xu, Jun Zhu, and Bo Zhang. Triple generative adversarial nets. In *Advances in Neural information processing systems*, 2017.
- <span id="page-11-9"></span>[8] Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J Colwell, and Adrian Weller. Rethinking attention with performers. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-11-13"></span>[9] Patryk Chrabaszcz, Ilya Loshchilov, and Frank Hutter. A downsampled variant of imagenet as an alternative to the cifar datasets, 2017.
- <span id="page-11-12"></span>[10] Adam Coates, Andrew Ng, and Honglak Lee. An analysis of single-layer networks in unsupervised feature learning. In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2011.
- <span id="page-11-5"></span>[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of North American Chapter of the Association for Computational Linguistics (NAACL)*, 2019.
- <span id="page-11-2"></span>[12] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In *Advances in Neural Information Processing Systems*, 2021.
- <span id="page-11-6"></span>[13] Danilo Jimenez Rezende Diederik Kingma, Shakir Mohamed and Max Welling. Semisupervised learning with deep generative models. In *Neural Information Processing Systems*

*(NeurIPS)*, 2014.

- <span id="page-12-1"></span>[14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-12-9"></span>[15] Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, Zicheng Liu, and Michael Zeng. An Empirical Study of Training End-to-End Vision-and-Language Transformers. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.
- <span id="page-12-15"></span>[16] Yilun Du, Shuang Li, Joshua Tenenbaum, and Igor Mordatch. Improved Contrastive Divergence Training of Energy Based Models. In *International Conference on Machine Learning (ICML)*, 2021.
- <span id="page-12-6"></span>[17] Yilun Du and Igor Mordatch. Implicit generation and generalization in energy-based models. In *Neural Information Processing Systems (NeurIPS)*, 2019.
- <span id="page-12-5"></span>[18] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In *IEEE/CVF conference on computer vision and pattern recognition*, 2021.
- <span id="page-12-14"></span>[19] Ruiqi Gao, Yang Song, Ben Poole, Ying Nian Wu, and Diederik P. Kingma. Learning Energy-Based Models by Diffusion Recovery Likelihood. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-12-2"></span>[20] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *Neural Information Processing Systems (NeurIPS)*, 2014.
- <span id="page-12-19"></span>[21] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In *International Conference on Learning Representations (ICLR)*, 2015.
- <span id="page-12-11"></span>[22] Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. In *International Conference on Learning Representations (ICLR)*, 2019.
- <span id="page-12-7"></span>[23] Will Grathwohl, Kuan-Chieh Wang, Joern-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. In *International Conference on Learning Representations (ICLR)*, 2020.
- <span id="page-12-10"></span>[24] Will Sussman Grathwohl, Jacob Jin Kelly, Milad Hashemi, Mohammad Norouzi, Kevin Swersky, and David Duvenaud. No mcmc for me: Amortized sampling for fast and stable training of energy-based models. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-12-17"></span>[25] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In *International Conference on Machine Learning (ICML)*, 2017.
- <span id="page-12-8"></span>[26] Kai Han, Yunhe Wang, Hanting Chen, Xinghao Chen, Jianyuan Guo, Zhenhua Liu, Yehui Tang, An Xiao, Chunjing Xu, Yixing Xu, Yiman Zhang, and Dacheng Tao. A survey on visual transformer. *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2020.
- <span id="page-12-3"></span>[27] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
- <span id="page-12-18"></span>[28] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In *International Conference on Learning Representations (ICLR)*, 2016.
- <span id="page-12-13"></span>[29] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In *Neural Information Processing Systems (NeurIPS)*, 2017.
- <span id="page-12-12"></span>[30] Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. *Neural computation*, 2002.
- <span id="page-12-0"></span>[31] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-12-16"></span>[32] Liang Hou, Qi Cao, Huawei Shen, Siyuan Pan, Xiaoshuang Li, and Xueqi Cheng. Conditional gans with auxiliary discriminative classifier. In *International Conference on Machine Learning (ICML)*, 2022.
- <span id="page-12-4"></span>[33] Yifan Jiang, Shiyu Chang, and Zhangyang Wang. Transgan: Two pure transformers can make one strong gan, and that can scale up. In *Neural Information Processing Systems*, 2021.
- <span id="page-12-20"></span>[34] Heewoo Jun, Rewon Child, Mark Chen, John Schulman, Aditya Ramesh, Alec Radford, and Ilya Sutskever. Distribution augmentation for generative modeling. In *International Conference on Machine Learning(ICML)*, 2020.
- <span id="page-12-21"></span>[35] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In *International Conference on Learning Representations (ICLR)*, 2018.

- <span id="page-13-3"></span>[36] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Training generative adversarial networks with limited data. In *Proc. Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-13-1"></span>[37] Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In *International Conference on Machine Learning (ICML)*, 2021.
- <span id="page-13-11"></span>[38] Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved variational inference with inverse autoregressive flow. In *Neural Information Processing Systems (NeurIPS)*, 2016.
- <span id="page-13-22"></span>[39] Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009.
- <span id="page-13-0"></span>[40] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R.E. Howard, W. Hubbard, and L.D. Jackel. Backpropagation applied to handwritten zip code recognition. In *Neural Computation*, 1989.
- <span id="page-13-2"></span>[41] Kwonjoon Lee, Huiwen Chang, Lu Jiang, Han Zhang, Zhuowen Tu, and Ce Liu. Vitgan: Training gans with vision transformers. In *International Conference on Learning Representations(ICLR)*, 2022.
- <span id="page-13-13"></span>[42] Seung Hoon Lee, Seunghyun Lee, and Byung Cheol Song. Vision transformer for small-size datasets, 2021.
- <span id="page-13-19"></span>[43] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. Energy-based out-of-distribution detection. *Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-13-16"></span>[44] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021.
- <span id="page-13-20"></span>[45] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In *International Conference on Learning Representations*, 2018.
- <span id="page-13-5"></span>[46] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In *International Conference on Learning Representations (ICLR)*, 2018.
- <span id="page-13-18"></span>[47] Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur, and Balaji Lakshminarayanan. Do deep generative models know what they don't know? *arXiv preprint arXiv:1810.09136*, 2018.
- <span id="page-13-10"></span>[48] Eric T. Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Görür, and Balaji Lakshminarayanan. Hybrid models with deep and invertible features. In *International Conference on Machine Learning, (ICML)*, 2019.
- <span id="page-13-21"></span>[49] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In *International Conference on Machine Learning (ICML)*, 2021.
- <span id="page-13-15"></span>[50] Erik Nijkamp, Ruiqi Gao, Pavel Sountsov, Srinivas Vasudevan, Bo Pang, Song-Chun Zhu, and Ying Nian Wu. Mcmc should mix: Learning energy-based model with neural transport latent space mcmc. In *International Conference on Learning Representations (ICLR)*, 2022.
- <span id="page-13-12"></span>[51] Bo Pang, Tian Han, Erik Nijkamp, Song-Chun Zhu, and Ying Nian Wu. Learning latent space energy-based prior model. In *Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-13-9"></span>[52] Rajat Raina, Yirong Shen, Andrew Mccallum, and Andrew Y Ng. Classification with hybrid generative/discriminative models. In *Advances in neural information processing systems*, 2004.
- <span id="page-13-4"></span>[53] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In *International Conference on Medical image computing and computer-assisted intervention*, 2015.
- <span id="page-13-14"></span>[54] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In *Neural Information Processing Systems (NeurIPS)*, 2016.
- <span id="page-13-17"></span>[55] Edgar Schonfeld, Bernt Schiele, and Anna Khoreva. A u-net based discriminator for generative adversarial networks. In *IEEE/CVF conference on computer vision and pattern recognition*, 2020.
- <span id="page-13-6"></span>[56] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In *International Conference on Machine Learning (ICML)*, 2015.
- <span id="page-13-7"></span>[57] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In *Neural Information Processing Systems (NeurIPS)*, 2019.
- <span id="page-13-8"></span>[58] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In *International Conference on Learning Representations*, 2020.

- <span id="page-14-10"></span>[59] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016.
- <span id="page-14-7"></span>[60] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In *International Conference on Learning Representations (ICLR)*, 2014.
- <span id="page-14-5"></span>[61] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In *International Conference on Machine Learning*. PMLR, 2021.
- <span id="page-14-3"></span>[62] Arash Vahdat and Jan Kautz. Nvae: A deep hierarchical variational autoencoder. In *Neural Information Processing Systems (NeurIPS)*, 2020.
- <span id="page-14-1"></span>[63] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Neural Information Processing Systems (NeurIPS)*, 2017.
- <span id="page-14-2"></span>[64] Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, and Lidia S. Chao. Learning deep transformer models for machine translation. In *ACL*, 2019.
- <span id="page-14-6"></span>[65] Haiping Wu, Bin Xiao, Noel Codella, Mengchen Liu, Xiyang Dai, Lu Yuan, and Lei Zhang. Cvt: Introducing convolutions to vision transformers. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.
- <span id="page-14-4"></span>[66] Zhisheng Xiao, Karsten Kreis, Jan Kautz, and Arash Vahdat. VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models. In *International Conference on Learning Representations (ICLR)*, 2021.
- <span id="page-14-0"></span>[67] Xiulong Yang and Shihao Ji. JEM++: Improved Techniques for Training JEM. In *International Conference on Computer Vision (ICCV)*, 2021.

# A Image Datasets

The image benchmark datasets used in our experiments are described below:

- 1. CIFAR10 [\[39\]](#page-13-22) contains 60,000 RGB images of size  $32 \times 32$  from 10 classes, in which 50,000 images are for training and 10,000 images are for test.
- 2. CIFAR100 [\[39\]](#page-13-22) also contains 60,000 RGB images of size  $32 \times 32$ , except that it contains 100 classes with 500 training images and 100 test images per class.
- 3. STL10 [\[10\]](#page-11-12) 500 training images from 10 classes as CIFAR10, 800 test images per class.
- 4. Tiny-ImageNet contains 100000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images, 50 validation images and 50 test images.
- 5. CelebA-HQ [\[35\]](#page-12-21) is a human face image dataset. In our experiment, we use the downsampled version with size  $128 \times 128$ .
- 6. Imagenet 32x32 [\[9\]](#page-11-13) is a downsampled variant of ImageNet with 1,000 classes. It contains the same number of images as vanilla ImageNet, but the image size is  $32 \times 32$ .

# B Experimental Details

As we discuss in the main content, all our experiments are based on vanilla ViT in  $[42]$ <sup>[5](#page-14-8)</sup> and DDPM<sup>[6](#page-14-9)</sup> and follow their settings. We use SGD for all datasets with an initial learning rate of 0.1. We reduce the learning rate using the cosine scheduler. Table [9](#page-15-0) lists the hyper-parameters in our experiments. We also tried  $T = 4000$  and  $L_2$  loss to train our GenViT and HybViT, and their final results are comparable.

<span id="page-14-8"></span><sup>5</sup> [https://github.com/aanna0701/SPT\\_LSA\\_ViT](https://github.com/aanna0701/SPT_LSA_ViT)

<span id="page-14-9"></span> $^6$ <https://github.com/lucidrains/denoising-diffusion-pytorch>

<span id="page-15-0"></span>

| Variable                       | Value      |
|--------------------------------|------------|
| Learning rate                  | 0.1        |
| <b>Batch Size</b>              | 128        |
| Warmup Epochs                  | 10         |
| Coefficient $\alpha$ in HybViT | 1, 10, 100 |
| Configurations of ViT          |            |
| Dimensions                     | 384        |
| Depth                          | 9          |
| Heads                          | 12         |
| Patch Size                     | 4.8        |
| Configurations of DDPM         |            |
| Number of Timesteps $T$        | 1000       |
| Loss Type                      | $L_1$      |
| Noise Schedule                 | cosine     |

Table 9: Hyper-parameters of ViT, GenViT and HybViT

![](_page_15_Figure_2.jpeg)

<span id="page-15-1"></span>Figure 8: The evolution of HybViT's classification accuracy, FID as a function of training epochs on CIFAR10 and ImageNet 32x32.

![](_page_15_Figure_4.jpeg)

Figure 9: The comparison between samples with FID=40 and FID=20. The difference is visually imperceptible for human.

# C Model Evaluation

## C.1 Qualitative Analysis of Samples

First, we investigate the gap between ViT, GenViT and HybViT in Fig [8.](#page-15-1) We select two benchmark datasets CIFAR10 and ImageNet 32x32. It can be observed that the improvement of generation quality is relatively small after 10% training epochs. The difference is almost visually imperceptible for human between samples with FID=40 and FID=20 as shown in Fig. Hence, we think accelerating the convergence rates of our models is an interesting direction in the future.

Following the setting of JEM [\[23\]](#page-12-7), we conduct a qualitative analysis of samples on CIFAR10. We define an energy function of  $x$  as  $p_{\theta}(x) \propto E(x) = \log \sum_{y} e^{f_{\theta}(x)[y]} = LSE(\bar{f}_{\theta}(x))$ , the negative of the energy function in [\[43,](#page-13-19) [23\]](#page-12-7). We use a CIFAR10-trained HybViT model to generate 10,000 images from scratch, then feed them back into the HybViT model to compute  $E(x)$  and  $p(y|x)$ . We show the examples and distribution by class in Fig [10](#page-16-0) and Fig [11.](#page-16-1) We can observe that the worst examples of Plane can be completely blank. Additional HybViT generated class-conditional (best and worst) samples of CIFAR10 are provided in Figures [15](#page-19-0)[-24.](#page-22-0)

![](_page_16_Figure_4.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $E(\boldsymbol{x})$  $E(\boldsymbol{x})$  $p(y|\boldsymbol{x})$  $p(y|\boldsymbol{x})$ 

<span id="page-16-0"></span>Figure 10: HybViT generated class-conditional (best and worst) samples. Each row corresponds to 1 class.

![](_page_16_Figure_7.jpeg)

<span id="page-16-1"></span>Figure 11: Histograms (oriented horizontally for easier visual alignment) of  $x$  arranged by class for CIFAR10.

Table 10: Ablation Study of Data Augmentation on CIFAR10.

<span id="page-16-2"></span>

| Model  | Aug    | Acc $\% \uparrow$ | IS $\uparrow$ | $FID \perp$ |
|--------|--------|-------------------|---------------|-------------|
| ViT    | Strong | 96.5              | -             |             |
|        | Weak   | 87.1              |               |             |
|        | Strong | 95.9              | 7.68          | 26.4        |
| HybViT | Weak   | 84.6              | 7.85          | 24.9        |

## C.2 Data Augmentation

We study the effect of data augmentation. ViT is known to require a too large amount of training data and/or repeated strong data augmentations to obtain acceptable visual representation. Table [10](#page-16-2) compares the performance between strong augmented data and conventional Inception-style preprocessed(namely weak augmentation) data [\[59\]](#page-14-10). We can conclude that the strong data augmentation is really essential for high classification performance and the effect on generation is negative but tiny. Note that the data augmentation is only used for classification, and for DDPM, we don't apply any data augmentation.

# C.3 Out-of-Distribution Detection

Another useful OOD score function is the maximum probability from a classifier's predictive distribution:  $s_{\theta}(x) = \max_{y} p_{\theta}(y|x)$ . The results can be found in Table [11](#page-17-0) (bottom row).

| $s_{\boldsymbol{\theta}}(\boldsymbol{x})$ | Model             | <b>SVHN</b> | CIFAR <sub>10</sub> Interp | CIFAR <sub>100</sub> | CelebA |
|-------------------------------------------|-------------------|-------------|----------------------------|----------------------|--------|
|                                           | WideResNet [43]   | .91         |                            | .87                  | .78    |
|                                           | <b>IGEBM</b> [17] | .63         | .70                        | .50                  | .70    |
|                                           | <b>JEM [23]</b>   | .67         | .65                        | .67                  | .75    |
| $\log p_{\theta}(\boldsymbol{x})$         | $JEM++$ [67]      | .85         | .57                        | .68                  | .80    |
|                                           | <b>VERA</b> [24]  | .83         | .86                        | .73                  | .33    |
|                                           | $ImCD$ [16]       | .91         | .65                        | .83                  |        |
|                                           | ViT               | .93         | .93                        | .82                  | .81    |
|                                           | HybViT            | .93         | .92                        | .84                  | .76    |
|                                           | WideResNet        | .93         | .77                        | .85                  | .62    |
|                                           | <b>IGEBM</b> [17] | .43         | .69                        | .54                  | .69    |
| $\max_{y} p_{\theta}(y \boldsymbol{x})$   | <b>JEM [23]</b>   | .89         | .75                        | .87                  | .79    |
|                                           | $JEM++$ [67]      | .94         | .77                        | .88                  | .90    |
|                                           | ViT               | .91         | .95                        | .82                  | .74    |
|                                           | HybViT            | .91         | .94                        | .85                  | .67    |

<span id="page-17-0"></span>Table 11: OOD detection results. Models are trained on CIFAR10. Values are AUROC.

# C.4 Robustness

Given ViT models trained with different data augmentations, we can investigate their robustness since weak data augmentations are commonly used in CNNs. Table [12](#page-17-1) shows an interesting phenomena that HybViT with weak data augmentation is much robust than other models, especially under  $L_2$ attack. We suppose it's because the noising process feeds huge amount of noisy samples to HybViT, then HybViT learns from the noisy data implicitly to improve the flatness and robustness.

<span id="page-17-1"></span>Table 12: Classification accuracies when models are under  $L_{\infty}$  and  $L_2$  PGD attacks with different  $\epsilon$ 's. All models are trained on CIFAR10.

| Model      | Clean $(\% )$ | $L_{\infty} \epsilon = 1/255$ | $\overline{2}$ | 4        | 8    | 12  | 16       | 22       | 30       |
|------------|---------------|-------------------------------|----------------|----------|------|-----|----------|----------|----------|
| ViT        | 96.5          | 70.8                          | 46.7           | 21<br>-7 | 7.0  | 1.4 | 0.1      | $\theta$ | $\Omega$ |
| - Weak Aug | 87.1          | 67.3                          | 41.8           | 14.8     | 1.4  | 0.1 | $\Omega$ | 0        | $\Omega$ |
| HybViT     | 95.9          | 70.4                          | 48.0           | 21.9     | 5.5  | 1.3 | 0.3      | 0        | 0        |
| - Weak Aug | 84.6          | 71.3                          | 55.6           | 30.3     | 6.7  | 0.6 | 0.1      | $\theta$ | $\Omega$ |
| Model      | Clean $(\%)$  | $L_2 \epsilon = 50/255$       | 100            | 150      | 200  | 250 | 300      | 350      | 400      |
| ViT        | 96.5          | 52.3                          | 9.2            |          | 0.3  | 0.1 | 0.1      | $\Omega$ | $\Omega$ |
| - Weak Aug | 87.1          | 53.9                          | 21.4           | 5.5      | 1.0  | 0.1 | $\Omega$ | $\Omega$ | $\Omega$ |
| HybViT     | 95.9          | 58.7                          | 16.3           | 3.4      | 1.0  | 0.2 | 0.1      | 0.1      | $\Omega$ |
| - Weak Aug | 84.6          | 65.8                          | 42.3           | 25.7     | 13.2 | 6.4 | 3.4      | 1.5      | 0.7      |

# C.5 Calibration

Figures in [12](#page-18-0) provide a comparison of ViT and HybViT with the baselines WRN and JEM, and also corresponding ViTs trained without strong data augmentations. It can be observed that strong data augmentations can better calibrate the predictions of ViT and HybViT, but further make them under-confident.

![](_page_18_Figure_2.jpeg)

<span id="page-18-0"></span>Figure 12: Calibration results on CIFAR10. The smaller ECE is, the better.

## C.6 Training Speed

We further report the empirical training speeds of our models and baseline methods for ImageNet 32x32. Our methods are memory efficient since it only requires a single GPU, and much faster.

| Table 13: Run-time comparison on ImageNet 32x32. All experiments are performed on a single GPU |  |  |  |
|------------------------------------------------------------------------------------------------|--|--|--|
| for 100 epochs.                                                                                |  |  |  |

| Model         | NoP(M)             | Runtime |  |
|---------------|--------------------|---------|--|
| ViT           | 11.6               | 3 days  |  |
| GenViT        | 11.6               | 2 days  |  |
| HybViT        | 11.6               | 5 days  |  |
| <b>IGEBM</b>  | 32 GPUs for 5 days |         |  |
| <b>KL-EBM</b> | 8 GPUs for 3 days  |         |  |

# D Additional Generated Samples

Additional generated samples of CIFAR10, CIFAR100, ImageNet 32x32, TinyImageNet, STL10, and CelebA 128 are provided in Figure [13.](#page-19-1) We further provide some generated images for ImageNet 128x128 and vanilla ImageNet 224x224 are shown in [14.](#page-19-2) , The patch size are set as 8 and 14 for ImageNet 128 and 224 respectively. Similar to previous discussion about patch size, we find the generation quality is very low. Due to limited computation resource and low generation quality, we only show a preliminary generative results on ImageNet-128 and vanilla ImageNet 224x224.

| - 1               |                  |   |
|-------------------|------------------|---|
| ∸                 |                  |   |
|                   | ļ                |   |
| $\bullet$         |                  |   |
| <b>Page</b>       |                  | е |
|                   |                  |   |
|                   |                  |   |
| -                 | B                | B |
| B<br><b>Borne</b> | $\mathbf \sigma$ | n |

Figure 13: Additional generated images on benchmark datasets

<span id="page-19-1"></span>

![](_page_19_Picture_4.jpeg)

![](_page_19_Picture_5.jpeg)

(d) TinyImagenet (e) STL10 (f) CelebA 128

![](_page_19_Picture_8.jpeg)

![](_page_19_Picture_10.jpeg)

![](_page_19_Picture_12.jpeg)

(g) ImageNet 128x128 (h) ImageNet 224x224

<span id="page-19-2"></span>Figure 14: Generated Images

![](_page_19_Figure_15.jpeg)

(a) Samples with highest  $p(\bm{x})$ (b) Samples with lowest  $p(\bm{x})$ (c) Samples with highest  $p(y|\bm{x})$ (d) Samples with lowest  $p(y|\bm{x})$ 

<span id="page-19-0"></span>Figure 15: HybViT generated class-conditional samples of Plane

![](_page_20_Picture_0.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\boldsymbol{x})$  $p(y|\boldsymbol{x})$ 

Figure 16: HybViT generated class-conditional samples of Car

![](_page_20_Figure_3.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\boldsymbol{x})$  $p(y|\boldsymbol{x})$ 

Figure 17: HybViT generated class-conditional samples of Bird

![](_page_20_Figure_6.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\bm{x})$  $p(y|\bm{x})$ 

Figure 18: HybViT generated class-conditional samples of Cat

![](_page_20_Figure_9.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(x)$  $p(x)$  $p(y|\bm{x})$  $p(y|\boldsymbol{x})$ 

Figure 19: HybViT generated class-conditional samples of Deer

![](_page_21_Picture_0.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\boldsymbol{x})$  $p(y|\boldsymbol{x})$ 

Figure 20: HybViT generated class-conditional samples of Dog

![](_page_21_Figure_3.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\boldsymbol{x})$  $p(y|\boldsymbol{x})$ 

Figure 21: HybViT generated class-conditional samples of Frog

![](_page_21_Figure_6.jpeg)

(a) Samples with highest (b) Samples with lowest (c) Samples with highest (d) Samples with lowest  $p(\boldsymbol{x})$  $p(\boldsymbol{x})$  $p(y|\bm{x})$  $p(y|\bm{x})$ 

Figure 22: HybViT generated class-conditional samples of Horse

![](_page_21_Figure_9.jpeg)

Figure 23: HybViT generated class-conditional samples of Ship

![](_page_22_Picture_0.jpeg)

(a) Samples with highest  $p(\bm{x})$ (b) Samples with lowest  $p(\bm{x})$ (c) Samples with highest  $p(y|\bm{x})$ (d) Samples with lowest  $p(y|\bm{x})$ 

<span id="page-22-0"></span>Figure 24: HybViT generated class-conditional samples of Truck