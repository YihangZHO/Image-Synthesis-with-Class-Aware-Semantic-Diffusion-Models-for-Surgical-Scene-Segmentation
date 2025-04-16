# Image-Synthesis-with-Class-Aware-Semantic-Diffusion-Models-for-Surgical-Scene-Segmentation

Official code for the paper: **Image Synthesis with Class-Aware Semantic Diffusion Models for Surgical Scene Segmentation**

---

## 📁 Datasets & Resources

- **CholecSeg8K train/test sets**  
  [Download here](https://drive.google.com/file/d/1U-RcSu_pui6sOk15ldu7pLPJenIyWP_E/view?usp=sharing)

- **CholecSeg8K colorful masks**  
  [Download here](https://drive.google.com/file/d/1pkzxc5g0mVtw4jLuRrdx1YnlFQfU2891/view?usp=sharing)

- **Image generation checkpoint**  
  [Download from Google Drive](https://drive.google.com/drive/folders/1Mh3PqbO6Rv9twxbnve1Snv_YtH35z8rF?usp=sharing)

---

## 🧪 Code Usage

- **Training**: See [`train.ipynb`](./train.ipynb)  
- **Generation / Testing**: See [`test.ipynb`](./test.ipynb)

---

## 🙏 Acknowledgements

Our model is built upon the following excellent open-source projects:

- 🤗 [Hugging Face Diffusers](https://github.com/huggingface/diffusers) – the baseline of this project and code for mask synthesis is based on its implementation of Stable Diffusion.
- 🧠 [Semantic Diffusion Model by Weilun Wang](https://github.com/WeilunWang/semantic-diffusion-model) – semantic-aware conditioning framework

We thank the authors for their contributions to the open-source community.

---

Feel free to open an issue or contact us if you have any questions.  
Pull requests are welcome!
