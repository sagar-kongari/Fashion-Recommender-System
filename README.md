# 🥼 Fashion recommender system using RESNET50

## 💮 Introduction
A recommendation system is a class of machine learning that uses data to help predict, narrow down, and find what people are looking for among an exponentially growing number of options. In general, these algorithms are aimed at suggesting relevant items to users, like movies to watch or products to buy. There are many types of recommendation systems, the two main are:
- Collaborative based filtering
- Content based filtering

> This project uses “A content-based goods/apparels image recommendation system” as a baseline model to generate similar products. The same content-based image retrieval technique is extended to Deep Learning models and architectures to achieve better results and generate most similar recommendations.

## 

```python
# function to convert image to numeric array
def feature_array(pic_path, model):
    pic = image.load_img(pic_path, target_size=(224,224))
    pic_array = image.img_to_array(pic)
    expand_pic_array = np.expand_dims(pic_array, axis=0)
    final_pic = preprocess_input(expand_pic_array)
    result = model.predict(final_pic).flatten()
    norm_result = result/norm(result)
    return norm_result
```
```mermaid
---
title: FlowChart
---
flowchart LR
  id1[Load\nImage] --> id2[Convert\nImage to an Array] --> id3[Expand Dimensions\nfor batch input] --> id4[Align\nRGB to BGR] --> id5[Flatten] --> id6[Normalization]
```
