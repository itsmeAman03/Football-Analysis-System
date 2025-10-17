## ⚽ Football Object Detection using YOLOv11-Large

This model is designed to detect key entities on a football field:  
- 🧤 **Goalkeeper**  
- 🧍 **Player**  
- ⚪ **Ball**  
- 🧑‍⚖️ **Referee**

Built on the powerful **YOLOv11-Large** architecture with transfer learning, the first half of the layers were frozen to retain general features while fine-tuning the rest on football-specific data.

---

### 🔧 Configuration

- **Architecture**: YOLOv11-Large
- **Frozen Layers**: 2/3rd
- **Pre-trained Weights**: Used
- **Epochs Trained**: 50
- **Learning Rate**: ~3.73e-5 (pg0/pg1/pg2)
- **Train Duration**: 539 minutes
- **Model Size**: 25.3M parameters, 87.3 GFLOPs

---

### 📊 Evaluation Metrics

> *Values are reported as: Final / Best during training*

| Metric                     | Final      | Best (Min)    |
|----------------------------|------------|---------------|
| **mAP@50-95 (val)**        | 0.53167    | 0.53167       |
| **mAP@50 (val)**           | 0.81680    | 0.81737       |
| **Precision (val)**        | 0.90306    | 0.91265       |
| **Recall (val)**           | 0.74100    | 0.78776       |
| **Box Loss (train)**       | 0.84756    | 0.83256       |
| **Class Loss (train)**     | 0.41376    | 0.40873       |
| **DFL Loss (train)**       | 0.80578    | 0.80386       |
| **Box Loss (val)**         | 0.86958    | 0.86958       |
| **Class Loss (val)**       | 0.42202    | 0.42202       |
| **DFL Loss (val)**         | 0.81992    | 0.81951       |

---

### 🧪 Notes

- The model converged smoothly with no signs of overfitting.
- High **precision (0.91)** and solid **recall (0.78)** suggest strong localization and classification.
- Model complexity (87.3 GFLOPs) makes it suitable for powerful edge or server-side deployment.

---

### 🚀 Potential Use Cases

- **Real-time football analytics**
- **Referee decision assist systems**
- **Player movement tracking & ball possession**
- **Broadcast visuals enhancement**

