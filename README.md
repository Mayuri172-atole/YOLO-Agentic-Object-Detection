# YOLO-Agentic-Object-Detection
 Advanced YOLO-based object detection, segmentation, and pose estimation with Agentic AI integration using AutoGen, enabling multimodal image analysis and conversational interaction.

Got it — you want to prepare this **YOLO + Agentic YOLO pipeline** for GitHub with a proper repo name, description, and README.
Here’s the setup for your GitHub repo.

---

## **Repository Name**

`YOLO-Agentic-Object-Detection`

---

## **Short Description**

Advanced YOLO-based object detection, segmentation, and pose estimation with Agentic AI integration using AutoGen, enabling multimodal image analysis and conversational interaction.

---

# YOLO & Agentic Object Detection

This repository demonstrates **object detection, segmentation, and pose estimation** using the latest YOLO models, enhanced with **Agentic AI** capabilities via AutoGen. The system allows multimodal interactions — processing images and answering queries conversationally.
Features
- **YOLO Integration**
  - YOLOv11 object detection
  - YOLO segmentation
  - YOLO pose estimation
- **Agentic AI**
  - AutoGen-powered assistant for computer vision tasks
  - Natural language interaction with image processing results
  - Termination conditions for controlled responses
- **Multimodal Input**
  - Processes text + images
  - Detects objects, segments them, and estimates poses
- **Custom Tools**
  - `yolo_tool` for structured detection output
  - Integration with OpenAI-compatible chat models (Gemini API in example)

## 🛠 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/YOLO-Agentic-Object-Detection.git
cd YOLO-Agentic-Object-Detection

pip install ultralytics
pip install autogen-agentchat==0.5.7 autogen-ext[openai] python-dotenv autogen-core pydantic
pip install pillow
````
 Usage

YOLO Object Detection

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model("/path/to/image.jpg")
for result in results:
    result.show()
```

YOLO Segmentation Example

```python
model = YOLO("yoloe-v8s-seg.pt")
names = ["bicycle", "person"]
model.set_classes(names, model.get_text_pe(names))
results = model("/path/to/image.jpg")
for result in results:
    result.show()
```

 Agentic YOLO with AutoGen

```python
from ultralytics import YOLO
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

# Load YOLO model
yolo_model = YOLO("yolo11n.pt")

# Define YOLO Tool
def yolo_tool(image_path: str):
    results = yolo_model(image_path)
    results[0].show()
    classes = list({results[0].names[int(box.cls)] for box in results[0].boxes})
    detections = [{
        "class_id": int(box.cls),
        "class_name": results[0].names[int(box.cls)],
        "confidence": round(float(box.conf), 2)
    } for box in results[0].boxes]
    return {"detected_classes": classes, "detailed_detections": detections}
```

 Multi-Agent Setup

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

team = RoundRobinGroupChat(
    participants=[yolo_agent],
    max_turns=4,
    termination_condition=TextMentionTermination("TERMINATE")
)
```

 Multimodal Input

```python
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from PIL import Image as PILImage

pil_image = PILImage.open("/path/to/image.jpg")
img = Image(pil_image)

multi_modal_message = MultiModalMessage(
    content=["Describe objects and count people", img],
    source="user"
)
```

 Requirements

* Python 3.9+
* ultralytics
* autogen-agentchat
* autogen-ext\[openai]
* pydantic
* pillow

Example Outputs

* **Detection results** with bounding boxes
* **Segmentation masks** for objects
* **Pose estimation** for humans
* **Natural language summaries** from the Agent

