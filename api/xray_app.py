from fastapi import FastAPI,UploadFile,File
import torch
import io
from PIL import Image
from torchvision import transforms
from api.label_map  import labels

#Load Model
model=torch.load('model/resnet_cnn_model.pt',weights_only=False, map_location="cpu")
model.eval()

# Preprocessing: same as you used in training
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

app=FastAPI(title='X-RAY Classifier')

@app.get("/")
def root():
    return {"message": "X-RAY Classifier API is running. Use POST /predict."}
@app.get("/health")
def health_check():
    return{
        'status':'OK',
        'version':'1.0.0',
        'model_loaded':model is not None
    }

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
        predicted_label=labels[int(pred.item())]
        confidence_score=float(conf.item())
        predicted_class= int(pred.item()),

    return {
        "predicted_class": predicted_class,
        "confidence_score": confidence_score,
        "predicted_label": predicted_label
    }
