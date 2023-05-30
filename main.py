# Importing the needed modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import cv2
import torchvision.transforms as T
from torchvision import transforms
import os







def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.input = conv_block(in_channels, 64)

        self.conv1 = conv_block(64, 64, pool=True)
        self.res1 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop1 = nn.Dropout(0.5)
        
        self.conv2 = conv_block(64, 64, pool=True)
        self.res2 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop2 = nn.Dropout(0.5)
        
        self.conv3 = conv_block(64, 64, pool=True)
        self.res3 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop3 = nn.Dropout(0.5)        
        
        self.conv4 = conv_block(64, 64, pool=True)
        self.res4 = nn.Sequential(conv_block(64, 32), conv_block(32, 64))
        self.drop4 = nn.Dropout(0.5)
        
        self.classifier = nn.Sequential(nn.MaxPool2d(6), 
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))
        
    def forward(self, xb):
        out = self.input(xb)

        out = self.conv1(out)
        out = self.res1(out) + out
        out = self.drop1(out)
        
        out = self.conv2(out)
        out = self.res2(out) + out
        out = self.drop2(out)
        
        out = self.conv3(out)
        out = self.res3(out) + out
        out = self.drop3(out)
        
        out = self.conv4(out)
        out = self.res4(out) + out
        out = self.drop4(out)
        
        return self.classifier(out)

model=ResNet(3,7)


   
model.load_state_dict(torch.load('fer_modelDict_15.pt',map_location=torch.device('cpu')))


model.eval()








emotion=['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']




transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomResizedCrop(128),
        # transforms.Grayscale(),
        
        # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))      
    ])















from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


camera = cv2.VideoCapture(0)

    
@app.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    pre_pred=0
    frame_number=0
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for person,(x,y,w,h) in enumerate(faces,1):
                if person==1:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    pred_image=frame[y:y+h, x:x+w]
                    
                    img = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)

                    image_tenor=torch.unsqueeze(transform(im_pil),0)
                    pred_tensor=model(image_tenor)


                    frame_number+=1
                    if frame_number==2:
                        pred=torch.argmax(pred_tensor,1)
                        pre_pred=pred
                        cv2.putText(frame, 
                                    f'{emotion[pred]}', 
                                    (50, 50), 
                                    font, 1, 
                                    (0, 255, 255), 
                                    2, 
                                    cv2.LINE_4)
                        frame_number=0
                    else:
                        cv2.putText(frame, 
                                    f'{emotion[pre_pred]}', 
                                    (50, 50), 
                                    font, 1, 
                                    (0, 255, 255), 
                                    2, 
                                    cv2.LINE_4)
                else:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    pred_image=frame[y:y+h, x:x+w]
                    
                    img = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)

                    image_tenor=torch.unsqueeze(transform(im_pil),0)
                    pred_tensor=model(image_tenor)
                    cv2.putText(frame, 
                                f'Too Many To Processe', 
                                (50, 50), 
                                font, 1, 
                                (0, 255, 255), 
                                2, 
                                cv2.LINE_4)




                    # Convert the frame to JPEG format
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()

                    yield (frame)
        _, buffer = cv2.imencode('.jpg', frame)        
        frame = buffer.tobytes()
        yield (frame)



@app.route('/video_feed')
def video_feed():
    def generator():
        for frame in gen_frames():
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

    return Response(generator(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)