
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch

from face_detector import FaceDetector




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# تموذج التعرف على الوجه غير المقنع
NORMAL_RESNET = InceptionResnetV1(pretrained='vggface2', device=DEVICE).eval()
# نموذج التعرف على الوجه
MASKED_MODEL = torch.load('arcface1.pt', map_location=DEVICE)

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

FACE_DETECTOR = FaceDetector(
        factor=0.7,
        min_face_size= 400,
        keep_all=True,
        device =DEVICE)

TRANSFORM = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128, 128)),      
                            transforms.ToTensor()
                            ])






