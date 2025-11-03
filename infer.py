import torch, cv2, matplotlib.pyplot as plt
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

img = cv2.imread("test_blueprint.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inp = torch.tensor(img_rgb.transpose(2,0,1)/255.0, dtype=torch.float32).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = torch.sigmoid(model(inp))[0,0].cpu().numpy()

plt.subplot(1,2,1); plt.imshow(img_rgb)
plt.subplot(1,2,2); plt.imshow(pred>0.5, cmap="gray")
plt.show()
