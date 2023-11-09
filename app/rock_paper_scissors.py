import cv2
import torch
import torch.nn as nn
import numpy as np
import random
from torchvision import transforms
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 512)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame)
    tensor_frame = transform(pil_image)
    tensor_frame = tensor_frame.unsqueeze(0)
    return tensor_frame

def predict_move(model, frame):
    with torch.no_grad():
        prediction = model(frame)
    _, predicted_class = torch.max(prediction, 1)
    return class_to_str[predicted_class.item()]

def get_computer_move():
    return random.choice(['Rock', 'Paper', 'Scissors'])

def determine_winner(user_move, app_move):
    if user_move == app_move:
        return 'It\'s a Draw!', 0
    elif (user_move == 'Rock' and app_move == 'Scissors') or \
        (user_move == 'Paper' and app_move == 'Rock') or \
        (user_move == 'Scissors' and app_move == 'Paper'):
        return 'You Win!', 1
    else:
        return 'App Wins!', -1

def display_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def main():
    model = load_model('best_model.pth')
    cap = cv2.VideoCapture(0)
    user_score, app_score = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        user_move = predict_move(model, preprocess_frame(frame))
        display_text(frame, f'Your Move: {user_move}', (10, 50), (0, 255, 0))
        cv2.imshow('Rock Paper Scissors', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            app_move = get_computer_move()
            result, score = determine_winner(user_move, app_move)
            user_score += max(score, 0)
            app_score += max(-score, 0)

            display_text(frame, f'Your Move: {user_move}', (10, 50), (0, 255, 0))
            display_text(frame, f'App Move: {app_move}', (10, 100), (0, 0, 255))
            display_text(frame, result, (10, 150))
            display_text(frame, f'Score: You {user_score} - App {app_score}', (10, 200), (255, 255, 0))
            cv2.imshow('Rock Paper Scissors', frame)
            cv2.waitKey(3000)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class_to_str = {0: 'Rock', 1: 'Paper', 2: 'Scissors'}
    main()
