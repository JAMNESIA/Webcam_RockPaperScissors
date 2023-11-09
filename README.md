# Rock Paper Scissors trough your webcam!

I made this project for the minor AI at Avans Hogeschool Breda. The game is made with python and the library OpenCV. It uses a CNN to detect the hand gestures. The CNN was made with pytorch and trained with the [Rock Paper Scissors Dataset](https://www.kaggle.com/drgfreeman/rockpaperscissors).

## Installation

Run the notebook and train the model. Move the file `best_model.pt` to the app folder.

Go to the app folder.

```bash
cd app
```

Install the requirements.

```bash
pip install -r requirements.txt
```

Run the app.

```bash
python rock_paper_scissors.py
```

## Usage

When you run the app you will see a window with the webcam feed. You can press the spacebar to take a picture. The CNN will predict the gesture and show the result in the window. You can press the spacebar again to take another picture. You can press the `q` key to close the app.
