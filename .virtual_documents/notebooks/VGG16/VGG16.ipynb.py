# Importing necessary packages

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os


# Construct the parser and initialise arguments
parser = argparse.ArgumentParser(description='COVID-CNN')

parser.add_argument('--input_size', default=224, type=int, help='Dimensions of the input image')
parser.add_argument('--n_channels', default=3, type=int, help='Number of channels in the image')
parser.add_argument('--train_file', default='train_split.txt', type=str, help='Name of train metadata file')
parser.add_argument('--test_file', default='test_split.txt', type=str, help='Name of test metadata file')
parser.add_argument('--data_dir', default='data', type=str, help='Path to data folder containing datasets')
parser.add_argument('--train_data_dir', default='train', type=str, help='Path to folder containing training dataset')
parser.add_argument('--test_data_dir', default='test', type=str, help='Path to folder containing testing dataset')
parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to train for')
parser.add_argument('--bs', default=8, type=int, help='Batch size')
parser.add_argument('--lr', default=0.0002, type=float, help='Learning rate')
parser.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to save loss/accuracy plot")
parser.add_argument("-mp", "--model_plot", type=str, default="model_plot.png", help="Path to save model's plot")

args, unknown = parser.parse_known_args()

# Declaring constants
EPOCHS = args.epochs
BS = args.bs
LR = args.lr


# Reading training dataset csv and extracting labels
df_train = pd.read_csv(args.train_file, sep=' ', header=0)
train_labels = df_train['class']

# Reading testing dataset csv and extracting labels
df_test = pd.read_csv(args.test_file, sep=' ', header=0)
test_labels = df_test['class']

# Extracting unique classes
classes = test_labels.unique()

# Plotting distribution of classes in training dataset
unique, counts = np.unique(test_labels, return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency Training Dataset')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()

# Plotting distribution of classes in testing dataset
unique, counts = np.unique(test_labels, return_counts=True)
plt.bar(unique, counts)

plt.title('Class Frequency Testing Dataset')
plt.xlabel('Class')
plt.ylabel('Frequency')

plt.show()


# Initialising data augmentation object to improve the dataset
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    os.path.join(args.data_dir, args.train_data_dir),
    target_size=(args.input_size, args.input_size),
    color_mode="rgb",
    batch_size=BS,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=14
)

validation_generator = train_data_gen.flow_from_directory(
    os.path.join(args.data_dir,args.train_data_dir),
    target_size=(args.input_size, args.input_size),
    color_mode="rgb",
    batch_size=BS,
    class_mode="categorical",
    subset='validation',
    shuffle=True,
    seed=14
)

test_generator = test_data_gen.flow_from_directory(
    os.path.join(args.data_dir,args.test_data_dir),
    target_size=(args.input_size, args.input_size),
    color_mode="rgb",
    batch_size=1,
    class_mode="categorical",
    shuffle=False
)


# Loading VGG16 model trained on imagenet without head
baseModel = VGG16(
    weights="imagenet", include_top=False,
    input_tensor=Input(shape=(args.input_size, args.input_size,
                              args.n_channels))
)

# Constructing the head for classification (to be trained)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)

# Fixing the weights of base model
for layer in baseModel.layers:
    layer.trainable = False

model = Model(inputs=baseModel.input, outputs=headModel)

# Saving a plot of the model
plot_model(model, to_file=args.model_plot, show_shapes=True, show_layer_names=True)


# Compiling the model
opt = Adam(learning_rate=LR)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

# Training model
print("[INFO] Training model")
H = model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // BS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BS,
    epochs=EPOCHS)


# Predicting on testing data
print("[INFO] Predicting on testing data")
test_file_names = test_generator.filenames
nb_samples = len(test_file_names)
test_generator.reset()
pred = model.predict_generator(test_generator, steps=nb_samples, verbose=1)
pred = np.argmax(pred, axis=1)

# Getting class label map
label_map = (test_generator.class_indices)
for c in classes:
    label_map.update({label_map.get(c):c})
    label_map.pop(label_map.get(label_map.get(c)))
temp = []
for i in range(len(label_map)):
    temp.append(label_map.get(i))
label_map = temp
del(temp)

# Printing classification report
print(classification_report(test_generator.classes, pred, target_names=label_map))


# Generating confusion matrix and calculating: accuracy, sensitivity and specificity
cm = confusion_matrix(test_generator.classes, pred)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1] + cm[2, 2]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2])
TN = cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2]
FP = cm[1, 0] + cm[2, 0]
specificity = TN / (TN + FP)

# Printing found values
print(cm)
print("Overall Accuracy: {:.4f}".format(acc))
print("COVID Sensitivity: {:.4f}".format(sensitivity))
print("COVID Specificity: {:.4f}".format(specificity))


# Plotting training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on COVIDx Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
args.plot = "plot_vgg16_sgd.png"
plt.savefig(args.plot)
