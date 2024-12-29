import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt 


category=['dogs','cats']

my_path = r'F:\Image_processing_mit_Python\image_classification_KNN\coursera\training-an-image-classifier'

train_data=[]
train_label=[]

for i in category:
    file = os.path.join(my_path, i)
    label = category.index(i)
    for j in os.listdir(file):
        image_path = os.path.join(file,j)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (32, 32))
        image = image.flatten()
        train_data.append(image)
        train_label.append(label)
        
        
train_images = np.array(train_data).astype('float32')
train_labels = np.array(train_label)
train_labels = train_labels.astype(int)
train_labels = train_labels.reshape((train_labels.size,1))


test_size = 0.2
train_samples, test_samples, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=test_size, random_state=0)
        
knn = cv2.ml.KNearest_create()
knn.train(train_samples, cv2.ml.ROW_SAMPLE, train_labels)

## get different values of K
k_values = [1,2, 3, 4, 5]
k_result = []
for k in k_values:
    ret,result,neighbours,dist = knn.findNearest(test_samples,k=k)
    k_result.append(result)
flattened = []
for res in k_result:
    flat_result = [item for sublist in res for item in sublist]
    flattened.append(flat_result)
    
    
accuracy_res = []
con_matrix = []
## we will use a loop because we have multiple value of k
for k_res in k_result:
    label_names = [0, 1]
    cmx = confusion_matrix(test_labels, k_res, labels=label_names)
    con_matrix.append(cmx)
    ## get values for when we predict accurately
    matches = k_res==test_labels
    correct = np.count_nonzero(matches)
    ## calculate accuracy
    accuracy = correct*100.0/result.size
    accuracy_res.append(accuracy)
## stor accuracy for later when we create the graph
res_accuracy = {k_values[i]: accuracy_res[i] for i in range(len(k_values))}
list_res = sorted(res_accuracy.items())
        

## for each value of k we will create a confusion matrix
for cm in con_matrix:
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1,0])
    cm_disp.plot()
      
k_best = max(list_res,key=lambda item:item[1])[0]
print(k_best)
    
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk


def classify_image(image_path, knn_model, k_best, categories=['dogs', 'cats']):
    
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Image not found."
    
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (32, 32))
    image = image.flatten().astype('float32')
    
    # Reshape to match the input format for KNN (1 sample, 1024 features)
    image = image.reshape(1, -1)
    
    
    ret, result, neighbours, dist = knn_model.findNearest(image, k=k_best)
    
   
    label = int(result[0][0])  
    return categories[label]


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path).resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk  # Keep a reference to avoid garbage collection
        result = classify_image(file_path, knn, k_best)
        result_label.config(text=f"Result: {result}")


root = tk.Tk()
root.title("Cat vs Dog Classifier")


panel = Label(root)
panel.pack()

btn = tk.Button(root, text="Upload Image", command=browse_file)
btn.pack()

result_label = tk.Label(root, text="Result: ", font=("Helvetica", 16))
result_label.pack()


root.mainloop()

