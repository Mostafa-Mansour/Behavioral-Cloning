def get_data(data_path):
    import numpy as np
    import cv2
    import csv
    images=[]
    angels=[]
    correction_angle=0.2
    csv_file_path=data_path+"driving_log.csv"
    #print(csv_file_path)
    #exit()

    with open(csv_file_path) as csv_file:
        lines=csv.reader(csv_file)
        for line in lines:
            for i in range(3):
                tokens=line[i].split("/")
                img_path=data_path+"IMG/"+tokens[-1]
                img=cv2.imread(img_path)
                #gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                images.append(img)
                
            angle=float(line[3])
            angels.append(angle)
            angels.append(angle+correction_angle)
            angels.append(angle-correction_angle)

    x_train=np.array(images)
    y_train=np.array(angels)
    return x_train,y_train

if __name__=='__main__':
    import numpy as np
    import cv2
    import csv
    x_train,y_train=get_data('/home/ros-indigo/Desktop/behavioral_cloning/')
    print(x_train.shape)
    print(y_train.shape)







