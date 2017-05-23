def get_data(data_path):
    import numpy as np
    import cv2
    import csv
    images=[]
    angels=[]
    correction_angle=0.2
    csv_file_path=data_path+"driving_log.csv"
    
    
    with open(csv_file_path) as csv_file:
        lines=csv.reader(csv_file)
        for line in lines:
            for i in range(3):
                tokens=line[i].split("\\")
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
    import matplotlib.pyplot as plt
    x_train,y_train=get_data('./abdo/')
    print(x_train.shape)
    print(y_train.shape)
    #hsv=cv2.cvtColor(x_train[0,:,:,:],cv2.COLOR_BGR2HSV)
    shape_=x_train[0].shape
    #print(x_train[0].shape)
    #exit()
    f,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(20,10))
    cropped_center=x_train[0][70:shape_[0]][:][:]
    #print(cropped_center.shape)
    ax1.imshow(cv2.cvtColor((x_train[1][70:shape_[0]-30][:][:]),cv2.COLOR_BGR2RGB))
    ax1.set_title("Left camera")
    ax2.imshow(cv2.cvtColor((x_train[0][70:shape_[0]-30][:][:]),cv2.COLOR_BGR2RGB))
    ax2.set_title("Center camera")
    ax3.imshow(cv2.cvtColor((x_train[2][70:shape_[0]-30][:][:]),cv2.COLOR_BGR2RGB))
    ax3.set_title("Right camera")
    plt.show()








