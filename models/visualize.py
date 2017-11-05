import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_image_list(imgs):
    
    imgnum = len(imgs)
    
    plt.ion()
    column_num = 5
    
    row_num = 1 + imgnum//column_num
    
    
    for row in range(row_num):
        # plt.figure(figsize = (10,17))
        plt.figure(figsize = (15,20))
        gs1 = gridspec.GridSpec(1,column_num)
        gs1.update(wspace=0.025, hspace=0.05)
        for col in range(column_num):
            idx = col + row*column_num
            if(imgnum <= idx):
                # print("too small matched: {0}, {1}".format(len(cat_img_paths), i))
                return
        
            try:
                ax1 = plt.subplot(gs1[col])
                plt.axis('on')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_aspect('equal')
            
            
                image = plt.imread(imgs[idx])
                plt.imshow(image)
            except:
                    pass
        plt.show()