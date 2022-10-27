import matplotlib.pyplot as plt
import numpy as np
import os
"""
threshold_data = np.genfromtxt('Results/tandir_kitti_v1.csv', delimiter=',')

threshold_data = threshold_data[:40,:]

plt.plot(threshold_data[:,0],threshold_data[:,1], 'ro')
plt.plot(threshold_data[:,0],threshold_data[:,2], 'bo')
plt.plot(threshold_data[:,0],threshold_data[:,3], 'go')
plt.plot(threshold_data[:,0],threshold_data[:,4], 'yo')
plt.legend(['True Positive', 'False Positive', 'False Negative', 'True Negative'])
plt.title("Confusion Matrix")


threshold_data[1:,0] = np.linspace(0,99,99)/1000
plt.plot(threshold_data[1:,0],threshold_data[1:,5], 'ro')
plt.plot(threshold_data[1:,0],threshold_data[1:,6], 'bo')
plt.plot(threshold_data[1:,0],threshold_data[1:,7], 'go')
plt.legend(['Precision', 'Sensitivity', 'Specificity'])
plt.title("Confusion Matrix Evaluation")

plt.xlabel('Threshold [rad]')
# plt.xlabel('Ellipsoid Size in Pixels')
# plt.xlabel('Point Quality')
#plt.ylabel('Points Registered')
plt.grid()

#plt.savefig('Results/Conf_Matrix_bbox_var3.png', dpi = 300)
plt.savefig('Results/Conf_Eval_tandir_kitti_v1.png', dpi = 300)
plt.show()
"""
def flatten(t):
    return [item for sublist in t for item in sublist]

load_pattern = flatten([[
    f'results{"/*" * i}.csv',
    f'results{"/*" * i}.tif',
] for i in range(1,2)])

filenames = next(os.walk('results/'), (None, None, []))[2]  # [] if no file

for i, file in enumerate(filenames):
    threshold_data = np.genfromtxt('results/' + filenames[i], delimiter=',')
    print(file)
    # naming scheme
    file = file[:-4]
    file = file.split(sep="_")

    if file[0] == "raytrace":
        file[0] = "Ray-Trace"
        plt.xlabel("Ellipsoid Circumference [pixels]")
    else:
        file[0] = "Directional Change"
        plt.xlabel("Threshold for Tangent of Directional Change [tan(" + r'$\gamma$' + ")]")

    if file[1] == "davis":
        file[1] = "DAVIS"
    else:
        file[1] = "KITTI"

    plt.ylabel("Accuracy")


    #Plot of confusionmatrix evaluation

    plt.plot(threshold_data[1:, 0], threshold_data[1:, 5], 'ro')
    plt.plot(threshold_data[1:, 0], threshold_data[1:, 6], 'bo')
    plt.plot(threshold_data[1:, 0], threshold_data[1:, 7], 'go')
    plt.legend(['Precision', 'Sensitivity', 'Specificity'], loc="center right")

    if threshold_data[1,10] == 0:
        plt.savefig("plots/CME_" + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + " - EMC.png",
                    dpi=300, bbox_inches="tight")
    else:
        plt.savefig("plots/CME_" + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + ".png",
                    dpi=300, bbox_inches="tight")

    # plt.savefig("plots/CME_" + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]), dpi=300)

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    """
    #Time analysis
    time_mean = np.mean(threshold_data[:,8:],axis=0)
    total_time = time_mean[5]
    time_mean[5] = time_mean[5]-time_mean[0]-time_mean[1]-time_mean[2]-time_mean[3]-time_mean[4]
    porcent = 100. * time_mean / time_mean.sum()

    if file[2] == "motorbike":
        time_mean = time_mean/40
        average_time = total_time/40
    if file[2] == "car-shadow":
        time_mean = time_mean / 35
        average_time = total_time/35
    if file[2] == "car-turn":
        time_mean = time_mean/70
        average_time = total_time/70
    if file[2] == "bus":
        time_mean = time_mean/70
        average_time = total_time/70
    else:
        time_mean = time_mean / 40
        average_time = total_time/40
    switch = 0

    if time_mean[6] == 0:
        labels = ['Preprocessing', 'Feature Extraction', 'Algorithm', 'Classification', 'Other']
        time_mean = np.delete(time_mean, 2, 0)
        porcent = 100. * time_mean / time_mean.sum()
        switch = 1

        patches, texts = plt.pie(time_mean, startangle=90, radius=1.2)
        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, porcent)]

        plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1.04, 0.5),
                   fontsize=8)
        plt.title("Time Analysis - " + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + '\n' +
                  'Average Time per Frame ' + str(np.around(average_time, decimals=3)) + 's')

    else:
        labels = ['Preprocessing', 'Feature Extraction', 'Algorithm', 'Classification', 'Other', 'Ego-Motion-Estimation',
                  'Ego-Motion-Compensation']
        time_mean = np.delete(time_mean, 2, 0)

        porcent = 100. * time_mean / time_mean.sum()

        patches, texts = plt.pie(time_mean, startangle=90, radius=1.2)
        labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(labels, porcent)]

        plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1.04, 0.5),
                   fontsize=8)
        plt.title("Time Analysis - " + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + ' - EMC' + '\n' +
                  'Average Time per Frame ' + str(np.around(average_time, decimals=3)) + 's')

    if switch == 1:
        plt.savefig("plots/TA_" + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + ".png",
                    dpi=300, bbox_inches="tight")
    else:
        plt.savefig("plots/TA_" + str(file[0]) + " - " + str(file[1]) + " - " + str(file[2]) + " - EMC.png",
                    dpi=300, bbox_inches="tight")
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    """
