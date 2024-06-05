import cv2 as cv
import torch
import numpy as np

KP_NOSE = 3
KP_SPINE_SHOULDER = 20
KP_LEFT_ELLBOW = 5
KP_RIGHT_ELLBOW = 8
KP_LEFT_WRIST = 6
KP_RIGHT_WRIST = 10
KP_LEFT_SHOULDER = 4
KP_RIGHT_SHOULDER = 8
KP_LEFT_HIP = 12
KP_RIGHT_HIP = 16
KP_LEFT_KNEE = 13
KP_RIGHT_KNEE = 17
KP_LEFT_ANKLE = 14
KP_RIGHT_ANKLE = 18

#region helper drawing methods
def __center__(p1, p2):
    c = torch.zeros((3))
    c[0] = (p1[0] + 0.5 * (p2[0] - p1[0]))
    c[1] = (p1[1] + 0.5 * (p2[1] - p1[1]))
    c[2] = (p1[2] + 0.5 * (p2[2] - p1[2]))

    return c

def __draw_line__(img, kpts, i1, i2):
    if kpts[i1, 0].item() == 0 and kpts[i1, 1].item() == 0 and kpts[i2, 0].item() == 0 and kpts[i2, 1].item() == 0:
        return img
    
    p1x = int(img.shape[0] / 2 + kpts[i1, 0].item() * 100)
    p1y = int(img.shape[1] / 2 - kpts[i1, 1].item() * 100)
    p2x = int(img.shape[0] / 2 + kpts[i2, 0].item() * 100)
    p2y = int(img.shape[1] / 2 - kpts[i2, 1].item() * 100)
    
    img = cv.line(img, (p1x, p1y), (p2x, p2y), color=(0, 0, 0), thickness=2)
    return img

def __draw_skeleton__(img, subject):
    # body
    img = __draw_line__(img, subject, KP_LEFT_SHOULDER, KP_SPINE_SHOULDER)
    img = __draw_line__(img, subject, KP_RIGHT_SHOULDER, KP_SPINE_SHOULDER)
    img = __draw_line__(img, subject, KP_RIGHT_SHOULDER, KP_RIGHT_HIP)
    img = __draw_line__(img, subject, KP_LEFT_SHOULDER, KP_LEFT_HIP)
    img = __draw_line__(img, subject, KP_LEFT_HIP, KP_RIGHT_HIP)

    # left arm
    img = __draw_line__(img, subject, KP_LEFT_SHOULDER, KP_LEFT_ELLBOW)
    img = __draw_line__(img, subject, KP_LEFT_ELLBOW, KP_LEFT_WRIST)

    # left arm
    img = __draw_line__(img, subject, KP_RIGHT_SHOULDER, KP_RIGHT_ELLBOW)
    img = __draw_line__(img, subject, KP_RIGHT_ELLBOW, KP_RIGHT_WRIST)

    # left leg
    img = __draw_line__(img, subject, KP_LEFT_HIP, KP_LEFT_KNEE)
    img = __draw_line__(img, subject, KP_LEFT_KNEE, KP_LEFT_ANKLE)

    # right leg
    img = __draw_line__(img, subject, KP_RIGHT_HIP, KP_RIGHT_KNEE)
    img = __draw_line__(img, subject, KP_RIGHT_KNEE, KP_RIGHT_ANKLE)

    # head
    img = __draw_line__(img, subject, KP_SPINE_SHOULDER, KP_NOSE)
    
    return img
#endregion

def show_skeleton_sequences(X:torch.Tensor, T:torch.Tensor):
    # X.shape: (N, num_frames, num_joints, xyz) => (batch_size, default=64, 75, 3)
    # T.shape: (N, num_frames, num_classes) => (batch_size, default=64, default=64)

    for x, t in zip(X, T):
        for frame_x, frame_t in zip(x, t):
            img = np.ones((300, 300, 3))
            subjects = [frame_x[0:25], frame_x[25:50], frame_x[50:75]]

            for subject in subjects:
                img = __draw_skeleton__(img, subject)
            # for i, joint in enumerate(subject1):
            #     x = joint[0].item()
            #     y = joint[1].item()
            #     z = joint[2].item()
            #     pass

            cv.imshow('skeleton', img)
            cv.waitKey(30)