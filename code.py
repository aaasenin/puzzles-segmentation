from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2 # use old opencv, last version lacks SIFT, ex: pip install opencv-python==3.4.2.17
from PIL import Image, ImageEnhance

image_dir = 'your image directory'
train_image_dir = 'train image directory'

def is_outlier(points, thresh=0.1):
    maximum = np.max(points)
    
    return points / maximum < thresh

image = io.imread(image_dir)


# image preprocessing for fore/backgroung building
image = cv2.GaussianBlur(image, (351, 351), cv2.BORDER_DEFAULT)

clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  
l, a, b = cv2.split(lab)  
l2 = clahe.apply(l)  # apply CLAHE to the L-channel
lab = cv2.merge((l2, a, b))
image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# building fore/backgroung markers
dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
_, dist = cv2.threshold(dist, 0.25, 1.0, cv2.THRESH_BINARY)
dist = np.array(dist).astype(np.uint8)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dist, None, None, 
                                                                     None, 8, cv2.CV_32S)
areas = stats[1:, cv2.CC_STAT_AREA]
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if areas[i] >= 1000:  
        result[labels == i + 1] = 255

kernel = np.ones((20, 20),np.uint8)
result = cv2.dilate(result, kernel, iterations = 3)
kernel = np.ones((5, 5),np.uint8)
result = cv2.erode(result, kernel, iterations = 15)
kernel = np.ones((20, 20),np.uint8)
background = cv2.dilate(result, kernel, iterations = 20)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(result, None, None, 
                                                                     None, 8, cv2.CV_32S)
areas = stats[1:,cv2.CC_STAT_AREA]

outliers_mask = is_outlier(areas, thresh=0.1)
result = np.zeros((labels.shape), np.uint8)

for i in range(0, nlabels - 1):
    if not outliers_mask[i]:  
        result[labels == i + 1] = 255

unknown = cv2.subtract(background, result)
ret, markers = cv2.connectedComponents(result)
markers = markers + 1
markers[unknown == 255] = 0


# image preprocessing for watershed
image = io.imread(image_dir)

im = Image.fromarray(image)
enhancer = ImageEnhance.Contrast(im)
enhanced_im = enhancer.enhance(3.5)
image = np.array(enhanced_im)

image = cv2.medianBlur(image, 7)


# watershed
markers = cv2.watershed(image,markers)
mask = (markers != 1).astype(np.uint8)


def crop_image(img, mask=None):   
    if mask is None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = (gray_img != 0).astype(np.uint8)
    x, y, w, h = cv2.boundingRect(mask)
    
    return (img[y : y + h + 1, x : x + w + 1, :]).astype(np.uint8)


train_image = io.imread(train_image_dir)


# train_image preproc
im = Image.fromarray(train_image)
enhancer = ImageEnhance.Sharpness(im)
enhanced_im = enhancer.enhance(5)
train_image = np.array(enhanced_im)

train_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
train_gray = (train_gray != 0).astype(np.uint8)
train_gray = cv2.medianBlur(train_gray, 5)
kernel = np.ones((5,5),np.uint8)
train_gray = cv2.dilate(train_gray, kernel, iterations = 1)


# building train_markers and train_answers
train_ret, train_markers = cv2.connectedComponents(train_gray)
train_markers = train_markers + 1

train_idxes = [1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 12, 13, 14, 15, 11, 16, 18, 20, 17, 19]
train_masks = []
train_answers = ['P1B1', 'P2B2', 'P2B1', 'P1B2', 'P2B2', 
                 'P1B2', 'P1B2', 'P0B2', 'P2B1', 'P1B2',
                 'P1B1', 'P3B0', 'P1B3', 'P3B1', 'P2B1',
                 'P1B3', 'P1B1', 'P2B2', 'P2B1', 'P2B1']

for comp_idx in np.unique(train_markers):
    if comp_idx != 1:
        mask = (train_markers == comp_idx).astype(np.uint8)
        bin_mask = (mask != 0).astype(np.uint8)
        train_masks.append(bin_mask)


def compare_images(idxed_image, train_image, ratio_thresh):
# comparing images based on number of matched sift key points
# both idxed and train images should be cropped and masked

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=100000)
    kp_1, desc_1 = sift.detectAndCompute(idxed_image, None)
    kp_2, desc_2 = sift.detectAndCompute(train_image, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(desc_1, desc_2, 2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    
    decision_array = np.zeros(len(train_masks))

    for match in good_matches:
        x, y = kp_2[match.trainIdx].pt

        for bin_mask_idx in range(len(train_masks)):
            x_m, y_m, w_m, h_m = cv2.boundingRect(train_masks[bin_mask_idx])

            if x >= x_m and x <= x_m + w_m and y >= y_m and y <= y_m + h_m:
                decision_array[train_idxes[bin_mask_idx] - 1] += 1
                break
    
    sorted_decision_array = sorted(decision_array)
    if sorted_decision_array[-1] != sorted_decision_array[-2]:
        return np.argmax(decision_array) + 1
    else:
        # ambiguous answer, try one more time recursively
        return compare_images(idxed_image, train_image, ratio_thresh)


image = io.imread(image_dir)

# image preproc 
im = Image.fromarray(image)
enhancer = ImageEnhance.Sharpness(im)
enhanced_im = enhancer.enhance(5)
image = np.array(enhanced_im)

image = np.ascontiguousarray(image, dtype=np.uint8)

# several iterations to make algorithm answers more robust
num_iter = 1

for comp_idx in np.unique(markers):
    if comp_idx != -1 and comp_idx != 1:
        mask = (markers == comp_idx).astype(np.uint8)
        bin_mask = (mask != 0).astype(np.uint8)
        
        idxed_image = cv2.bitwise_and(image,image,mask=mask)
        idxed_image = crop_image(idxed_image, bin_mask)
            
        iter_ans_arr = []
        for _ in range(num_iter):
            iter_res = compare_images(idxed_image, train_image, 0.7)
            iter_ans_arr.append(iter_res)
          
        curr_ans = np.bincount(iter_ans_arr).argmax()
        
        text_x, text_y, text_w, text_h = cv2.boundingRect(bin_mask)
        image = cv2.putText(image, 
                            train_answers[curr_ans - 1], 
                            (text_x + text_w // 2 - 75, text_y + text_h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            2, (255, 255, 255), thickness=10)
        image = cv2.rectangle(image, 
                              (text_x, text_y), (text_x + text_w, text_y + text_h), 
                              (255, 255, 255), thickness=8) 

plt.figure(figsize=(20,20))
plt.imshow(image, cmap='gray')
