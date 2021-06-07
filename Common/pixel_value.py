
import cv2

red=(0,0,255)
white=(255,255,255)

image = cv2.imread('./opencv_server/2021-06-07_10_20_08.jpg', cv2.IMREAD_GRAYSCALE) # 영상 읽기
if image is None: raise Exception("영상 파일 읽기 오류")

print(image.shape)
img=cv2.Laplacian(image, -1, ksize=5)
#https://dsbook.tistory.com/205


(x,y),(w,h) = (0, 0), (image.shape[1], image.shape[0])                   # 좌표는 x, y
roi_img = img[y:y+h, x:x+w]                   # 행렬 접근은 y, x

#print(“[roi_img] =\n”, roi_img) # 행렬 원소 바로 출력 가능




sum=0
print("[roi_img] =")
for row in roi_img:                    # 원소 순회 방식 출력
    for p in row:
        #print("%4d" % p, end="")       # 순회 원소 하나씩 출력
        sum += p
    #print()
print(sum)
print(int(sum)/(image.shape[0]*image.shape[1]))

cv2.rectangle(img,(40,40,200,30),white,cv2.FILLED)
cv2.putText(img, 'sum : '+str(round(int(sum)/(image.shape[0]*image.shape[1]),4)), (60,60), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,red,2)
#print(type((int(sum))))
#print('gray의 총합/면적 : ', bin(int(sum)/(588*496)))

cv2.rectangle(img, (x,y, w,h), 255, 1) # 관심 영역에 사각형 표시
cv2.imshow("image", img)
cv2.waitKey(0)
