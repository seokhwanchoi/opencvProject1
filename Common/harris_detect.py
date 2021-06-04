import numpy as np, cv2

def cornerHarris(image, ksize, k):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize)   # 세 번째 인수 1-> x방향 미분 -> 수직 마스크, 수직 방향 에지 검출
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize)   # 네 번째 인수 1-> y방향 미분 -> 수평 마스크, 수평 방향 에지 검출

    a = cv2.GaussianBlur(dx * dx, (5, 5), 0)                     # 가우시안 블러링 수행
    b = cv2.GaussianBlur(dy * dy, (5, 5), 0)
    c = cv2.GaussianBlur(dx * dy, (5, 5), 0)
    
    corner = (a * b - c * c) - k * (a + b) ** 2        # 코너 응답 함수 계산 -행렬 연산 적용
    return corner

def drawCorner(corner, image, thresh): #임계값 이상 코너 표시 
    cnt = 0
    corner = cv2.normalize(corner, 0, 100, cv2.NORM_MINMAX) #정규화   https://deep-learning-study.tistory.com/121
    #  최솟값이 0이고 최댓값이 100을 갖도록 히스토그램 스트레칭
    corners = []
    for i in range(1, corner.shape[0]-1 ):  #비최대치 억제 : 현재 화소가 이웃하는 화소들보다 크면 에지로 보존하고, 그렇지 않으면 에지가 아닌 것으로 간주해 제거하는 것
        for j in range(1, corner.shape[1]-1 ):
            neighbor = corner[i-1:i+2, j-1:j+2].flatten()      #이웃 화소 가져옴
            max = np.max(neighbor[1::2])                       #상하좌우 값만
            if thresh < corner[i, j] > max : corners.append((j,i)) #코너 확정 좌표 저장

    for pt in corners:                         #코너 확정 좌표 순회
        cv2.circle(image, pt, 3, (0, 230, 0), -1)    # 좌표 표시
    print("임계값: %2d , 코너 개수: %2d" %(thresh, len(corners)) )
    return image

def onCornerHarris(thresh):
    img1 = drawCorner(corner1, np.copy(image), thresh)
    img2 = drawCorner(corner2, np.copy(image), thresh)


    dst = cv2.repeat(img1, 1, 2)                            # 두 개 영상을 하나의 윈도우에 표시
    dst[:, img1.shape[1]:, :] = img2
    cv2.imshow("harris detect", dst)

image = cv2.imread('florence-5208579_640.jpg', cv2.IMREAD_COLOR)
if image is None: raise Exception("영상 파일 읽기 에러")

blockSize = 4                                                # 이웃 화소 범위
apertureSize = 3                                             # 소벨 마스크 크기
k = 0.04
thresh = 2                                                   # 코너 응답 임계값
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corner1 = cornerHarris(gray, apertureSize, k)                # 사용자 정의 함수
corner2 = cv2.cornerHarris(gray, blockSize, apertureSize, k) # OpenCV 제공 함수
# gray : 코너 검출 대상 그레이스케일 이미지(float32타입이어야)  ///  blockSize : 코너 검출을 위해 고려할 이웃 픽셀의 범위
# apertureSize : 소벨 미분에 사용된 인자 값   /// k : 해리스 코너 검출 수학식 R에서 k값
# https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=samsjang&logNo=220637582963

onCornerHarris(thresh)
cv2.createTrackbar("Threshold", "harris detect", thresh, 20, onCornerHarris)
cv2.waitKey(0)
