import streamlit as st
import cv2
import math
from PIL import Image
import numpy as np

# イメージロードし、カラー、グレイスケール、二値画像
# 返却値 : カラー(img),グレイスケール(gray),二値画像(bw)
def supplyImage(imageFile):
    img = np.array(Image.open(imageFile))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bw = cv2.bitwise_not(bw)
    # モルフォロジー変換
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel5x5, iterations=2)
    return  img, gray, bw

class WidthScanner:
    def __init__(self, coef, mx, my):
        self._imx = int(mx)
        self._imy = int(my)
        self.delta = -1 / coef[0]
        self.b = my - mx * self.delta

    def getX(self, y):
        x = (y - self.b) / self.delta
        return x

    def getPoints(self, binimg):
        h, w = binimg.shape
        x1 = 0
        y1 = 0
        for y in range(self._imy, 0, -1):
            x = int(self.getX(y))
            if binimg[y, x] == 0:
                break
            else:
                x1 = x
                y1 = y

        x2 = 0
        y2 = 0
        for y in range(self._imy, h, 1):
            x = int(self.getX(y))
            if binimg[y, x] == 0:
                break
            else:
                x2 = x
                y2 = y
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return (x1, y1), (x2, y2), dist
    
def projectionPCA(xpts, ypts):
    data_pts = np.empty((xpts.shape[0], 2), dtype=np.float64)
    data_pts[:, 0] = xpts
    data_pts[:, 1] = ypts
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    compressed_data = cv2.PCAProject(data_pts, mean, eigenvectors)
    rv = np.polyfit(compressed_data[:, 0], compressed_data[:, 1], 3)
    a, b, c, d = rv
    expr = np.poly1d(rv)
    minx = np.min(compressed_data[:, 0])
    maxx = np.max(compressed_data[:, 0])
    total_curve_length = 0
    prev_x, prev_y = minx, expr(minx)
    for x in np.linspace(minx, maxx, 1000):
        y = expr(x)
        if x != minx:
            total_curve_length += np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        prev_x, prev_y = x, y
    polynomial_y = expr(compressed_data[:, 0])
    mean_y = (np.max(polynomial_y) + np.min(polynomial_y)) / 2
    distances = polynomial_y - mean_y
    max_curve_dist = np.max(distances)
    min_curve_dist = np.min(distances)
    first_point_dist = distances[0]
    last_point_dist = distances[-1]
    if first_point_dist > 0 and last_point_dist > 0:
        val1 = abs(min_curve_dist)
        val2 = abs(first_point_dist) + abs(last_point_dist)
    elif first_point_dist < 0 and last_point_dist < 0:
        val1 = abs(max_curve_dist)
        val2 = abs(first_point_dist) + abs(last_point_dist)
    else:
        val1 = abs(mean_y)
        val2 = abs(first_point_dist) + abs(last_point_dist)
    val = val1 + val2
    return a, b, c, d, total_curve_length, val, val1, val2, compressed_data, polynomial_y
    
def main(image):
    img, gray, bw = supplyImage(image)
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(bw)
    target_lb_id = 1
    for t in range(1, nLabels):
        if data[t, 4] > 1000:
            target_lb_id = t
            break
    tobjx = data[target_lb_id, 0]
    tobjw = data[target_lb_id, 2]
    bres = np.zeros_like(bw)
    bres[labelImages == target_lb_id] = 255
    ypts, xpts = np.where(bres == 255)
    rv = np.polyfit(xpts, ypts, 3)
    expr = np.poly1d(rv)
    total_len = 0
    scan_flag = False
    vy = expr(tobjx)
    while bres[int(vy), tobjx] != 255 and tobjx < tobjx + tobjw:
        vy = expr(tobjx)
        tobjx += 1
    for x in range(tobjx, tobjx + tobjw):
        if not scan_flag and bres[int(vy), x] == 255:
            scan_flag = True
            vy = expr(x)
            bx, by = x, vy
        elif scan_flag:
            if bres[int(vy), x] == 255:
                vy = expr(x)
                sdist = math.sqrt((bx - x) ** 2 + (by - vy) ** 2)
                total_len += sdist
                bx, by = x, vy
            else:
                break
    a, b, c, d, total_curve_length, val, val1, val2, compressed_data, polynomial_y = projectionPCA(xpts, ypts)
    straight_ratio_curve_length = total_curve_length / tobjw
    return straight_ratio_curve_length
    
st.subheader("閾値を変えられる分類モデル")
st.write("---")
lst =  ['21.jpg', '29.jpg', '26.jpg', '14.jpg', '27.jpg', '12.jpg', '52.jpg', '24.jpg', '25.jpg', '11.jpg', '28.jpg', '55.jpg', '18.jpg', '54.jpg', '17.jpg', '23.jpg', '119.jpg', '99.jpg', '111.jpg', '92.jpg'] 
value_dict = {
    "21.jpg": 0.998915589,
    "29.jpg": 1.003953426,
    "26.jpg": 1.013938291,
    "14.jpg": 1.016510942,
    "27.jpg": 1.020197185,
    "12.jpg": 1.021105115,
    "52.jpg": 1.022290982,
    "24.jpg": 1.023274646,
    "25.jpg": 1.029273638,
    "11.jpg": 1.032448549,   
    "28.jpg": 1.03740688,
    "55.jpg": 1.046257309,
    "18.jpg": 1.062940948,
    "54.jpg": 1.07263839,
    "17.jpg": 1.096459305,
    "23.jpg": 1.103308259,
    "119.jpg": 1.150363466,
    "99.jpg": 1.168651699,
    "111.jpg": 1.199886481,
    "92.jpg": 1.217907459 
}
 
num_items = len(lst)
lcol=[]
col= st.columns(4)
if 'value' not in st.session_state:
        st.session_state.value = -1
if 'hoge' not in st.session_state:
        st.session_state.hoge = 0
def syasinhyouzi():
    if st.button("図{}".format(i+1), key=i+1):
        st.write("ここから悪品") 
        st.session_state.value = value_dict.get(lst[i])
        st.session_state.hoge = i+1
    st.image(lst[i], use_column_width=True)
for i in list(range(0,num_items,4)):
    with col[0]:
        syasinhyouzi()            
for i in list(range(1,num_items,4)):
    with col[1]:
        syasinhyouzi()             
for i in list(range(2,num_items,4)):
    with col[2]:
        syasinhyouzi() 
for i in list(range(3,num_items,4)):
    with col[3]:
        syasinhyouzi()

if st.session_state.value < 0:
    st.sidebar.title("閾値の設定")
    st.sidebar.write("閾値を設定していません")
else:
    st.sidebar.title("閾値の設定") 
    st.sidebar.write("図{}".format(st.session_state.hoge)+"から悪品") 
    st.sidebar.write(f"閾値の値: {st.session_state.value}") 
    
if __name__ == '__main__':  
    st.sidebar.write("---")
    st.sidebar.title("画像の取得") 
    picture = st.sidebar.camera_input("Take a picture")
    st.sidebar.write("---")
    st.sidebar.title("画像の解析") 
    st.sidebar.write("評価したい画像：")
    if picture:
        st.sidebar.image(picture, use_column_width=True)
        straight_ratio_curve_length = main(picture)
        #rounded_straight_ratio_curve_length = round(straight_ratio_curve_length, 8)
        st.sidebar.write("評価したい値：")
        st.sidebar.write(f"{straight_ratio_curve_length}")
        st.sidebar.write("---")
        st.sidebar.title("評価の結果") 
        if st.session_state.value < 0:
            st.sidebar.title("閾値を設定してください") 
        else:
            if straight_ratio_curve_length<=st.session_state.value:   
                st.sidebar.image("良品.jpg", use_column_width=True)
            else:
                st.sidebar.image("悪品.jpg", use_column_width=True)
    else:
        st.sidebar.write("画像が存在しません")       

  
    