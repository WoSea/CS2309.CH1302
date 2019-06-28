# Import các thư viện
import math
from sklearn import neighbors 
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

# Danh sách các đuôi mở rộng cho phép của tệp hình ảnh
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Định nghĩa hàm Train(), trả về một model chứa kết quả huấn luyện cho KNN trên tập dự liệu trong folder train
def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):

    X = [] # mảng chứa các khuôn mặt để huấn luyện
    y = [] # mảng chứa các label tương ứng
    # Vòng lặp for duyệt các hình ảnh khuôn mặt trong tập huấn luyện
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue 
        
        # Với mỗi thư mục trên tạo một vòng lặp duyệt các file hình ảnh trong đó
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path) 
            face_bounding_boxes = face_recognition.face_locations(image)


            if len(face_bounding_boxes) != 1:
                #Nếu không có hoặc có quá nhiều khuôn mặt trong hình ảnh test sẽ không hợp lệ và thực hiện skip
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # mã hóa khuôn mặt và đưa vào tập huấn huyện (mảng X)
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir) 
    # Xác định bao nhiêu neighbors trong KNN classifier,  
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Gọi hàm neighbors.KneighborsClassifier để tạo và tập huấn cho KNN classifier trên tập huấn luyện X, truyền vào trọng số n_neighbors đã tính, tham số algorithm và weights
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

     
    # save model đã train
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)  
    # lưu kết quả huấn luyện của KNN classifier => file trained_knn_model.clf
    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):

    # Kiểm tra đường dẫn hình ảnh  
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
        
    # Yêu cầu truyền vào model knn_clf đã train trước đó  (kết quả hàm train)
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path") 
        
    # Load KNN model đã train (trained_knn_model.clf)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load hình ảnh cần nhận dạng và tìm các vị trí có khuôn mặt  
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # Trả về 0 nếu không tìm ra các khuôn mặt 
    if len(X_face_locations) == 0:
        return []

    # Encoding cho các khuôn mặt của ảnh test
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Gọi hàm kneighbors để tìm kết quả phù match tốt từ model knn_clf cho các khuôn mặt cần nhận dạng.
    # Sau đó chọn ra các kết quả nằm trong ngưỡng closest_distances
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Kết quả sẽ trả về bounding box khuôn mặt và Label tương ứng. Những khuôn mặt không xác định sẽ được gán 'unknown'
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)] 

 # Hàm hiển thị kết quả, vị trí khuôn mặt tìm được và label tên tương ứng
def show_prediction_labels_on_image(image_file, predictions):
    
    #Sử dụng thư viện Image và ImageDraw để vẽ label và box lên hình
    img_path = os.path.join("C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/images/test", image_file)
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    # Vòng lặp: với mỗi tên label, tọa độ vị trí khuôn mặt lấy được từ hàm predict, ta vẽ bounding box quanh khuôn mặt lên hình
    for name, (top, right, bottom, left) in predictions:
        # vẽ bounding box cho khuôn mặt sử dụng Pillow module 
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

         # Do có lỗi trong Pillow nên cần encode UTF-8
        name = name.encode("UTF-8")

        # Vẽ thêm Label bên dưới bounding box
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    # Xóa drawing library từ memory
    del draw

    # Hiển thị kết quả
    pil_image.show()

    # lưu vào thư mục result
    pil_image.save(os.path.join("C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/images/result", image_file))


 # Thực thi các cài đặt, hiển thị thông báo kết quả
if __name__ == "__main__":
     
    # Bước 1: Train the KNN classifier và save vào thư mục project 
    print("Training KNN classifier...")
    classifier = train("C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/images/train", model_save_path="C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/trained_knn_model.clf", n_neighbors=2)
    print("Training complete!\n\n")

   # STEP 2: Sử dụng model đã train, chạy chuẩn đoán cho các hình ảnh test  
    for image_file in os.listdir("C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/images/test"):
        full_file_path = os.path.join("C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/images/test", image_file)

        print("Looking for faces in {}".format(image_file))
 
        # Tìm các khuôn mặt trong hình test 
        predictions = predict(full_file_path, model_path="C:/Users/duyph/PYTHONSOURCECODE/FACERECOGNITIONMINI-CHALLENGE/trained_knn_model.clf")

        # hiển thị thông báo ra console
        count = 0
        for name, (top, right, bottom, left) in predictions:
            print("Found {}".format(name))
            if name != "unknown":
                count += 1 
        print("Recognized {}/{} faces found".format(count, len(predictions)))
        print("")

         # Hiển thị thông tin trên ảnh test, kết quả ảnh đã test trong thư mục result
        show_prediction_labels_on_image(image_file, predictions)